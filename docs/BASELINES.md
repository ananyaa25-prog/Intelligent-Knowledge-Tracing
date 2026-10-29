# Baselines (v3)

All baselines live in `src/baselines/` and share one interface (`src/baselines/base.py`):

```python
model = make_classical("bkt", num_skills=123, seed=s).fit(train_split)
p = model.predict(history, next_skill, next_difficulty)   # P(correct) in [0, 1]
```

- `train_split` is a split dict from `data/processed/windowed_data.pkl` (or a list of
  per-student sequences); `history` is a list of `(skill, correct, difficulty)` tuples or
  dicts with internal skill ids 1..123 and difficulty buckets 0 unknown / 1 easy / 2 medium / 3 hard.
- Same split, targets and buckets as the main model: every model predicts each position
  t >= 1 of a test student from that student's history before t, exactly the targets of
  `src/train.py` `run_epoch` (`correct[:, 1:]` under `mask[:, 1:]`, sequences truncated at 500).
- Difficulty buckets: on the official split the stored train-derived buckets are used. For the
  re-split protocol of Stage 4 they are re-derived from that seed's train students
  (`derive_difficulty_from_sequences`, same rule as `src/dataset.py`). On the official split the
  re-derived buckets agree with the stored ones on 86.3% of train interactions; the difference is
  because the stored buckets were computed from the full CSV and the tensors are truncated at 500
  steps per student (plus tie order). This is recorded, not hidden.
- Tests: `tests/test_baselines_classical.py` (range, shape, seed determinism, empty history,
  online path == prefix path, EM log-likelihood non-decreasing, metrics).

## Classical baselines (Stage 2)

| Key | File | Model | Uses difficulty bucket | Seed-dependent |
|---|---|---|---|---|
| `majority` | majority.py | constant = train correctness rate (AUC 0.5 by construction) | no | no |
| `skill_mean` | majority.py | train correctness of (skill, bucket), m-estimate smoothing (m=5) to skill and global rate | yes | no |
| `logreg_counts` | logistic.py | logistic regression on log1p counts (correct / wrong on target skill and overall) + one-hot skill + one-hot bucket | yes | no |
| `pfa` | pfa.py | Performance Factors Analysis: per-skill intercept and per-skill weights on raw prior success / failure counts (x0.1 for conditioning) + bucket term | yes | no |
| `elo` | elo.py | Elo rating: sigmoid(theta + theta_skill - b_(skill,bucket)), step A/(1+Bn) with A=1, B=0.05; item ratings learnt on train (student order shuffled by seed), frozen at test | yes | yes (order) |
| `bkt` | bkt.py | Bayesian Knowledge Tracing, own EM (Baum-Welch, 25 iterations max), per-skill L0, T, G, S; G, S <= 0.3; no forgetting | no (classic) | yes (EM init) |
| `bkt_diff` | bkt.py | BKT with guess/slip per (skill, bucket), in the spirit of KT-IDEM | yes | yes (EM init) |

Smoke-run on the official split (seed 1, from `reports/v3/benchmark_classical.md` once Stage 4 is run;
numbers there are authoritative).

## References (verify before citing page numbers)

- Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4).
- Pavlik, P. I., Cen, H., & Koedinger, K. R. (2009). Performance Factors Analysis - A new alternative to knowledge tracing. *AIED 2009*.
- Pardos, Z. A., & Heffernan, N. T. (2011). KT-IDEM: Introducing item difficulty to the knowledge tracing model. *UMAP 2011*.
- Pelánek, R. (2016). Applications of the Elo rating system in adaptive educational systems. *Computers & Education*, 98.

## Deep baselines and our variants (Stage 3)

Shared trainer `src/baselines/deep_common.py`: Adam 1e-3, batch 32, masked BCE, grad clip 5,
best-validation-AUC weights kept in memory, seeded init and batch order. Batches group
students of similar length (`length_bucketed_batches`): same data and number of updates as
random batches, much less padding (about 2-5x faster on CPU). Every network maps
`(skill, correct, mask, rt, attempt, hint, difficulty) -> P for positions 1..L-1`, the same
contract as `GatedKTModel.forward`, so all models are trained and scored on identical targets.
Hidden size 64 everywhere (same as GatedKT). Checkpoints, if saved, go to `models_v3/` only
(`DeepKT.save` refuses `data/processed/`).

| Key | File | Model | Difficulty bucket |
|---|---|---|---|
| `dkt_classic` | dkt.py | DKT (Piech et al., 2015) as published: emb(skill, correct) -> GRU -> one output unit per skill | no |
| `dkt_gru` | dkt.py | DKT GRU; input also gets emb(bucket); head on [h_t, emb(next skill)+emb(next bucket)] | yes |
| `dkt_lstm` | dkt.py | same with LSTM | yes |
| `dkvmn_lite` | dkvmn.py | simplified DKVMN (Zhang et al., 2017): 20 key/value slots, erase/add write, read before each answer | yes |
| `sakt_lite` | sakt.py | simplified SAKT (Pandey & Karypis, 2019): 1 attention block, 4 heads, causal mask, learned positions | yes |
| `full_gated` | ours.py | our GatedKT (graph + gate), unchanged `src/model.py` | yes |
| `graph_no_gate` | ours.py | ours, graph kept, gate replaced by averaging (= "no gate" ablation) | yes |
| `dkt_only` | ours.py | ours without graph and gate ("DKT+difficulty" variant of v2) | yes |
| `no_graph` | ours.py | ours with gate, graph context = zeros | yes |
| `no_metadata` | ours.py | ours with response time / attempts / hints replaced by constants (train-mean RT, 1 attempt, no hint) | yes |
| `no_difficulty` | ours.py | ours with all buckets set to 0 (unknown) | no |

"-lite" marks models simplified relative to the published architecture; they are not
re-implementations with the original hyper-parameters and must not be described as such.

`OursVariant.from_checkpoint(cond, seed)` loads the existing v2 checkpoints
`data/processed/best_<cond>_seed<seed>.pt` (cond in full_gated / graph_no_gate / dkt_only,
seeds 42 / 123 / 2024) read-only; a test checks the file hash is unchanged. On the official
test split the seed-42 full_gated checkpoint gives AUC 0.8715 through this wrapper (the v2
report's 3-seed mean was 0.8713), which confirms the wrapper scores the same targets.

Tests: `tests/test_baselines_deep.py` (range, shape, seed determinism, empty history,
predict() == batched path, ablation input transforms, read-only checkpoint loading).

Additional references (verify before citing page numbers):
- Piech, C., et al. (2015). Deep knowledge tracing. *NeurIPS 2015*.
- Zhang, J., Shi, X., King, I., & Yeung, D.-Y. (2017). Dynamic key-value memory networks for knowledge tracing. *WWW 2017*.
- Pandey, S., & Karypis, G. (2019). A self-attentive model for knowledge tracing. *EDM 2019*.
