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
