# Related work: comparison table

"Run here" means the approach (or a simplified re-implementation, marked -lite) was trained and
evaluated in this project on the same split; numbers for those rows are in
reports/v3/stage5/benchmark.md and reports/v3/benchmark_classical.md. All other rows are
**from published descriptions, not re-run**, and none of their published numbers are compared with
ours (different data, preprocessing and splits make such numbers non-comparable).

| Approach | Kind | What it models | Item difficulty | Skill relations | Chooses the next question | Status here |
|---|---|---|---|---|---|---|
| BKT (Corbett & Anderson, 1995) | HMM per skill | known / not known, learn, guess, slip | no (classic); KT-IDEM variant: per-item guess/slip | no | no (tracing only) | run here: `bkt`, `bkt_diff` (own EM) |
| PFA (Pavlik, Cen & Koedinger, 2009) | logistic regression | prior successes / failures per skill | via item/skill intercepts | no | no | run here: `pfa` |
| Elo rating for education (Pelánek, 2016) | online rating | student ability vs item difficulty | yes | no | can target a success rate | run here: `elo` (baseline) and `POLICY=elo` (simulation) |
| DKT (Piech et al., 2015) | RNN (LSTM) | hidden knowledge state | no (classic) | implicit | no | run here: `dkt_classic`, `dkt_gru`, `dkt_lstm` (+bucket input) |
| DKVMN (Zhang et al., 2017) | key-value memory network | concept memory, read/erase/add | no | implicit (key memory) | no | run here as simplified `dkvmn_lite` |
| SAKT (Pandey & Karypis, 2019) | self-attention | attention over past interactions | no | implicit | no | run here as simplified `sakt_lite` |
| AKT (Ghosh, Heffernan & Lan, 2020) | context-aware attention with Rasch-style embeddings | monotonic attention, forgetting | yes (Rasch embeddings) | implicit | no | from published description, not re-run |
| ALEKS (based on Knowledge Space Theory, Doignon & Falmagne) | commercial adaptive system | knowledge state as a set of mastered items | yes | yes (prerequisite structure) | yes | from published descriptions, not re-run; product details: verify before citing |
| Khan Academy-style mastery practice | commercial platform | mastery levels per skill from recent answers | partly | course structure | yes (practice sets) | from public product descriptions, not re-run; no peer-reviewed source checked: verify before citing |
| pyKT benchmark library (Liu et al., 2022) | benchmark protocol | many deep KT models under one protocol | - | - | - | not used; cited as the kind of standardised protocol our benchmark imitates (verify version before citing) |
| **Ours: GatedKT + adaptive session** | GRU short/long-term encoders, metadata MLP, PPMI skill graph (GAT), learned vector gate | next-answer probability per (skill, difficulty bucket) | yes (train-derived buckets) | yes (PPMI co-occurrence graph) | yes (topic by lowest P / evidence blend; level by staircase or policy flag) | this project; evidence in reports/v3/ |

## What the comparison does and does not show

- On ASSISTments 2009 (official split) every row marked "run here" is compared with ours under one
  protocol and paired significance tests (reports/v3/stage5/benchmark.md). Simplified "-lite" models are not
  the published architectures with their tuned hyper-parameters, so a gap to them is not a gap to the
  original papers.
- ALEKS and Khan Academy are full products with content, teachers and years of data; this project is
  a research prototype. The table lists design differences only.
- Our model's inputs include train-derived difficulty buckets per problem. Several of the classical and
  deep baselines were given the same buckets (column "Item difficulty" in docs/BASELINES.md) so that the
  comparison is not just "with vs without difficulty information".

## References (only those we are confident exist; check details before citing)

- Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4, 253-278.
- Pavlik, P. I., Cen, H., & Koedinger, K. R. (2009). Performance factors analysis - A new alternative to knowledge tracing. *AIED 2009*.
- Pardos, Z. A., & Heffernan, N. T. (2011). KT-IDEM: Introducing item difficulty to the knowledge tracing model. *UMAP 2011*.
- Pelánek, R. (2016). Applications of the Elo rating system in adaptive educational systems. *Computers & Education*, 98, 169-179.
- Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. *NeurIPS 2015*.
- Zhang, J., Shi, X., King, I., & Yeung, D.-Y. (2017). Dynamic key-value memory networks for knowledge tracing. *WWW 2017*.
- Pandey, S., & Karypis, G. (2019). A self-attentive model for knowledge tracing. *EDM 2019*.
- Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-aware attentive knowledge tracing. *KDD 2020*.
- Doignon, J.-P., & Falmagne, J.-C. (1985). Spaces for the assessment of knowledge. *International Journal of Man-Machine Studies*, 23. (verify before citing)
- Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., & Luo, W. (2022). pyKT: A Python library to benchmark deep learning based knowledge tracing models. *NeurIPS 2022 Datasets and Benchmarks*. (verify author list before citing)
- Wang, Z., et al. (2020). Instructions and guide for diagnostic questions: The NeurIPS 2020 Education Challenge. arXiv:2007.12061. (source of the Eedi data; verify before citing)
