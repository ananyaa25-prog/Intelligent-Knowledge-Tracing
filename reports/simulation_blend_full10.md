# Policy simulation (T6)

Generated 2026-10-09 06:08 UTC by `scripts/simulate.py` (seed 2026, 150 paired synthetic students per policy, 43s). Session mode full10, topic selection evidence_blend; level rules as in adaptive_session.py (revised D27).

> **Scope.** Synthetic students are NOT drawn from the model's training distribution (ASSISTments). This validates the decision logic (which level gets served, how statuses are assigned) only. It says nothing about real learning gains and is not a test of knowledge-tracing accuracy.

Student model: g ~ N(0, 0.8); a_t = g + N(0, 0.7); P_true = 0.05 + 0.95 * sigmoid(a_t - b), b = easy -1 / medium 0 / hard +1. The real served predictor (`best_full_gated.pt`) drives topic choice and (for POLICY=model) the level. Profile mode: profile_reliable = true. Policy parameters were fixed before running (TARGET_P = 0.7, T_PROF = 0.7, T_READY = 0.6, MIN_ANSWERS = 2).

| Metric | random | staircase | model |
|---|---|---|---|
| 1. In-band share (true P in [0.55, 0.85]; excl. 2 warm-ups) | 37.2% (95% CI 34.2%-40.3%) | 41.6% (95% CI 38.1%-45.0%) | 44.9% (95% CI 41.0%-48.8%) |
| 2. Status accuracy (exact) | 49.4% (95% CI 46.6%-55.8%) | 52.8% (95% CI 50.2%-59.1%) | 50.2% (95% CI 46.8%-56.3%) |
| 2b. Status within one step | 94.2% | 93.7% | 93.6% |
| 2c. Coverage (topics with a status / 10) | 42.8% | 39.5% | 42.1% |
| 3. Answers per topic before first status | 2.00 | 2.00 | 2.00 |
| 4. Ready-for-next precision (flags truly ready) | 59.6% of 57 flags | 51.9% of 54 flags | 54.7% of 64 flags |
| Level mix served (easy / medium / hard) | 34%/34%/32% | 23%/62%/14% | 22%/74%/4% |
| Model-policy fallbacks to staircase | - | - | 0 |

Paired difference model - staircase (per student, 95% bootstrap CI): in-band share +0.033 (+0.017 to +0.049); status accuracy -0.030 (-0.069 to +0.007).

## Gate S (pre-registered)

Model policy is used for the demo only if Gate M passes (profile_reliable = true) AND model in-band share >= staircase - 0.02 (44.9% vs 39.6%) AND model status accuracy >= staircase - 0.03 (50.2% vs 49.8%).

**Result: Gate S PASSES.**

Metric 3 note: with the pre-registered rule a topic receives a status as soon as it has MIN_ANSWERS = 2 answers, so metric 3 equals 2 by construction; what differs between policies is coverage (2c), i.e. how many topics reach that many answers.

![in-band](simulation_inband_blend_full10.png)

![metrics](simulation_status_blend_full10.png)
