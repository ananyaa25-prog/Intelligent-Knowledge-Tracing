# Policy simulation (T6)

Generated 2026-10-09 06:09 UTC by `scripts/simulate.py` (seed 2026, 150 paired synthetic students per policy, 39s). Session mode core5, topic selection evidence_blend; level rules as in adaptive_session.py (revised D27).

> **Scope.** Synthetic students are NOT drawn from the model's training distribution (ASSISTments). This validates the decision logic (which level gets served, how statuses are assigned) only. It says nothing about real learning gains and is not a test of knowledge-tracing accuracy.

Student model: g ~ N(0, 0.8); a_t = g + N(0, 0.7); P_true = 0.05 + 0.95 * sigmoid(a_t - b), b = easy -1 / medium 0 / hard +1. The real served predictor (`best_full_gated.pt`) drives topic choice and (for POLICY=model) the level. Profile mode: profile_reliable = true. Policy parameters were fixed before running (TARGET_P = 0.7, T_PROF = 0.7, T_READY = 0.6, MIN_ANSWERS = 2).

| Metric | random | staircase | model |
|---|---|---|---|
| 1. In-band share (true P in [0.55, 0.85]; excl. 2 warm-ups) | 35.2% (95% CI 32.1%-38.3%) | 42.7% (95% CI 39.5%-45.9%) | 50.2% (95% CI 46.7%-53.8%) |
| 2. Status accuracy (exact) | 50.9% (95% CI 47.3%-55.8%) | 47.1% (95% CI 43.1%-52.8%) | 47.6% (95% CI 43.6%-52.5%) |
| 2b. Status within one step | 93.3% | 93.0% | 93.0% |
| 2c. Coverage (topics with a status / 10) | 41.4% | 40.7% | 42.9% |
| 3. Answers per topic before first status | 2.00 | 2.00 | 2.00 |
| 4. Ready-for-next precision (flags truly ready) | 32.3% of 96 flags | 32.6% of 95 flags | 36.4% of 110 flags |
| Level mix served (easy / medium / hard) | 32%/35%/33% | 34%/43%/23% | 42%/51%/7% |
| Model-policy fallbacks to staircase | - | - | 0 |

Paired difference model - staircase (per student, 95% bootstrap CI): in-band share +0.075 (+0.057 to +0.093); status accuracy +0.002 (-0.043 to +0.049).

## Gate S (pre-registered)

Model policy is used for the demo only if Gate M passes (profile_reliable = true) AND model in-band share >= staircase - 0.02 (50.2% vs 40.7%) AND model status accuracy >= staircase - 0.03 (47.6% vs 44.1%).

**Result: Gate S PASSES.**

Metric 3 note: with the pre-registered rule a topic receives a status as soon as it has MIN_ANSWERS = 2 answers, so metric 3 equals 2 by construction; what differs between policies is coverage (2c), i.e. how many topics reach that many answers.

![in-band](simulation_inband_demo_core5_blend.png)

![metrics](simulation_status_demo_core5_blend.png)
