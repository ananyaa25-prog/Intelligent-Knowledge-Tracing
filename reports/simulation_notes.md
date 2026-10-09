# Simulation: interpretation notes

All numbers are copied from the generated reports (seed 2026, 150 paired synthetic students per policy). Synthetic students validate decision logic only.

## History of this evaluation (what changed and why)

| Version | What changed | Gate S | Where |
|---|---|---|---|
| v1 | Original model policy: level closest to 70% among {current, one down, one up} | PASS (in-band 57.6% vs staircase 32.6%) | reports/archive_v1/ |
| v1 + D26 | Evidence-conflict rule in the status logic (dry run) | PASS | reports/archive_v1/ (before/after recorded in the first FINAL_REPORT revision) |
| **v2 (current)** | After user testing: model-policy level rules follow the student's answers on the topic (D27), cautious evidence-conflict rule (D30); simulation re-run | **FAIL** -> demo uses the staircase | reports/simulation.md |

Why v1 looked so good: the v1 model policy served Easy 64% of the time. The predictor's estimates for short quiz histories are mostly below 70% at Medium, so "closest to 70%" chose Easy, even after correct answers. That raised the share of questions in the target band for these synthetic students, but a real tester found it stuck on Easy and never reached Hard. v2 forbids easier levels after a correct answer and steps up after two correct answers. With that, the model policy's in-band advantage shrinks (+25 -> +7.5 points) and its status accuracy is slightly below the staircase's.

## Current results (v2)

**A. Pre-registered comparison (full10, original topic selection)** — reports/simulation.md

| | random | staircase | model |
|---|---|---|---|
| In-band share (true P 0.55-0.85) | 38.5% | 32.6% | 40.1% |
| Exact status accuracy | 46.0% | 51.5% | 48.1% |
| Within one step | 93.0% | 93.2% | 93.9% |
| Coverage | 46.5% | 40.0% | 43.0% |
| Ready-for-next precision | 58.2% (55 flags) | 56.4% (55) | 44.9% (49) |
| Level mix E/M/H | 33/35/32% | 2/59/39% | 4/78/18% |

Paired model - staircase: in-band +0.075 (95% CI +0.058 to +0.091); status accuracy -0.033 (CI -0.080 to +0.019). **Gate S fails** (model status accuracy 48.1% < 51.5% - 3 = 48.5%), so the demo uses POLICY=staircase. Model-guided difficulty did not beat the staircase in simulation; the model still drives topic selection.

**B. Blended topic selection, full10** (D31) — reports/simulation_blend_full10.md

| | random | staircase | model |
|---|---|---|---|
| In-band share | 37.2% | 41.6% | 44.9% |
| Exact status accuracy | 49.4% | 52.8% | 50.2% |
| Coverage | 42.8% | 39.5% | 42.1% |
| Level mix E/M/H | 34/34/32% | 23/62/14% | 22/74/4% |

For the staircase, blending the model prediction with the student's own answers per topic raised in-band share from 32.6% to 41.6% and status accuracy from 51.5% to 52.8%, and let the staircase actually use Easy (23% vs 2%). This supports Rule T (reports/DEMO_CONFIG.md), which was added after user testing, not pre-registered.

**C. The demo configuration (core5 + blended topic selection, with calibration off in the simulator)** — reports/simulation_demo_core5_blend.md

| | random | staircase | model |
|---|---|---|---|
| In-band share | 35.2% | 42.7% | 50.2% |
| Exact status accuracy | 50.9% | 47.1% | 47.6% |
| Ready-for-next precision | 32.3% (96 flags) | 32.6% (95) | 36.4% (110) |
| Level mix E/M/H | 32/35/33% | 34/43/23% | 42/51/7% |

## Observations

1. Status labels are rough for every policy: about 46-53% exact, about 93-94% within one step.
2. **Ready-for-harder flags are weak**, especially in core5 (32-36% truly ready). Treat them as a suggestion to try, not a certification.
3. Topic selection by "lowest predicted success" alone keeps returning to topics the student answered correctly, because the model's per-topic estimates in this quiz are dominated by recent answers on any topic (reports/answer_response.md). Blending in the student's own record (D31) fixes the visible symptom; the underlying model behaviour is unchanged (no retraining).
4. Metric 3 (answers before a status) is 2.00 by construction (MIN_ANSWERS = 2); coverage is the informative quantity.

Run notes: simulation sizes are the specified 150 students per policy (about 70-150 s per run). Earlier runs are archived in reports/archive_v1/.
