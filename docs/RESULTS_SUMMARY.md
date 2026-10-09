# Results summary (one page, for slides)

Every number below is copied from a generated report (file in brackets). Synthetic-student results validate decision logic only.

## Model (unchanged; not retrained)
- Validation AUC, mean of 3 seeds: DKT-only **0.8915**, graph without gate **0.8900**, full gated **0.8950**. The gate adds +0.0035 over DKT; the graph alone did not help. [data/processed/retrain.log]
- Test AUC, mean of 3 seeds: 0.8677 / 0.8657 / 0.8713. [ADAPTIVE_KT_REPORT.md]

## How the prediction reacts to answers [reports/answer_response.md]
- A correct last answer gives a higher prediction than a wrong one in about 95-97% of validation cases (+0.22 on the same skill).
- Repeated correct practice on ASSISTments data raises the prediction (0.67 -> 0.83), tracking the observed data.
- For a quiz topic not yet answered, the estimate is unreliable (about 0.97 at Easy, dropping to a skill-specific value after one answer). This is why the first question on a topic is now Medium and topic choice also uses the student's own record.

## Difficulty awareness (Gate M)
- P(Easy) >= P(Medium) >= P(Hard) violated in **0.0%** of 600 validation cases and **0.0%** of 6,000 quiz-topic cases. -> model-based profile enabled. [reports/monotonicity.md]

## Calibration (Gate C)
- Temperature T = 1.167. On held-out validation students: ECE **0.0249 -> 0.0117** (-53%), Brier 0.1183 -> 0.1176, AUC 0.8946 unchanged. -> calibration enabled in the demo. [reports/calibration.md]

## Difficulty policy (Gate S): 150 paired synthetic students per policy, revised rules [reports/simulation.md]
| | random | staircase | model-guided |
|---|---|---|---|
| Questions in the target band (true P 0.55-0.85) | 38.5% | 32.6% | 40.1% |
| Exact status accuracy | 46.0% | **51.5%** | 48.1% |
| Status within one step | 93.0% | 93.2% | 93.9% |

**Gate S fails** (model status accuracy 48.1% is more than 3 points below the staircase's 51.5%). **Model-guided difficulty did not beat the staircase in simulation; the demo uses the staircase, and the model still drives topic selection.** An earlier version of the model policy passed Gate S (57.6% in-band) but did so by serving Easy 64% of the time; a user test showed it stayed on Easy after correct answers, so its rules were revised. [reports/simulation_notes.md]

## Topic selection (Rule T, added after user testing) [reports/simulation_blend_full10.md]
- Blending the model's prediction with the student's own record on each topic, with the staircase: in-band 32.6% -> **41.6%**, status accuracy 51.5% -> 52.8%.

## Demo configuration [reports/DEMO_CONFIG.md]
`POLICY=staircase`, `APPLY_CALIBRATION=true`, `SESSION_MODE=core5`, `TOPIC_SELECTION=evidence_blend`.

## End-to-end and UI checks
- Real FastAPI + Node, scripted sessions: default 287/287 checks pass; core5 variants 306/306. [reports/e2e_http_check.md]
- Scenario dry runs (strong / struggling / mixed / returning student) on the demo configuration: 10 of 10 seeds pass. [scripts/dry_run.py]
- Browser QA at 1280px and 375px: no console errors, no horizontal overflow, no correct/incorrect shown during the quiz. [reports/screenshots/qa_result.json]

## Real-user evidence
- One real tester's sessions exposed the "stuck on Easy" and "no prediction in session 2" problems, which are now fixed. There is no systematic pilot yet: the log has 62 usable answers, but almost all from developer/tester sessions (ids such as 10, 12, 88109-ak), not a pilot sample; do not present these numbers as evidence. [reports/pilot.md, docs/PILOT_GUIDE.md]
