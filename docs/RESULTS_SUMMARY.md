# Results summary (one page, for slides)

Every number below is copied from a generated report (file in brackets). Synthetic-student results validate decision logic only.

## Model (unchanged; not retrained)
- Validation AUC, mean of 3 seeds: DKT-only **0.8915**, graph without gate **0.8900**, full gated **0.8950**. The gate adds +0.0035 over DKT; the graph alone did not help. [data/processed/retrain.log]
- Test AUC, mean of 3 seeds: 0.8677 / 0.8657 / 0.8713. [ADAPTIVE_KT_REPORT.md]

## Difficulty awareness (Gate M)
- P(Easy) >= P(Medium) >= P(Hard) violated in **0.0%** of 600 validation cases and **0.0%** of 6,000 quiz-topic cases. Mean gaps: Easy-Medium +0.13 to +0.17, Medium-Hard +0.16. -> model-based profile enabled. [reports/monotonicity.md]

## Calibration (Gate C)
- Temperature T = 1.167. On held-out validation students: ECE **0.0249 -> 0.0117** (-53%), Brier 0.1183 -> 0.1176, AUC 0.8946 unchanged. -> calibration enabled in the demo. [reports/calibration.md]

## Policy simulation (Gate S): 150 paired synthetic students per policy
| | random | staircase | model-guided |
|---|---|---|---|
| Questions in the target band (true P 0.55-0.85) | 38.5% | 32.6% | **57.6%** |
| Exact status accuracy | 45.0% | 40.3% | 45.6% |
| Status within one step | 90.5% | 87.3% | 93.0% |
| Topics with a status (coverage) | 46.5% | 40.0% | 43.0% |
| "Ready for harder" flags truly ready | 62.3% | 59.1% | 56.9% |

Paired model - staircase: in-band **+0.251** (95% CI +0.202 to +0.302); status accuracy +0.037 (CI -0.005 to +0.080, i.e. no clear difference). -> model-guided difficulty used in the demo. [reports/simulation.md]

## End-to-end and UI checks
- Real FastAPI + Node, scripted sessions: staircase 287/287 checks pass; model + calibration + core5 306/306. [reports/e2e_http_check.md]
- Browser QA at 1280px and 375px: no console errors, no horizontal overflow, no correct/incorrect shown during the quiz. [reports/screenshots/qa_result.json]

## Real-user evidence
- None yet: 26 usable logged answers after excluding test accounts -> "not meaningful yet". [reports/pilot.md] A pilot is planned ([docs/PILOT_GUIDE.md](PILOT_GUIDE.md)).
