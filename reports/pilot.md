# Pilot log analysis

Generated 2026-10-09 06:19 UTC from `session_logs.jsonl` by `scripts/analyze_logs.py`.
Excluded ids matching `(smoke|e2e|pred_check|test|^sim_)`: 19 answer events excluded.

> Check the ids below: developer or tester accounts that do not match the exclusion pattern are included. Only a planned pilot with pseudonymous ids (docs/PILOT_GUIDE.md) counts as evidence.

| Quantity | Value |
|---|---|
| Students / sessions | 3 / 6 |
| Student ids included | 10, 12, 88109-ak |
| Answers (all) / usable (non-probe, with prediction) | 80 / 62 |
| Brier score (lower is better; 0.25 = always 50%) | 0.118 |
| Accuracy of (p >= 0.5) | 82.3% |
| Mean predicted vs observed correct | 31.1% vs 45.2% |

## Reliability table (non-probe answers)

| predicted p | n | mean predicted | observed correct |
|---|---|---|---|
| 0.0-0.2 | 30 | 5.8% | 3.3% |
| 0.2-0.4 | 6 | 30.4% | 66.7% |
| 0.4-0.6 | 11 | 49.2% | 81.8% |
| 0.6-0.8 | 14 | 67.9% | 92.9% |
| 0.8-1.0 | 1 | 80.1% | 100.0% |

## Predicted vs actual by level

| level | n | mean predicted | observed correct |
|---|---|---|---|
| easy | 38 | 38.8% | 60.5% |
| medium | 23 | 19.0% | 21.7% |
| hard | 1 | 19.7% | 0.0% |

## Probe validation (graph prediction before a never-asked topic)

6 probe answers with a prediction; mean predicted 47.9%, observed correct 50.0%, Brier 0.378.

## Perceived difficulty by policy (feedback events)

| policy | too easy | about right | too hard | mean helpfulness (1-5) |
|---|---|---|---|---|
| model | 0 | 1 | 0 | 4.000 |

What this can and cannot show: see docs/PILOT_GUIDE.md. A small pilot can show whether predictions are roughly calibrated for these quiz questions and how students perceive the difficulty; it cannot show learning gains.
