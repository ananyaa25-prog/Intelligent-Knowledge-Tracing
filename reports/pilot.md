# Pilot log analysis

Generated 2026-10-09 04:15 UTC from `session_logs.jsonl` by `scripts/analyze_logs.py`.
Excluded ids matching `(smoke|e2e|pred_check|test|^sim_)`: 19 answer events excluded.

> **Not meaningful yet:** only 26 usable answers with a pre-answer prediction (minimum 30). The numbers below are shown for completeness only and must not be used as evidence.

| Quantity | Value |
|---|---|
| Students / sessions | 2 / 2 |
| Answers (all) / usable (non-probe, with prediction) | 30 / 26 |
| Brier score (lower is better; 0.25 = always 50%) | 0.006 |
| Accuracy of (p >= 0.5) | 100.0% |
| Mean predicted vs observed correct | 5.3% vs 0.0% |

## Reliability table (non-probe answers)

| predicted p | n | mean predicted | observed correct |
|---|---|---|---|
| 0.0-0.2 | 25 | 4.6% | 0.0% |
| 0.2-0.4 | 1 | 23.2% | 0.0% |
| 0.4-0.6 | 0 | n/a | n/a |
| 0.6-0.8 | 0 | n/a | n/a |
| 0.8-1.0 | 0 | n/a | n/a |

## Predicted vs actual by level

| level | n | mean predicted | observed correct |
|---|---|---|---|
| easy | 8 | 2.2% | 0.0% |
| medium | 17 | 6.0% | 0.0% |
| hard | 1 | 19.7% | 0.0% |

## Probe validation (graph prediction before a never-asked topic)

0 probe answers with a prediction; mean predicted n/a, observed correct n/a, Brier n/a.

## Perceived difficulty by policy (feedback events)

No feedback events yet.

What this can and cannot show: see docs/PILOT_GUIDE.md. A small pilot can show whether predictions are roughly calibrated for these quiz questions and how students perceive the difficulty; it cannot show learning gains.
