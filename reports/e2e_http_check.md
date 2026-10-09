# End-to-end HTTP check

Generated 2026-10-09 06:15 UTC by `scripts/e2e_http_check.py`: real FastAPI + Node proxy on free ports, temporary DB copy and log, one scripted session per configuration through `/api/*`.

| Configuration | env | questions | checks | result |
|---|---|---|---|---|
| staircase | `{}` | 15 | 287 | PASS |
| candidate | `{"POLICY": "model", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5"}` | 16 | 306 | PASS |
| demo | `{"POLICY": "staircase", "TARGET_P": "0.70", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5", "TOPIC_SELECTION": "evidence_blend"}` | 16 | 306 | PASS |

Note: warm-up questions 1-2 carry no prediction by design (the model needs 2 answers). The `/answer` response includes `correct` (pre-existing API field); the UI stores it only for the results screen and never shows it during the quiz (SHOW_VERDICT=false).

- staircase: example decision reason: "Topic 'Polynomial Factors' had the lowest predicted success (16%) among 10 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 34%, medium 16%, hard 6%."
- candidate: example decision reason: "Topic 'Equivalent Fractions' had the lowest predicted success (22%) among 5 eligible topics. Level medium: predicted 22%, closest to the 70% target among allowed levels (easy 41%, medium 22%, hard 11%). First question on this topic starts at Medium."
- demo: example decision reason: "Topic 'Equivalent Fractions' had the lowest combined estimate (9%: model 9% combined with 0 of 0 earlier answers on it correct) among 5 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 19%, medium 9%, hard 4%."
