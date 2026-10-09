# End-to-end HTTP check

Generated 2026-10-09 04:34 UTC by `scripts/e2e_http_check.py`: real FastAPI + Node proxy on free ports, temporary DB copy and log, one scripted session per configuration through `/api/*`.

| Configuration | env | questions | checks | result |
|---|---|---|---|---|
| staircase | `{}` | 15 | 287 | PASS |
| candidate | `{"POLICY": "model", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5"}` | 16 | 306 | PASS |
| demo | `{"POLICY": "model", "TARGET_P": "0.70", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5"}` | 16 | 306 | PASS |

Note: warm-up questions 1-2 carry no prediction by design (the model needs 2 answers). The `/answer` response includes `correct` (pre-existing API field); the UI stores it only for the results screen and never shows it during the quiz (SHOW_VERDICT=false).

- staircase: example decision reason: "Topic 'Pythagorean Theorem' had the lowest predicted success (2%) among 10 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 4%, medium 2%, hard 1%."
- candidate: example decision reason: "Topic 'Solving Systems of Linear Equations' had the lowest predicted success (4%) among 5 eligible topics. Level easy: predicted 9%, closest to the 70% target among allowed levels (easy 9%, medium 4%, hard 2%). Hard unlocks after a correct Medium or Hard answer on this topic."
- demo: example decision reason: "Topic 'Percents' had the lowest predicted success (6%) among 5 eligible topics. Level easy: predicted 13%, closest to the 70% target among allowed levels (easy 13%, medium 6%, hard 3%). Hard unlocks after a correct Medium or Hard answer on this topic."
