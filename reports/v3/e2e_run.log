# End-to-end HTTP check

Generated 2026-10-29 13:44 UTC by `scripts/e2e_http_check.py`: real FastAPI + Node proxy on free ports, temporary DB copy and log, one scripted session per configuration through `/api/*`.

| Configuration | env | questions | checks | result |
|---|---|---|---|---|
| demo | `{"POLICY": "uncertainty", "TARGET_P": "0.70", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5", "TOPIC_SELECTION": "evidence_blend", "QUESTION_SOURCE": "mixed"}` | 16 | 325 | PASS |
| mixed | `{"QUESTION_SOURCE": "mixed"}` | 15 | 305 | PASS |
| demo_mixed | `{"POLICY": "uncertainty", "TARGET_P": "0.70", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5", "TOPIC_SELECTION": "evidence_blend", "QUESTION_SOURCE": "mixed"}` | 16 | 325 | PASS |
| section_primary | `{}` | 15 | 337 | PASS |
| section_secondary | `{}` | 15 | 335 | PASS |
| section_undergraduate | `{}` | 12 | 274 | PASS |

Note: warm-up questions 1-2 carry no prediction by design (the model needs 2 answers). The `/answer` response includes `correct` (pre-existing API field); the UI stores it only for the results screen and never shows it during the quiz (SHOW_VERDICT=false).

- demo: example decision reason: "Topic 'Solving Systems of Linear Equations' had the lowest combined estimate (6%: model 6% combined with 0 of 0 earlier answers on it correct) among 5 eligible topics. Level medium: ensemble prediction closest to 70% among allowed levels (easy 11%, medium 4%, hard 2%). A progression rule (based on your answers on this topic) limited the levels."
- mixed: example decision reason: "Topic 'Polynomial Factors' had the lowest predicted success (8%) among 10 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 19%, medium 8%, hard 3%."
- demo_mixed: example decision reason: "Topic 'Solving Systems of Linear Equations' had the lowest combined estimate (14%: model 14% combined with 0 of 0 earlier answers on it correct) among 5 eligible topics. Level medium: ensemble prediction closest to 70% among allowed levels (easy 30%, medium 15%, hard 7%). A progression rule (based on your answers on this topic) limited the levels."
- section_primary: example decision reason: "Topic 'Addition and subtraction' chosen from your answers so far (0 of 1 correct on it; topics with fewer correct answers come first) among 5 eligible topics. Level easy set by the staircase rule (one step per answer on this topic). No model prediction is used for this section."
- section_secondary: example decision reason: "Topic 'Geometry (angles, Pythagoras, circles)' had the lowest predicted success (16%) among 5 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 35%, medium 16%, hard 6%."
- section_undergraduate: example decision reason: "Topic 'Linear algebra (2x2 matrices)' chosen from your answers so far (0 of 1 correct on it; topics with fewer correct answers come first) among 4 eligible topics. Level easy set by the staircase rule (one step per answer on this topic). No model prediction is used for this section."
