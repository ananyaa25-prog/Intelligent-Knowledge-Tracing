# End-to-end HTTP check

Generated 2026-10-29 11:02 UTC by `scripts/e2e_http_check.py`: real FastAPI + Node proxy on free ports, temporary DB copy and log, one scripted session per configuration through `/api/*`.

| Configuration | env | questions | checks | result |
|---|---|---|---|---|
| mixed | `{"QUESTION_SOURCE": "mixed"}` | 15 | 306 | PASS |
| demo_mixed | `{"POLICY": "staircase", "TARGET_P": "0.70", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5", "TOPIC_SELECTION": "evidence_blend", "QUESTION_SOURCE": "mixed"}` | 16 | 324 | PASS |
| section_primary | `{}` | 15 | 337 | PASS |
| section_secondary | `{}` | 15 | 335 | PASS |
| section_undergraduate | `{}` | 12 | 274 | PASS |

Note: warm-up questions 1-2 carry no prediction by design (the model needs 2 answers). The `/answer` response includes `correct` (pre-existing API field); the UI stores it only for the results screen and never shows it during the quiz (SHOW_VERDICT=false).

- mixed: example decision reason: "Topic 'Equivalent Fractions' had the lowest predicted success (4%) among 10 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 12%, medium 4%, hard 2%."
- demo_mixed: example decision reason: "Topic 'Equivalent Fractions' had the lowest combined estimate (13%: model 13% combined with 0 of 0 earlier answers on it correct) among 5 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 28%, medium 13%, hard 6%."
- section_primary: example decision reason: "Topic 'Multiplication and division' had the lowest combined estimate (33%: model 50% combined with 0 of 1 earlier answers on it correct) among 5 eligible topics. Level easy set by the staircase rule (one step per answer on this topic); predicted easy 50%, medium 50%, hard 50%. Transfer mode: no model prediction for this section; topics are chosen from your own answers."
- section_secondary: example decision reason: "Topic 'Geometry (angles, Pythagoras, circles)' had the lowest predicted success (14%) among 5 eligible topics. Level medium set by the staircase rule (one step per answer on this topic); predicted easy 32%, medium 14%, hard 6%."
- section_undergraduate: example decision reason: "Topic 'Calculus basics (derivatives)' had the lowest combined estimate (33%: model 50% combined with 0 of 1 earlier answers on it correct) among 4 eligible topics. Level easy set by the staircase rule (one step per answer on this topic); predicted easy 50%, medium 50%, hard 50%. Transfer mode: no model prediction for this section; topics are chosen from your own answers."
