# Build log (adaptive-v2)

Resume rule: start from the first task whose status is not DONE/SKIPPED.

Baseline (before any change, run on an isolated copy of the baseline commit):
- pytest: 51 collected, **36 passed, 15 failed**. The 15 failures pre-date this work: they drive the legacy fixed 10-question `/session/{id}/answer` flow (no `/next` first), which the adaptive service rejects with 400 "No pending question", plus `test_4_numerical_parity_against_evaluate_test` (IndexError) and gate-endpoint tests that need a legacy-completed session.
- Node: `test_server.js` FAIL (expects 400 for a missing position, gets 404 from the adaptive service); `test_frontend_smoke.js` FAIL (legacy /answer flow -> 400); `test_dashboard_smoke.js` FAIL (legacy flow). 0/3 suites pass at baseline.

| Task | Status | Summary | Tests (pytest pass/fail; node suites pass/total) |
|---|---|---|---|
| T0 recon + restore point + baseline | DONE | git baseline on master, branch feature/adaptive-v2; DB/log backed up to data/backups; .venv; tests isolated via tests/conftest.py (KT_DB_PATH/KT_LOG_PATH temp copies) | 36/15; 0/3 |
| T1 policy_config + monotonicity | DONE | policy_config.py (env-driven, defaults = old behaviour); check_monotonicity.py: 600 val cases + 6000 quiz-topic cases, violation rate 0.0% / 0.0% -> profile_reliable=true | 42/15; - |
| T2 mastery module + /profile | DONE | mastery.py (reliable + evidence-only rules, monotone projection, one batched predictor call); v2_routes.py: GET /session/{id}/profile, /student/{id}/profile; Node pass-through routes for all new endpoints + test_v2_routes.js (fake upstream) | 53/15 (failure set identical to baseline); v2 node 8/8 |
| T3 model-driven difficulty + decision object | DONE | POLICY=staircase/model/random in adaptive_session.py, guardrails G1/G2, fallback to staircase, decision object on /next and in logged question events (policy, decision, student_id, predicted_before_probe); defaults byte-identical to baseline trace | 66/15 (identical failure set) |
| T6 simulator + policy comparison | DONE | scripts/simulate.py, 150 paired synthetic students x 3 policies; in-band 57.6% model vs 32.6% staircase vs 37.7% random; status accuracy 41.6% / 37.8% / 40.8% (CI overlap); Gate S passes; notes in reports/simulation_notes.md | unchanged |
