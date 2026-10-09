# Final report: GatedKT adaptive-v2 upgrade

Branch `feature/adaptive-v2` (baseline commit on `master`). All numbers are copied from generated reports or test output.

## Task status

| Task | Status | Notes |
|---|---|---|
| T0 recon, restore point, baseline | DONE | git baseline on `master` (byte-exact line endings); DB and log backed up with the SQLite backup API to `data/backups/`; project-local `.venv`; tests isolated to temp DB/log copies (`tests/conftest.py`); docs/ARCHITECTURE_NOTES.md |
| T1 policy_config + monotonicity | DONE | `src/serving/policy_config.py`; violation rate 0.0% (600 val cases) / 0.0% (6,000 quiz-topic cases) |
| T2 mastery module + /profile | DONE | `mastery.py`, `GET /session/{id}/profile`, `GET /student/{id}/profile` |
| T3 model-driven difficulty + decision object | DONE | POLICY=staircase/model/random, guardrails, staircase fallback, `decision` on /next and in logs; default path byte-identical to the baseline engine (trace test) |
| T4 quiz UI | DONE | prediction card, difficulty pill, segment bar, shimmer, adaptation panel; no verdict during the quiz |
| T5 results UI | DONE | stat cards, Strong / Needs improvement / Not enough evidence, difficulty profile, measured vs predicted, "Did the prediction work?", feedback |
| T6 simulator + policy comparison | DONE | 150 paired synthetic students x 3 policies |
| T8a teacher overview + dashboard + seeded demo data | DONE | `/class/overview`, status heatmap, Needs attention, Live / Simulated toggle; seed script tested on a temp DB only |
| T8b calibration | DONE | temperature scaling, held-out evaluation |
| T8c core5 mode | DONE | 14 core main questions + 2 probes = 16; e2e PASS |
| T8d feedback + log analysis + pilot guide | DONE | `/session/{id}/feedback`, `scripts/analyze_logs.py`, docs/PILOT_GUIDE.md |
| T8e landing cards | DONE | Context / Who benefits / Problem / Action |
| T8f vendor JS libs | DONE | chart.js 4.4.4 + vis-network 9.1.9 in `server/public/vendor/`, CDN as fallback |
| T7 defence pack | DONE | README, PROJECT_OVERVIEW, DEMO_SCRIPT, VIVA_QA (28 Qs), LIMITATIONS, RESULTS_SUMMARY, PILOT_GUIDE |
| T9 verification | DONE | see below |
| COULD: ensemble spread, session replay | SKIPPED | not started; time spent on verification and fixes instead |

## Tests (before -> after)

| Suite | Baseline | After |
|---|---|---|
| pytest (`tests/`) | 51 collected: 36 passed, 15 failed | 93 collected: **78 passed**, 15 failed |
| Failing pytest set | 15 legacy-flow tests | **identical** 15 tests (diffed by name) |
| `server/tests/test_server.js` | FAIL (expects 400, gets 404 from adaptive service) | FAIL, same assertion |
| `server/tests/test_frontend_smoke.js` | FAIL (legacy /answer flow -> 400) | FAIL, same assertion |
| `server/tests/test_dashboard_smoke.js` | FAIL (legacy flow) | FAIL, same assertion |
| `server/tests/test_v2_routes.js` (new) | - | 8/8 PASS |
| `scripts/e2e_http_check.py` (new) | - | staircase 287/287, candidate 306/306, demo 306/306 PASS |
| Browser QA 1280px + 375px (new) | - | 0 console errors, no horizontal overflow, no verdict text, all new sections render with live API data |

The 15 pytest failures and 3 Node suites failed before any change because they drive the old fixed 10-question `/answer` flow, which the adaptive service rejects. They were left unchanged (docs/DECISIONS.md D9). The 42 new pytest tests cover: defaults identical to the pre-upgrade engine (fixed seeds), staircase unchanged, model-policy choice, both guardrails and the fallback, random policy, every status rule including evidence downgrades, the evidence-conflict rule (D26) and the evidence-only fallback, monotone projection, one batched predictor call, calibration preserving ranking, the core5 topic set and totals, every new endpoint's shape, overview separation and cache invalidation, feedback logging, log-analysis exclusions, and the seed script's real-DB guard.

## Demo configuration and gates (reports/DEMO_CONFIG.md)

| Gate | Numbers | Result |
|---|---|---|
| M: violation_rate <= 10% | 0.0% | PASS -> profile_reliable = true |
| S: model in-band >= staircase - 0.02 and status accuracy >= staircase - 0.03 | in-band 57.6% vs 32.6%; status acc 45.6% vs 40.3% | PASS -> POLICY=model |
| C: ECE -10% relative, AUC unchanged | ECE 0.0249 -> 0.0117 (-53%); AUC 0.8946 = 0.8946 | PASS -> APPLY_CALIBRATION=true |
| K: T8c DONE and core5 e2e passes | 306/306 checks | PASS -> SESSION_MODE=core5 |

`demo.env`: `POLICY=model`, `TARGET_P=0.70`, `APPLY_CALIBRATION=true`, `SESSION_MODE=core5`. Without demo.env (`start_demo.py --defaults`) the behaviour is the original one.

## Headline results

- **Model (unchanged):** validation AUC, mean of 3 seeds: dkt_only 0.8915, graph_no_gate 0.8900, full_gated 0.8950. The gate adds +0.0035; the graph alone did not help.
- **Monotonicity:** 0.0% violations; mean gaps Easy-Medium +0.130 / +0.167, Medium-Hard +0.159 / +0.163 (next-skill / quiz-topic samples).
- **Calibration:** T = 1.167; held-out Brier 0.1183 -> 0.1176, ECE 0.0249 -> 0.0117, AUC 0.8946 unchanged.
- **Simulation (150 paired synthetic students per policy):**

| | random | staircase | model |
|---|---|---|---|
| In-band share (true P 0.55-0.85) | 38.5% | 32.6% | 57.6% |
| Status accuracy (exact) | 45.0% | 40.3% | 45.6% |
| Within one step | 90.5% | 87.3% | 93.0% |
| Coverage | 46.5% | 40.0% | 43.0% |
| Ready precision | 62.3% (69 flags) | 59.1% (66) | 56.9% (72) |
| Level mix E/M/H | 33/35/32% | 2/59/39% | 64/34/2% |

  Paired model - staircase: in-band +0.251 (95% CI +0.202 to +0.302); status accuracy +0.037 (CI -0.005 to +0.080).
- **Real logs:** 26 usable answers after exclusions -> "not meaningful yet" (reports/pilot.md).

## Dry run (after the first report)

`scripts/dry_run.py` plays three scripted students (strong = always correct, struggling = always wrong, mixed = correct only on fraction/percent topics) through the real API with demo.env on a temp DB. Problem found and fixed: the mixed student was labelled "Needs support" on topics answered 4/4 correctly, because the model's estimate is dominated by recent answers on other topics. Fix: one-step evidence-conflict rule (docs/DECISIONS.md D26) plus a test. Simulation and gates were re-run; status numbers above are after the fix (before/after in reports/simulation_notes.md). After the fix, 5 repeated dry runs (3 demo config, 2 default config) passed all scenario checks: the strong student reaches Hard and Proficient/Mastered with green predictions, the struggling student stays on Easy and Needs support, and the mixed student gets both strong and weak topics.

## Known weaknesses and things that looked suspicious

1. **0.0% monotonicity violations** looked suspiciously clean. The model adds a learned difficulty embedding to the target representation, so a stable ordering is expected once training has learned it. This shows ordering, not that the gap sizes fit the quiz's template questions.
2. **Status labels are weak:** about 40-46% exact accuracy in simulation for every policy. In the seeded demo data, simulated *strong* students are mostly rated Developing (few reach Proficient), because the model's per-topic estimates stay modest after only 2-4 answers. The UI frames labels as rule-based estimates with answer counts.
3. **Model policy served Easy 64% of the time** in simulation, driving much of the in-band gain. Real-student behaviour is unknown.
4. **Staircase/topic-selection interaction:** the staircase served Easy only 2%, because topic selection always picks the lowest predicted P at each topic's current level, so topics stepped down to Easy are rarely revisited. This is a pre-existing design property that was kept unchanged as required.
5. The demo **combination** (model + calibration + core5) was checked end-to-end but not re-simulated as a combination.
6. Adaptive sessions are never marked "completed" (pre-existing), so legacy counters ("Welcome back", the legacy matrix) ignore adaptive sessions.
7. The live DB contains developer accounts (`10`, `demo_student_qa`, `dash_st_*`) that the exclusion regex does not catch, so they appear in the live dashboard. The regex also excludes any id containing "test".
8. In browser QA the student always answered "1", so only the red and neutral prediction states and the Easy/Medium pills occurred naturally. Green/amber cards and the Hard pill screenshots are marked `_injected` (a real question re-rendered with a substituted value).
9. Process note: the simulation was first run with 75 students on a wrong timing estimate, then re-run at the specified 150; only the 150-student run is reported.

## Files

Created: `src/serving/{policy_config,mastery,v2_routes,calibration}.py`; `scripts/{_common,_services,check_monotonicity,simulate,fit_calibration,seed_demo_data,analyze_logs,e2e_http_check,apply_gates,start_demo,visual_qa}.py`, `scripts/visual_qa.js`; `server/public/{adaptive_ui,dashboard_overview}.js`, `server/public/vendor/*`; `server/tests/test_v2_routes.js`; `tests/{conftest,test_policy_config,test_mastery,test_policy_engine,test_v2_api,test_calibration,test_analyze_logs}.py`, `tests/data_trace_staircase_baseline.json`; `demo.env`; `requirements-serving.txt`; `docs/*`; `reports/*`; `data/processed/{profile_reliability,calibration}.json`.

Changed (additive): `src/serving/adaptive_session.py`, `src/serving/inference_service.py` (CRLF kept), `src/serving/db.py` (optional KT_DB_PATH), `server/src/index.js` (4 pass-through routes), `server/public/{index.html,app.js,dashboard.html,dashboard.js,style.css}`, `README.md` (replaced the pytest-cache placeholder), `.gitignore` (the original was a stray pytest file containing `*`; kept as `.gitignore.pytest_orig.bak`).

Not touched: `data/raw/`, all `*.pt`, `windowed_data.pkl`, `skill_graph.pkl` (verified byte-identical), and the real `kt_sessions.db` / `session_logs.jsonl` (hashes unchanged since the start-of-work backup; no test or script wrote to them). No retraining.

## Commands

```bash
# start the demo (demo.env); --defaults for the original behaviour
.venv/Scripts/python scripts/start_demo.py
# all tests
.venv/Scripts/python -m pytest tests -q
cd server && node tests/test_v2_routes.js && cd ..
.venv/Scripts/python scripts/e2e_http_check.py
.venv/Scripts/python scripts/visual_qa.py --env-file demo.env --seed-sim    # needs: npm install --prefix .tools/qa puppeteer-core@23
# simulated demo students (real DB) and their removal
.venv/Scripts/python scripts/seed_demo_data.py --yes-real-db
.venv/Scripts/python scripts/seed_demo_data.py --remove --yes-real-db
# regenerate evidence
.venv/Scripts/python scripts/check_monotonicity.py
.venv/Scripts/python scripts/fit_calibration.py
.venv/Scripts/python scripts/simulate.py --students 150
.venv/Scripts/python scripts/apply_gates.py
.venv/Scripts/python scripts/analyze_logs.py
```

(On macOS/Linux use `.venv/bin/python`. Node is installed via WinGet on this machine and is located automatically by the scripts; for `npm`/`node` in a terminal add its folder to PATH.)

## Things only the human can do

1. **Run the pilot** (15-30 people, pseudonymous ids) following docs/PILOT_GUIDE.md, then `scripts/analyze_logs.py`.
2. **Get one real teacher's input** on the dashboard and the "Needs attention" suggestions. No teacher has been consulted; do not quote one you have not asked.
3. **Replace placeholders in your slides** with the real numbers from docs/RESULTS_SUMMARY.md (and pilot numbers only once they exist).
4. **Rehearse the demo** with docs/DEMO_SCRIPT.md, including the fallback screenshots.
5. Optionally clean the developer test accounts out of the live DB before presenting (after backing it up), so the live dashboard shows only real participants.
