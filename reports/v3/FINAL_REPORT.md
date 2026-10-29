# GatedKT v3 - final report

Branch `feature/v3-benchmark` (from v2 commit 78f97e7). Every number below is copied from a generated file in
reports/v3/ (named in brackets). Protected files (data/raw, existing *.pt, windowed_data.pkl, skill_graph.pkl,
kt_sessions.db) were only read; new outputs are in models_v3/, reports/v3/, data/processed_v3/.

## Stage status

| Stage | Status | Reason / summary |
|---|---|---|
| 1 Recon and restore point | DONE | branch, docs/ARCHITECTURE_NOTES_v3.md, v3 folders, scripts/run_tests_v3.py, legacy-failure list |
| 2 Classical baselines | DONE | majority, skill mean, LR on counts, PFA, Elo, BKT (own EM), BKT+difficulty; shared interface; tests |
| 3 Deep baselines | DONE | DKT classic/GRU/LSTM, DKVMN-lite, SAKT-lite, wrappers for our variants (read-only checkpoints); tests |
| 4 Cheap-model benchmark | DONE | 100 seeds, two protocols [benchmark_classical.md]; a BKT numerical bug was found and fixed |
| 5 Deep benchmark + significance | DONE (reduced) | 20 seeds (target 50 not feasible; 10 first, 10 added in a follow-up), 5 epochs instead of 15, 85.6 min in total (above the 30-40 min target); 1000-resample bootstrap; paired tests with Holm [benchmark.md] |
| 6 Sections (backend) | DONE | sections.py, /session/start "section", GET /sections, per-band AUC [by_band.md] |
| 7 Real datasets | PARTIAL | only Eedi / NeurIPS 2020 was publicly downloadable; 6 others need login or licence acceptance [docs/DATASETS_TODO.md]; Eedi benchmark per section + transfer [external_benchmark.md] |
| 8 Question generator | DONE | 23 parametric topics, QUESTION_SOURCE flag (default bank), 100 tests [generator_check.md] |
| 9 Policy comparison | DONE | 6 policies, 100 paired synthetic students, pre-registered gates [policy_comparison.md, DEMO_CONFIG.md] |
| 10 UI integration | DONE | picker, labels, MC choices; e2e 6 configs PASS; browser QA 1280/375 PASS [e2e_http_check.md, ui_check.md] |
| 11 Docs, audit, report | DONE | README, overview claims table, limitations, results, demo script, related work, pilot guide, viva Q31-45, honesty audit PASS [honesty_audit.md] |

## Tests

| | pytest | Node suites |
|---|---|---|
| Before v3 (baseline) | 85 passed / 15 failed | 1/4 pass (test_v2_routes.js) |
| After v3 | 259 passed / 15 failed | 1/4 pass (test_v2_routes.js) |

The 15 pytest failures and 3 Node suites are the same legacy fixed-quiz tests that already failed before v2
(tests/legacy_failures_v3.txt); `scripts/run_tests_v3.py` reports 0 new failures. 174 new tests: classical
baselines 28, deep baselines 18, sections 15, adapters 4, generator 100, v3 policies 9.

## Benchmark (ASSISTments 2009, official split) [benchmark.md, benchmark_classical.md]

20 seeds, same 5-epoch budget for every deep model; uncertainty from seeds (t-CI) and, separately, from a
1000-resample bootstrap over the 313 test students.

| Model | Test AUC mean ± std (20 seeds) | full_gated - model | Verdict (Holm-corrected paired t-test + both CIs) |
|---|---|---|---|
| full_gated (ours) | 0.8533 ± 0.0018 | - | - |
| no_graph (ablation) | 0.8526 ± 0.0018 | +0.0007 | improves (negligible; comparable with 10 seeds) |
| no_metadata (ablation) | 0.8523 ± 0.0017 | +0.0010 | improves (tiny) |
| dkt_classic | 0.8521 ± 0.0009 | +0.0012 | comparable (bootstrap CI includes 0) |
| dkt_gru | 0.8478 ± 0.0017 | +0.0055 | improves |
| dkt_lstm | 0.8475 ± 0.0017 | +0.0058 | improves |
| dkt_only (ours, no graph/gate) | 0.8469 ± 0.0019 | +0.0064 | improves |
| graph_no_gate (no-gate ablation) | 0.8457 ± 0.0020 | +0.0075 | improves |
| dkvmn_lite | 0.8456 ± 0.0027 | +0.0077 | improves |
| no_difficulty (ablation) | 0.8444 ± 0.0013 | +0.0089 | improves |
| elo | 0.8084 ± 0.0005 | +0.0449 | improves |
| bkt_diff | 0.7959 ± 0.0007 | +0.0573 | improves |
| logreg_counts | 0.7930 | +0.0603 | improves |
| bkt | 0.7820 ± 0.0002 | +0.0713 | improves |
| sakt_lite | 0.7773 ± 0.0033 | +0.0760 | improves |
| pfa | 0.7657 | +0.0875 | improves |

Reference (fully trained v2 checkpoints, 3 seeds): full_gated 0.8713, dkt_only 0.8677, graph_no_gate 0.8657.
Classical models over 100 random re-splits: Elo 0.8094 ± 0.0133, LR-counts 0.7915, BKT+difficulty 0.7849, BKT
0.7701, PFA 0.7515, skill mean 0.6723. Per grade band: full_gated 0.8759 on primary-level skills (classic DKT 0.8814), 0.8365 on
secondary-level skills [by_band.md].

**Eedi (real primary- and secondary-age learners, 3 seeds, 4 epochs)** [external_benchmark.md]: primary
full_gated 0.7765, DKT-GRU 0.7764, Elo 0.7728; secondary Elo 0.7601, full_gated 0.7581, DKT-GRU 0.7575.
Cross-level transfer: full_gated primary -> secondary 0.7306, secondary -> primary 0.7419.

## Policy comparison (100 paired synthetic students, demo setting) [policy_comparison.md]

| Policy | In-band share | Pooled status accuracy | In-band diff vs staircase (95% t-CI) |
|---|---|---|---|
| random | 34.5% | 51.3% | -8.7% [-11.9%, -5.4%] |
| staircase | 43.2% | 48.3% | - |
| Elo/IRT target | 51.5% | 49.5% | +8.3% [+6.2%, +10.5%] |
| max-information | 46.4% | 49.8% | +3.2% [+1.0%, +5.5%] |
| ZPD 0.70 (model) | 50.3% | 47.3% | +7.2% [+4.8%, +9.6%] |
| uncertainty-aware | 45.9% | 50.4% | +2.8% [+0.7%, +4.8%] |

Status-accuracy differences vs the staircase are not significant for any policy (Holm-corrected p = 1.0). The
uncertainty-aware policy fell back to the staircase on 59% of level decisions.

## Gate outcomes [DEMO_CONFIG.md]

- Gate P (pre-registered, committed before the run): uncertainty PASS, model PASS -> **POLICY=uncertainty**
  (higher pooled status accuracy, 50.4% vs 47.3%). This is a non-inferiority result in simulation only.
- Gate Q: generator tests 100 passed, e2e with QUESTION_SOURCE=mixed PASS (mixed 306 checks, demo_mixed) ->
  **QUESTION_SOURCE=mixed**.
- Unchanged from v2: APPLY_CALIBRATION=true, SESSION_MODE=core5, TOPIC_SELECTION=evidence_blend, TARGET_P=0.70.

## Known weaknesses

- Our model is not better than classic DKT on ASSISTments under the same budget, and not better than DKT or Elo on
  Eedi. The skill graph adds a negligible +0.0007 (significant only with 20 seeds); the gate and the difficulty input give small benefits.
- Reduced training budget (5 epochs) and 20 instead of 50 seeds for the deep models; -lite baselines are not the
  published architectures; AKT was not run.
- Only one external dataset; Eedi lacks response time / attempts / hints; sections by an age rule.
- Primary is labelled "validated on real data" but the served model is ASSISTments-only, so primary sessions are
  evidence-only; undergraduate has no data (transfer mode).
- Generated items: difficulty by construction, never answered by students or reviewed by a teacher.
- Policy results come from synthetic students; the demo policy mostly behaves like the staircase.
- No pilot, no learning-outcome or teacher evidence.

## Files changed (v2 commit 78f97e7 -> HEAD, screenshots excluded)

101 files, 86 added and 15 modified (+11731 / -53 lines). New: src/baselines/ (base, majority, logistic, pfa,
elo, bkt, deep_common, dkt, dkvmn, sakt, ours, deep), src/data/adapters/ (common, eedi), src/questions/generator.py,
src/serving/sections.py, src/serving/policies_v3.py, scripts (run_tests_v3, benchmark_seeds, benchmark_deep,
build_by_band, fetch_datasets, run_external_benchmark, check_generator, simulate_v3, apply_gates_v3,
visual_qa_v3.py/.js, honesty_audit), server/public/sections_ui.js, 6 new test files, docs (ARCHITECTURE_NOTES_v3,
BASELINES, QUESTION_GENERATOR, RELATED_WORK_TABLE, DATASETS_TODO), reports/v3/*. Modified (additive, defaults =
previous behaviour): adaptive_session.py, inference_service.py, v2_routes.py, policy_config.py, server/src/index.js,
app.js, index.html, dashboard.html, style.css, e2e_http_check.py, start_demo.py, demo.env, README and v2 docs
(appended v3 sections), .gitignore.

## Exact commands

```bash
# tests
.venv/Scripts/python scripts/run_tests_v3.py
# benchmark
.venv/Scripts/python scripts/benchmark_seeds.py classical --seeds 100 --workers 5
.venv/Scripts/python scripts/benchmark_seeds.py deep --seeds 20 --epochs 5 --workers 6 --bootstrap 1000   # cached seeds are reused
.venv/Scripts/python scripts/build_by_band.py
.venv/Scripts/python scripts/fetch_datasets.py
.venv/Scripts/python scripts/run_external_benchmark.py --seeds 3 --epochs 4 --max_students 3000 --workers 5
.venv/Scripts/python scripts/check_generator.py
# simulator and gates
.venv/Scripts/python scripts/simulate_v3.py --students 100 --seed 2026 --exploratory
.venv/Scripts/python scripts/apply_gates_v3.py
# end-to-end, UI, audit
.venv/Scripts/python scripts/e2e_http_check.py --only demo,demo_mixed,mixed,section_primary,section_secondary,section_undergraduate --out-dir reports/v3
.venv/Scripts/python scripts/visual_qa_v3.py --env-file demo.env --sections default,primary,secondary
.venv/Scripts/python scripts/honesty_audit.py
# demo
.venv/Scripts/python scripts/start_demo.py            # demo.env (uncertainty, mixed questions, core5)
.venv/Scripts/python scripts/start_demo.py --defaults # original behaviour
```

## Things only the human can do

1. Run a real pilot (docs/PILOT_GUIDE.md): consent, 15-30 students, logged sessions, then scripts/analyze_logs.py.
2. Collect or obtain real data for the levels the served model does not cover (undergraduate; primary for the
   served model), e.g. by downloading XES3G5M / Junyi / KDD Cup / Statics2011 after accepting their licences
   (docs/DATASETS_TODO.md), and retrain.
3. Ask a teacher to review a sample of generated questions per topic and level (appropriate / wrong level / unclear).
4. Check the Eedi licence terms before sharing any derived data, and verify citations marked "verify before citing".
5. Update the slides with the v3 tables (docs/RESULTS_SUMMARY.md) and the honest verdicts ("comparable to DKT").
