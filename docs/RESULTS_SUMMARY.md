# Results summary (one page, for slides)

Every number below is copied from a generated report (file in brackets). Synthetic-student results validate decision logic only.

## Model (unchanged; not retrained)
- Validation AUC, mean of 3 seeds: DKT-only **0.8915**, graph without gate **0.8900**, full gated **0.8950**. The gate adds +0.0035 over DKT; the graph alone did not help. [data/processed/retrain.log]
- Test AUC, mean of 3 seeds: 0.8677 / 0.8657 / 0.8713. [ADAPTIVE_KT_REPORT.md]

## How the prediction reacts to answers [reports/answer_response.md]
- A correct last answer gives a higher prediction than a wrong one in about 95-97% of validation cases (+0.22 on the same skill).
- Repeated correct practice on ASSISTments data raises the prediction (0.67 -> 0.83), tracking the observed data.
- For a quiz topic not yet answered, the estimate is unreliable (about 0.97 at Easy, dropping to a skill-specific value after one answer). This is why the first question on a topic is now Medium and topic choice also uses the student's own record.

## Difficulty awareness (Gate M)
- P(Easy) >= P(Medium) >= P(Hard) violated in **0.0%** of 600 validation cases and **0.0%** of 6,000 quiz-topic cases. -> model-based profile enabled. [reports/monotonicity.md]

## Calibration (Gate C)
- Temperature T = 1.167. On held-out validation students: ECE **0.0249 -> 0.0117** (-53%), Brier 0.1183 -> 0.1176, AUC 0.8946 unchanged. -> calibration enabled in the demo. [reports/calibration.md]

## Difficulty policy (Gate S): 150 paired synthetic students per policy, revised rules [reports/simulation.md]
| | random | staircase | model-guided |
|---|---|---|---|
| Questions in the target band (true P 0.55-0.85) | 38.5% | 32.6% | 40.1% |
| Exact status accuracy | 46.0% | **51.5%** | 48.1% |
| Status within one step | 93.0% | 93.2% | 93.9% |

**Gate S fails** (model status accuracy 48.1% is more than 3 points below the staircase's 51.5%). **Model-guided difficulty did not beat the staircase in simulation; the demo uses the staircase, and the model still drives topic selection.** An earlier version of the model policy passed Gate S (57.6% in-band) but did so by serving Easy 64% of the time; a user test showed it stayed on Easy after correct answers, so its rules were revised. [reports/simulation_notes.md]

## Topic selection (Rule T, added after user testing) [reports/simulation_blend_full10.md]
- Blending the model's prediction with the student's own record on each topic, with the staircase: in-band 32.6% -> **41.6%**, status accuracy 51.5% -> 52.8%.

## Demo configuration [reports/DEMO_CONFIG.md]
`POLICY=staircase`, `APPLY_CALIBRATION=true`, `SESSION_MODE=core5`, `TOPIC_SELECTION=evidence_blend`.

## End-to-end and UI checks
- Real FastAPI + Node, scripted sessions: default 287/287 checks pass; core5 variants 306/306. [reports/e2e_http_check.md]
- Scenario dry runs (strong / struggling / mixed / returning student) on the demo configuration: 10 of 10 seeds pass. [scripts/dry_run.py]
- Browser QA at 1280px and 375px: no console errors, no horizontal overflow, no correct/incorrect shown during the quiz. [reports/screenshots/qa_result.json]

## Real-user evidence
- One real tester's sessions exposed the "stuck on Easy" and "no prediction in session 2" problems, which are now fixed. There is no systematic pilot yet: the log has 62 usable answers, but almost all from developer/tester sessions (ids such as 10, 12, 88109-ak), not a pilot sample; do not present these numbers as evidence. [reports/pilot.md, docs/PILOT_GUIDE.md]

## v3 results (all numbers from reports/v3/)

**Benchmark, ASSISTments official split, 20 seeds, 5 epochs for every deep model** [reports/v3/stage5/benchmark.md]
- full_gated 0.8533 ± 0.0018 (95% CI 0.8524-0.8541). Classic DKT 0.8521, no-graph 0.8526, DKT-GRU 0.8478, DKVMN-lite 0.8456, Elo 0.8084, BKT+difficulty 0.7959, SAKT-lite 0.7773, PFA 0.7657.
- Paired tests (Holm): improves over all classical baselines, SAKT-lite, DKVMN-lite, DKT-GRU/LSTM; **comparable** to classic DKT (+0.0012; the student-bootstrap CI includes 0).
- Ablations: gate +0.0075, difficulty +0.0089 (meaningful); metadata +0.0010 and graph +0.0007 (significant with 20 seeds but negligible; the graph was not significant with 10 seeds).
- Fully trained v2 checkpoints (3 seeds, reference): 0.8713 / 0.8677 / 0.8657.

**Classical baselines, 100 random re-splits** [reports/v3/benchmark_classical.md]: Elo 0.8094 ± 0.0133, LR-counts 0.7915, BKT+difficulty 0.7849, BKT 0.7701, PFA 0.7515, skill mean 0.6723.

**By grade band (ASSISTments)** [reports/v3/by_band.md]: full_gated primary-level skills 0.8759 (classic DKT 0.8814), secondary 0.8365; no undergraduate skills exist in the data.

**Real data for other levels (Eedi, 3 seeds)** [reports/v3/external_benchmark.md]: primary-age full_gated 0.7765, DKT-GRU 0.7764, Elo 0.7728; secondary-age Elo 0.7601, full_gated 0.7581. Cross-level transfer costs 0.02-0.05 AUC.

**Undergraduate learners (Statics2011, engineering statics, 332 students, 3 seeds, 5 epochs)** [reports/v3/external_benchmark.md]: Elo 0.7977, LR-counts 0.7919, PFA 0.7819, BKT+difficulty 0.7752, full_gated 0.7652, DKT-GRU 0.7553, SAKT-lite 0.7342. Our model is below the classical baselines on this small dataset.

**Difficulty policies, 100 paired synthetic students, demo setting** [reports/v3/policy_comparison.md]: in-band share staircase 43.2%, ZPD model 50.3%, Elo 51.5%, max-info 46.4%, uncertainty 45.9%, random 34.5%; status-accuracy differences not significant. Demo: POLICY=uncertainty, QUESTION_SOURCE=mixed by the pre-registered gates [reports/v3/DEMO_CONFIG.md]; the uncertainty policy uses the staircase on 59% of decisions.

**Generator** [reports/v3/generator_check.md]: 23 topics; 100 tests pass; complexity ordered 23/23 (by construction), text proxy 17/23.

**Checks**: e2e through the Node proxy PASS for 6 configurations [reports/v3/e2e_http_check.md]; browser QA 1280/375 PASS [reports/v3/ui_check.md].

**Improved model GatedKT+ (pre-registered, reports/v3/IMPROVED_PREREG.md; summary reports/v3/IMPROVED_SUMMARY.md)**:
per-question difficulty, no skill graph, dropout/weight decay, Elo blend fitted on validation; up to 15 epochs with
early stopping for every deep model. gkt_plus_elo test AUC: ASSISTments 0.8702 (DKT classic 0.8613, Elo 0.8084);
Eedi primary 0.7901 (DKT-GRU 0.7788, Elo 0.7728); Eedi secondary 0.7725 (DKT-GRU 0.7620, Elo 0.7602); Statics2011
0.8111 (Elo 0.7975), pretrained on ASSISTments 0.8154. "Better" than both DKT classic and Elo on all four datasets
by the pre-registered rule. The served app model is unchanged.

**Hints and attempts inside the GRUs and the gate (pre-registered, reports/v3/META_PREREG.md; reports/v3/meta_assist.md, ASSISTments, 10 seeds)**: gkt_plus_meta 0.8687 vs gate-only gkt_plus 0.8696 (comparable, -0.0008), no-metadata 0.8683 (comparable), GRU-only 0.8689 (comparable); still better than DKT classic (+0.0074) and DKT-GRU (+0.0076). Hints and attempts do change the model's predictions (seed-0 model: wrong after a hint 0.4481 vs wrong without help 0.4086, mean P(next) after a correct answer 0.7707), but they do not raise test AUC. The app now has a hint button and retries (demo.env: HINTS_ENABLED=true, MAX_ATTEMPTS=2); the served model is unchanged (reports/v3/meta_serving_check.md).

**Seed variance of our three variants (reports/v3/seeds.md; 50 seeds, fixed official split, 5 epochs with early stopping, docs/DECISIONS.md D42)**: full gated 0.8532 +/- 0.0015 (min 0.8503, max 0.8559), DKT-only 0.8469 +/- 0.0018, graph without gate 0.8459 +/- 0.0017. Full gated improves over DKT-only (+0.0062, 95% CI [+0.0058, +0.0067], d_z 3.70, higher on 50/50 seeds) and over graph without gate (+0.0073, [+0.0070, +0.0077], d_z 5.66, 50/50); both gains are 4-5x the seed-to-seed std. In this short-budget run the graph without the gate is slightly below DKT-only. With a new student split per seed (10 splits) the std rises to 0.0138 and full gated still improves (+0.0062 vs DKT-only, +0.0067 vs graph without gate). Absolute AUCs are lower than the longer-trained checkpoints because of the 5-epoch budget.
