# Project overview

## Context
Foundational maths practice (percents, fractions, rates, linear systems, exponents, Pythagoras) in settings where one teacher looks after many learners and cannot follow each student's work question by question.

## Stakeholders
- **Students**: want practice that is neither trivially easy nor hopeless, and a clear picture of what to work on next.
- **Teachers**: need to know *which topic* and *which difficulty level* each student is stuck on, without marking every attempt.
- **Course coordinator**: needs a class-level view of where support is needed, with an honest indication of how much evidence there is behind each label.

## Problem
A fixed quiz gives one overall score. It does not show which topic is the gap, or whether a student can handle the basics but not harder questions. It also spends questions on topics the student already handles well.

## Action (what the system does)
1. **Predict before answering.** Before each question the knowledge-tracing model estimates P(correct) for every candidate topic and level, using the student's answers so far. The prediction is shown to the student as a probability, with no verdict.
2. **Use the prediction.** The next topic is the one with the lowest estimate among eligible topics (caps: 3 per topic in full10, 4 in core5, no more than 2 in a row). In the demo configuration that estimate blends the model's prediction with the student's own record on the topic (the model counts like 2 answers), so topics already answered correctly are not re-drilled. The **difficulty** follows the staircase in the demo (one step up after a correct answer, one down after a wrong one), because model-guided difficulty did not beat the staircase in simulation (Gate S). Answers from earlier sessions feed every prediction, and each topic resumes at its previous level.
3. **Profile.** After the session, each topic gets a rule-based status from the model's predicted P at Easy / Medium / Hard (Proficient at a level = P >= 70%) **and** the evidence actually observed: at least 2 answers for any status; Proficient needs a Medium/Hard attempt; Mastered needs a correct Hard answer. Topics never asked are shown separately as "Predicted, not measured" (from the skill-relation graph) and checked by probe questions.
4. **Teacher action.** The dashboard shows a students x topics status heatmap with evidence counts, and a "Needs attention" list with rule-based suggestions (e.g. "Revisit basics, assign Easy practice").

## System loop
student history -> predict P(correct) for candidates -> choose topic + difficulty (decision object logged) -> student answers (no verdict shown) -> history updated -> ... -> per-topic profile (status, readiness, evidence) -> teacher overview + suggestions.

## What is new in this project
These are contributions at the **system and evaluation** level, not new algorithms:
1. **An integrated loop with a visible per-question decision log.** Every question records which topics were scored, the predicted P at each level, the chosen level and why (a reason string built only from those numbers). This is visible to the student ("How the system is adapting") and stored in the log for analysis.
2. **A difficulty-aware counterfactual mastery profile with evidence gating.** The model is asked "what would P be at Easy / Medium / Hard?" for each topic. The answers are made monotone, and a status is certified only where the student was actually tested near that level. The profile is enabled only after checking that the model orders difficulty correctly (reports/monotonicity.md).
3. **Separation of measured vs graph-inferred knowledge, checked by probes.** Topics that were never asked are labelled "Predicted, not measured". Probe questions record the prediction made before answering, so these inferences can be validated in pilot data.
4. **Policy comparison and calibration reporting with pre-registered gates.** Random vs staircase vs model-guided difficulty were compared on paired synthetic students. Calibration was measured on held-out data. The demo configuration was chosen mechanically by gates fixed in advance.

## What is NOT new
- Adaptive difficulty / staircase procedures and targeting a success probability are standard in adaptive testing and tutoring.
- Knowledge tracing (DKT-style recurrent models), graph-based knowledge tracing and gated fusion of encoders are existing ideas. This project trains and serves one such model.
- Temperature scaling is a standard calibration method.
- Mastery thresholds (e.g. 70%) are design choices, not discoveries.

## Claims vs evidence

| Claim | Evidence | Status |
|---|---|---|
| The prediction is made before the student answers and is used to choose the next topic | Code path `AdaptiveSession._choose_topic`; e2e check asserts predicted_p on every non-probe question after the 2 warm-ups (reports/e2e_http_check.md) | supported |
| The model chooses the difficulty in the demo | Gate S failed with the revised model policy (reports/DEMO_CONFIG.md); the demo uses the staircase for difficulty | not supported (staircase used) |
| The full gated model is slightly better than DKT-only (val AUC 0.8950 vs 0.8915) | data/processed/retrain.log (3 seeds) | supported (small effect, +0.0035) |
| The skill graph alone improves prediction | graph_no_gate 0.8900 < dkt_only 0.8915 | not supported |
| The model orders difficulty correctly (Easy >= Medium >= Hard) | 0.0% violations in 600 + 6,000 cases (reports/monotonicity.md) | supported (ordering only; gap sizes not validated for quiz items) |
| Temperature scaling improves calibration without changing ranking | ECE 0.0249 -> 0.0117, AUC unchanged (reports/calibration.md) | supported on ASSISTments validation data; untested on quiz answers |
| Model-guided difficulty serves more questions at a suitable difficulty than the staircase | revised rules: in-band 40.1% vs 32.6%, paired +0.075 (CI +0.058 to +0.091); the earlier +0.251 came from serving Easy 64% of the time (reports/simulation_notes.md) | supported in simulation only, small |
| Model-guided difficulty gives more accurate statuses | 48.1% vs 51.5% (staircase), paired -0.033, CI -0.080 to +0.019 | not supported |
| Blending the student's own record into topic choice helps | staircase in-band 32.6% -> 41.6%, status accuracy 51.5% -> 52.8% (reports/simulation_blend_full10.md); added after user testing | supported in simulation only (post hoc) |
| The prediction moves in the right direction after an answer | correct > wrong in about 95-97% of cases (reports/answer_response.md) | supported; estimates for never-answered topics are unreliable |
| Status labels are accurate | exact accuracy about 46-53% in simulation; within one step about 93-94% | not supported as exact labels; read as rough estimates |
| "Ready for harder questions" is reliable | 45-60% of flags truly ready in full10 simulations, 32-36% in core5 | not supported; a suggestion only |
| Graph predictions for unasked topics are accurate for real students | probe events are logged, but there is no pilot data (reports/pilot.md) | inconclusive (no data) |
| The system improves learning | no learning-outcome study | not supported (not tested) |
| Teachers find the dashboard useful | no teacher study | not supported (not tested) |

## v3 claims vs evidence (numbers from reports/v3/)

| Claim | Evidence | Status |
|---|---|---|
| The full gated model beats the classical baselines (majority, skill mean, LR on counts, PFA, Elo, BKT, BKT+difficulty) on ASSISTments | 20 paired seeds, Holm-corrected paired t-test p < 1e-4, seed CI and student-bootstrap CI of the difference exclude 0 for every one; e.g. vs Elo +0.0449 (reports/v3/benchmark.md) | supported |
| It beats the deep baselines | improves over DKT-GRU (+0.0055), DKT-LSTM (+0.0058), DKVMN-lite (+0.0077), SAKT-lite (+0.0760); comparable to classic DKT (+0.0012, Holm p 0.0213 but the student-bootstrap CI includes 0) | partly supported (not vs classic DKT) |
| The gate helps | full vs no-gate ablation +0.0075, significant (benchmark.md) | supported (small effect) |
| The skill graph helps | full vs no-graph ablation +0.0007, Holm p 0.0316 with 20 seeds (was not significant with 10) | statistically detectable, negligible size |
| Difficulty buckets help | full vs no-difficulty +0.0089, significant | supported |
| Response-time / attempt / hint metadata help | full vs no-metadata +0.0010, significant but tiny | supported, negligible size |
| Results are stable across seeds | full_gated test AUC 0.8533 ± 0.0018 over 20 seeds; classical 100 re-splits, e.g. Elo 0.8094 ± 0.0133 (benchmark_classical.md) | supported (seed spread small; split spread larger) |
| The model works for other learner levels | Eedi primary-age learners: full_gated 0.7765 vs DKT-GRU 0.7764 vs Elo 0.7728; secondary-age: Elo 0.7601 vs full_gated 0.7581 (3 seeds, external_benchmark.md) | inconclusive: it works, but no advantage over DKT or Elo is shown |
| A model trained on one level transfers to another | Eedi full_gated primary -> secondary 0.7306 (in-level 0.7581); secondary -> primary 0.7419 (in-level 0.7765) | not supported: transfer costs about 0.02-0.05 AUC |
| The approach works for undergraduate learners | Statics2011 (engineering statics, 332 students): full_gated 0.7652 vs Elo 0.7977, LR-counts 0.7919, PFA 0.7819 (external_benchmark.md) | not supported: evaluated, but below the classical baselines; different subject from the app's maths topics |
| Model-guided difficulty puts more questions in the target band than the staircase (simulation) | in-band +7.2% (model), +8.3% (Elo), +2.8% (uncertainty), all significant (policy_comparison.md) | supported in simulation only |
| Any policy gives more accurate statuses than the staircase | all status-accuracy differences non-significant (Holm p = 1.0) | inconclusive |
| The uncertainty-aware policy uses the model ensemble | it fell back to the staircase on 59% of level decisions | partly: mostly behaves like the staircase |
| Generated questions are harder at harder levels | complexity ordered by construction (23/23 topics); independent text proxy ordered for 17/23 (generator_check.md); no student data | inconclusive (not validated with students or teachers) |
| Generated answers are correct | 100 tests incl. independent re-computation of answers from the question text | supported for the tested item types |
