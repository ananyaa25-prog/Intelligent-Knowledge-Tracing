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
