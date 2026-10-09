# Viva questions and honest answers

Numbers are taken from the reports named in brackets.

**1. Is this a novel algorithm?**
No. Knowledge tracing, graph-based KT, gated fusion, adaptive difficulty and temperature scaling all exist. The contribution is an integrated, evaluated system: the prediction is made before each question and used to choose topic and difficulty, every decision is logged with its numbers, the mastery profile is gated by evidence, measured topics are kept separate from graph-inferred ones, and policies and calibration were checked against pre-registered gates. (docs/PROJECT_OVERVIEW.md)

**2. Why is the AUC gain so small?**
The full gated model reaches 0.8950 validation AUC vs 0.8915 for DKT-only (mean of 3 seeds), +0.0035. Once the baseline knows which skill it is predicting, most of the signal is already captured. The std across seeds is about 0.0004-0.0010, so the gain is consistent but small. We do not present it as a large improvement. (retrain.log)

**3. Why did the graph alone not help?**
graph_no_gate scored 0.8900, below DKT-only. Averaging the graph signal in with equal weight seems to add noise. Only with the learned gate, which can down-weight it, does the combined model come out ahead. We report it as a negative result.

**4. How do you know the prediction is actually used?**
The topic is chosen as the one with the lowest estimate among eligible topics (in the demo, the model's prediction blended with the student's own record on that topic). The prediction shown is made before the student answers. Each question's `decision` object (topic scores, model scores, level predictions, chosen level, reason) is returned by `/next`, shown in the UI and written to the log. The end-to-end check asserts it on every question. Difficulty in the demo follows the staircase, because the model-guided version failed Gate S. (reports/e2e_http_check.md, reports/DEMO_CONFIG.md)

**5. How do you know the system works?**
We know the parts behave as designed: contract checks pass end-to-end; scripted dry runs (strong, struggling, mixed, returning student) behave sensibly; the model orders difficulty correctly (0.0% violations) and moves in the right direction after answers; and calibration improves on held-out data. We do **not** know that it helps real students learn; that needs a pilot and a proper study.

**6. Why target a 70% predicted success rate?**
It is a design choice: high enough to keep students from failing most questions, low enough to keep questions informative. It was fixed before the simulation and not tuned afterwards. It is configurable (TARGET_P).

**7. Why exactly three levels?**
The model's difficulty feature has three data-derived buckets (per-skill tertiles of training success; bucket 0 = unknown). Three levels map directly onto these buckets. More levels would need finer buckets and more data per level.

**8. Why T_PROF = 0.70 and MIN_ANSWERS = 2?**
Also design choices. 70% is a common "can usually do it" threshold. 2 answers is the minimum before any label, which already leaves many topics unlabelled (coverage 40-47% in simulation). Raising it would make labels more trustworthy but rarer.

**9. What if the prediction is wrong?**
It is shown as a probability with "not a verdict" wording. Guardrails limit the damage: two wrong answers in a row on a topic force a step down; Hard requires a correct Medium/Hard answer first; on any model failure the staircase takes over. Statuses also require observed evidence, not just predictions.

**10. How accurate are the status labels?**
Not very, as exact labels: about 46-53% exact agreement with the synthetic ground truth, about 93-94% within one step (reports/simulation*.md). That is why the UI calls them rule-based estimates, shows the answer count and offers suggestions rather than verdicts. The status also never goes above what the student's own answers support (D30).

**11. What does "Mastered" mean here?**
The model's predicted P at Hard is at least 70% **and** the student answered at least one Hard question correctly, with at least 2 answers on the topic. If the model estimate is high but the evidence is missing, the status is downgraded and an evidence note says why. It is never shown without that evidence.

**12. Why is the counterfactual P(easy/medium/hard) trustworthy?**
We checked that the model orders difficulty correctly: 0 violations in 600 validation cases and 6,000 quiz-topic cases, with gaps of about 0.13-0.17 per level (reports/monotonicity.md). That shows ordering, not that the gap sizes are right for the quiz's template questions. A monotone projection is applied just in case.

**13. The simulation's synthetic students are not real. Why trust it?**
Only for what it can show: given students whose true success follows a known curve, does the decision logic serve suitable questions and assign sensible statuses? It cannot show learning gains or real-student accuracy, and the report says so.

**14. Why did you drop model-guided difficulty from the demo?**
The first version picked the level whose predicted success was closest to 70%. The model's estimates for short quiz histories are mostly below 70%, so it chose Easy 64% of the time. That looked good in simulation (57.6% in-band), but a real tester got correct answers and stayed on Easy, never reaching Hard. We revised the rules so the level follows the student's answers (never easier after a correct answer, step up after two correct). Re-run with the same pre-registered gate, the model policy's status accuracy (48.1%) was more than 3 points below the staircase's (51.5%), so Gate S failed and the demo uses the staircase. We report both runs. (reports/simulation_notes.md)

**15. Why does the staircase almost never serve Easy (2%)?**
Topic selection always picks the lowest predicted success at each topic's *current* level. Once a topic drops to Easy its predicted success rises, so it is rarely chosen again. This is a property of the original design, which we kept unchanged as required; we report it as a weakness. (reports/simulation_notes.md)

**16. What does calibration change for the student?**
Displayed probabilities become slightly less extreme (T = 1.167). On held-out validation data ECE fell from 0.0249 to 0.0117 and Brier from 0.1183 to 0.1176. Ranking, and therefore AUC (0.8946), is unchanged by construction. It was measured on ASSISTments, not on quiz answers.

**17. Cold start: what happens for a new student?**
The first 2 questions are warm-ups (random topic, Medium, no prediction). After that the model predicts from those answers. Profiles need at least 2 answers per topic; fewer gives "Not enough evidence".

**18. Why is question selection based only on the current session?**
That is how the original engine rebuilt sessions. The profile and the teacher view use the cumulative history across sessions. Using cumulative history for selection would be a natural next step.

**19. How do you handle topics that were never asked?**
They are shown as "Predicted, not measured" (hatched), using the model's graph-informed prediction at Medium. In core5 mode, 5 topics are measured and the others are inferred, and 2 probe questions check the inference. The prediction before each probe is logged so it can be validated later.

**20. Why not show correct/incorrect after each question?**
To keep the focus on practice rather than judgement, and to keep the prediction display from becoming a scoreboard. Level changes still implicitly hint at correctness; we accept and document that. Results are shown at the end.

**21. What can a 15-30 person pilot show?**
Whether predictions are roughly calibrated on these quiz questions, how predicted vs actual success looks by level, whether probe predictions hold up, and how students perceive the difficulty. It cannot show learning gains or meaningful differences between policies. (docs/PILOT_GUIDE.md)

**22. What about privacy?**
Ids are free text with no login. We recommend pseudonymous ids with the mapping kept offline. Data stays in a local SQLite file and a JSONL log. There is no external transmission.

**23. How do you prevent test data from contaminating results?**
Tests and evaluation scripts run on temporary copies of the DB and log. Analytics exclude ids matching a test pattern. Simulated demo students are tagged `sim_`/`sample` and shown only behind a clearly labelled toggle. The pattern can still miss oddly named test accounts (the live DB contains some).

**24. What evidence do you have from real teachers or students?**
None yet. There is no teacher quote and no pilot result. The current log has 62 usable answers, but almost all from developer/tester sessions (ids such as 10, 12, 88109-ak), not a pilot sample; one tester's feedback did lead to concrete fixes (question 30 and docs/DECISIONS.md D27-D31).

**25. Why were the demo settings chosen?**
By four gates written before the results: M (difficulty ordering), S (model policy not worse than the staircase in simulation), C (calibration improves ECE by >= 10% without changing AUC), K (core5 passes end-to-end). M, C and K pass; S fails with the revised model rules, so difficulty uses the staircase. One extra rule (T: blended topic selection) was added after user testing and is labelled as post hoc. (reports/DEMO_CONFIG.md)

**26. Did you tune anything after seeing results?**
No policy parameter or threshold was changed after the simulation. One process correction: the first simulation run used 75 students because of a wrong timing estimate; it was re-run at the specified 150 and only that run is reported (docs/DECISIONS.md D21).

**27. What would you do next?**
Run the pilot; get a teacher's feedback on the dashboard; use cumulative history for question selection; revisit the topic-selection/staircase interaction; collect quiz-specific difficulty data to calibrate the template levels.

**28. Which model checkpoint is served?**
`best_full_gated.pt`, byte-identical to the seed-42 full gated checkpoint. No retraining was done in this work.

**29. Is the percentage shown before each question correct? It sometimes drops after I answer correctly.**
It is the model's estimate for *the next* question's topic and level. Two effects make it drop: (1) the next question is deliberately on the topic you currently find hardest, so it usually shows a lower number than the topic you just got right; (2) for a topic you have not answered yet, the model's estimate is unreliable (about 0.97 at Easy, then it settles to a skill-specific value after one answer). In like-for-like comparisons a correct answer gives a higher prediction than a wrong one in about 95-97% of validation cases. (reports/answer_response.md)

**30. Does a returning student start from scratch?**
No (fixed after user testing). Answers from earlier sessions feed every prediction from question 1, each topic resumes at its previous level, and finished sessions are marked completed so they appear in the teacher's matrix.
