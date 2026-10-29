# Pilot guide (15-30 participants)

Purpose: collect a small amount of **real** answer data so that we can check (a) whether the pre-answer predictions are roughly calibrated for *these* quiz questions and (b) how students perceive the difficulty. Nothing in this repository is pilot evidence yet: `reports/pilot.md` currently says "not meaningful yet" (26 usable answers, and the existing log is dominated by developer test accounts).

## 1. Before the pilot

1. Decide the configuration. The recommended one is in `demo.env` (see `reports/DEMO_CONFIG.md` for why). Start everything with `python scripts/start_demo.py`.
2. Make sure test accounts will not pollute the analysis: ids containing `smoke`, `e2e`, `pred_check`, `test` or starting with `sim_` are excluded automatically (`EXCLUDE_STUDENT_REGEX`). Note this also excludes any real id that happens to contain "test" (e.g. "contest") — avoid such ids.
3. Prepare **pseudonymous IDs** in advance, e.g. `p01` ... `p30`. Keep the list mapping people to IDs offline (paper or a private file), never in this system. Do not use names, emails or student numbers as IDs.
4. Optional but recommended: back up the database first (`python -c "import sqlite3;s=sqlite3.connect('data/processed/kt_sessions.db');d=sqlite3.connect('data/backups/pre_pilot.db');s.backup(d)"`).

## 2. What to tell participants (script)

> "This is a short maths practice quiz (about 15 questions, 10-15 minutes). Before each question the system shows its estimate of your chance of getting it right. That number is a prediction from your earlier answers, not a judgement of you, and it is fine to get questions wrong. You will not see right/wrong during the quiz; you will see a summary at the end. Your answers are stored under the ID we give you, not your name. Participation is voluntary and you can stop at any time. At the end there is an optional 2-question feedback form."

Get consent in the way your institution requires. Do not run the pilot with minors without the required permissions.

## 3. During the pilot

- One session per participant (15-17 questions, ~10-15 minutes). A second session on another day is optional and allows looking at change over time.
- Ask participants to work without help and without a calculator unless you decide otherwise (be consistent and note what you decided).
- Encourage the optional feedback widget on the results page (perceived difficulty + helpfulness).
- If something breaks, note the time and the participant ID; do not "fix" data by hand.

## 4. After the pilot

```
python scripts/analyze_logs.py            # writes reports/pilot.md
```

How to read `reports/pilot.md`:

- **Usable answers**: non-probe answers that had a prediction before the student answered. Below 30 the report says "not meaningful yet"; even at 30-500 treat everything as indicative.
- **Brier score**: mean squared gap between predicted probability and outcome (0 = perfect, 0.25 = always saying 50%). Lower is better.
- **Accuracy of (p >= 0.5)**: how often "more likely right than wrong" matched the outcome. Easy to inflate when almost all answers are wrong (or right), so read it together with the observed rate.
- **Reliability table**: within each predicted-probability band, compare mean predicted with observed share correct. Large systematic gaps mean the model is mis-calibrated for these questions (the model was trained on ASSISTments problems, not on this quiz's templates).
- **Predicted vs actual by level**: checks whether Easy/Medium/Hard behave as the model expects for the quiz templates.
- **Probe validation**: predictions made for never-asked topics (from the related-topic graph) vs what happened on the probe question. Few probes per session, so this needs many sessions.
- **Perceived difficulty by policy**: share of "too easy / about right / too hard" answers.

## 5. What NOT to conclude

- No claim about **learning gains**: one or two short sessions cannot measure learning, and there is no control group.
- No claim that the system is "accurate" in general: 15-30 students on 10 topics is a very small, non-representative sample.
- No comparison between policies unless participants were randomly assigned to them and the groups are large enough; with 15-30 people, any difference is likely noise.
- Do not report teacher opinions you did not collect. If you want a teacher's view, ask a real teacher and quote them with permission.

## 6. Data handling

- Data stays in `data/processed/kt_sessions.db` and `data/processed/session_logs.jsonl` on the machine running the demo.
- Delete the pilot data when it is no longer needed for the project, according to your consent form.

## v3 additions

- **Sections:** run each pilot group in one learner level (Primary / Secondary / Undergraduate) and note it; section answers are stored with section topic ids (10xxx / 20xxx / 30xxx). Primary and undergraduate sessions show no model predictions (evidence-only).
- **Teacher review of generated items (before the pilot):** print 5 items per topic and level with `QuestionStream` from `src/questions/generator.py` (see docs/QUESTION_GENERATOR.md) and ask a teacher to mark each as appropriate / wrong level / unclear. Report the counts; do not change levels silently.
- **What would validate the generator:** correctness falling from easy to medium to hard within each topic on logged answers (scripts/analyze_logs.py groups answers by level).
- **What would validate a non-ASSISTments section for the served model:** training or fine-tuning on real data from that level and re-running scripts/run_external_benchmark.py - not done yet.
