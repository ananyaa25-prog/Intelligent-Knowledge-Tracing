# Demo script (6-8 minutes)

## Before the demo (10 minutes earlier)
1. `.venv/Scripts/python scripts/start_demo.py` -> wait for the URL box (quiz on `http://127.0.0.1:3000/`).
2. Optional, for the teacher view: `.venv/Scripts/python scripts/seed_demo_data.py --yes-real-db` (12 simulated students; remove afterwards with `--remove --yes-real-db`).
3. Open two tabs: the quiz and `/dashboard.html`. Have `reports/screenshots/` and `docs/RESULTS_SUMMARY.md` open as a fallback.
4. Use fresh student ids for each scenario (e.g. `demo_strong_1`; avoid ids containing "test", "e2e" or "smoke", which the analytics exclude).

The demo configuration (demo.env) is: model-guided difficulty, calibration on, core5 mode (14 questions on 5 core topics + 2 probes on other topics = 16 questions). Answers are free text; you need to actually solve the questions to answer them correctly.

## 0:00-1:00 Framing (landing page)
- Point at the four cards: **Context -> Who benefits -> Problem -> Action**.
- Say: "This is an integrated system built from standard parts: a knowledge-tracing model, a skill graph and a gate. The model is not new. What we built and evaluated is the loop: predict before each question, use the prediction to choose the topic and difficulty, and turn the answers into a per-topic profile that a teacher can act on."

## 1:00-3:30 Scenario A: strong student (answer correctly)
- Enter `demo_strong_1`, start. Warm-up questions 1-2: the card says "Warm-up: no prediction yet". Explain that the model needs 2 answers first.
- From question 3: point at the **prediction card** ("Predicted by GatedKT before you answer"; hover the (i): a probability, not a verdict) and the **difficulty pill**.
- Open **"How the system is adapting"**: read the decision line aloud (e.g. "Topic X had the lowest predicted success (..%) among 5 eligible topics. Level easy: predicted ..%, closest to the 70% target..."). Point at the step chart and the topic chips.
- Answer correctly a few times. When the pill changes, note the neutral toast "Difficulty adjusted for this topic". Say that correct/incorrect is never shown during the quiz.
- Finish (16 questions). Results: summary cards -> **Strong topics** box -> difficulty profile (three bars per topic with the 70% line, dots where answered) -> **Measured vs predicted** (5 measured + probes, the rest hatched "Predicted, not measured") -> **Did the prediction work?** with the "only N questions" caveat.

## 3:30-5:00 Scenario B: struggling student (answer wrongly / "1")
- New id `demo_struggling_1`. Answer "1" to everything. The cards turn red ("Challenging for you right now"), and the level steps down to Easy (guardrail: two wrong answers in a row on a topic force a step down; Hard stays locked).
- Results: **Needs improvement** box with "Needs support" statuses and the suggestion "Revisit basics, assign Easy practice". Point at the evidence chip ("2 answers - low confidence") and the sentence "Status labels are rule-based estimates...".

## 5:00-6:30 Scenario C: mixed student + teacher view
- (If time) mixed answers: correct on fractions, wrong on linear systems; show one topic in each box.
- Dashboard: scroll to **Class overview**. Toggle **Live** (your demo students; test accounts excluded) and **Simulated demo** (clearly labelled synthetic students). Point at icon + text + colour per cell, the answer counts, and the **Needs attention** list.

## 6:30-8:00 Evidence (RESULTS_SUMMARY.md)
- Gate M: 0.0% difficulty-ordering violations -> the profile uses model estimates.
- Gate S (simulation, synthetic students): model-guided 57.6% of questions in the target band vs staircase 32.6%; status accuracy not clearly different (45.6% vs 40.3%, CI includes 0).
- Gate C: ECE 0.0249 -> 0.0117 with AUC unchanged.
- Model: gate +0.0035 AUC over DKT; the graph alone did not help.
- Close with what is missing: a real pilot and a teacher's input (docs/PILOT_GUIDE.md).

## If something fails live
- Services do not start: check that ports 8000/3000 are free, or run `scripts/start_demo.py --py-port 8010 --node-port 3010`.
- Model or page errors: switch to the screenshots in `reports/screenshots/` (quiz states at 1280px and 375px, results, dashboard, simulated dashboard). Files with `_injected` in the name show a real question with a substituted prediction value; say so if you use them.
- Dashboard empty: use the "Simulated demo" toggle (after seeding) or `dashboard_simulated_1280.png`.
- Want the original behaviour: `scripts/start_demo.py --defaults`.
