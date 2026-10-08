# Adaptive Knowledge Tracing Quiz — Implementation Report

This document explains every change made to the repo to address the professor's
five points of feedback and deliver the three requested phases (A: verify &
retrain, B: adaptive session engine, C: logging & evaluation).

**Scope summary:** the §3 patch set (`kt_patch/src/`) described in the brief
did **not** exist in the working tree or in `aiproj.zip`. It was implemented
from scratch exactly as specified, then retrained and wired end-to-end.

---

## 1. Environment

Two dependencies required by the existing pipeline were missing from the active
interpreter (pyenv Python 3.14, torch 2.12, CPU-only):

```bash
pip install scikit-learn torch-geometric
```

No code depends on these being present at import time beyond what the original
pipeline already required (`train.py`/`evaluate_test.py` use scikit-learn;
`graph_encoder.py` uses `torch_geometric.nn.GATConv`).

---

## 2. Phase A — target-aware model, difficulty, retrain

The core flaw: the target skill only reached the output through the *gate*
weights, so predictions for different topics were nearly identical, and the
`dkt_only` ablation had no target-skill signal at all (confounding it).

### 2.1 `src/gate_fusion.py`

- `output_head` changed from `Linear(hidden_dim, 1)` to
  `Linear(hidden_dim + graph_dim, 1)`.
- `forward()` now takes `h_target` and predicts from
  `sigmoid(output_head(cat([h_final, h_target])))`.
- Self-test now also asserts the prediction **changes when `h_target` is
  perturbed** (target-sensitivity check).

### 2.2 `src/model.py`

- Added `self.diff_emb = nn.Embedding(4, hidden_dim)` — difficulty buckets
  `0=unknown/padding, 1=easy, 2=medium, 3=hard`.
- `forward(..., difficulty=None, edge_index=None, edge_weight=None)`:
  - History inputs now add the difficulty embedding:
    `x = skill_emb + correct_emb + diff_emb(difficulty[:, :-1])`.
  - Target representation is explicit: `h_target` = graph-refined target-skill
    embedding (or raw skill embedding when the graph is off) **plus** the target
    difficulty embedding.
  - All three conditions feed `h_target` to the head. The no-gate ablation still
    averages `h_short/h_long/h_graph`, but the head now sees the target skill.
- Added `_final_states(...)` and `predict_candidates(...)`: encodes a student's
  history once and scores a list of candidate `(skill_id, difficulty_bucket)`
  pairs as the **next** question. This is the pre-answer API.
- Self-test now checks (a) `forward()` and `predict_candidates()` agree on the
  same last-step target, and (b) predictions differ across target skills.

> Note: `forward`'s trailing args are now effectively keyword-only in the call
> sites (`edge_index=…, edge_weight=…`) because `difficulty` was inserted before
> them. All call sites were updated accordingly.

### 2.3 `src/dataset.py`

- `problem_id` added to `RAW_REQUIRED_COLUMNS` and kept through cleaning.
- New `derive_difficulty(train_df)` computes difficulty **from the train split
  only** (no test leakage): per `(skill, problem)` mean correctness, tertiled
  within each skill by rank; problems with `< MIN_PROBLEM_SAMPLES (5)` train
  responses → bucket `0` (unknown).
- Student split is computed **before** difficulty derivation so the buckets come
  from train students only, then applied uniformly to all three splits.
- `build_sequences` now emits `difficulty_seq` and `problem_seq`.
- New `difficulty_shares(...)` reports per-split bucket fractions.

**Difficulty shares per split:**

| split | n | unknown(0) | easy(1) | medium(2) | hard(3) |
|---|---|---|---|---|---|
| train | 271042 | 0.048 | 0.290 | 0.341 | 0.320 |
| val   |  35835 | 0.071 | 0.290 | 0.314 | 0.326 |
| test  |  31333 | 0.076 | 0.280 | 0.336 | 0.309 |

### 2.4 `src/train.py`

- `EPOCHS` raised from 5 to **15** (best-val checkpointing already in place).
- `run_epoch` passes `difficulty` and uses keyword args for the graph tensors.

### 2.5 Retraining + test results

All 3 conditions × 3 seeds (42 / 123 / 2024) retrained; old `.pt` checkpoints
were incompatible and replaced. `best_full_gated.pt` was refreshed from the new
seed-42 run for serving.

**Test AUC (evaluate_test.py):**

| Condition | seed 42 | seed 123 | seed 2024 | mean | std |
|---|---|---|---|---|---|
| dkt_only | 0.8681 | 0.8670 | 0.8680 | **0.8677** | 0.0005 |
| graph_no_gate | 0.8647 | 0.8666 | 0.8656 | 0.8657 | 0.0008 |
| full_gated | 0.8715 | 0.8719 | 0.8704 | **0.8713** | 0.0006 |

**Honest finding (reported, not tuned):** once the baseline knows the target
skill, the graph *alone* no longer helps — `graph_no_gate` now sits slightly
**below** `dkt_only`. The full gated model is still best (+0.0036 over DKT).
The gate, not the graph, is the load-bearing contribution.

---

## 3. Phase B — adaptive session engine

### 3.1 `src/serving/question_templates.py`

Every one of the 10 generators now accepts a `level` argument
(`easy` / `medium` / `hard`), each with its own parameter ranges, and returns a
`level` field plus the bucket mapping. `LEVEL_TO_BUCKET = {easy:1, medium:2,
hard:3}`. `generate_question(position, level="medium")` threads it through.
Template levels are an **approximation** of the data-derived buckets.

### 3.2 `src/serving/adaptive_session.py` (new)

`AdaptiveSession` + `AdaptiveSessionManager` implement the decided design:

- **10 topics** (the fixed skill sequence), each with easy/medium/hard generators.
- **15 main questions.** First **2 are warm-up** (random topic, medium level,
  `predicted_p = null` — the model needs ≥2 interactions). After that the next
  topic is chosen by `predict_candidates` (lowest predicted P(correct) among
  eligible topics).
- **Caps:** ≤3 questions per topic and ≤2 in a row on the same topic.
- **Staircase level:** start medium; correct → up one level, wrong → down one
  (clamped to easy/hard). Reacts **only** to the actual answer, never the
  prediction.
- Questions are generated one at a time (not up front).
- **Probes:** after 15 main questions, 1–2 probe questions on topics never
  asked, each with a pre-answer prediction; the manager also exposes unasked
  topics for graph-propagated predictions.
- `from_history(...)` rebuilds state by replaying stored interactions, so a
  restart can resume from SQLite.

### 3.3 `src/serving/db.py`

- `interactions` table gained columns `difficulty_bucket`, `level`, `p_before`,
  `is_probe` (with migration for existing DBs).
- New `record_adaptive_interaction(...)`.
- `get_session_interactions` / `get_full_history` now return the new fields
  (including `topic == skill_id`, `difficulty_bucket`, `p_before`, `is_probe`).

### 3.4 `src/serving/inference_service.py`

- `InteractionRecord` gained `difficulty_bucket: int = 0`.
- `_process_history` now builds a `difficulty_seq` and passes it (with keyword
  args) to the model; the missing-skills block now uses `predict_candidates`.
- Added `SESSION_LOG_PATH`, `_log_line`, `_store_adaptive`, `_history_to_tensors`,
  `_predictor`, and a global `adaptive_manager`.
- Replaced the fixed-order `/session/start` and `/session/{id}/answer`, and added
  `POST /session/{id}/next` returning `{topic, level, question_text, predicted_p,
  is_probe, ...}`. When the session is complete, `/next` returns the
  graph-propagated `unasked_predictions`.
- `AnswerRequest.position` made optional (position is tracked server-side).

### 3.5 `server/src/index.js`

- Added `/api/session/:sessionId/next` proxy.
- Relaxed `validateAnswerPayload` so `position` is optional (1–20).

### 3.6 Frontend (`server/public/index.html`, `app.js`)

- `index.html`: adaptive subtitle, a `#predictionBanner` showing
  *"The model predicts a X% chance you answer this correctly"* before answering.
- `app.js`: replaced the fixed 10-question flow with one-question-at-a-time:
  `start` → `fetchNextQuestion()` → render with predicted P → answer (no
  verdict) → next. Results screen uses a fixed `SKILL_SEQUENCE` and renders
  graph-propagated predictions for unasked topics.

---

## 4. Phase C — logging & evaluation

### 4.1 Logging

Every interaction is appended to `data/processed/session_logs.jsonl` with
`(event, session_id, student_id, position, topic, level, difficulty_bucket,
p_before, outcome, is_probe, timestamp)` — both a `question` event (with
`p_before`) and an `answer` event (with `outcome`).

### 4.2 `scripts/evaluate_sessions.py` (new)

Reads the log and reports: AUC, accuracy@0.5, Brier score, a calibration table
overall and by level, confident misses (`p_before > 0.8` but wrong), and a
probe-question predicted-vs-actual comparison (the graph-propagation test).

### 4.3 Tests

- `tests/test_adaptive_session.py` (new, 7 tests): warm-up has no prediction,
  lowest-p selection, staircase + clamping, topic caps + consecutive rule,
  probe-on-unasked, manager logging/storing, full-session run.
- `tests/test_e2e_adaptive.py` (new): drives a full session through the real
  FastAPI app + retrained model and asserts ≥15 questions and a written log.

---

## 5. Verification results

- `python src/gate_fusion.py` — PASS (target-sensitive).
- `python src/model.py` — PASS (forward ↔ predict_candidates agree; output
  depends on target skill; no NaNs on real data).
- `python src/dataset.py` — PASS (shares above).
- `python src/train.py` → `python src/evaluate_test.py` — table above.
- `python tests/test_adaptive_session.py` — PASS.
- `python tests/test_db.py tests/test_session.py` — PASS (16 tests total with
  adaptive).
- `python tests/test_e2e_adaptive.py` — PASS (15-question session + log).
- `python scripts/evaluate_sessions.py` — runs and reports metrics.
- Live smoke test: `/session/start`, `/session/{id}/next`,
  `/session/{id}/answer`, `/session/{id}/predict` all return correctly through
  the Node proxy (port 3001) to the Python service (port 8000).

---

## 6. How to run

```bash
# Terminal 1 — Python inference service (port 8000)
python src/serving/inference_service.py

# Terminal 2 — Node proxy + frontend (port 3001; use 3000 if free)
PORT=3001 node server/src/index.js
```

Open http://localhost:3001 and enter a student ID. After answering, run:

```bash
python scripts/evaluate_sessions.py
```

---

## 7. Known limitations / notes

- **3 legacy tests fail** by design: `test_inference_service.py::test_4_...` and
  `test_cumulative_inference.py` (×2) assert the deprecated fixed-order flow and
  the pre-patch `forward` signature. They were not updated because they test
  behavior this change intentionally replaced.
- **Template levels are approximate** bucket proxies; the logged-answer
  calibration step (`evaluate_sessions.py` by-level table) is the reconciliation
  point, not a claim of exactness.
- **Probe questions only fire when topics remain unasked.** If the model happens
  to touch all 10 topics inside 15 questions, `/next` completes with an empty
  `unasked_predictions` (correct behavior; the probe path is unit-tested).
- **Sessions are held in memory** (SQLite persists interactions); a restart can
  resume history via `from_history`, but a pending un-answered question is not
  persisted.
- Difficulty was derived from **train only**; the ~5–8% "unknown" bucket
  corresponds to problems with <5 train responses.
