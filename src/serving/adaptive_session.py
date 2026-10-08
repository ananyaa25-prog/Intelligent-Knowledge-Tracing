"""
src/serving/adaptive_session.py
--------------------------------
Adaptive session engine.

Replaces the fixed 10-question, fixed-order flow with a model-driven session:

- 10 topics, each with easy / medium / hard generators (level -> difficulty bucket).
- A session is 15 main questions. The first 2 topics are random (warm-up; the
  model needs >= 2 interactions). After that, the next TOPIC is chosen by
  predict_candidates (lowest predicted P(correct) among eligible topics).
- Caps: at most 3 questions per topic, at most 2 in a row on the same topic.
- Per-topic level is a staircase: start medium, correct moves up one level,
  wrong moves down one (clamped to easy/hard).
- Questions are generated one at a time; predicted_p is computed BEFORE the
  question is shown.
- At the end, 1-2 probe questions are asked on topics never asked, and
  graph-based predictions are produced for all unasked topics.

The level reacts ONLY to the actual answer (never to the prediction) under the
default POLICY=staircase. Optional policies (see policy_config.py):

- POLICY=model: topic selection is unchanged; for the chosen topic the level is
  picked among {current level, one step down, one step up} as the level whose
  predicted P(correct) is closest to TARGET_P, with guardrails (two consecutive
  wrong answers on the topic force one step down; Hard only after the topic's
  last Medium/Hard answer was correct). Any predictor failure falls back to the
  staircase.
- POLICY=random: uniform random level (evaluation baseline only).
- SESSION_MODE=core5: main questions only from CORE_TOPICS (max 4 per topic),
  other topics only via probes + graph predictions.

Every question carries a `decision` object (policy, topic scores, level
candidates, chosen level, reason built only from those numbers).
"""

import json
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from serving.question_templates import generate_question, LEVEL_TO_BUCKET
from serving.policy_config import get_config, CORE5_MAX_PER_TOPIC, CORE5_TOTAL_QUESTIONS

LEVEL_INDEX = {"easy": 0, "medium": 1, "hard": 2}
LEVEL_NAMES = ["easy", "medium", "hard"]

MAIN_QUESTIONS = 15
WARMUP_QUESTIONS = 2
MAX_PER_TOPIC = 3
MAX_CONSECUTIVE = 2
PROBE_QUESTIONS = 2

# The 10 topics (mirrors question_bank.json skill_sequence).
TOPICS: List[Dict[str, Any]] = [
    {"position": 1, "skill_name": "Percents", "raw_skill_id": 204},
    {"position": 2, "skill_name": "Percent Discount", "raw_skill_id": 203},
    {"position": 3, "skill_name": "Rate", "raw_skill_id": 217},
    {"position": 4, "skill_name": "Polynomial Factors", "raw_skill_id": 346},
    {"position": 5, "skill_name": "Simplifying Expressions positive exponents", "raw_skill_id": 371},
    {"position": 6, "skill_name": "Solving Systems of Linear Equations", "raw_skill_id": 350},
    {"position": 7, "skill_name": "Solving Systems of Linear Equations by Graphing", "raw_skill_id": 378},
    {"position": 8, "skill_name": "Ordering Fractions", "raw_skill_id": 50},
    {"position": 9, "skill_name": "Equivalent Fractions", "raw_skill_id": 48},
    {"position": 10, "skill_name": "Pythagorean Theorem", "raw_skill_id": 27},
]

TOPIC_BY_ID: Dict[int, Dict[str, Any]] = {t["raw_skill_id"]: t for t in TOPICS}
TOPIC_IDS: List[int] = [t["raw_skill_id"] for t in TOPICS]


class AdaptiveSession:
    """Per-session adaptive state. Pure state + logic; no DB or model I/O."""

    def __init__(self, session_id: str, student_id: str, policy: Optional[str] = None,
                 session_mode: Optional[str] = None, target_p: Optional[float] = None,
                 core_topics: Optional[List[int]] = None):
        self.session_id = session_id
        self.student_id = student_id
        cfg = get_config()
        self.policy = policy or cfg.policy
        self.session_mode = session_mode or cfg.session_mode
        self.target_p = float(target_p if target_p is not None else cfg.target_p)
        if self.session_mode == "core5":
            core = [t for t in (core_topics or cfg.core_topics) if t in TOPIC_BY_ID]
            self.main_topics: List[int] = core or list(TOPIC_IDS)
            self.max_per_topic = CORE5_MAX_PER_TOPIC
            self.main_questions = CORE5_TOTAL_QUESTIONS - PROBE_QUESTIONS
        else:
            self.main_topics = list(TOPIC_IDS)
            self.max_per_topic = MAX_PER_TOPIC
            self.main_questions = MAIN_QUESTIONS
        self.last_topic_scores: Dict[int, float] = {}
        self.policy_fallbacks = 0
        self.levels: Dict[int, str] = {t: "medium" for t in TOPIC_IDS}
        self.topic_count: Dict[int, int] = {t: 0 for t in TOPIC_IDS}
        self.asked_topics: set = set()
        self.history: List[Dict[str, Any]] = []
        self.pending: Optional[Dict[str, Any]] = None
        self.probe_queue: List[int] = []
        self.phase = "main"   # "main" -> "probe" -> "done"

    # ---- persistence ------------------------------------------------------

    @classmethod
    def from_history(cls, session_id: str, student_id: str, interactions: List[Dict[str, Any]],
                     **kwargs) -> "AdaptiveSession":
        """Rebuild state by replaying stored interactions (deterministic)."""
        s = cls(session_id, student_id, **kwargs)
        for it in interactions:
            topic = int(it["topic"] if "topic" in it else it["skill_id"])
            if topic not in TOPIC_BY_ID:
                continue
            correct = int(it["correct"])
            s.topic_count[topic] = s.topic_count.get(topic, 0) + 1
            s.asked_topics.add(topic)
            if s.policy != "staircase" and it.get("level") in LEVEL_INDEX:
                s.levels[topic] = it["level"]  # non-staircase: step from the served level
            cur = s.levels[topic]
            idx = LEVEL_INDEX[cur]
            if correct:
                idx = min(idx + 1, 2)
            else:
                idx = max(idx - 1, 0)
            s.levels[topic] = LEVEL_NAMES[idx]
            s.history.append(it)
        return s

    # ---- selection --------------------------------------------------------

    def _eligible_topics(self) -> List[int]:
        eligible = [t for t in self.main_topics if self.topic_count[t] < self.max_per_topic]
        if len(self.history) >= 2 and self.history[-1]["topic"] == self.history[-2]["topic"]:
            blocked = self.history[-1]["topic"]
            relaxed = [t for t in eligible if t != blocked]
            if relaxed:
                eligible = relaxed
        return eligible

    def _choose_topic(self, predictor) -> Tuple[int, str, Optional[float]]:
        eligible = self._eligible_topics()
        if not eligible:
            eligible = [t for t in self.main_topics if self.topic_count[t] < self.max_per_topic] or self.main_topics

        candidates = [(t, LEVEL_TO_BUCKET[self.levels[t]]) for t in eligible]
        self.last_topic_scores = {}

        if predictor is not None and len(self.history) >= 2 and candidates:
            scores = predictor(self.history, candidates)
            self.last_topic_scores = {c[0]: float(sc) for c, sc in zip(candidates, scores)}
            idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            topic = candidates[idx][0]
            return topic, self.levels[topic], float(scores[idx])

        topic = random.choice(eligible)
        return topic, self.levels[topic], None

    # ---- difficulty policy --------------------------------------------------

    def _score_levels(self, topic: int, predictor) -> Dict[str, float]:
        """P(correct) for the topic at easy/medium/hard (one predictor call)."""
        if predictor is None or len(self.history) < 2:
            return {}
        cands = [(topic, LEVEL_TO_BUCKET[lvl]) for lvl in LEVEL_NAMES]
        scores = predictor(self.history, cands)
        return {lvl: float(sc) for lvl, sc in zip(LEVEL_NAMES, scores)}

    def _model_level(self, topic: int, scores: Dict[str, float]) -> Tuple[str, Optional[str]]:
        """Pick the level whose predicted P is closest to target_p among the
        current level and one step down/up, after guardrails. Returns
        (level, guardrail_name_or_None)."""
        base = LEVEL_INDEX[self.levels[topic]]
        allowed = sorted({max(base - 1, 0), base, min(base + 1, 2)})
        topic_hist = [h for h in self.history if int(h["topic"]) == topic]
        guard = None
        # G1: two consecutive wrong answers on this topic force one step down.
        if len(topic_hist) >= 2 and int(topic_hist[-1]["correct"]) == 0 and int(topic_hist[-2]["correct"]) == 0:
            last_idx = LEVEL_INDEX.get(topic_hist[-1].get("level"), base)
            allowed = [max(last_idx - 1, 0)]
            guard = "step_down_after_two_wrong"
        # G2: Hard only if the topic's last Medium/Hard answer was correct.
        mh = [h for h in topic_hist if h.get("level") in ("medium", "hard")]
        hard_ok = bool(mh) and int(mh[-1]["correct"]) == 1
        if not hard_ok and 2 in allowed:
            allowed = [i for i in allowed if i != 2] or [1]
            guard = guard or "hard_locked"
        best = min(allowed, key=lambda i: (abs(scores[LEVEL_NAMES[i]] - self.target_p), i != base, i))
        return LEVEL_NAMES[best], guard

    def _decide_level(self, topic: int, predictor) -> Tuple[str, Dict[str, float], Optional[str], bool]:
        """Returns (level, level_candidates, guardrail, fell_back)."""
        level = self.levels[topic]
        try:
            cand = self._score_levels(topic, predictor)
        except Exception as e:
            print(f"[AdaptiveSession] level scoring failed ({e}); staircase level kept")
            cand = {}
        if self.policy == "random":
            return random.choice(LEVEL_NAMES), cand, None, False
        if self.policy == "model":
            if len(cand) != 3:
                self.policy_fallbacks += 1
                print("[AdaptiveSession] model policy fallback to staircase (no level scores)")
                return level, cand, None, True
            lvl, guard = self._model_level(topic, cand)
            return lvl, cand, guard, False
        return level, cand, None, False

    def _build_decision(self, topic: int, level: str, cand: Dict[str, float],
                        guard: Optional[str], fell_back: bool, kind: str) -> Dict[str, Any]:
        """Decision object; the reason text is built only from these numbers/rules."""
        name = TOPIC_BY_ID[topic]["skill_name"]

        def pct(v):
            return f"{round(v * 100)}%"

        parts = []
        topic_scores = {str(t): round(v, 4) for t, v in self.last_topic_scores.items()} if kind == "main" else {}
        if kind == "warmup":
            parts.append("Warm-up question: topic picked at random at Medium; the model needs 2 answers before it can predict.")
        elif kind == "probe":
            p = cand.get("medium")
            parts.append(f"Probe on '{name}', a topic not asked yet"
                         + (f"; predicted {pct(p)} recorded before answering to check the related-topic prediction." if p is not None else "."))
        else:
            if self.last_topic_scores:
                parts.append(f"Topic '{name}' had the lowest predicted success ({pct(self.last_topic_scores[topic])}) "
                             f"among {len(self.last_topic_scores)} eligible topics.")
            cand_txt = ", ".join(f"{l} {pct(cand[l])}" for l in LEVEL_NAMES if l in cand)
            if self.policy == "model" and not fell_back:
                parts.append(f"Level {level}: predicted {pct(cand[level])}, closest to the {pct(self.target_p)} target "
                             f"among allowed levels ({cand_txt}).")
                if guard == "hard_locked":
                    parts.append("Hard unlocks after a correct Medium or Hard answer on this topic.")
                elif guard:
                    parts.append("A guardrail rule limited this topic's level.")
            elif self.policy == "random":
                parts.append(f"Level {level} chosen at random (evaluation baseline).")
            else:
                if fell_back:
                    parts.append("Model level scoring unavailable; staircase rule used.")
                parts.append(f"Level {level} set by the staircase rule (one step per answer on this topic)"
                             + (f"; predicted {cand_txt}." if cand_txt else "."))
        return {
            "policy": self.policy,
            "topic_scores": topic_scores,
            "level_candidates": {k: round(v, 4) for k, v in cand.items()},
            "chosen_level": level,
            "reason": " ".join(parts),
            "guardrail": guard,
            "fallback": bool(fell_back),
        }

    def total_questions(self) -> int:
        """Number of questions the student is expected to see in this session."""
        if self.phase == "main":
            unasked = [t for t in TOPIC_IDS if t not in self.asked_topics]
            return self.main_questions + min(PROBE_QUESTIONS, len(unasked))
        return len(self.history) + (1 if self.pending else 0) + len(self.probe_queue)

    # ---- question generation ----------------------------------------------

    def _make_question(self, topic: int, level: str, is_probe: bool, predicted_p: Optional[float]) -> Dict[str, Any]:
        meta = TOPIC_BY_ID[topic]
        q = generate_question(meta["position"], level)
        position = len(self.history) + 1
        pending = {
            "position": position,
            "topic": topic,
            "topic_position": meta["position"],
            "skill_name": meta["skill_name"],
            "level": level,
            "difficulty_bucket": LEVEL_TO_BUCKET[level],
            "question_text": q["question_text"],
            "correct_answer": q["correct_answer"],
            "predicted_p": predicted_p,
            "is_probe": is_probe,
        }
        self.pending = pending
        return pending

    def next_question(self, predictor=None) -> Optional[Dict[str, Any]]:
        if self.phase == "done":
            return None

        if self.phase == "main":
            if len(self.history) < self.main_questions:
                if len(self.history) < WARMUP_QUESTIONS:
                    # Warm-up: random topic not yet asked, medium level, no prediction.
                    unasked = [t for t in self.main_topics if t not in self.asked_topics]
                    topic = random.choice(unasked or self.main_topics)
                    self.asked_topics.add(topic)
                    self.topic_count[topic] += 1
                    q = self._make_question(topic, "medium", False, None)
                    q["decision"] = self._build_decision(topic, "medium", {}, None, False, "warmup")
                    return q

                topic, level, predicted_p = self._choose_topic(predictor)
                self.asked_topics.add(topic)
                self.topic_count[topic] += 1
                level, cand, guard, fell_back = self._decide_level(topic, predictor)
                if self.policy != "staircase" and level in cand:
                    predicted_p = cand[level]  # prediction for the level actually served
                if self.policy != "staircase":
                    self.levels[topic] = level  # the staircase steps from the served level
                q = self._make_question(topic, level, False, predicted_p)
                q["decision"] = self._build_decision(topic, level, cand, guard, fell_back, "main")
                return q

            # Main phase complete -> transition to probe phase exactly once.
            self.phase = "probe"
            unasked = [t for t in TOPIC_IDS if t not in self.asked_topics]
            if not unasked:
                self.phase = "done"
                return None
            self.probe_queue = random.sample(unasked, min(PROBE_QUESTIONS, len(unasked)))

        # phase == "probe"
        if self.probe_queue:
            topic = self.probe_queue.pop(0)
            predicted_p = None
            if predictor is not None and len(self.history) >= 2:
                predicted_p = float(predictor(self.history, [(topic, LEVEL_TO_BUCKET["medium"])])[0])
            q = self._make_question(topic, "medium", True, predicted_p)
            cand = {"medium": predicted_p} if predicted_p is not None else {}
            q["decision"] = self._build_decision(topic, "medium", cand, None, False, "probe")
            return q

        self.phase = "done"
        return None

    def record_answer(self, submitted_answer: Any, response_time_ms: float, hint_used: int = 0) -> Optional[Dict[str, Any]]:
        if self.pending is None:
            raise ValueError("No pending question for this session; call next_question first.")

        pending = self.pending
        topic = pending["topic"]
        is_correct = _check_answer(submitted_answer, pending["correct_answer"])

        # Staircase reacts to the actual answer only.
        idx = LEVEL_INDEX[self.levels[topic]]
        idx = min(idx + 1, 2) if is_correct else max(idx - 1, 0)
        self.levels[topic] = LEVEL_NAMES[idx]

        interaction = {
            "position": pending["position"],
            "topic": topic,
            "skill_id": topic,
            "skill_name": pending["skill_name"],
            "level": pending["level"],
            "difficulty_bucket": pending["difficulty_bucket"],
            "correct": 1 if is_correct else 0,
            "p_before": pending["predicted_p"],
            "response_time_ms": float(response_time_ms),
            "attempt_count": 1,
            "hint_used": int(hint_used),
            "question_text": pending["question_text"],
            "correct_answer": pending["correct_answer"],
            "is_probe": int(pending["is_probe"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(interaction)
        self.pending = None
        return interaction

    def is_complete(self) -> bool:
        if self.phase == "main" and len(self.history) >= self.main_questions:
            return False  # probes may still be pending
        return self.phase == "done" or (self.phase == "probe" and not self.probe_queue)

    def unasked_topics(self) -> List[int]:
        return [t for t in TOPIC_IDS if t not in self.asked_topics]


def _check_answer(submitted: Any, correct: Any) -> bool:
    if submitted is None:
        return False
    sub_str = str(submitted).strip().lower()
    corr_str = str(correct).strip().lower()
    if sub_str == corr_str:
        return True
    try:
        return abs(float(sub_str) - float(corr_str)) <= 0.01
    except (ValueError, TypeError):
        return False


class AdaptiveSessionManager:
    """Holds adaptive sessions in memory and persists interactions via a
    `store` callback (SQLite) and `log` callback (JSONL file)."""

    def __init__(self, store=None, log=None, on_answer=None):
        self.store = store or (lambda session_id, student_id, inter: None)
        self.log = log or (lambda inter: None)
        self.on_answer = on_answer  # optional hook (e.g. profile-cache invalidation)
        self.sessions: Dict[str, AdaptiveSession] = {}

    def start(self, student_id: Optional[str] = None, session_id: Optional[str] = None,
              **session_kwargs) -> Tuple[str, str]:
        sid = session_id or str(uuid.uuid4())
        st_id = (student_id or str(uuid.uuid4())[:8]).strip() or "guest_student"
        self.sessions[sid] = AdaptiveSession(sid, st_id, **session_kwargs)
        return sid, st_id

    def get(self, session_id: str) -> AdaptiveSession:
        if session_id in self.sessions:
            return self.sessions[session_id]
        raise KeyError(f"Adaptive session '{session_id}' not found")

    def next_question(self, session_id: str, predictor=None) -> Optional[Dict[str, Any]]:
        sess = self.get(session_id)
        q = sess.next_question(predictor)
        if q is not None:
            event = {
                "event": "question",
                "session_id": session_id,
                "position": q["position"],
                "topic": q["topic"],
                "level": q["level"],
                "difficulty_bucket": q["difficulty_bucket"],
                "p_before": q["predicted_p"],
                "is_probe": q["is_probe"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "student_id": sess.student_id,
                "policy": sess.policy,
                "session_mode": sess.session_mode,
                "decision": q.get("decision"),
            }
            if q["is_probe"]:
                event["predicted_before_probe"] = q["predicted_p"]
            self.log(event)
        return q

    def record_answer(self, session_id: str, submitted_answer: Any, response_time_ms: float,
                      hint_used: int = 0) -> Dict[str, Any]:
        sess = self.get(session_id)
        inter = sess.record_answer(submitted_answer, response_time_ms, hint_used)
        self.store(session_id, sess.student_id, inter)
        if self.on_answer is not None:
            try:
                self.on_answer(sess.student_id)
            except Exception:
                pass
        self.log({
            "event": "answer",
            "session_id": session_id,
            "student_id": sess.student_id,
            "position": inter["position"],
            "topic": inter["topic"],
            "level": inter["level"],
            "difficulty_bucket": inter["difficulty_bucket"],
            "p_before": inter["p_before"],
            "outcome": inter["correct"],
            "is_probe": inter["is_probe"],
            "timestamp": inter["timestamp"],
        })
        return inter

    def history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.get(session_id).history
