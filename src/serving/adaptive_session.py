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

The level reacts ONLY to the actual answer (never to the prediction).
"""

import json
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from serving.question_templates import generate_question, LEVEL_TO_BUCKET

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

    def __init__(self, session_id: str, student_id: str):
        self.session_id = session_id
        self.student_id = student_id
        self.levels: Dict[int, str] = {t: "medium" for t in TOPIC_IDS}
        self.topic_count: Dict[int, int] = {t: 0 for t in TOPIC_IDS}
        self.asked_topics: set = set()
        self.history: List[Dict[str, Any]] = []
        self.pending: Optional[Dict[str, Any]] = None
        self.probe_queue: List[int] = []
        self.phase = "main"   # "main" -> "probe" -> "done"

    # ---- persistence ------------------------------------------------------

    @classmethod
    def from_history(cls, session_id: str, student_id: str, interactions: List[Dict[str, Any]]) -> "AdaptiveSession":
        """Rebuild state by replaying stored interactions (deterministic)."""
        s = cls(session_id, student_id)
        for it in interactions:
            topic = int(it["topic"] if "topic" in it else it["skill_id"])
            if topic not in TOPIC_BY_ID:
                continue
            correct = int(it["correct"])
            s.topic_count[topic] = s.topic_count.get(topic, 0) + 1
            s.asked_topics.add(topic)
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
        eligible = [t for t in TOPIC_IDS if self.topic_count[t] < MAX_PER_TOPIC]
        if len(self.history) >= 2 and self.history[-1]["topic"] == self.history[-2]["topic"]:
            blocked = self.history[-1]["topic"]
            relaxed = [t for t in eligible if t != blocked]
            if relaxed:
                eligible = relaxed
        return eligible

    def _choose_topic(self, predictor) -> Tuple[int, str, Optional[float]]:
        eligible = self._eligible_topics()
        if not eligible:
            eligible = [t for t in TOPIC_IDS if self.topic_count[t] < MAX_PER_TOPIC] or TOPIC_IDS

        candidates = [(t, LEVEL_TO_BUCKET[self.levels[t]]) for t in eligible]

        if predictor is not None and len(self.history) >= 2 and candidates:
            scores = predictor(self.history, candidates)
            idx = int(min(range(len(scores)), key=lambda i: scores[i]))
            topic = candidates[idx][0]
            return topic, self.levels[topic], float(scores[idx])

        topic = random.choice(eligible)
        return topic, self.levels[topic], None

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
            if len(self.history) < MAIN_QUESTIONS:
                if len(self.history) < WARMUP_QUESTIONS:
                    # Warm-up: random topic not yet asked, medium level, no prediction.
                    unasked = [t for t in TOPIC_IDS if t not in self.asked_topics]
                    topic = random.choice(unasked or TOPIC_IDS)
                    self.asked_topics.add(topic)
                    self.topic_count[topic] += 1
                    return self._make_question(topic, "medium", False, None)

                topic, level, predicted_p = self._choose_topic(predictor)
                self.asked_topics.add(topic)
                self.topic_count[topic] += 1
                return self._make_question(topic, level, False, predicted_p)

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
            return self._make_question(topic, "medium", True, predicted_p)

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
        if self.phase == "main" and len(self.history) >= MAIN_QUESTIONS:
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

    def __init__(self, store=None, log=None):
        self.store = store or (lambda session_id, student_id, inter: None)
        self.log = log or (lambda inter: None)
        self.sessions: Dict[str, AdaptiveSession] = {}

    def start(self, student_id: Optional[str] = None, session_id: Optional[str] = None) -> Tuple[str, str]:
        sid = session_id or str(uuid.uuid4())
        st_id = (student_id or str(uuid.uuid4())[:8]).strip() or "guest_student"
        self.sessions[sid] = AdaptiveSession(sid, st_id)
        return sid, st_id

    def get(self, session_id: str) -> AdaptiveSession:
        if session_id in self.sessions:
            return self.sessions[session_id]
        raise KeyError(f"Adaptive session '{session_id}' not found")

    def next_question(self, session_id: str, predictor=None) -> Optional[Dict[str, Any]]:
        sess = self.get(session_id)
        q = sess.next_question(predictor)
        if q is not None:
            self.log({
                "event": "question",
                "session_id": session_id,
                "position": q["position"],
                "topic": q["topic"],
                "level": q["level"],
                "difficulty_bucket": q["difficulty_bucket"],
                "p_before": q["predicted_p"],
                "is_probe": q["is_probe"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        return q

    def record_answer(self, session_id: str, submitted_answer: Any, response_time_ms: float,
                      hint_used: int = 0) -> Dict[str, Any]:
        sess = self.get(session_id)
        inter = sess.record_answer(submitted_answer, response_time_ms, hint_used)
        self.store(session_id, sess.student_id, inter)
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
