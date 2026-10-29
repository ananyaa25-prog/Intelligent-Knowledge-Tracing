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
  picked as the allowed level whose predicted P(correct) is closest to TARGET_P.
  Allowed levels follow the student's answers on that topic (rules revised after
  user testing, docs/DECISIONS.md D27): the first question on a topic is Medium;
  after a correct answer the level never gets easier ({same, one up}); after a
  wrong answer it never gets harder ({one down, same}); two correct in a row
  force one step up, two wrong in a row force one step down; Hard only after the
  topic's last Medium/Hard answer was correct. Any predictor failure falls back
  to the staircase.
- Cross-session history (D28): answers from the student's previous sessions
  (prior_history) are used for every prediction and to resume each topic's
  level, so a returning student gets no blind warm-up.
- POLICY=random: uniform random level (evaluation baseline only).
- SESSION_MODE=core5: main questions only from CORE_TOPICS (max 4 per topic),
  other topics only via probes + graph predictions.

Every question carries a `decision` object (policy, topic scores, level
candidates, chosen level, reason built only from those numbers).
"""

import json
import random
import uuid

import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from serving.question_templates import generate_question, LEVEL_TO_BUCKET
BUCKET_TO_LEVEL = {1: "easy", 2: "medium", 3: "hard"}
from serving.policy_config import (get_config, CORE5_MAX_PER_TOPIC, CORE5_TOTAL_QUESTIONS,
                                   EVIDENCE_PRIOR_WEIGHT)
from serving.sections import (normalise_section, section_topics, assist_skill_for_topic,
                              SECTIONS, VALIDATED)

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
                 core_topics: Optional[List[int]] = None,
                 prior_history: Optional[List[Dict[str, Any]]] = None,
                 topic_selection: Optional[str] = None, section: Optional[str] = None,
                 question_source: Optional[str] = None):
        self.session_id = session_id
        self.student_id = student_id
        cfg = get_config()
        # v3 sections: None = the original 10 ASSISTments topics (unchanged behaviour).
        self.section = normalise_section(section)
        self.topics: List[Dict[str, Any]] = section_topics(self.section) if self.section else TOPICS
        self.topic_by_id: Dict[int, Dict[str, Any]] = {t["raw_skill_id"]: t for t in self.topics}
        self.topic_ids: List[int] = [t["raw_skill_id"] for t in self.topics]
        self.validated = self.section is None or SECTIONS[self.section]["validation"] == VALIDATED
        # whether the served ASSISTments model is used for this section (D37); the label above
        # only reports whether a real dataset for the level was evaluated
        self.uses_model = self.section is None or SECTIONS[self.section].get("serving") == "model"
        self.question_source = question_source or getattr(cfg, "question_source", "bank")
        self.policy = policy or cfg.policy
        self.session_mode = session_mode or cfg.session_mode
        self.target_p = float(target_p if target_p is not None else cfg.target_p)
        if self.session_mode == "core5":
            core = [t for t in (core_topics or cfg.core_topics) if t in self.topic_by_id]
            self.main_topics: List[int] = core or list(self.topic_ids)
            self.max_per_topic = CORE5_MAX_PER_TOPIC
            self.main_questions = CORE5_TOTAL_QUESTIONS - PROBE_QUESTIONS
        else:
            self.main_topics = list(self.topic_ids)
            self.max_per_topic = MAX_PER_TOPIC
            self.main_questions = MAIN_QUESTIONS
        if self.section is not None:
            # a section with few topics cannot fill the session within the per-topic cap
            self.main_questions = min(self.main_questions, len(self.main_topics) * self.max_per_topic)
        self.last_topic_scores: Dict[int, float] = {}
        self.last_topic_model_scores: Dict[int, float] = {}
        self.policy_fallbacks = 0
        self.low_confidence_count = 0
        self.topic_selection = topic_selection or cfg.topic_selection
        if self.section and not self.uses_model:
            # transfer / evidence-only mode: the model has no data for these topics, so rank topics by the
            # student's own answers (evidence blend with a neutral prior), D35.
            self.topic_selection = "evidence_blend"
            self.policy = "staircase"   # levels follow the answers only (no model levels)
        self.levels: Dict[int, str] = {t: "medium" for t in self.topic_ids}
        self.topic_count: Dict[int, int] = {t: 0 for t in self.topic_ids}
        self.asked_topics: set = set()
        self.history: List[Dict[str, Any]] = []
        self.pending: Optional[Dict[str, Any]] = None
        self.probe_queue: List[int] = []
        self.phase = "main"   # "main" -> "probe" -> "done"
        # Answers from the student's earlier sessions (oldest first); used for
        # predictions and to resume each topic's level. Empty = old behaviour.
        self.prior_history: List[Dict[str, Any]] = []
        for it in (prior_history or []):
            norm = _normalise_item(it, self.topic_by_id)
            if norm is not None:
                self.prior_history.append(norm)
        for it in self.prior_history:
            self._replay_level(int(it["topic"]), it)

    # ---- persistence ------------------------------------------------------

    @classmethod
    def from_history(cls, session_id: str, student_id: str, interactions: List[Dict[str, Any]],
                     **kwargs) -> "AdaptiveSession":
        """Rebuild state by replaying stored interactions (deterministic)."""
        if kwargs.get("section") is None:
            kwargs["section"] = infer_section(interactions)
        s = cls(session_id, student_id, **kwargs)
        for it in interactions:
            topic = int(it["topic"] if "topic" in it else it["skill_id"])
            if topic not in s.topic_by_id:
                continue
            correct = int(it["correct"])
            s.topic_count[topic] = s.topic_count.get(topic, 0) + 1
            s.asked_topics.add(topic)
            s._replay_level(topic, it)
            s.history.append(it)
        return s

    def _replay_level(self, topic: int, it: Dict[str, Any]) -> None:
        """Apply one stored answer to the topic's level state (same rule as record_answer)."""
        if topic not in self.levels:
            return
        if self.policy != "staircase" and _item_level(it) in LEVEL_INDEX:
            self.levels[topic] = _item_level(it)  # non-staircase: step from the served level
        idx = LEVEL_INDEX[self.levels[topic]]
        idx = min(idx + 1, 2) if int(it["correct"]) else max(idx - 1, 0)
        self.levels[topic] = LEVEL_NAMES[idx]

    def context(self) -> List[Dict[str, Any]]:
        """History the predictor sees: earlier sessions + this session."""
        return self.prior_history + self.history if self.prior_history else self.history

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

        if predictor is not None and len(self.context()) >= 2 and candidates:
            scores = predictor(self.context(), candidates)
            self.last_topic_model_scores = {c[0]: float(sc) for c, sc in zip(candidates, scores)}
            rank = list(scores)
            if self.topic_selection == "evidence_blend":
                # D31: combine the model prediction with the student's own record on
                # each topic, so topics answered correctly are not re-drilled while a
                # topic answered wrongly is revisited.
                ctx = self.context()
                rank = []
                for (t, _), sc in zip(candidates, scores):
                    th = [h for h in ctx if _item_topic(h) == t]
                    n_c = sum(int(h["correct"]) for h in th)
                    rank.append((EVIDENCE_PRIOR_WEIGHT * float(sc) + n_c) / (EVIDENCE_PRIOR_WEIGHT + len(th)))
            self.last_topic_scores = {c[0]: float(r) for c, r in zip(candidates, rank)}
            idx = int(min(range(len(rank)), key=lambda i: rank[i]))
            topic = candidates[idx][0]
            return topic, self.levels[topic], float(scores[idx])

        topic = random.choice(eligible)
        return topic, self.levels[topic], None

    # ---- difficulty policy --------------------------------------------------

    def _score_levels(self, topic: int, predictor) -> Dict[str, float]:
        """P(correct) for the topic at easy/medium/hard (one predictor call)."""
        if predictor is None or len(self.context()) < 2:
            return {}
        cands = [(topic, LEVEL_TO_BUCKET[lvl]) for lvl in LEVEL_NAMES]
        scores = predictor(self.context(), cands)
        return {lvl: float(sc) for lvl, sc in zip(LEVEL_NAMES, scores)}

    def _model_level(self, topic: int, scores: Dict[str, float],
                     target: Optional[float] = None) -> Tuple[str, Optional[str]]:
        """Pick the allowed level whose predicted P is closest to target_p.
        Allowed levels follow the student's answers on this topic (D27).
        Returns (level, guardrail_name_or_None)."""
        topic_hist = [h for h in self.context() if _item_topic(h) == topic]
        # G0: first question on a topic is Medium (no answers on it yet; the
        # model's estimate for an unseen topic is unreliable, reports/answer_response.md).
        if not topic_hist:
            return "medium", "first_question_on_topic"
        last = topic_hist[-1]
        last_idx = LEVEL_INDEX.get(_item_level(last) or "medium", 1)
        if int(last["correct"]):
            allowed = sorted({last_idx, min(last_idx + 1, 2)})   # never easier after a correct answer
        else:
            allowed = sorted({max(last_idx - 1, 0), last_idx})   # never harder after a wrong answer
        guard = None
        if len(topic_hist) >= 2 and int(topic_hist[-2]["correct"]) == int(last["correct"]):
            if int(last["correct"]):
                allowed = [min(last_idx + 1, 2)]                  # G3: two correct in a row -> step up
                guard = "step_up_after_two_correct"
            else:
                allowed = [max(last_idx - 1, 0)]                  # G1: two wrong in a row -> step down
                guard = "step_down_after_two_wrong"
        # G2: Hard only if the topic's last Medium/Hard answer was correct.
        mh = [h for h in topic_hist if _item_level(h) in ("medium", "hard")]
        hard_ok = bool(mh) and int(mh[-1]["correct"]) == 1
        if not hard_ok and 2 in allowed:
            allowed = [i for i in allowed if i != 2] or [1]
            guard = guard or "hard_locked"
        tgt = self.target_p if target is None else float(target)
        best = min(allowed, key=lambda i: (abs(scores[LEVEL_NAMES[i]] - tgt), i != last_idx, i))
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
        if self.policy in ("elo", "maxinfo", "uncertainty"):
            return self._decide_level_v3(topic, predictor, level, cand)
        if self.policy == "model":
            if len(cand) != 3:
                self.policy_fallbacks += 1
                print("[AdaptiveSession] model policy fallback to staircase (no level scores)")
                return level, cand, None, True
            lvl, guard = self._model_level(topic, cand)
            return lvl, cand, guard, False
        return level, cand, None, False

    def _decide_level_v3(self, topic: int, predictor, level: str,
                         cand: Dict[str, float]) -> Tuple[str, Dict[str, float], Optional[str], bool]:
        """v3 policies (src/serving/policies_v3.py). Same allowed-level guardrails as
        POLICY=model; only the score source / target differs. Any failure -> staircase."""
        from serving import policies_v3 as pv
        self.last_policy_scores = {}
        self.last_low_confidence = False
        try:
            if self.policy == "elo":
                scores = pv.elo_scores(self.context(), topic, _item_topic)
                self.last_policy_scores = scores
                lvl, guard = self._model_level(topic, scores)
                return lvl, cand, guard, False
            if len(cand) != 3:
                raise ValueError("no model level scores")
            if self.policy == "maxinfo":
                self.last_policy_scores = dict(cand)
                lvl, guard = self._model_level(topic, cand, target=pv.MAXINFO_TARGET)
                return lvl, cand, guard, False
            # uncertainty-aware: ensemble mean, staircase level if the members disagree
            ens = pv.ensemble_scores(getattr(predictor, "ensemble", None), self.context(),
                                     [(topic, LEVEL_TO_BUCKET[l]) for l in LEVEL_NAMES])
            if not ens:
                raise ValueError("ensemble unavailable")
            mean = {l: float(np.mean([m[i] for m in ens])) for i, l in enumerate(LEVEL_NAMES)}
            spread = {l: float(max(m[i] for m in ens) - min(m[i] for m in ens)) for i, l in enumerate(LEVEL_NAMES)}
            self.last_policy_scores = mean
            self.last_spread = spread
            lvl, guard = self._model_level(topic, mean)
            if spread[lvl] > pv.UNCERTAINTY_MAX_SPREAD:
                self.last_low_confidence = True
                self.low_confidence_count += 1
                return level, cand, "low_confidence_prediction", False
            return lvl, cand, guard, False
        except Exception as e:
            self.policy_fallbacks += 1
            print(f"[AdaptiveSession] {self.policy} policy fallback to staircase ({e})")
            return level, cand, None, True

    def _build_decision(self, topic: int, level: str, cand: Dict[str, float],
                        guard: Optional[str], fell_back: bool, kind: str) -> Dict[str, Any]:
        """Decision object; the reason text is built only from these numbers/rules."""
        name = self.topic_by_id[topic]["skill_name"]

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
            if self.last_topic_scores and self.topic_selection == "evidence_blend":
                th = [h for h in self.context() if _item_topic(h) == topic]
                n_c = sum(int(h["correct"]) for h in th)
                parts.append(f"Topic '{name}' had the lowest combined estimate ({pct(self.last_topic_scores[topic])}: model "
                             f"{pct(self.last_topic_model_scores[topic])} combined with {n_c} of {len(th)} earlier answers on it "
                             f"correct) among {len(self.last_topic_scores)} eligible topics.")
            elif self.last_topic_scores:
                parts.append(f"Topic '{name}' had the lowest predicted success ({pct(self.last_topic_scores[topic])}) "
                             f"among {len(self.last_topic_scores)} eligible topics.")
            cand_txt = ", ".join(f"{l} {pct(cand[l])}" for l in LEVEL_NAMES if l in cand)
            if self.policy == "model" and not fell_back:
                parts.append(f"Level {level}: predicted {pct(cand[level])}, closest to the {pct(self.target_p)} target "
                             f"among allowed levels ({cand_txt}).")
                if guard == "hard_locked":
                    parts.append("Hard unlocks after a correct Medium or Hard answer on this topic.")
                elif guard == "first_question_on_topic":
                    parts.append("First question on this topic starts at Medium.")
                elif guard:
                    parts.append("A progression rule (based on your answers on this topic) set the level.")
            elif self.policy == "random":
                parts.append(f"Level {level} chosen at random (evaluation baseline).")
            elif self.policy in ("elo", "maxinfo", "uncertainty") and not fell_back:
                ps = getattr(self, "last_policy_scores", {}) or {}
                ps_txt = ", ".join(f"{l} {pct(ps[l])}" for l in LEVEL_NAMES if l in ps)
                if self.policy == "elo":
                    parts.append(f"Level {level}: Elo estimate from your answers closest to {pct(self.target_p)} "
                                 f"among allowed levels ({ps_txt}).")
                elif self.policy == "maxinfo":
                    parts.append(f"Level {level}: predicted success closest to 50% (most informative) among "
                                 f"allowed levels ({ps_txt}).")
                elif guard == "low_confidence_prediction":
                    parts.append(f"Low-confidence prediction: the model ensemble disagrees, so level {level} "
                                 "follows the staircase rule.")
                else:
                    parts.append(f"Level {level}: ensemble prediction closest to {pct(self.target_p)} among "
                                 f"allowed levels ({ps_txt}).")
                if guard and guard != "low_confidence_prediction":
                    parts.append("A progression rule (based on your answers on this topic) limited the levels.")
            else:
                if fell_back:
                    parts.append("Model level scoring unavailable; staircase rule used.")
                parts.append(f"Level {level} set by the staircase rule (one step per answer on this topic)"
                             + (f"; predicted {cand_txt}." if cand_txt else "."))
        return {
            "policy": self.policy,
            "topic_selection": self.topic_selection,
            "topic_scores": topic_scores,
            "level_candidates": {k: round(v, 4) for k, v in cand.items()},
            "chosen_level": level,
            "reason": " ".join(parts),
            "guardrail": guard,
            "fallback": bool(fell_back),
            # v3 additive fields
            **({"policy_scores": {k: round(v, 4) for k, v in (getattr(self, "last_policy_scores", {}) or {}).items()},
                "low_confidence": bool(getattr(self, "last_low_confidence", False))}
               if self.policy in ("elo", "maxinfo", "uncertainty") and kind == "main" else {}),
        }

    def total_questions(self) -> int:
        """Number of questions the student is expected to see in this session."""
        if self.phase == "main":
            unasked = [t for t in self.topic_ids if t not in self.asked_topics]
            return self.main_questions + min(PROBE_QUESTIONS, len(unasked))
        return len(self.history) + (1 if self.pending else 0) + len(self.probe_queue)

    # ---- question generation ----------------------------------------------

    def _make_question(self, topic: int, level: str, is_probe: bool, predicted_p: Optional[float]) -> Dict[str, Any]:
        meta = self.topic_by_id[topic]
        used = {h.get("question_text") for h in self.history}
        if self._use_generator(meta):
            q = self._generated_question(meta, level, used)
        else:
            q = generate_question(meta["position"], level)
            if q["question_text"] in used:
                # Avoid repeating a question within the session (D29). Restore the RNG
                # afterwards so topic/level choices are unaffected by the retries.
                state_after_first = random.getstate()
                for _ in range(15):
                    q2 = generate_question(meta["position"], level)
                    if q2["question_text"] not in used:
                        q = q2
                        break
                random.setstate(state_after_first)
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
        if q.get("source") == "generated":
            # v3 generated item: answer type / choices are public; the checker spec stays server-side.
            pending["question_source"] = "generated"
            pending["answer_type"] = q.get("answer_type", "numeric")
            if q.get("choices"):
                pending["choices"] = list(q["choices"])
            pending["_item"] = q
        self.pending = pending
        return pending

    # ---- v3: sections and generated questions -------------------------------

    def _use_generator(self, meta: Dict[str, Any]) -> bool:
        """Section topics always use the parametric generator (they have no bank).
        Default topics use it only if QUESTION_SOURCE=generated, or for about half of
        the questions with QUESTION_SOURCE=mixed."""
        if self.section is not None:
            return True
        if self.question_source == "generated":
            return True
        if self.question_source == "mixed":
            return len(self.history) % 2 == 1
        return False

    def _generated_question(self, meta: Dict[str, Any], level: str, used: set) -> Dict[str, Any]:
        from questions.generator import generate, topic_key_for_default
        key = meta.get("key") or topic_key_for_default(meta["raw_skill_id"])
        # Own RNG seeded from one draw of the global RNG, so a section session is
        # reproducible under random.seed() and retries do not shift later choices.
        rng = random.Random(random.random())
        q = generate(key, level=level, rng=rng, exclude_texts=used)
        q["source"] = "generated"
        return q

    def _model_predictor(self, predictor):
        """Predictor the engine uses. Default section: unchanged. Validated section:
        topic ids are translated to their ASSISTments skill (history items of
        unmapped topics are dropped). Transfer mode: a neutral 0.5 predictor so topic
        ranking uses the student's own answers only (no model numbers are shown)."""
        if self.section is None or predictor is None:
            return predictor
        if not self.uses_model:
            return lambda history, cands: [0.5] * len(cands)

        def wrapped(history, cands):
            hist = []
            for h in history:
                m = assist_skill_for_topic(_item_topic(h) or 0)
                if m is not None:
                    hist.append(dict(h, skill_id=m, topic=m))
            out = [0.5] * len(cands)
            idx = [i for i, (t, _) in enumerate(cands) if assist_skill_for_topic(t) is not None]
            if hist and idx:
                scores = predictor(hist, [(assist_skill_for_topic(cands[i][0]), cands[i][1]) for i in idx])
                for j, i in enumerate(idx):
                    out[i] = float(scores[j])
            return out
        return wrapped

    def check(self, submitted: Any, pending: Dict[str, Any]) -> bool:
        if pending.get("_item") is not None:
            from questions.generator import check_answer
            return check_answer(submitted, pending["_item"])
        return _check_answer(submitted, pending["correct_answer"])

    def section_payload(self) -> Dict[str, Any]:
        from serving.sections import section_info
        info = section_info(self.section)
        return {"section": self.section, "section_label": info["label"],
                "validation": info["validation"], "validated": info["validated"],
                "model_predictions": bool(self.uses_model)}

    def next_question(self, predictor=None) -> Optional[Dict[str, Any]]:
        q = self._next_question(self._model_predictor(predictor))
        if q is not None and self.section is not None and not self.uses_model:
            q["predicted_p"] = None   # transfer mode: no model prediction is shown or logged
            if self.pending is not None:
                self.pending["predicted_p"] = None
            if q.get("decision"):
                q["decision"]["reason"] = (q["decision"]["reason"] + " Transfer mode: no model prediction "
                                           "for this section; topics are chosen from your own answers.").strip()
        return q

    def _next_question(self, predictor=None) -> Optional[Dict[str, Any]]:
        if self.phase == "done":
            return None

        if self.phase == "main":
            if len(self.history) < self.main_questions:
                if len(self.context()) < WARMUP_QUESTIONS:
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
            unasked = [t for t in self.topic_ids if t not in self.asked_topics]
            if not unasked:
                self.phase = "done"
                return None
            self.probe_queue = random.sample(unasked, min(PROBE_QUESTIONS, len(unasked)))

        # phase == "probe"
        if self.probe_queue:
            topic = self.probe_queue.pop(0)
            predicted_p = None
            if predictor is not None and len(self.context()) >= 2:
                predicted_p = float(predictor(self.context(), [(topic, LEVEL_TO_BUCKET["medium"])])[0])
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
        is_correct = self.check(submitted_answer, pending)

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
        return [t for t in self.topic_ids if t not in self.asked_topics]


def _item_topic(it: Dict[str, Any]) -> Optional[int]:
    try:
        return int(it.get("topic", it.get("skill_id")))
    except (TypeError, ValueError):
        return None


def _item_level(it: Dict[str, Any]) -> Optional[str]:
    lvl = it.get("level")
    if lvl in LEVEL_INDEX:
        return lvl
    try:
        return BUCKET_TO_LEVEL.get(int(it.get("difficulty_bucket") or 0))
    except (TypeError, ValueError):
        return None


def infer_section(interactions: List[Dict[str, Any]]) -> Optional[str]:
    """Section of a stored session from its topic ids (section topics use 10xxx /
    20xxx / 30xxx ids, ASSISTments topics are < 1000); None = default."""
    from serving.sections import TOPIC_INDEX
    for it in interactions or []:
        t = _item_topic(it)
        if t is not None and t in TOPIC_INDEX:
            return TOPIC_INDEX[t]["section"]
    return None


def _normalise_item(it: Dict[str, Any], topic_by_id: Optional[Dict[int, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Stored interaction -> dict usable by the predictor and level rules."""
    t = _item_topic(it)
    if t is None or t not in (topic_by_id if topic_by_id is not None else TOPIC_BY_ID):
        return None
    out = dict(it)
    out["topic"] = t
    out["skill_id"] = t
    out.setdefault("level", _item_level(it))
    return out


def _calibration_active() -> bool:
    try:
        from serving.calibration import is_active
        return is_active()
    except Exception:
        return False


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
        """session_kwargs may include prior_history (earlier sessions' answers)."""
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
                "calibration_applied": _calibration_active(),
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
