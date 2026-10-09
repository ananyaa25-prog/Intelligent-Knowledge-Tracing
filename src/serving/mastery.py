"""
src/serving/mastery.py
----------------------
Per-topic mastery / readiness profile (T2).

For every topic the student has answered we report observed evidence
(answers, levels attempted, correct counts by level) and, when the model's
difficulty ordering has been checked (profile_reliability.json), the model's
counterfactual P(correct) at easy / medium / hard from ONE batched predictor
call. Statuses are rule-based estimates:

  reliable mode (model counterfactuals available)
    not_enough_evidence  n_answers < MIN_ANSWERS
    needs_support        p_easy < T_PROF
    developing           p_easy >= T_PROF and p_med < T_PROF
    proficient           p_med >= T_PROF and >= 1 answer at medium or hard
    mastered             p_hard >= T_PROF and >= 1 CORRECT answer at hard
    A status whose evidence requirement is not met is downgraded one step
    (repeatedly, until the requirement holds) and evidence_note explains why.
    If the model status clearly contradicts the observed answers on the topic
    (needs_support with >= 75% correct, or proficient/mastered with <= 25%
    correct), it moves one step toward the evidence (evidence_note says so).
    ready_for_next = status in {developing, proficient} and P(next level) >= T_READY

  fallback (profile_reliable = false): evidence-only rules, p fields null.

Pure functions only; the predictor and histories are passed in.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from serving.policy_config import (
    PROFILE_RELIABILITY_PATH, RECOMMENDED_ACTIONS, PolicyConfig, get_config,
)
from serving.question_templates import LEVEL_TO_BUCKET

LEVELS = ["easy", "medium", "hard"]
STATUS_ORDER = ["needs_support", "developing", "proficient", "mastered"]
BUCKET_TO_LEVEL = {1: "easy", 2: "medium", 3: "hard"}

# Fallback accuracy cut-offs: the brief's 0.67 / 0.34 are rounded 2/3 and 1/3,
# so exactly 2 of 3 correct counts as >= 0.67 (see docs/DECISIONS.md D13).
FALLBACK_PROF_ACC = 0.665
FALLBACK_SUPPORT_ACC = 0.34

# Evidence-conflict cut-offs for the model-based status (D26).
EVIDENCE_HIGH_ACC = 0.75
EVIDENCE_LOW_ACC = 0.25

STATUS_DISCLAIMER = "Status labels are rule-based estimates from the model and your answers."


def load_profile_reliability(path=PROFILE_RELIABILITY_PATH) -> Dict[str, Any]:
    """Return {'reliable': bool, 'violation_rate': float|None, 'source': str}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "reliable": bool(data.get("reliable", False)),
            "violation_rate": data.get("violation_rate"),
            "source": str(path),
        }
    except Exception:
        return {"reliable": False, "violation_rate": None, "source": "missing"}


def _level_of(item: Dict[str, Any]) -> Optional[str]:
    lvl = item.get("level")
    if lvl in LEVELS:
        return lvl
    try:
        return BUCKET_TO_LEVEL.get(int(item.get("difficulty_bucket") or 0))
    except (TypeError, ValueError):
        return None


def topic_evidence(history: List[Dict[str, Any]], topic: int) -> Dict[str, Any]:
    """Observed evidence for one topic from a chronological history."""
    items = []
    for it in history:
        try:
            sid = int(it.get("skill_id", it.get("topic")))
        except (TypeError, ValueError):
            continue
        if sid == int(topic):
            items.append(it)
    cbl = {lvl: {"n": 0, "correct": 0} for lvl in LEVELS}
    levels_attempted: List[str] = []
    for it in items:
        lvl = _level_of(it)
        if lvl is None:
            continue
        cbl[lvl]["n"] += 1
        cbl[lvl]["correct"] += int(it.get("correct", 0))
        if lvl not in levels_attempted:
            levels_attempted.append(lvl)
    levels_attempted.sort(key=LEVELS.index)
    n_correct = sum(int(it.get("correct", 0)) for it in items)
    last = items[-1] if items else None
    return {
        "n_answers": len(items),
        "n_correct": n_correct,
        "correct_by_level": cbl,
        "levels_attempted": levels_attempted,
        "last_correct": (int(last.get("correct", 0)) == 1) if last else None,
        "last_level": _level_of(last) if last else None,
    }


def monotone_project(p_easy: float, p_med: float, p_hard: float):
    """Enforce easy >= medium >= hard. Returns (pe, pm, ph, adjusted)."""
    pm = min(p_med, p_easy)
    ph = min(p_hard, pm)
    adjusted = (pm != p_med) or (ph != p_hard)
    return p_easy, pm, ph, adjusted


def _model_status(pe: float, pm: float, ph: float, t_prof: float) -> str:
    if ph >= t_prof:
        return "mastered"
    if pm >= t_prof:
        return "proficient"
    if pe >= t_prof:
        return "developing"
    return "needs_support"


def reliable_status(ev: Dict[str, Any], p: Dict[str, float], cfg: PolicyConfig):
    """Status from model counterfactuals + evidence gating.
    Returns (status, ready_for_next, evidence_note)."""
    if ev["n_answers"] < cfg.min_answers:
        return "not_enough_evidence", False, None
    status = _model_status(p["easy"], p["medium"], p["hard"], cfg.t_prof)
    cbl = ev["correct_by_level"]
    notes = []
    # Evidence conflict (added after the dry run, docs/DECISIONS.md D26): the model's
    # estimate is strongly driven by the student's most recent answers on ANY topic,
    # so it can contradict what the student did on THIS topic. Move one step toward
    # the observed evidence and say so.
    acc = ev["n_correct"] / ev["n_answers"]
    if status == "needs_support" and acc >= EVIDENCE_HIGH_ACC:
        status = "developing"
        notes.append(f"Model estimate is low, but {ev['n_correct']} of {ev['n_answers']} answers on this topic "
                     "were correct; shown as Developing.")
    elif status in ("proficient", "mastered") and acc <= EVIDENCE_LOW_ACC:
        status = STATUS_ORDER[STATUS_ORDER.index(status) - 1]
        notes.append(f"Model estimate is high, but only {ev['n_correct']} of {ev['n_answers']} answers on this topic "
                     f"were correct; shown one level lower.")
    # Evidence gates: never certify a level the student was not tested near.
    if status == "mastered" and cbl["hard"]["correct"] < 1:
        status = "proficient"
        notes.append("Model estimate reaches Hard, but no Hard question was answered correctly yet; shown as Proficient.")
    if status == "proficient" and (cbl["medium"]["n"] + cbl["hard"]["n"]) < 1:
        status = "developing"
        notes.append("Model estimate reaches Medium, but no Medium or Hard question was answered yet; shown as Developing.")
    ready = False
    if status == "developing":
        ready = p["medium"] >= cfg.t_ready
    elif status == "proficient":
        ready = p["hard"] >= cfg.t_ready
    return status, ready, (" ".join(notes) or None)


def fallback_status(ev: Dict[str, Any], cfg: PolicyConfig):
    """Evidence-only status (no model counterfactuals)."""
    n = ev["n_answers"]
    if n < cfg.min_answers:
        return "not_enough_evidence", False, None
    acc = ev["n_correct"] / n if n else 0.0
    cbl = ev["correct_by_level"]
    hard_ok = cbl["hard"]["correct"] >= 1
    medhard_ok = (cbl["medium"]["correct"] + cbl["hard"]["correct"]) >= 1
    if n >= 3 and acc >= 0.75 and hard_ok:
        status = "mastered"
    elif acc >= FALLBACK_PROF_ACC and medhard_ok:
        status = "proficient"
    elif acc <= FALLBACK_SUPPORT_ACC:
        status = "needs_support"
    else:
        status = "developing"
    ready = False
    if status in ("developing", "proficient") and ev["levels_attempted"]:
        highest = ev["levels_attempted"][-1]
        ready = bool(ev["last_correct"]) and ev["last_level"] == highest
    return status, ready, "Evidence-only estimate (model difficulty ordering not verified)."


def build_profile(
    full_history: List[Dict[str, Any]],
    topics: List[Dict[str, Any]],
    predictor: Optional[Callable] = None,
    session_history: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional[PolicyConfig] = None,
    reliable: Optional[bool] = None,
    session_id: Optional[str] = None,
    student_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the profile (contract B).

    topics: list of {"raw_skill_id", "skill_name"} (the assessed topics).
    predictor: _predictor(history, [(raw_id, bucket)]) -> [p]; may be None.
    reliable: override; defaults to profile_reliability.json.
    """
    cfg = cfg or get_config()
    if reliable is None:
        reliable = load_profile_reliability()["reliable"]
    session_history = session_history or []

    evidences = {int(t["raw_skill_id"]): topic_evidence(full_history, int(t["raw_skill_id"])) for t in topics}
    asked = [t for t in topics if evidences[int(t["raw_skill_id"])]["n_answers"] > 0]
    unasked = [t for t in topics if evidences[int(t["raw_skill_id"])]["n_answers"] == 0]

    # ONE batched predictor call: asked topics x 3 levels (+ unasked topics at medium).
    pmap: Dict[tuple, float] = {}
    model_ok = predictor is not None and len(full_history) >= 2
    if model_ok:
        cands = []
        if reliable:
            for t in asked:
                for lvl in LEVELS:
                    cands.append((int(t["raw_skill_id"]), LEVEL_TO_BUCKET[lvl]))
        for t in unasked:
            cands.append((int(t["raw_skill_id"]), LEVEL_TO_BUCKET["medium"]))
        if cands:
            try:
                scores = predictor(full_history, cands)
                pmap = {c: float(s) for c, s in zip(cands, scores)}
            except Exception as e:  # never break the profile on a model error
                print(f"[mastery] predictor failed, using evidence-only profile: {e}")
                pmap = {}
                model_ok = False
    use_model = bool(reliable and model_ok and pmap)

    session_counts: Dict[int, int] = {}
    for it in session_history:
        try:
            sid = int(it.get("skill_id", it.get("topic")))
        except (TypeError, ValueError):
            continue
        session_counts[sid] = session_counts.get(sid, 0) + 1

    out_topics = []
    for t in asked:
        tid = int(t["raw_skill_id"])
        ev = evidences[tid]
        p_out = None
        adjusted = False
        if use_model:
            pe = pmap[(tid, LEVEL_TO_BUCKET["easy"])]
            pm = pmap[(tid, LEVEL_TO_BUCKET["medium"])]
            ph = pmap[(tid, LEVEL_TO_BUCKET["hard"])]
            pe, pm, ph, adjusted = monotone_project(pe, pm, ph)
            p_out = {"easy": round(pe, 4), "medium": round(pm, 4), "hard": round(ph, 4)}
            status, ready, note = reliable_status(ev, {"easy": pe, "medium": pm, "hard": ph}, cfg)
        else:
            status, ready, note = fallback_status(ev, cfg)
        out_topics.append({
            "topic": tid,
            "skill_name": t.get("skill_name"),
            "n_answers": ev["n_answers"],
            "n_this_session": session_counts.get(tid, 0),
            "levels_attempted": ev["levels_attempted"],
            "correct_by_level": ev["correct_by_level"],
            "p": p_out,
            "status": status,
            "ready_for_next": bool(ready),
            "monotone_adjusted": bool(adjusted),
            "evidence_note": note,
            "recommended_action": RECOMMENDED_ACTIONS[status],
        })

    out_unasked = []
    for t in unasked:
        tid = int(t["raw_skill_id"])
        p = pmap.get((tid, LEVEL_TO_BUCKET["medium"])) if model_ok else None
        out_unasked.append({
            "topic": tid,
            "skill_name": t.get("skill_name"),
            "predicted_p": round(p, 4) if p is not None else None,
            "measured": False,
        })

    return {
        "session_id": session_id,
        "student_id": student_id,
        "profile_reliable": bool(use_model),
        "thresholds": cfg.thresholds(),
        "status_note": STATUS_DISCLAIMER,
        "recommended_action_note": "Recommended actions are rule-based suggestions.",
        "topics": out_topics,
        "unasked": out_unasked,
    }
