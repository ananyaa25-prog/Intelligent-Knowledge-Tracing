"""Tests for the difficulty policies, decision object and core5 mode (T3, T8c)."""

import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving.adaptive_session import (AdaptiveSession, AdaptiveSessionManager, MAIN_QUESTIONS,
                                      TOPIC_IDS, TOPIC_BY_ID)

TRACE = Path(__file__).resolve().parent / "data_trace_staircase_baseline.json"


def trace_predictor(history, cands):
    n = len(history)
    return [((t * 7 + b * 13 + n * 3) % 97) / 100.0 for t, b in cands]


def _trace(**kwargs):
    out = []
    for seed in (1, 2, 3):
        random.seed(seed)
        s = AdaptiveSession("s", "u", **kwargs)
        k = 0
        while True:
            q = s.next_question(trace_predictor)
            if q is None:
                break
            ans = q["correct_answer"] if (k * 5 + seed) % 3 else "wrong"
            s.record_answer(ans, 1000.0)
            out.append([seed, q["topic"], q["level"], q["predicted_p"], q["is_probe"]])
            k += 1
    return out


def test_defaults_identical_to_pre_upgrade_trace(monkeypatch):
    """With all defaults the engine reproduces the trace recorded from the
    baseline (pre-upgrade) code with the same seeds and fake predictor."""
    for k in ["POLICY", "SESSION_MODE", "TARGET_P"]:
        monkeypatch.delenv(k, raising=False)
    expected = json.loads(TRACE.read_text(encoding="utf-8"))
    assert _trace() == expected


def test_explicit_staircase_matches_defaults():
    expected = json.loads(TRACE.read_text(encoding="utf-8"))
    assert _trace(policy="staircase", session_mode="full10") == expected


def _session_at(policy, history_items, levels=None, **kw):
    s = AdaptiveSession("s", "u", policy=policy, **kw)
    for i, (topic, level, correct) in enumerate(history_items, 1):
        s.history.append({"position": i, "topic": topic, "skill_id": topic, "level": level,
                          "difficulty_bucket": {"easy": 1, "medium": 2, "hard": 3}[level], "correct": correct,
                          "response_time_ms": 1000.0, "attempt_count": 1, "hint_used": 0, "is_probe": 0})
        s.topic_count[topic] += 1
        s.asked_topics.add(topic)
    for t, lv in (levels or {}).items():
        s.levels[t] = lv
    return s


def level_pred(table):
    return lambda h, c: [table.get(b, 0.5) for _, b in c]


T = TOPIC_IDS[0]
U = TOPIC_IDS[1]


def test_model_picks_level_closest_to_target():
    s = _session_at("model", [(U, "medium", 1), (T, "medium", 1)], levels={T: "medium"})
    lvl, cand, guard, fb = s._decide_level(T, level_pred({1: 0.9, 2: 0.8, 3: 0.68}))
    assert (lvl, guard, fb) == ("hard", None, False)  # 0.68 closest to 0.70; last medium answer correct
    lvl, *_ = s._decide_level(T, level_pred({1: 0.9, 2: 0.72, 3: 0.4}))
    assert lvl == "medium"


def test_model_after_wrong_never_harder_and_hard_locked():
    s = _session_at("model", [(U, "medium", 1), (T, "medium", 0)], levels={T: "medium"})
    lvl, cand, guard, fb = s._decide_level(T, level_pred({1: 0.95, 2: 0.9, 3: 0.7}))
    assert lvl in ("easy", "medium") and lvl != "hard"
    s2 = _session_at("model", [(U, "medium", 1), (T, "hard", 0)])
    lvl, _, guard, _ = s2._decide_level(T, level_pred({1: 0.95, 2: 0.9, 3: 0.7}))
    assert lvl == "medium" and guard == "hard_locked"  # last Medium/Hard answer on T was wrong


def test_model_first_question_on_topic_is_medium():
    s = _session_at("model", [(U, "medium", 1), (U, "medium", 1)])
    lvl, _, guard, _ = s._decide_level(T, level_pred({1: 0.7, 2: 0.2, 3: 0.1}))
    assert (lvl, guard) == ("medium", "first_question_on_topic")


def test_model_never_easier_after_correct():
    s = _session_at("model", [(U, "medium", 1), (T, "medium", 1)])
    lvl, *_ = s._decide_level(T, level_pred({1: 0.7, 2: 0.3, 3: 0.1}))
    assert lvl in ("medium", "hard")  # easy would be closest to 70% but is not allowed


def test_model_two_correct_step_up_reaches_hard():
    s = _session_at("model", [(T, "easy", 1), (U, "medium", 1), (T, "easy", 1)])
    lvl, _, guard, _ = s._decide_level(T, level_pred({1: 0.7, 2: 0.3, 3: 0.1}))
    assert (lvl, guard) == ("medium", "step_up_after_two_correct")
    s = _session_at("model", [(T, "medium", 1), (U, "medium", 1), (T, "medium", 1)])
    lvl, _, guard, _ = s._decide_level(T, level_pred({1: 0.9, 2: 0.6, 3: 0.2}))
    assert (lvl, guard) == ("hard", "step_up_after_two_correct")


def test_model_falls_back_to_staircase_on_predictor_failure():
    s = _session_at("model", [(U, "medium", 1), (T, "medium", 1)], levels={T: "hard"})

    def boom(h, c):
        raise RuntimeError("down")
    lvl, cand, guard, fb = s._decide_level(T, boom)
    assert (lvl, fb) == ("hard", True) and s.policy_fallbacks == 1


def test_model_full_session_decision_objects_and_prediction_matches_level():
    random.seed(5)
    s = AdaptiveSession("s", "u", policy="model")
    pred = level_pred({1: 0.85, 2: 0.66, 3: 0.45})
    n = 0
    while True:
        q = s.next_question(pred)
        if q is None:
            break
        d = q["decision"]
        assert d["policy"] == "model" and d["chosen_level"] == q["level"] and isinstance(d["reason"], str)
        if not q["is_probe"] and q["position"] > 2:
            assert set(d["level_candidates"]) == {"easy", "medium", "hard"}
            assert abs(q["predicted_p"] - d["level_candidates"][q["level"]]) < 1e-3
            assert d["topic_scores"]
        s.record_answer(q["correct_answer"] if n % 2 else "x", 1000.0)
        n += 1
    assert n >= MAIN_QUESTIONS


def test_random_policy_uses_all_levels():
    random.seed(11)
    levels = set()
    for _ in range(5):
        s = AdaptiveSession("s", "u", policy="random")
        while True:
            q = s.next_question(level_pred({1: 0.8, 2: 0.6, 3: 0.4}))
            if q is None:
                break
            if not q["is_probe"] and q["position"] > 2:
                levels.add(q["level"])
            s.record_answer("x", 1000.0)
    assert levels == {"easy", "medium", "hard"}


def test_staircase_decision_reports_candidates_but_rule_unchanged():
    random.seed(3)
    s = AdaptiveSession("s", "u", policy="staircase")
    for _ in range(2):
        s.next_question(None)
        s.record_answer("x", 1000.0)
    q = s.next_question(level_pred({1: 0.9, 2: 0.7, 3: 0.2}))
    assert q["decision"]["policy"] == "staircase"
    assert q["level"] == s.levels[q["topic"]]
    assert q["decision"]["level_candidates"] == {"easy": 0.9, "medium": 0.7, "hard": 0.2}


def test_core5_topic_set_and_totals():
    random.seed(9)
    core = [50, 48, 204, 203, 350]
    s = AdaptiveSession("s", "u", session_mode="core5", core_topics=core)
    pred = level_pred({1: 0.8, 2: 0.6, 3: 0.4})
    totals = set()
    main, probes = [], []
    while True:
        q = s.next_question(pred)
        if q is None:
            break
        totals.add(s.total_questions())
        (probes if q["is_probe"] else main).append(q["topic"])
        s.record_answer("x", 1000.0)
    assert set(main) <= set(core)
    assert len(main) == 14 and len(probes) == 2
    assert all(p not in core for p in probes)
    assert all(main.count(t) <= 4 for t in core)
    assert totals == {16}


def test_full10_total_questions_and_probe_log_key():
    logs = []
    mgr = AdaptiveSessionManager(log=logs.append)
    random.seed(4)
    sid, _ = mgr.start(student_id="u")
    pred = level_pred({1: 0.8, 2: 0.6, 3: 0.4})
    seen_total = []
    while True:
        q = mgr.next_question(sid, predictor=pred)
        if q is None:
            break
        seen_total.append(mgr.get(sid).total_questions())
        mgr.record_answer(sid, "x", 1000.0)
    n = len(mgr.history(sid))
    assert seen_total[-1] == n and all(15 <= t <= 17 for t in seen_total)
    qev = [e for e in logs if e["event"] == "question"]
    assert all("policy" in e and "decision" in e and e["student_id"] == "u" for e in qev)
    probe_ev = [e for e in qev if e["is_probe"]]
    assert all("predicted_before_probe" in e for e in probe_ev)


def _prior(items):
    return [{"skill_id": t, "correct": c, "level": l, "difficulty_bucket": {"easy": 1, "medium": 2, "hard": 3}[l],
             "response_time_ms": 4000.0, "attempt_count": 1, "hint_used": 0} for t, l, c in items]


def test_prior_history_removes_warmup_and_feeds_predictor():
    seen = []

    def pred(h, c):
        seen.append(len(h))
        return [0.5] * len(c)
    random.seed(1)
    s = AdaptiveSession("s", "u", prior_history=_prior([(T, "medium", 1), (U, "medium", 0), (T, "hard", 1)]))
    q = s.next_question(pred)
    assert q["predicted_p"] is not None and q["position"] == 1  # no blind warm-up for a returning student
    assert seen and min(seen) == 3                               # predictor saw the 3 earlier answers
    assert s.levels[T] == "hard"                                 # staircase resumed from earlier sessions


def test_prior_history_ignored_when_empty_keeps_old_behaviour():
    expected = json.loads(TRACE.read_text(encoding="utf-8"))
    out = []
    for seed in (1, 2, 3):
        random.seed(seed)
        s = AdaptiveSession("s", "u", prior_history=[])
        k = 0
        while True:
            q = s.next_question(trace_predictor)
            if q is None:
                break
            s.record_answer(q["correct_answer"] if (k * 5 + seed) % 3 else "wrong", 1000.0)
            out.append([seed, q["topic"], q["level"], q["predicted_p"], q["is_probe"]])
            k += 1
    assert out == expected


def test_no_repeated_question_text_within_session():
    for seed in range(6):
        random.seed(seed)
        s = AdaptiveSession("s", "u", policy="model", session_mode="core5")
        texts = []
        while True:
            q = s.next_question(level_pred({1: 0.8, 2: 0.6, 3: 0.4}))
            if q is None:
                break
            texts.append(q["question_text"])
            s.record_answer(q["correct_answer"], 1000.0)
        assert len(texts) == len(set(texts)), [t for t in texts if texts.count(t) > 1]


def test_strong_student_reaches_hard_under_model_policy():
    random.seed(7)
    s = AdaptiveSession("s", "u", policy="model", session_mode="core5")
    levels = []
    while True:
        q = s.next_question(level_pred({1: 0.45, 2: 0.25, 3: 0.1}))  # pessimistic model
        if q is None:
            break
        levels.append(q["level"])
        s.record_answer(q["correct_answer"], 1000.0)
    assert "hard" in levels
