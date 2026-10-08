"""Tests for src/serving/mastery.py and the /profile endpoints (T2)."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving.mastery import (build_profile, fallback_status, monotone_project, reliable_status,
                             topic_evidence)
from serving.policy_config import PolicyConfig

CFG = PolicyConfig()
TOPICS = [{"raw_skill_id": 204, "skill_name": "Percents"}, {"raw_skill_id": 50, "skill_name": "Ordering Fractions"},
          {"raw_skill_id": 27, "skill_name": "Pythagorean Theorem"}]


def it(topic, level, correct):
    return {"skill_id": topic, "topic": topic, "level": level, "correct": correct,
            "difficulty_bucket": {"easy": 1, "medium": 2, "hard": 3}[level],
            "response_time_ms": 4000.0, "attempt_count": 1, "hint_used": 0}


def ev(*items):
    return topic_evidence(list(items), 204)


# ---- monotone projection --------------------------------------------------

def test_monotone_projection():
    assert monotone_project(0.8, 0.6, 0.4) == (0.8, 0.6, 0.4, False)
    pe, pm, ph, adj = monotone_project(0.5, 0.7, 0.9)
    assert (pe, pm, ph, adj) == (0.5, 0.5, 0.5, True)
    pe, pm, ph, adj = monotone_project(0.8, 0.6, 0.65)
    assert (pm, ph, adj) == (0.6, 0.6, True)


# ---- reliable-mode status rules ------------------------------------------

def P(e, m, h):
    return {"easy": e, "medium": m, "hard": h}


def test_not_enough_evidence():
    s, ready, note = reliable_status(ev(it(204, "medium", 1)), P(0.9, 0.9, 0.9), CFG)
    assert s == "not_enough_evidence" and ready is False


def test_needs_support():
    s, _, _ = reliable_status(ev(it(204, "easy", 0), it(204, "easy", 0)), P(0.5, 0.3, 0.2), CFG)
    assert s == "needs_support"


def test_developing_and_ready():
    e = ev(it(204, "easy", 1), it(204, "medium", 0))
    s, ready, _ = reliable_status(e, P(0.8, 0.65, 0.4), CFG)
    assert s == "developing" and ready is True       # p_med 0.65 >= T_READY 0.60
    s, ready, _ = reliable_status(e, P(0.8, 0.55, 0.4), CFG)
    assert s == "developing" and ready is False


def test_proficient_requires_medium_or_hard_attempt():
    e_ok = ev(it(204, "medium", 1), it(204, "medium", 1))
    s, ready, note = reliable_status(e_ok, P(0.9, 0.75, 0.62), CFG)
    assert s == "proficient" and ready is True and note is None
    e_easy_only = ev(it(204, "easy", 1), it(204, "easy", 1))
    s, ready, note = reliable_status(e_easy_only, P(0.9, 0.75, 0.62), CFG)
    assert s == "developing" and note  # evidence downgrade


def test_mastered_requires_correct_hard():
    e_hard = ev(it(204, "medium", 1), it(204, "hard", 1))
    s, _, note = reliable_status(e_hard, P(0.95, 0.9, 0.8), CFG)
    assert s == "mastered" and note is None
    e_wrong_hard = ev(it(204, "medium", 1), it(204, "hard", 0))
    s, _, note = reliable_status(e_wrong_hard, P(0.95, 0.9, 0.8), CFG)
    assert s == "proficient" and "Hard" in note
    e_easy = ev(it(204, "easy", 1), it(204, "easy", 1))
    s, _, note = reliable_status(e_easy, P(0.95, 0.9, 0.8), CFG)
    assert s == "developing"  # downgraded twice: no hard correct, no medium/hard attempt


# ---- fallback (evidence-only) rules ---------------------------------------

def test_fallback_rules():
    s, r, _ = fallback_status(ev(it(204, "medium", 1)), CFG)
    assert s == "not_enough_evidence"
    s, r, _ = fallback_status(ev(it(204, "medium", 1), it(204, "hard", 1), it(204, "hard", 1)), CFG)
    assert s == "mastered"
    s, r, _ = fallback_status(ev(it(204, "medium", 1), it(204, "hard", 1), it(204, "hard", 0)), CFG)
    assert s == "proficient" and r is False  # last answer at highest level was wrong
    s, r, _ = fallback_status(ev(it(204, "easy", 1), it(204, "medium", 1)), CFG)
    assert s == "proficient" and r is True
    s, r, _ = fallback_status(ev(it(204, "easy", 0), it(204, "easy", 0), it(204, "easy", 1)), CFG)
    assert s == "needs_support"
    s, r, _ = fallback_status(ev(it(204, "easy", 1), it(204, "easy", 0)), CFG)
    assert s == "developing"


# ---- build_profile ---------------------------------------------------------

def fake_predictor(history, candidates):
    table = {1: 0.85, 2: 0.72, 3: 0.40}
    return [table.get(b, 0.5) for _, b in candidates]


def test_build_profile_reliable_shape_and_single_call():
    calls = []

    def pred(h, c):
        calls.append(len(c))
        return fake_predictor(h, c)

    hist = [it(204, "medium", 1), it(204, "hard", 0), it(50, "medium", 1)]
    prof = build_profile(hist, TOPICS, predictor=pred, session_history=hist, cfg=CFG, reliable=True,
                         session_id="s", student_id="u")
    assert len(calls) == 1  # ONE batched call
    assert calls[0] == 2 * 3 + 1  # 2 asked topics x 3 levels + 1 unasked topic
    assert prof["profile_reliable"] is True
    assert prof["thresholds"] == {"T_PROF": 0.7, "T_READY": 0.6, "MIN_ANSWERS": 2}
    by = {t["topic"]: t for t in prof["topics"]}
    assert set(by) == {204, 50}
    t = by[204]
    for k in ["topic", "skill_name", "n_answers", "n_this_session", "levels_attempted", "correct_by_level", "p",
              "status", "ready_for_next", "monotone_adjusted", "evidence_note", "recommended_action"]:
        assert k in t
    assert t["status"] == "proficient" and t["ready_for_next"] is False  # p_hard 0.40 < 0.60
    assert t["correct_by_level"]["hard"] == {"n": 1, "correct": 0}
    assert by[50]["status"] == "not_enough_evidence"
    assert prof["unasked"] == [{"topic": 27, "skill_name": "Pythagorean Theorem", "predicted_p": 0.72, "measured": False}]


def test_build_profile_unreliable_fallback_has_null_p():
    hist = [it(204, "medium", 1), it(204, "hard", 1), it(204, "hard", 1)]
    prof = build_profile(hist, TOPICS, predictor=fake_predictor, cfg=CFG, reliable=False)
    assert prof["profile_reliable"] is False
    assert all(t["p"] is None for t in prof["topics"])
    assert prof["topics"][0]["status"] == "mastered"


def test_build_profile_predictor_failure_degrades():
    def boom(h, c):
        raise RuntimeError("model down")
    hist = [it(204, "medium", 1), it(204, "medium", 1)]
    prof = build_profile(hist, TOPICS, predictor=boom, cfg=CFG, reliable=True)
    assert prof["profile_reliable"] is False
    assert prof["topics"][0]["p"] is None


# ---- endpoints -------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from serving.inference_service import app
    with TestClient(app) as c:
        yield c


def _play(client, student_id, n):
    sid = client.post("/session/start", json={"student_id": student_id, "source": "test"}).json()["session_id"]
    for _ in range(n):
        q = client.post(f"/session/{sid}/next", json={}).json()
        if q.get("is_complete"):
            break
        client.post(f"/session/{sid}/answer", json={"submitted_answer": "1", "response_time_ms": 3000.0, "hint_used": 0})
    return sid


def test_profile_endpoints_shape(client):
    sid = _play(client, "pytest_profile_student", 6)
    r = client.get(f"/session/{sid}/profile")
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ["session_id", "student_id", "profile_reliable", "thresholds", "topics", "unasked"]:
        assert k in d
    assert d["session_id"] == sid and d["student_id"] == "pytest_profile_student"
    assert sum(t["n_this_session"] for t in d["topics"]) == 6
    assert len(d["topics"]) + len(d["unasked"]) == 10
    for u in d["unasked"]:
        assert u["measured"] is False
    r2 = client.get("/student/pytest_profile_student/profile")
    assert r2.status_code == 200 and r2.json()["student_id"] == "pytest_profile_student"
    assert client.get("/session/does-not-exist/profile").status_code == 404
