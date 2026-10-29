"""v3 Stage 6: learner-level sections (config, grade-band rule, session engine, API).
The default (no section) behaviour is covered by the existing tests."""

import random
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving.adaptive_session import AdaptiveSession, TOPIC_IDS, infer_section  # noqa: E402
from serving.sections import (SECTIONS, TRANSFER, VALIDATED, TOPIC_INDEX, assist_skill_for_topic,  # noqa: E402
                              band_for_name, model_items, normalise_section, section_info)


def test_section_config_complete():
    assert set(SECTIONS) == {"primary", "secondary", "undergraduate"}
    ids = [t["id"] for s in SECTIONS.values() for t in s["topics"]]
    assert len(ids) == len(set(ids)) and all(i >= 10000 for i in ids)
    for name, s in SECTIONS.items():
        assert s["grade_label"] and s["topics"] and set(s["difficulty"]) == {"easy", "medium", "hard"}
        assert s["validation"] in (VALIDATED, TRANSFER)
    assert SECTIONS["secondary"]["validation"] == VALIDATED
    assert SECTIONS["undergraduate"]["validation"] == TRANSFER
    assert SECTIONS["primary"]["validation"] == VALIDATED          # Eedi primary-age data (Stage 7, D37)
    assert SECTIONS["primary"]["serving"] == "evidence_only"         # ...but the served model is ASSISTments-only
    assert SECTIONS["secondary"]["serving"] == "model" and SECTIONS["undergraduate"]["serving"] == "evidence_only"
    assert all(not t["assist_skills"] for t in SECTIONS["undergraduate"]["topics"])
    assert section_info(None)["validated"] is True


def test_normalise_section():
    assert normalise_section(None) is None and normalise_section("") is None and normalise_section("default") is None
    assert normalise_section(" Primary ") == "primary"
    with pytest.raises(ValueError):
        normalise_section("kindergarten")


def test_band_rule():
    assert band_for_name("Addition Whole Numbers") == "primary"
    assert band_for_name("Equivalent Fractions") == "primary"
    assert band_for_name("Pythagorean Theorem") == "secondary"
    assert band_for_name("Solving Systems of Linear Equations") == "secondary"
    assert band_for_name(None) == "unassigned" and band_for_name("  ") == "unassigned"


def test_model_items_translate_and_drop():
    hist = [{"skill_id": 204, "correct": 1}, {"skill_id": 20003, "correct": 0}, {"skill_id": 30001, "correct": 1}]
    out = model_items(hist)
    assert [h["skill_id"] for h in out] == [204, assist_skill_for_topic(20003)]
    assert model_items([{"skill_id": 99999, "correct": 1}])[0]["skill_id"] == 99999   # unchanged


def _play(section, predictor, seed=0, correct_rate=0.6):
    random.seed(seed)
    s = AdaptiveSession("t", "stud", section=section, session_mode="full10")
    texts = []
    while True:
        q = s.next_question(predictor)
        if q is None:
            break
        texts.append(q["question_text"])
        ans = q["correct_answer"] if random.random() < correct_rate else "-12345"
        s.record_answer(ans, 5000)
        if len(texts) > 40:
            raise AssertionError("session did not end")
    return s, texts


@pytest.mark.parametrize("section", ["primary", "secondary", "undergraduate"])
def test_section_session_runs(section):
    calls = []

    def fake_predictor(history, cands):
        calls.append(([h["skill_id"] for h in history], [c[0] for c in cands]))
        return [0.6 for _ in cands]

    s, texts = _play(section, fake_predictor)
    assert len(texts) >= 10 and len(set(texts)) == len(texts)          # no repeated item
    assert {h["topic"] for h in s.history} <= set(s.topic_ids)
    assert all(h["topic"] >= 10000 for h in s.history)
    assert infer_section(s.history) == section
    if section == "secondary":
        assert calls and all(i < 1000 for hist, cands in calls for i in hist + cands)   # ASSISTments ids only
        assert any(h["p_before"] is not None for h in s.history)
    else:
        assert not calls                                                  # transfer mode: model not used
        assert all(h["p_before"] is None for h in s.history)
        assert s.policy == "staircase" and s.topic_selection == "evidence_blend"


def test_correct_answer_marked_correct_in_section():
    random.seed(3)
    s = AdaptiveSession("t", "stud", section="undergraduate")
    q = s.next_question(None)
    inter = s.record_answer(q["correct_answer"], 4000)
    assert inter["correct"] == 1
    q = s.next_question(None)
    assert s.record_answer("not a number", 4000)["correct"] == 0


def test_default_session_unchanged_topics():
    random.seed(1)
    s = AdaptiveSession("t", "stud")
    assert s.section is None and s.topic_ids == TOPIC_IDS
    q = s.next_question(None)
    assert q["topic"] in TOPIC_IDS and "question_source" not in q


def test_section_history_does_not_leak_into_default_session():
    prior = [{"skill_id": 10001, "topic": 10001, "correct": 1, "level": "medium"},
             {"skill_id": 204, "topic": 204, "correct": 0, "level": "medium"}]
    s = AdaptiveSession("t", "stud", prior_history=prior)
    assert [h["topic"] for h in s.prior_history] == [204]
    p = AdaptiveSession("t", "stud", prior_history=prior, section="primary")
    assert [h["topic"] for h in p.prior_history] == [10001]


# ---- API --------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from serving.inference_service import app
    with TestClient(app) as c:
        yield c


def test_api_sections_and_start(client):
    r = client.get("/sections")
    assert r.status_code == 200 and len(r.json()["sections"]) == 3
    d = client.post("/session/start", json={"student_id": "pytest_sec_default", "source": "test"}).json()
    assert d["section"] is None and d["validated"] is True
    bad = client.post("/session/start", json={"student_id": "pytest_sec_bad", "source": "test", "section": "phd"})
    assert bad.status_code == 400


@pytest.mark.parametrize("section", ["primary", "secondary", "undergraduate"])
def test_api_section_flow(client, section):
    d = client.post("/session/start", json={"student_id": f"pytest_sec_{section}", "source": "test",
                                            "section": section}).json()
    assert d["section"] == section and d["validation"] == SECTIONS[section]["validation"]
    sid = d["session_id"]
    for i in range(4):
        q = client.post(f"/session/{sid}/next", json={}).json()
        assert q["section"] == section and q["question_source"] == "generated"
        assert q["topic"] in TOPIC_INDEX and "correct_answer" not in q
        assert q["answer_type"] in ("numeric", "fraction", "mc")
        a = client.post(f"/session/{sid}/answer", json={"submitted_answer": "0", "response_time_ms": 3000}).json()
        assert a["position"] == i + 1
    prof = client.get(f"/session/{sid}/profile").json()
    assert prof["section"] == section
    if section != "secondary":   # transfer mode: evidence only, no model numbers
        assert prof["profile_reliable"] is False
        assert all(u["predicted_p"] is None for u in prof["unasked"])
    seen = {t["topic"] for t in prof["topics"]} | {int(u.get("topic", u.get("raw_skill_id", 0))) for u in prof["unasked"]}
    assert seen == {t["id"] for t in SECTIONS[section]["topics"]}
    # the student's legacy endpoints keep working with section answers in the history
    assert client.post(f"/student/pytest_sec_{section}/predict").status_code == 200
