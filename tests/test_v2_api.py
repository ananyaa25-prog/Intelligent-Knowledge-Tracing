"""API-level tests for the adaptive-v2 contract against the real model (temp DB via conftest)."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from serving.inference_service import app

OLD_FIELDS = ["position", "topic", "skill_name", "level", "difficulty_bucket", "question_text",
              "predicted_p", "is_probe", "is_complete"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _start(client, sid_name):
    r = client.post("/session/start", json={"student_id": sid_name, "source": "test"})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_next_contract_a(client, monkeypatch):
    monkeypatch.delenv("POLICY", raising=False)
    sid = _start(client, "pytest_contract_a")
    n = 0
    while n < 40:
        d = client.post(f"/session/{sid}/next", json={}).json()
        if d["is_complete"]:
            assert d["total_questions"] == n
            break
        for k in OLD_FIELDS:
            assert k in d
        assert isinstance(d["total_questions"], int) and d["total_questions"] >= 15
        assert d["session_mode"] == "full10" and d["policy"] == "staircase"
        dec = d["decision"]
        for k in ["policy", "topic_scores", "level_candidates", "chosen_level", "reason"]:
            assert k in dec
        assert dec["chosen_level"] == d["level"]
        if not d["is_probe"] and d["position"] > 2:
            assert d["predicted_p"] is not None
        # No verdict leaks in the question payload.
        assert "correct_answer" not in d and "correct" not in d
        client.post(f"/session/{sid}/answer", json={"submitted_answer": "1", "response_time_ms": 2000.0})
        n += 1
    assert n >= 15


def test_next_model_policy(client, monkeypatch):
    monkeypatch.setenv("POLICY", "model")
    sid = _start(client, "pytest_model_policy")
    for _ in range(5):
        d = client.post(f"/session/{sid}/next", json={}).json()
        assert d["policy"] == "model"
        if d["position"] > 2:
            assert set(d["decision"]["level_candidates"]) == {"easy", "medium", "hard"}
            assert abs(d["decision"]["level_candidates"][d["level"]] - d["predicted_p"]) < 1e-3
        client.post(f"/session/{sid}/answer", json={"submitted_answer": "1", "response_time_ms": 2000.0})


def _play(client, student_id, source, n=6):
    sid = client.post("/session/start", json={"student_id": student_id, "source": source}).json()["session_id"]
    for _ in range(n):
        d = client.post(f"/session/{sid}/next", json={}).json()
        if d.get("is_complete"):
            break
        client.post(f"/session/{sid}/answer", json={"submitted_answer": "1", "response_time_ms": 2000.0})
    return sid


def test_class_overview_contract_c_and_separation(client):
    _play(client, "sim_pytest_overview", "sample")
    _play(client, "learner_live_kid", "live")
    _play(client, "e2e_excluded_kid", "live")
    live = client.get("/class/overview?source=live").json()
    sample = client.get("/class/overview?source=sample").json()
    for d in (live, sample):
        for k in ["students", "status_counts", "needs_attention"]:
            assert k in d
    live_ids = {s["student_id"] for s in live["students"]}
    sample_ids = {s["student_id"] for s in sample["students"]}
    assert "learner_live_kid" in live_ids
    assert "e2e_excluded_kid" not in live_ids          # EXCLUDE_STUDENT_REGEX
    assert "sim_pytest_overview" in sample_ids and "sim_pytest_overview" not in live_ids
    assert not (live_ids & sample_ids)
    st = next(s for s in sample["students"] if s["student_id"] == "sim_pytest_overview")
    for c in st["topics"].values():
        assert set(c) == {"status", "n_answers"}
    for a in sample["needs_attention"]:
        for k in ["student_id", "topic", "skill_name", "status", "n_answers", "recommended_action"]:
            assert k in a
    assert client.get("/class/overview?source=bogus").status_code == 400


def test_overview_cache_invalidated_on_answer(client):
    sid = _play(client, "learner_cache_kid", "live", n=3)
    before = {s["student_id"]: s for s in client.get("/class/overview?source=live").json()["students"]}["learner_cache_kid"]
    client.post(f"/session/{sid}/next", json={})
    client.post(f"/session/{sid}/answer", json={"submitted_answer": "1", "response_time_ms": 2000.0})
    after = {s["student_id"]: s for s in client.get("/class/overview?source=live").json()["students"]}["learner_cache_kid"]
    assert after["n_interactions"] == before["n_interactions"] + 1


def test_feedback_contract_d_logs_event(client):
    import json
    import serving.inference_service as svc
    sid = _play(client, "pytest_feedback_kid", "test", n=2)
    r = client.post(f"/session/{sid}/feedback", json={"perceived": "too_hard", "helpful": 4})
    assert r.status_code == 200 and r.json()["ok"] is True
    last = [json.loads(l) for l in open(svc.SESSION_LOG_PATH, encoding="utf-8") if l.strip()][-1]
    assert last["event"] == "feedback" and last["session_id"] == sid
    assert (last["perceived"], last["helpful"]) == ("too_hard", 4)
    assert client.post(f"/session/{sid}/feedback", json={"perceived": "meh"}).status_code == 422
    assert client.post(f"/session/{sid}/feedback", json={"perceived": "about_right", "helpful": 9}).status_code == 422
    assert client.post("/session/nope/feedback", json={"perceived": "about_right"}).status_code == 404


def test_log_path_is_isolated_from_real_log():
    import serving.inference_service as svc
    real = Path(__file__).resolve().parent.parent / "data" / "processed" / "session_logs.jsonl"
    assert Path(svc.SESSION_LOG_PATH).resolve() != real.resolve()


def test_seed_script_refuses_real_db_without_flag():
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "KT_DB_PATH"}
    script = Path(__file__).resolve().parent.parent / "scripts" / "seed_demo_data.py"
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "Refusing" in r.stdout
