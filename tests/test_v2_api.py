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
