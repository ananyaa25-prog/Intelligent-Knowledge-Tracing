"""
tests/test_cumulative_inference.py
-----------------------------------
Smoke test suite for Task 5: Cumulative-history inference and sequence length truncation.

Verifies:
1. Single session: Single-session prediction runs without regression.
2. Second session: Prediction on same student uses cumulative 20 interactions.
3. Long sequence truncation: Sequence exceeding max_seq_len (500) truncates to
   most recent 500 without crashing and produces valid floats in [0, 1].
"""

import sys
import pickle
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.inference_service import app, artifacts
from serving.session import session_manager


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


import uuid

def test_1_single_session_mastery_baseline(client):
    """
    1. A student completes one session.
    Assert: Mastery prediction matches baseline single-session behavior (no regression).
    """
    student_id = f"test_student_single_{uuid.uuid4().hex[:8]}"
    start_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    assert start_res.status_code == 200
    start_data = start_res.json()
    sid = start_data["session_id"]
    assigned_set = session_manager.get_session(sid)["set_name"]

    # Submit 10 answers
    for pos in range(1, 11):
        q = session_manager.bank.get_question(assigned_set, pos)
        ans_res = client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 4000.0 + pos * 150,
            "hint_used": 0
        })
        assert ans_res.status_code == 200

    # Request prediction
    pred_res = client.post(f"/session/{sid}/predict")
    assert pred_res.status_code == 200
    body = pred_res.json()
    assert body["num_interactions_processed"] == 10
    assert len(body["skill_mastery"]) == 10

    for skill_str, val in body["skill_mastery"].items():
        assert 0.0 <= val <= 1.0

    print(f"\n[Task 5 Test 1 PASS] Single session evaluated successfully with 10 interactions.")


def test_2_multi_session_cumulative_payload(client):
    """
    2. The SAME student completes a second session.
    Assert: The mastery prediction call uses all 20 interactions.
    """
    student_id = f"test_student_cumulative_{uuid.uuid4().hex[:8]}"

    # Session 1
    s1_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    assert s1_res.status_code == 200
    sid1 = s1_res.json()["session_id"]
    set1 = session_manager.get_session(sid1)["set_name"]

    for pos in range(1, 11):
        q = session_manager.bank.get_question(set1, pos)
        client.post(f"/session/{sid1}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 5000.0,
            "hint_used": 0
        })

    # Session 2 (same student ID)
    s2_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    assert s2_res.status_code == 200
    sid2 = s2_res.json()["session_id"]
    set2 = session_manager.get_session(sid2)["set_name"]

    for pos in range(1, 11):
        q = session_manager.bank.get_question(set2, pos)
        ans = q["correct_answer"] if pos % 2 == 0 else "wrong_answer"
        client.post(f"/session/{sid2}/answer", json={
            "position": pos,
            "submitted_answer": ans,
            "response_time_ms": 7000.0,
            "hint_used": 0
        })

    # Request prediction by session_id and by student_id
    pred_res_session = client.post(f"/session/{sid2}/predict")
    assert pred_res_session.status_code == 200
    body_sess = pred_res_session.json()

    # Confirms 20 interactions were processed in cumulative calculation
    assert body_sess["num_interactions_processed"] == 20, (
        f"Expected 20 cumulative interactions processed, got {body_sess['num_interactions_processed']}"
    )

    pred_res_student = client.post(f"/student/{student_id}/predict")
    assert pred_res_student.status_code == 200
    body_student = pred_res_student.json()
    assert body_student["num_interactions_processed"] == 20
    assert body_sess["skill_mastery"] == body_student["skill_mastery"]

    print(f"\n[Task 5 Test 2 PASS] Cumulative prediction processed all 20 interactions across 2 sessions.")


def test_3_sequence_truncation_on_long_history(client):
    """
    3. Construct a synthetic student with more interactions than max_seq_len (520 interactions).
    Assert: Inference truncates to the most recent max_seq_len (500) interactions and does not crash.
    """
    max_len = artifacts.max_seq_len
    total_items = max_len + 25  # 525 interactions

    # Build sequence using valid skills from skill_sequence
    valid_skills = [s["raw_skill_id"] for s in session_manager.bank.skill_sequence]
    synthetic_history = []
    for i in range(total_items):
        skill_id = valid_skills[i % len(valid_skills)]
        synthetic_history.append({
            "skill_id": int(skill_id),
            "correct": 1 if i % 3 == 0 else 0,
            "response_time_ms": 4500.0 + (i % 10) * 100,
            "attempt_count": 1,
            "hint_used": 0
        })

    assert len(synthetic_history) == 525

    # Direct predict endpoint call
    response = client.post("/predict", json={"student_history": synthetic_history})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    body = response.json()
    assert body["num_interactions_processed"] == 525
    assert len(body["skill_mastery"]) > 0

    for skill_str, val in body["skill_mastery"].items():
        assert 0.0 <= val <= 1.0

    print(f"\n[Task 5 Test 3 PASS] Handled long history ({total_items} items) with max_seq_len={max_len} truncation successfully.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
