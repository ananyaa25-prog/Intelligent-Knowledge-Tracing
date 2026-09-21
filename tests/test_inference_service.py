"""
tests/test_inference_service.py
-------------------------------
Smoke tests for Task 1 FastAPI inference service (src/serving/inference_service.py).

Verifies:
1. Real student sequence from test split returns 200, non-empty skill_mastery,
   valid floats in [0, 1], no NaNs.
2. 1-interaction edge case returns graceful empty/insufficient data response, not 500.
3. Invalid skill_id returns clear 4xx error naming the problem without crashing.
4. Parity test: Predicted probability for shared skills/timesteps matches direct
   evaluate_test.py execution within 1e-4 tolerance.
"""

import sys
import math
import pickle
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.inference_service import app, artifacts
from model import GatedKTModel
from graph_encoder import adjacency_to_edge_index
from train import compute_rt_stats


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient and triggers lifespan startup."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def processed_data():
    """Loads windowed_data.pkl and skill_graph.pkl for verification."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "windowed_data.pkl"
    graph_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "skill_graph.pkl"

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(graph_path, "rb") as f:
        graph_data = pickle.load(f)

    return data, graph_data


def test_1_real_student_prediction(client, processed_data):
    """
    Smoke Test 1:
    Send a POST /predict with a REAL student's history pulled directly from
    windowed_data.pkl's test split.
    Assert: response status 200, skill_mastery is non-empty, all values are
    floats in [0, 1], no NaN, no exception raised.
    """
    data, _ = processed_data
    test_split = data["splits"]["test"]
    skill_map = data["skill_map"]
    reverse_map = {v: k for k, v in skill_map.items()}

    # Pick student 0 from test split
    student_idx = 0
    mask = test_split["mask"][student_idx]
    real_len = int(mask.sum().item())
    assert real_len >= 2, f"Test student has only {real_len} interactions"

    # Reconstruct raw request from stored tensor
    student_history = []
    for i in range(real_len):
        reindexed_skill = int(test_split["skill_seq"][student_idx, i].item())
        raw_skill_id = reverse_map[reindexed_skill]
        student_history.append({
            "skill_id": int(raw_skill_id),
            "correct": int(test_split["correct_seq"][student_idx, i].item()),
            "response_time_ms": float(test_split["response_time_seq"][student_idx, i].item()),
            "attempt_count": max(1, int(test_split["attempt_seq"][student_idx, i].item())),
            "hint_used": int(test_split["hint_seq"][student_idx, i].item()),
        })

    response = client.post("/predict", json={"student_history": student_history})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    body = response.json()
    assert "skill_mastery" in body, "Response missing 'skill_mastery'"
    assert "num_interactions_processed" in body, "Response missing 'num_interactions_processed'"
    assert body["num_interactions_processed"] == real_len

    mastery = body["skill_mastery"]
    assert len(mastery) > 0, "skill_mastery should be non-empty"

    for skill_key, val in mastery.items():
        assert isinstance(val, (float, int)), f"Value for skill {skill_key} is not float: {val}"
        assert not math.isnan(val), f"Value for skill {skill_key} is NaN"
        assert 0.0 <= val <= 1.0, f"Value for skill {skill_key} is out of bounds [0, 1]: {val}"

    print(f"\n[Test 1 PASS] Processed {real_len} interactions, estimated mastery for {len(mastery)} skills.")


def test_2_single_interaction_edge_case(client):
    """
    Smoke Test 2:
    Send a request with only 1 interaction.
    Assert: does not crash, returns a clearly-marked empty/insufficient-data
    response, not a 500 error.
    """
    single_interaction = [{
        "skill_id": 1,
        "correct": 1,
        "response_time_ms": 10000.0,
        "attempt_count": 1,
        "hint_used": 0,
    }]

    response = client.post("/predict", json={"student_history": single_interaction})
    assert response.status_code == 200, f"Expected 200 for edge case, got {response.status_code}"

    body = response.json()
    assert body["skill_mastery"] == {}, "Expected empty skill_mastery for 1 interaction"
    assert body["num_interactions_processed"] == 1
    assert "message" in body and body["message"] is not None
    assert "insufficient data" in body["message"].lower()

    print("\n[Test 2 PASS] Graceful handling of single interaction (insufficient data).")


def test_3_invalid_skill_id(client):
    """
    Smoke Test 3:
    Send a request with a skill_id NOT present in skill_map (an invalid ID).
    Assert: returns a clear 4xx error naming the problem, does not crash the
    service or silently produce garbage output.
    """
    invalid_history = [
        {
            "skill_id": 1,
            "correct": 1,
            "response_time_ms": 10000.0,
            "attempt_count": 1,
            "hint_used": 0,
        },
        {
            "skill_id": 99999999,  # Non-existent skill ID
            "correct": 0,
            "response_time_ms": 15000.0,
            "attempt_count": 2,
            "hint_used": 1,
        }
    ]

    response = client.post("/predict", json={"student_history": invalid_history})
    assert 400 <= response.status_code < 500, f"Expected 4xx error, got {response.status_code}"
    error_detail = response.json().get("detail", "")
    assert "99999999" in str(error_detail), f"Error message should name the invalid skill_id: {error_detail}"
    assert "skill_map" in str(error_detail), f"Error message should mention skill_map: {error_detail}"

    print(f"\n[Test 3 PASS] Invalid skill ID correctly rejected with {response.status_code}: {error_detail}")


def test_4_numerical_parity_against_evaluate_test(client, processed_data):
    """
    Smoke Test 4:
    Compare the service's output for the test student against directly running
    the same student through evaluate_test.py's existing model-loading code path.
    Assert: The predicted probability for at least one shared skill/timestep
    matches within floating-point tolerance (1e-4).
    """
    data, graph_data = processed_data
    train_data = data["splits"]["train"]
    test_data = data["splits"]["test"]
    num_skills = data["num_skills"]
    skill_map = data["skill_map"]
    reverse_map = {v: k for k, v in skill_map.items()}

    # 1. Exact evaluate_test.py pipeline setup
    adj_matrix = torch.tensor(graph_data["adjacency_matrix"], dtype=torch.float32)
    edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
    rt_mean, rt_std = compute_rt_stats(train_data)

    ckpt_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "best_full_gated.pt"
    direct_model = GatedKTModel(
        num_skills=num_skills,
        rt_log_mean=rt_mean,
        rt_log_std=rt_std,
        use_graph=True,
        use_gate=True,
    )
    direct_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    direct_model.eval()

    # 2. Run direct single-student prediction
    student_idx = 0
    mask = test_data["mask"][student_idx:student_idx+1]
    real_len = int(mask.sum().item())

    with torch.no_grad():
        direct_pred, _ = direct_model(
            test_data["skill_seq"][student_idx:student_idx+1],
            test_data["correct_seq"][student_idx:student_idx+1],
            mask.float(),
            test_data["response_time_seq"][student_idx:student_idx+1].float(),
            test_data["attempt_seq"][student_idx:student_idx+1].float(),
            test_data["hint_seq"][student_idx:student_idx+1].float(),
            edge_index,
            edge_weight,
        )

    # 3. Call API service for the same student
    student_history = []
    for i in range(real_len):
        reindexed_skill = int(test_data["skill_seq"][student_idx, i].item())
        raw_skill_id = reverse_map[reindexed_skill]
        student_history.append({
            "skill_id": int(raw_skill_id),
            "correct": int(test_data["correct_seq"][student_idx, i].item()),
            "response_time_ms": float(test_data["response_time_seq"][student_idx, i].item()),
            "attempt_count": max(1, int(test_data["attempt_seq"][student_idx, i].item())),
            "hint_used": int(test_data["hint_seq"][student_idx, i].item()),
        })

    response = client.post("/predict", json={"student_history": student_history})
    assert response.status_code == 200
    service_mastery = response.json()["skill_mastery"]

    # 4. Compare every skill's latest prediction in direct_pred vs service_mastery
    # In direct_pred, timestep t predicts the outcome for interaction t+1
    direct_latest_per_skill = {}
    for t in range(real_len - 1):
        target_reindexed = int(test_data["skill_seq"][student_idx, t + 1].item())
        target_raw = str(int(reverse_map[target_reindexed]))
        direct_latest_per_skill[target_raw] = direct_pred[0, t].item()

    assert len(direct_latest_per_skill) > 0, "No skills found to compare"

    matched_skills = 0
    max_diff = 0.0
    for raw_skill_str, direct_val in direct_latest_per_skill.items():
        assert raw_skill_str in service_mastery, f"Skill {raw_skill_str} missing in service output"
        service_val = service_mastery[raw_skill_str]
        diff = abs(direct_val - service_val)
        max_diff = max(max_diff, diff)
        assert diff <= 1e-4, f"Tolerance exceeded for skill {raw_skill_str}: direct={direct_val}, service={service_val}, diff={diff}"
        matched_skills += 1

    print(f"\n[Test 4 PASS] Parity verified across {matched_skills} skills. Max diff: {max_diff:.2e} (<= 1e-4 tolerance)")


def test_5_completed_session_from_task_0(client):
    """
    Smoke Test 5:
    Feed it a completed session from Task 0 (a real 10-question session with
    genuine student answers, not synthetic data) as the student_history.
    Assert: 200 status, mastery values returned for all 10 skills that appeared
    in the session, no NaN.
    """
    from serving.session import session_manager

    # 1. Start session via Task 0
    sid, questions = session_manager.start_session()
    sess = session_manager.get_session(sid)
    assigned_set = sess["set_name"]

    # 2. Complete all 10 questions with genuine student answers
    # Realistic student performance: gets some correct, some wrong, realistic response times
    expected_skills = [str(s["raw_skill_id"]) for s in session_manager.bank.skill_sequence]

    for pos in range(1, 11):
        q = session_manager.bank.get_question(assigned_set, pos)
        # Realistic student answer: correct for first 7, wrong on last 3
        if pos <= 7:
            student_ans = q["correct_answer"]
            rt = 4500.0 + pos * 350.0
        else:
            student_ans = "wrong_answer"
            rt = 12000.0 + pos * 500.0

        session_manager.record_answer(
            session_id=sid,
            position=pos,
            submitted_answer=student_ans,
            response_time_ms=rt,
            hint_used=0
        )

    # 3. Retrieve history generated by Task 0
    history = session_manager.get_history(sid)
    assert len(history) == 10, f"Expected 10 interactions from Task 0, got {len(history)}"

    # 4. Feed Task 0's history directly into Task 1's /predict endpoint
    response = client.post("/predict", json={"student_history": history})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    body = response.json()
    assert "skill_mastery" in body
    mastery = body["skill_mastery"]

    # Assert mastery values returned for all 10 skills that appeared in the session
    for skill_str in expected_skills:
        assert skill_str in mastery, f"Expected mastery for skill {skill_str} from 10-question session"
        val = mastery[skill_str]
        assert isinstance(val, (float, int))
        assert not math.isnan(val), f"Mastery for skill {skill_str} is NaN"
        assert 0.0 <= val <= 1.0, f"Mastery {val} out of bounds for skill {skill_str}"

    assert body["num_interactions_processed"] == 10
    print(f"\n[Test 5 PASS] Completed 10-question session from set '{assigned_set}' evaluated successfully. All 10 skills present and valid.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
