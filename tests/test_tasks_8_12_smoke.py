"""
tests/test_tasks_8_12_smoke.py
--------------------------------
Comprehensive Smoke Test Suite for Tasks 8, 9, 10, 11, and 12.

Verifies:
- Task 8: PPMI Skill-relation graph extraction, exact edge weights vs skill_graph.pkl,
          weak skill highlighting (<50%), and honest new-student empty state.
- Task 9: Per-student mastery-over-time trend recalculation at historical session boundaries,
          multi-session non-constant trajectory, and single-session single-point handling.
- Task 10: Data-sufficiency indicators (interaction counts & low-confidence flags for <3 answers).
- Task 11: Demo hygiene -- reset script confirmation safety (--confirm) and test data isolation/tagging.
- Task 12: Live gate interpretability extraction for real session, timestep alignment,
           and graceful short-session (<3 interactions) handling.
"""

import sys
import pickle
import subprocess
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.inference_service import app, artifacts
from serving.session import session_manager
from serving.db import Database


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def raw_graph():
    graph_path = PROJECT_ROOT / "data" / "processed" / "skill_graph.pkl"
    with open(graph_path, "rb") as f:
        return pickle.load(f)


# =====================================================================
# TASK 8 SMOKE TESTS: Skill-relation graph visualization
# =====================================================================

def test_task_8_ppmi_subgraph_nodes_edges_and_exact_weights(client, raw_graph):
    """
    1. Render the graph for a completed real session.
    Assert:
    - Exactly 10 nodes appear (matching skill_sequence).
    - Edge weights match skill_graph.pkl's actual PPMI values for those skill pairs
      (spot-check at least 2-3 edges directly against the raw pickle file).
    """
    student_id = f"test_task8_student_{uuid.uuid4().hex[:6]}"

    # Complete a session
    s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    assert s_res.status_code == 200
    sid = s_res.json()["session_id"]
    assigned_set = session_manager.get_session(sid)["set_name"]

    for pos in range(1, 11):
        q = session_manager.bank.get_question(assigned_set, pos)
        # Alternate correctness to produce varied mastery
        ans = q["correct_answer"] if pos in [1, 2, 3, 7, 8] else "wrong_answer"
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": ans,
            "response_time_ms": 4000.0,
            "hint_used": 0
        })

    # Fetch subgraph with session_id
    graph_res = client.get(f"/graph/subgraph?session_id={sid}")
    assert graph_res.status_code == 200
    g_data = graph_res.json()

    # 1. Assert exactly 10 nodes
    nodes = g_data["nodes"]
    assert len(nodes) == 10, f"Expected 10 nodes, got {len(nodes)}"
    expected_skill_ids = [s["raw_skill_id"] for s in session_manager.bank.skill_sequence]
    actual_skill_ids = [n["raw_skill_id"] for n in nodes]
    assert actual_skill_ids == expected_skill_ids, "Nodes do not match 10-skill sequence"

    # 2. Spot-check 3 edge weights directly against raw skill_graph.pkl matrix
    adj = raw_graph["adjacency_matrix"]
    s_map = raw_graph["skill_map"]
    edges = g_data["edges"]
    assert len(edges) > 0, "Graph edges must not be empty"

    # Spot check pairs: (204, 203), (204, 217), (350, 378)
    spot_pairs = [(204, 203), (204, 217), (350, 378)]
    for raw1, raw2 in spot_pairs:
        idx1 = s_map[raw1]
        idx2 = s_map[raw2]
        expected_weight = round(float(adj[idx1, idx2]), 4)

        matching_edge = next(
            (e for e in edges if (e["source"] == raw1 and e["target"] == raw2) or
                                 (e["source"] == raw2 and e["target"] == raw1)),
            None
        )
        assert matching_edge is not None, f"Expected edge between {raw1} and {raw2} not found"
        assert matching_edge["weight"] == expected_weight, (
            f"Edge weight mismatch for ({raw1}, {raw2}): API returned {matching_edge['weight']}, "
            f"raw pickle matrix has {expected_weight}"
        )

    print(f"\n[Task 8 Smoke Test 1 PASS] Graph rendered with 10 nodes and verified PPMI weights against raw pickle.")


def test_task_8_weak_skill_highlighting_and_empty_state(client):
    """
    2. Student with at least one skill below weak threshold (<50%): assert skill is visually flagged (is_weak=True).
    3. Brand-new student ID with no completed session: assert honest empty-state response (has_completed_session=False).
    """
    # Weak skill test
    student_id = f"test_weak_student_{uuid.uuid4().hex[:6]}"
    s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid = s_res.json()["session_id"]
    assigned_set = session_manager.get_session(sid)["set_name"]

    # Submit all wrong answers to guarantee weak skills (<50%)
    for pos in range(1, 11):
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": "completely_wrong_ans",
            "response_time_ms": 12000.0,
            "hint_used": 0
        })

    graph_res = client.get(f"/graph/subgraph?student_id={student_id}")
    assert graph_res.status_code == 200
    g_data = graph_res.json()
    assert g_data["has_completed_session"] is True
    weak_nodes = [n for n in g_data["nodes"] if n.get("is_weak") is True]
    assert len(weak_nodes) > 0, "Expected at least one weak skill node with is_weak=True"
    for w in weak_nodes:
        assert w["mastery"] < 0.50

    # Brand-new student empty state test
    new_student_id = f"test_brand_new_{uuid.uuid4().hex[:8]}"
    new_graph_res = client.get(f"/graph/subgraph?student_id={new_student_id}")
    assert new_graph_res.status_code == 200
    new_g_data = new_graph_res.json()
    assert new_g_data["has_completed_session"] is False
    assert "message" in new_g_data
    assert "complete a session" in new_g_data["message"].lower()

    print(f"\n[Task 8 Smoke Test 2 & 3 PASS] Weak skill highlighting and new-student empty state verified.")


# =====================================================================
# TASK 9 SMOKE TESTS: Per-student mastery-over-time trend
# =====================================================================

def test_task_9_multi_session_trend_trajectory(client):
    """
    1. For a student with 3 completed sessions, assert returned trend data has
       one mastery value per session boundary, and that at least one skill's
       trajectory is non-constant across sessions.
    2. For a student with exactly 1 session, assert endpoint returns a single
       data point without crashing or fabricating additional points.
    """
    student_id = f"test_trend_student_{uuid.uuid4().hex[:6]}"

    # Session 1: Mostly correct
    s1_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid1 = s1_res.json()["session_id"]
    set1 = session_manager.get_session(sid1)["set_name"]
    for pos in range(1, 11):
        q = session_manager.bank.get_question(set1, pos)
        client.post(f"/session/{sid1}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 3500.0,
            "hint_used": 0
        })

    # Check 1-session trend endpoint
    t1_res = client.get(f"/student/{student_id}/trend")
    assert t1_res.status_code == 200
    t1_data = t1_res.json()
    assert t1_data["total_completed_sessions"] == 1
    assert len(t1_data["trend"]) == 1, "Expected exactly 1 data point for 1-session student"
    assert t1_data["trend"][0]["session_index"] == 1
    assert t1_data["trend"][0]["cumulative_interactions"] == 10

    # Session 2: All wrong (drastic performance change)
    s2_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid2 = s2_res.json()["session_id"]
    set2 = session_manager.get_session(sid2)["set_name"]
    for pos in range(1, 11):
        client.post(f"/session/{sid2}/answer", json={
            "position": pos,
            "submitted_answer": "incorrect_answer",
            "response_time_ms": 9000.0,
            "hint_used": 0
        })

    # Session 3: Mixed responses
    s3_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid3 = s3_res.json()["session_id"]
    set3 = session_manager.get_session(sid3)["set_name"]
    for pos in range(1, 11):
        q = session_manager.bank.get_question(set3, pos)
        ans = q["correct_answer"] if pos % 2 == 1 else "wrong"
        client.post(f"/session/{sid3}/answer", json={
            "position": pos,
            "submitted_answer": ans,
            "response_time_ms": 5000.0,
            "hint_used": 0
        })

    # Query 3-session trend
    t3_res = client.get(f"/student/{student_id}/trend")
    assert t3_res.status_code == 200
    t3_data = t3_res.json()
    assert t3_data["total_completed_sessions"] == 3
    assert len(t3_data["trend"]) == 3, f"Expected 3 session points, got {len(t3_data['trend'])}"

    # Assert cumulative interaction counts: 10, 20, 30
    assert t3_data["trend"][0]["cumulative_interactions"] == 10
    assert t3_data["trend"][1]["cumulative_interactions"] == 20
    assert t3_data["trend"][2]["cumulative_interactions"] == 30

    # Check non-constant trajectory across session boundaries
    # Skill 204 (Percents)
    m_s1 = t3_data["trend"][0]["mastery"]["204"]
    m_s2 = t3_data["trend"][1]["mastery"]["204"]
    m_s3 = t3_data["trend"][2]["mastery"]["204"]

    # Trajectory must not be flat
    assert not (m_s1 == m_s2 == m_s3), f"Trajectory for skill 204 was flat: {m_s1}, {m_s2}, {m_s3}"
    print(f"\n[Task 9 Smoke Test PASS] Trend computed across 3 session cutoffs: s1={m_s1:.4f}, s2={m_s2:.4f}, s3={m_s3:.4f}")


# =====================================================================
# TASK 10 SMOKE TESTS: Data-sufficiency indicators
# =====================================================================

def test_task_10_data_sufficiency_indicators(client):
    """
    1. Student who has seen a skill only once (1 session): assert skill is flagged low-data (is_low_data=True / count=1).
    2. Student who has seen a skill 5+ times (5 sessions): assert skill is NOT flagged low-data (count >= 3).
    """
    student_1 = f"test_data_low_{uuid.uuid4().hex[:6]}"
    s1_res = client.post("/session/start", json={"student_id": student_1, "source": "test"})
    sid1 = s1_res.json()["session_id"]
    set1 = session_manager.get_session(sid1)["set_name"]
    for pos in range(1, 11):
        q = session_manager.bank.get_question(set1, pos)
        client.post(f"/session/{sid1}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 4000.0,
            "hint_used": 0
        })

    # Prediction response data sufficiency
    pred_res1 = client.post(f"/session/{sid1}/predict")
    p1_data = pred_res1.json()
    assert "skill_interaction_counts" in p1_data
    assert "low_confidence_skills" in p1_data
    assert p1_data["skill_interaction_counts"]["204"] == 1
    assert "204" in p1_data["low_confidence_skills"]

    # Graph node data sufficiency
    g_res1 = client.get(f"/graph/subgraph?student_id={student_1}")
    node_204 = next(n for n in g_res1.json()["nodes"] if n["raw_skill_id"] == 204)
    assert node_204["interaction_count"] == 1
    assert node_204["is_low_data"] is True

    # 5-session student (each skill answered 5 times)
    student_5 = f"test_data_high_{uuid.uuid4().hex[:6]}"
    for sess_idx in range(5):
        s_res = client.post("/session/start", json={"student_id": student_5, "source": "test"})
        s_id = s_res.json()["session_id"]
        s_set = session_manager.get_session(s_id)["set_name"]
        for pos in range(1, 11):
            q = session_manager.bank.get_question(s_set, pos)
            client.post(f"/session/{s_id}/answer", json={
                "position": pos,
                "submitted_answer": q["correct_answer"],
                "response_time_ms": 4000.0,
                "hint_used": 0
            })

    pred_res5 = client.post(f"/student/{student_5}/predict")
    p5_data = pred_res5.json()
    assert p5_data["skill_interaction_counts"]["204"] == 5
    assert "204" not in p5_data["low_confidence_skills"]

    g_res5 = client.get(f"/graph/subgraph?student_id={student_5}")
    node_204_high = next(n for n in g_res5.json()["nodes"] if n["raw_skill_id"] == 204)
    assert node_204_high["interaction_count"] == 5
    assert node_204_high["is_low_data"] is False

    print(f"\n[Task 10 Smoke Test PASS] Low-data flag (<3 answers) and high-confidence transition verified.")


# =====================================================================
# TASK 11 SMOKE TESTS: Demo hygiene & Database reset
# =====================================================================

def test_task_11_reset_demo_db_script_safety_and_execution(tmp_path):
    """
    1. Run reset script WITHOUT --confirm. Assert: exits with code 1, does NOT wipe data.
    2. Run reset script WITH --confirm. Assert: tables are empty afterward.
    3. Tagging test: assert test students (source='test' or prefix test_) are filtered from live matrix view.
    """
    test_db_file = tmp_path / "hygiene_test.db"
    db_test = Database(test_db_file)
    sid = "sess_01"
    db_test.create_session(session_id=sid, student_id="live_student_demo", set_name="A", source="live")
    db_test.record_interaction(sid, "live_student_demo", 1, 204, 1, 4000.0, 1, 0)
    db_test.complete_session(sid)

    # 1. Run reset script WITHOUT --confirm
    res_no_confirm = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reset_demo_db.py"), "--db-path", str(test_db_file)],
        capture_output=True,
        text=True
    )
    assert res_no_confirm.returncode == 1, "Script without --confirm must exit code 1"
    assert "ERROR: Database reset aborted -- missing confirmation flag!" in res_no_confirm.stdout

    # Assert data was NOT deleted
    assert len(db_test.get_all_completed_students(source="all")) == 1

    # 2. Run reset script WITH --confirm
    res_with_confirm = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reset_demo_db.py"), "--confirm", "--db-path", str(test_db_file)],
        capture_output=True,
        text=True
    )
    assert res_with_confirm.returncode == 0, "Script with --confirm must exit code 0"
    assert "DATABASE RESET SUCCESSFUL" in res_with_confirm.stdout
    assert len(db_test.get_all_completed_students(source="all")) == 0

    print(f"\n[Task 11 Smoke Test PASS] Reset script safety refusal and confirmed wipe verified.")


def test_task_11_test_student_tagging_isolation(client):
    """
    Tagging verification:
    Create a student with prefix 'test_' and a student with prefix 'demo_'.
    Assert live dashboard matrix (?source=live) contains 'demo_' and excludes 'test_'.
    """
    # Create test student
    test_st = f"test_automated_fixture_{uuid.uuid4().hex[:6]}"
    s1 = client.post("/session/start", json={"student_id": test_st, "source": "test"}).json()["session_id"]
    set1 = session_manager.get_session(s1)["set_name"]
    for pos in range(1, 11):
        q = session_manager.bank.get_question(set1, pos)
        client.post(f"/session/{s1}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    # Create live student
    live_st = f"demo_live_user_{uuid.uuid4().hex[:6]}"
    s2 = client.post("/session/start", json={"student_id": live_st, "source": "live"}).json()["session_id"]
    set2 = session_manager.get_session(s2)["set_name"]
    for pos in range(1, 11):
        q = session_manager.bank.get_question(set2, pos)
        client.post(f"/session/{s2}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 4000.0,
            "hint_used": 0
        })

    # Query live matrix
    live_matrix = client.get("/students/matrix?source=live").json()
    live_ids = [s["student_id"] for s in live_matrix["students"]]

    assert live_st in live_ids, f"Expected live student '{live_st}' in live matrix"
    assert test_st not in live_ids, f"Test student '{test_st}' should be filtered from live matrix"

    # Query test matrix
    test_matrix = client.get("/students/matrix?source=test").json()
    test_ids = [s["student_id"] for s in test_matrix["students"]]
    assert test_st in test_ids, f"Test student '{test_st}' should be present in test matrix"

    print(f"\n[Task 11 Tagging PASS] Test-fixture tagging isolates automated runs from live dashboard.")


# =====================================================================
# TASK 12 SMOKE TESTS: Live gate interpretability
# =====================================================================

def test_task_12_live_gate_interpretability(client):
    """
    1. For a completed real session (10 interactions), assert gate detail has 9 timesteps (10-1 transitions),
       computes top-5 active dimensions, and generates gate plot image without error.
    2. For a short session (<3 interactions), assert system handles gracefully without crashing.
    """
    student_id = f"test_gate_student_{uuid.uuid4().hex[:6]}"
    s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid = s_res.json()["session_id"]
    assigned_set = session_manager.get_session(sid)["set_name"]

    # 1. Short session (only 2 interactions)
    for pos in range(1, 3):
        q = session_manager.bank.get_question(assigned_set, pos)
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    short_gate = client.get(f"/session/{sid}/gate_detail").json()
    assert short_gate["available"] is False
    assert "at least 3" in short_gate["message"].lower()

    # 2. Complete remaining 8 questions (10 total interactions)
    for pos in range(3, 11):
        q = session_manager.bank.get_question(assigned_set, pos)
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": q["correct_answer"],
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    full_gate = client.get(f"/session/{sid}/gate_detail").json()
    assert full_gate["available"] is True
    assert full_gate["num_interactions"] == 10
    assert full_gate["num_timesteps"] == 9  # 10 - 1
    assert len(full_gate["step_gate_means"]) == 9
    assert len(full_gate["top5_dimensions"]) == 5
    for d in full_gate["top5_dimensions"]:
        assert f"dim_{d}" in full_gate["top5_trajectories"]
        assert len(full_gate["top5_trajectories"][f"dim_{d}"]) == 9

    # Test gate plot image generation
    plot_res = client.get(f"/session/{sid}/gate_plot.png")
    assert plot_res.status_code == 200
    assert plot_res.headers["content-type"] == "image/png"
    assert len(plot_res.content) > 1000  # valid PNG binary

    print(f"\n[Task 12 Smoke Test PASS] Live gate extraction and 2-panel figure generation verified.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
