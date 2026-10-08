"""
tests/test_tasks_13_16_smoke.py
---------------------------------
Comprehensive Smoke Test Suite for Tasks 13, 14, 15, and 16.

Verifies:
- Task 13: Procedural Question Generation (question variety, mathematical correctness,
           anti-repeat safeguard, and server-side grading).
- Task 14: Mastery-over-Time Trend (multi-session trajectory, single-session point handling,
           and exact parity against /predict on truncated histories).
- Task 15: Demo Data Hygiene (source='test' vs source='live' isolation in dashboard,
           and reset_demo_db.py safety flag enforcement).
- Task 16: PPMI Weak-Skill Relational Callouts (surfacing connected weak skills with real PPMI weights,
           and no fabricated callouts when isolated).
"""

import sys
import pickle
import subprocess
import uuid
import re
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.inference_service import app, artifacts
from serving.session import session_manager, check_answer
from serving.question_templates import generate_question, GENERATOR_MAP
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
# TASK 13 SMOKE TESTS: Procedural Generation
# =====================================================================

def test_task_13_question_variety_position_1():
    """
    1. Generate 50 questions for position 1 (Percents) in a loop.
    Assert: Significant variety in generated question_text values (>= 20 unique strings out of 50).
    """
    generated_texts = set()
    for _ in range(50):
        q = generate_question(1)
        generated_texts.add(q["question_text"])

    assert len(generated_texts) >= 20, (
        f"Expected at least 20 unique question strings out of 50 for position 1, got {len(generated_texts)}"
    )


def test_task_13_all_10_positions_mathematical_correctness():
    """
    2. For each of the 10 positions, generate 10 questions and assert every computed
    correct_answer is actually mathematically correct given the generated parameters
    (independent re-computation in test).
    """
    for pos in range(1, 11):
        for _ in range(10):
            q = generate_question(pos)
            params = q.get("params", {})
            corr = q["correct_answer"]
            q_text = q["question_text"]

            if pos == 1:
                # Percents: What is {pct}% of {base}?
                pct = params["pct"]
                base = params["base"]
                expected = (pct * base) // 100
                assert int(corr) == expected
                assert f"What is {pct}% of {base}?" == q_text

            elif pos == 2:
                # Percent Discount: A {item} costs ${price}... sale price?
                price = params["price"]
                discount = params["discount"]
                expected = price - ((price * discount) // 100)
                assert int(corr) == expected
                assert f"{discount}% off" in q_text

            elif pos == 3:
                # Rate: travels {distance} in {time}...
                distance = params["distance"]
                time = params["time"]
                expected = distance // time
                assert int(corr) == expected

            elif pos == 4:
                # Polynomial Factors: Factor completely: {var}^2 - {a_sq} -> ({var}-{a})({var}+{a})
                var = params["var"]
                a = params["a"]
                a_sq = params["a_sq"]
                assert a * a == a_sq
                assert corr == f"({var}-{a})({var}+{a})"

            elif pos == 5:
                # Exponents: Simplify: {var}^{p1} * {var}^{p2} -> {var}^{p1+p2}
                var = params["var"]
                p1 = params["p1"]
                p2 = params["p2"]
                assert corr == f"{var}^{p1 + p2}"

            elif pos == 6:
                # Systems: x + y = {sum}, x - y = {diff}. What is x?
                x = params["x"]
                y = params["y"]
                sum_val = params["sum"]
                diff_val = params["diff"]
                assert sum_val == x + y
                assert diff_val == x - y
                assert int(corr) == x

            elif pos == 7:
                # Systems by Graphing: intersection x-value
                x = params["x"]
                m1 = params["m1"]
                c1 = params["c1"]
                m2 = params["m2"]
                c2 = params["c2"]
                # At x: m1 * x + c1 == m2 * x + c2
                y1 = m1 * x + c1
                y2 = m2 * x + c2
                assert y1 == y2
                assert int(corr) == x

            elif pos == 8:
                # Ordering Fractions: Which is larger: n1/d1 or n2/d2?
                f1_str = params["f1"]
                f2_str = params["f2"]
                n1, d1 = map(int, f1_str.split('/'))
                n2, d2 = map(int, f2_str.split('/'))
                expected_larger = f1_str if (n1 * d2 > n2 * d1) else f2_str
                assert corr == expected_larger

            elif pos == 9:
                # Equivalent Fractions: Which fraction is equivalent to n/d?
                base_n = params["base_n"]
                base_d = params["base_d"]
                ans_n = params["ans_n"]
                ans_d = params["ans_d"]
                assert ans_n * base_d == ans_d * base_n
                assert corr == f"{ans_n}/{ans_d}"

            elif pos == 10:
                # Pythagorean Theorem: legs leg_a, leg_b, hypotenuse hyp
                leg_a = params["leg_a"]
                leg_b = params["leg_b"]
                hyp = params["hypotenuse"]
                assert leg_a * leg_a + leg_b * leg_b == hyp * hyp
                assert int(corr) == hyp


def test_task_13_anti_repeat_and_back_to_back_session_variety(client):
    """
    3. Have the same student_id complete two sessions back to back.
    Assert: At least some positions show different question_text between the two sessions.
    """
    student_id = f"test_st_repeat_{uuid.uuid4().hex[:6]}"

    # Session 1
    s1_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    assert s1_res.status_code == 200
    s1_data = s1_res.json()
    s1_id = s1_data["session_id"]
    s1_questions = {q["position"]: q["question_text"] for q in s1_data["questions"]}

    # Complete session 1
    for pos in range(1, 11):
        client.post(f"/session/{s1_id}/answer", json={
            "position": pos,
            "submitted_answer": "10",
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    # Session 2 for SAME student
    s2_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    assert s2_res.status_code == 200
    s2_data = s2_res.json()
    s2_questions = {q["position"]: q["question_text"] for q in s2_data["questions"]}

    # Compare question texts between session 1 and session 2
    differences = 0
    for pos in range(1, 11):
        if s1_questions[pos] != s2_questions[pos]:
            differences += 1

    assert differences > 0, (
        f"Expected different question texts across back-to-back sessions for {student_id}, but all 10 were identical!"
    )


def test_task_13_answer_checking_grading_procedural_values():
    """
    4. Confirm existing answer-checking logic correctly grades answers against procedurally generated values.
    """
    for pos in range(1, 11):
        q = generate_question(pos)
        correct_ans = q["correct_answer"]

        # Exact match
        assert check_answer(correct_ans, correct_ans) is True
        # Trimmed match
        assert check_answer(f"  {correct_ans}  ", correct_ans) is True
        # Case-insensitive
        assert check_answer(correct_ans.upper(), correct_ans.lower()) is True

        # If numeric, numeric tolerance
        try:
            num = float(correct_ans)
            assert check_answer(num, correct_ans) is True
            assert check_answer(num + 0.005, correct_ans) is True
            assert check_answer(num + 5.0, correct_ans) is False
        except ValueError:
            pass


# =====================================================================
# TASK 14 SMOKE TESTS: Mastery-over-Time Trend
# =====================================================================

def test_task_14_mastery_trend_three_completed_sessions(client):
    """
    1. For a student with 3 real completed sessions (using procedural generator),
    assert trend endpoint returns 3 distinct mastery snapshots and that at least one skill
    shows a non-identical value across at least two snapshots.
    """
    student_id = f"test_trend_3sess_{uuid.uuid4().hex[:6]}"

    for s_idx in range(1, 4):
        s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
        sid = s_res.json()["session_id"]

        # In session 1: get everything wrong
        # In session 2: get positions 1-5 right, 6-10 wrong
        # In session 3: get everything right
        for pos in range(1, 11):
            sess_obj = session_manager.get_session(sid)
            q_info = sess_obj.get("questions", {}).get(pos, {})
            correct_str = q_info.get("correct_answer", "10")

            if s_idx == 1:
                ans = "wrong_answer"
            elif s_idx == 2:
                ans = correct_str if pos <= 5 else "wrong_answer"
            else:
                ans = correct_str

            client.post(f"/session/{sid}/answer", json={
                "position": pos,
                "submitted_answer": ans,
                "response_time_ms": 2500.0 + pos * 50,
                "hint_used": 0
            })
        client.post(f"/session/{sid}/predict")

    # Fetch trend
    trend_res = client.get(f"/student/{student_id}/trend")
    assert trend_res.status_code == 200
    t_data = trend_res.json()

    assert t_data["total_completed_sessions"] == 3
    assert len(t_data["trend"]) == 3

    # Verify trajectory changes over time (not flat identical)
    snap1 = t_data["trend"][0]["mastery"]
    snap2 = t_data["trend"][1]["mastery"]
    snap3 = t_data["trend"][2]["mastery"]

    changed_skills = 0
    for sk in snap1:
        v1 = snap1[sk]
        v3 = snap3.get(sk, v1)
        if abs(v1 - v3) > 1e-4:
            changed_skills += 1

    assert changed_skills >= 1, (
        "Expected at least one skill to change mastery across sessions 1 to 3, but all snapshots were identical!"
    )


def test_task_14_mastery_trend_single_session_handling(client):
    """
    2. For a student with exactly 1 session, assert a single point is returned,
    no crash, no fabricated second point.
    """
    student_id = f"test_trend_1sess_{uuid.uuid4().hex[:6]}"
    s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid = s_res.json()["session_id"]

    for pos in range(1, 11):
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": "test",
            "response_time_ms": 3000.0,
            "hint_used": 0
        })
    client.post(f"/session/{sid}/predict")

    trend_res = client.get(f"/student/{student_id}/trend")
    assert trend_res.status_code == 200
    t_data = trend_res.json()

    assert t_data["total_completed_sessions"] == 1
    assert len(t_data["trend"]) == 1
    assert "Single session" in t_data.get("message", "")


def test_task_14_mastery_trend_parity_against_truncated_predict(client):
    """
    3. Cross-check one snapshot's values against directly calling /predict with history
    manually truncated to that same session boundary -- must match identically.
    """
    student_id = f"test_parity_{uuid.uuid4().hex[:6]}"

    # Complete 2 sessions
    for _ in range(2):
        s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
        sid = s_res.json()["session_id"]
        for pos in range(1, 11):
            client.post(f"/session/{sid}/answer", json={
                "position": pos,
                "submitted_answer": "ans",
                "response_time_ms": 4000.0,
                "hint_used": 0
            })
        client.post(f"/session/{sid}/predict")

    trend_res = client.get(f"/student/{student_id}/trend")
    assert trend_res.status_code == 200
    t_data = trend_res.json()
    snap1_mastery = t_data["trend"][0]["mastery"]

    # Manually fetch session 1 history and call /predict directly
    full_hist = session_manager.get_full_history(student_id)
    s1_hist = full_hist[:10]  # first 10 interactions

    pred_res = client.post("/predict", json={"student_history": s1_hist})
    assert pred_res.status_code == 200
    direct_mastery = pred_res.json()["skill_mastery"]

    for sk_id, val in snap1_mastery.items():
        assert abs(val - direct_mastery[sk_id]) < 1e-5, (
            f"Skill {sk_id} mastery mismatch between trend snapshot ({val}) and direct /predict ({direct_mastery[sk_id]})"
        )


# =====================================================================
# TASK 15 SMOKE TESTS: Demo Data Hygiene & Reset Script
# =====================================================================

def test_task_15_test_data_isolation_in_dashboard(client):
    """
    1. Students tagged source='test' (or starting with test_) must NEVER appear in live matrix view.
    """
    test_id = f"test_isolated_st_{uuid.uuid4().hex[:6]}"
    s_res = client.post("/session/start", json={"student_id": test_id, "source": "test"})
    sid = s_res.json()["session_id"]
    for pos in range(1, 11):
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": "ans",
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    # Query live matrix
    live_res = client.get("/students/matrix?source=live")
    assert live_res.status_code == 200
    live_students = [s["student_id"] for s in live_res.json()["students"]]

    assert test_id not in live_students, f"Test student {test_id} leaked into live matrix view!"


def test_task_15_live_student_appears_in_live_dashboard(client):
    """
    2. A student created with source='live' DOES appear in the live view.
    """
    live_id = f"live_clean_st_{uuid.uuid4().hex[:6]}"
    s_res = client.post("/session/start", json={"student_id": live_id, "source": "live"})
    sid = s_res.json()["session_id"]
    for pos in range(1, 11):
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": "ans",
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    live_res = client.get("/students/matrix?source=live")
    assert live_res.status_code == 200
    live_students = [s["student_id"] for s in live_res.json()["students"]]

    assert live_id in live_students, f"Live student {live_id} should appear in live matrix view!"


def test_task_15_reset_script_safety_and_execution():
    """
    3. Run reset_demo_db.py without --confirm -> refuses and exits with non-zero code.
    4. Run reset_demo_db.py with --confirm -> succeeds and resets DB cleanly.
    """
    script_path = PROJECT_ROOT / "scripts" / "reset_demo_db.py"
    assert script_path.exists(), f"Reset script not found at {script_path}"

    # Test 3: Without --confirm
    res_no_confirm = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    assert res_no_confirm.returncode != 0, "reset_demo_db.py without --confirm should exit with non-zero error code"
    assert "missing confirmation flag" in res_no_confirm.stdout.lower() or "error" in res_no_confirm.stdout.lower()

    # Test 4: With --confirm
    res_confirm = subprocess.run(
        [sys.executable, str(script_path), "--confirm"],
        capture_output=True,
        text=True
    )
    assert res_confirm.returncode == 0, f"reset_demo_db.py --confirm failed: {res_confirm.stderr}"
    assert "DATABASE RESET SUCCESSFUL" in res_confirm.stdout


# =====================================================================
# TASK 16 SMOKE TESTS: Weak-Skill Relational Callouts
# =====================================================================

def test_task_16_weak_skill_relational_callout_for_related_pair(client, raw_graph):
    """
    1. Construct a test student weak in two genuinely PPMI-related skills:
       Percents (204) and Percent Discount (203).
       Raw PPMI weight between 204 and 203 is ~8.20.
       Assert: The relational callout appears, naming both skills and the actual PPMI value from skill_graph.pkl.
    """
    student_id = f"test_weak_pair_{uuid.uuid4().hex[:6]}"

    # Verify PPMI weight in raw_graph
    smap = raw_graph["skill_map"]
    adj = raw_graph["adjacency_matrix"]
    ppmi_val = float(adj[smap[204], smap[203]])
    assert ppmi_val > 1.0, f"Expected PPMI > 1.0 between 204 and 203, got {ppmi_val}"

    # Start session and answer wrong on positions 1 (Percents 204) and 2 (Percent Discount 203), correct on others
    s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid = s_res.json()["session_id"]

    for pos in range(1, 11):
        sess_obj = session_manager.get_session(sid)
        q_info = sess_obj.get("questions", {}).get(pos, {})
        corr_ans = q_info.get("correct_answer", "10")

        # Wrong on pos 1 & 2 -> weak mastery (<50%)
        # Right on pos 3-10 -> high mastery (>=50%)
        ans = "wrong" if pos in (1, 2) else corr_ans
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": ans,
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    # Query subgraph
    graph_res = client.get(f"/graph/subgraph?student_id={student_id}")
    assert graph_res.status_code == 200
    g_data = graph_res.json()

    callouts = g_data.get("weak_relational_callouts", [])
    weak_pairs = g_data.get("weak_skill_pairs", [])

    assert len(callouts) >= 1, f"Expected at least 1 relational callout for weak skills, got {callouts}"
    assert len(weak_pairs) >= 1, f"Expected at least 1 weak skill pair, got {weak_pairs}"

    # Verify that every weak pair's PPMI weight matches skill_graph.pkl directly
    for pair in weak_pairs:
        id1 = pair["skill1_id"]
        id2 = pair["skill2_id"]
        reported_ppmi = pair["ppmi_weight"]
        actual_ppmi = float(adj[smap[id1], smap[id2]])

        assert abs(reported_ppmi - actual_ppmi) < 1e-4, (
            f"Reported PPMI ({reported_ppmi}) differs from skill_graph.pkl ({actual_ppmi})"
        )

        # Confirm the corresponding callout string contains the skill names and formatted PPMI
        matching_callouts = [
            c for c in callouts
            if pair["skill1_name"] in c and pair["skill2_name"] in c and f"{reported_ppmi:.2f}" in c
        ]
        assert len(matching_callouts) == 1, (
            f"Could not find exact formatted callout for pair {pair} in {callouts}"
        )


def test_task_16_no_fabricated_callout_for_isolated_weak_skill(client):
    """
    2. Construct a test student weak in only ONE skill (e.g. Pythagorean Theorem 27),
    with no other weak skills.
    Assert: No fabricated relational callout appears (empty list).
    """
    student_id = f"test_isolated_weak_{uuid.uuid4().hex[:6]}"
    s_res = client.post("/session/start", json={"student_id": student_id, "source": "test"})
    sid = s_res.json()["session_id"]

    for pos in range(1, 11):
        sess_obj = session_manager.get_session(sid)
        q_info = sess_obj.get("questions", {}).get(pos, {})
        corr_ans = q_info.get("correct_answer", "10")

        # Wrong ONLY on pos 10 (Pythagorean Theorem 27), right on pos 1-9
        ans = "wrong" if pos == 10 else corr_ans
        client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": ans,
            "response_time_ms": 3000.0,
            "hint_used": 0
        })

    graph_res = client.get(f"/graph/subgraph?student_id={student_id}")
    assert graph_res.status_code == 200
    g_data = graph_res.json()

    callouts = g_data.get("weak_relational_callouts", [])
    assert len(callouts) == 0, f"Expected 0 relational callouts for isolated weak skill, got: {callouts}"
