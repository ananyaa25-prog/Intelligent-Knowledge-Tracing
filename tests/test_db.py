"""
tests/test_db.py
----------------
Smoke test suite for Task 4: SQLite persistence layer and multi-session student history.

Verifies:
1. Direct DB Query: 10 interactions written to SQLite with matching skill IDs in order.
2. Persistence Across Process Restart: Fresh SessionManager instance retains all 10 interactions.
3. Multi-Session Cumulative History: Same student completing Session 2 has 20 interactions in chronological order.
4. Student Isolation: 3 distinct students have strictly isolated interaction histories.
"""

import sys
import sqlite3
import tempfile
from pathlib import Path
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.db import Database
from serving.session import SessionManager, QuestionBank, _question_bank


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides an isolated SQLite file path for testing."""
    return tmp_path / "test_kt.db"


@pytest.fixture
def db(temp_db_path):
    return Database(temp_db_path)


@pytest.fixture
def manager(db):
    return SessionManager(_question_bank, db)


def test_1_direct_database_query_after_session(manager, temp_db_path):
    """
    1. Start a session for a NEW student_id. Complete all 10 answers.
    Assert: Querying the database directly shows exactly 10 interaction rows
    for that student_id, with correct skill_id values matching skill_sequence in order.
    """
    student_id = "student_alpha"
    sid, questions = manager.start_session(student_id=student_id, source="test")
    sess = manager.get_session(sid)
    assigned_set = sess["set_name"]

    for pos in range(1, 11):
        q = manager.bank.get_question(assigned_set, pos)
        ans = q["correct_answer"]
        manager.record_answer(
            session_id=sid,
            position=pos,
            submitted_answer=ans,
            response_time_ms=4000.0 + pos * 100,
            hint_used=0
        )

    # Direct query to SQLite file
    conn = sqlite3.connect(str(temp_db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    st_row = cursor.fetchone()
    assert st_row is not None, "Student record missing in SQLite"

    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,))
    sess_row = cursor.fetchone()
    assert sess_row is not None, "Session record missing in SQLite"
    assert sess_row["completed_at"] is not None, "Session should be marked completed"

    cursor.execute(
        "SELECT position, skill_id, correct FROM interactions WHERE student_id = ? ORDER BY position ASC",
        (student_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 10, f"Expected 10 rows in SQLite, found {len(rows)}"
    expected_skills = [s["raw_skill_id"] for s in manager.bank.skill_sequence]
    actual_skills = [r["skill_id"] for r in rows]

    assert actual_skills == expected_skills, (
        f"Skill sequence in SQLite mismatch!\nExpected: {expected_skills}\nActual:   {actual_skills}"
    )

    print(f"\n[Task 4 Test 1 PASS] Direct SQLite query confirmed 10 rows for {student_id} in exact skill order.")


def test_2_persistence_across_restart(temp_db_path):
    """
    2. RESTART TEST: Create a fresh SessionManager instance pointed at the SAME
    database file and query get_full_history for that student_id again.
    Assert: The 10 interactions are still there. (PROVES PERSISTENCE SURVIVES RESTART)
    """
    student_id = "student_persistent"

    # Instance 1: Create session and record 10 answers
    db1 = Database(temp_db_path)
    mgr1 = SessionManager(_question_bank, db1)
    sid, questions = mgr1.start_session(student_id=student_id, source="test")
    assigned_set = mgr1.get_session(sid)["set_name"]

    for pos in range(1, 11):
        q = mgr1.bank.get_question(assigned_set, pos)
        mgr1.record_answer(
            session_id=sid,
            position=pos,
            submitted_answer=q["correct_answer"],
            response_time_ms=5000.0,
            hint_used=0
        )

    # Simulate process shutdown and clean state
    del mgr1
    del db1

    # Instance 2: Fresh start connected to same SQLite DB
    db2 = Database(temp_db_path)
    mgr2 = SessionManager(_question_bank, db2)

    recovered_history = mgr2.get_full_history(student_id)
    assert len(recovered_history) == 10, (
        f"PERSISTENCE FAILURE: Expected 10 interactions after restart, got {len(recovered_history)}"
    )

    expected_skills = [s["raw_skill_id"] for s in mgr2.bank.skill_sequence]
    actual_skills = [item["skill_id"] for item in recovered_history]
    assert actual_skills == expected_skills

    summary = mgr2.get_student_summary(student_id)
    assert summary["session_count"] == 1
    assert summary["completed_session_count"] == 1
    assert summary["total_interactions"] == 10

    print(f"\n[Task 4 Test 2 PASS] Process restart verified: Fresh SessionManager recovered all 10 interactions.")


def test_3_same_student_multi_session_cumulative_history(manager):
    """
    3. Have the SAME student_id complete a SECOND session (10 more answers).
    Assert: get_full_history for that student_id returns 20 interactions total,
    in correct chronological order (first session's 10, then second session's 10).
    """
    student_id = "student_repeat"

    # Session 1
    sid1, _ = manager.start_session(student_id=student_id, source="test")
    set1 = manager.get_session(sid1)["set_name"]
    for pos in range(1, 11):
        q = manager.bank.get_question(set1, pos)
        manager.record_answer(sid1, pos, q["correct_answer"], 3000.0 + pos * 100)

    # Session 2 (same student ID)
    sid2, _ = manager.start_session(student_id=student_id, source="test")
    set2 = manager.get_session(sid2)["set_name"]
    for pos in range(1, 11):
        q = manager.bank.get_question(set2, pos)
        # Alternate answer correctness in second session
        ans = q["correct_answer"] if pos % 2 == 0 else "wrong_ans"
        manager.record_answer(sid2, pos, ans, 6000.0 + pos * 100)

    # Retrieve cumulative history
    full_history = manager.get_full_history(student_id)
    assert len(full_history) == 20, f"Expected 20 cumulative interactions, got {len(full_history)}"

    expected_skills = [s["raw_skill_id"] for s in manager.bank.skill_sequence]
    first_10_skills = [h["skill_id"] for h in full_history[:10]]
    second_10_skills = [h["skill_id"] for h in full_history[10:]]

    assert first_10_skills == expected_skills, "Session 1 skill order mismatch"
    assert second_10_skills == expected_skills, "Session 2 skill order mismatch"

    # Verify correctness matches the two distinct sessions
    assert all(h["correct"] == 1 for h in full_history[:10]), "Session 1 answers should all be correct"
    assert full_history[10]["correct"] == 0, "Session 2 pos 1 should be incorrect"
    assert full_history[11]["correct"] == 1, "Session 2 pos 2 should be correct"

    summary = manager.get_student_summary(student_id)
    assert summary["session_count"] == 2
    assert summary["completed_session_count"] == 2
    assert summary["total_interactions"] == 20

    print(f"\n[Task 4 Test 3 PASS] Multi-session accumulation verified: 20 interactions recorded across 2 sessions.")


def test_4_student_isolation(manager):
    """
    4. Start sessions for 3 different student_ids.
    Assert: Each student's get_full_history returns ONLY their own interactions,
    never mixed with another student's.
    """
    students = ["student_user_1", "student_user_2", "student_user_3"]

    for idx, st_id in enumerate(students):
        sid, _ = manager.start_session(student_id=st_id, source="test")
        sess = manager.get_session(sid)
        set_name = sess["set_name"]

        # Each student answers a different number of questions (e.g. 4, 7, 10)
        num_questions = 4 + idx * 3
        for pos in range(1, num_questions + 1):
            q = manager.bank.get_question(set_name, pos)
            manager.record_answer(sid, pos, q["correct_answer"], 4000.0)

    # Verify isolation
    hist1 = manager.get_full_history("student_user_1")
    hist2 = manager.get_full_history("student_user_2")
    hist3 = manager.get_full_history("student_user_3")

    assert len(hist1) == 4, f"Student 1 expected 4 interactions, got {len(hist1)}"
    assert len(hist2) == 7, f"Student 2 expected 7 interactions, got {len(hist2)}"
    assert len(hist3) == 10, f"Student 3 expected 10 interactions, got {len(hist3)}"

    completed_students = manager.db.get_all_completed_students(source="all")
    completed_ids = [s["student_id"] for s in completed_students]
    assert "student_user_3" in completed_ids
    assert "student_user_1" not in completed_ids  # Only 4 questions answered

    print(f"\n[Task 4 Test 4 PASS] Student isolation verified across 3 separate student histories.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
