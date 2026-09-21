"""
tests/test_session.py
---------------------
Smoke test suite for Task 0 (src/serving/session.py).

Verifies:
1. Start a session: exactly 10 questions in position order 1-10, question_text present,
   correct_answer NOT present.
2. Submit correct answer for position 1: recorded interaction has correct=1,
   skill_id matches skill_sequence position 1's raw_skill_id (204).
3. Submit wrong answer for position 2: recorded interaction has correct=0.
4. 20 repeated session-starts: not all 20 pick the same set; prints distribution.
5. Complete full 10-question session: history has exactly 10 interactions,
   with skill_ids in the EXACT same order as skill_sequence regardless of chosen set.
"""

import sys
from pathlib import Path
from collections import Counter
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.session import SessionManager, QuestionBank, _question_bank


@pytest.fixture
def manager():
    """Returns a fresh SessionManager instance for test isolation."""
    return SessionManager(_question_bank)


def test_1_start_session(manager):
    """
    1. Start a session.
    Assert: exactly 10 questions returned, in position order 1-10,
    question_text present for each, correct_answer NOT present in response payload.
    """
    sid, questions = manager.start_session()
    assert sid is not None and len(sid) > 0, "Session ID should be non-empty string"
    assert len(questions) == 10, f"Expected 10 questions, got {len(questions)}"

    for idx, q in enumerate(questions):
        expected_pos = idx + 1
        assert q["position"] == expected_pos, f"Question at index {idx} has position {q['position']}, expected {expected_pos}"
        assert "question_text" in q and len(q["question_text"]) > 0, f"Missing question_text at pos {expected_pos}"
        assert "correct_answer" not in q, f"SECURITY LEAK: correct_answer exposed in payload at pos {expected_pos}!"
        assert "raw_skill_id" in q, f"Missing raw_skill_id at pos {expected_pos}"

    print(f"\n[Task 0 Test 1 PASS] Session {sid[:8]} initialized with 10 questions, no answers revealed.")


def test_2_submit_correct_answer_pos_1(manager):
    """
    2. Submit a correct answer for position 1 (using the known correct_answer
    for whichever set was actually assigned -- read the session's assigned set
    server-side in the test to know which answer is correct).
    Assert: recorded interaction has correct=1, skill_id matches skill_sequence
    position 1's raw_skill_id exactly.
    """
    sid, _ = manager.start_session()
    sess = manager.get_session(sid)
    assigned_set = sess["set_name"]

    # Read the true correct answer server-side
    correct_ans = manager.bank.get_question(assigned_set, 1)["correct_answer"]
    pos_1_skill = manager.bank.get_skill_for_position(1)["raw_skill_id"]

    result = manager.record_answer(
        session_id=sid,
        position=1,
        submitted_answer=correct_ans,
        response_time_ms=5420.0,
        hint_used=0
    )

    assert result["correct"] == 1, f"Expected correct=1 for '{correct_ans}', got {result['correct']}"

    history = manager.get_history(sid)
    assert len(history) == 1
    last_interaction = history[0]

    assert last_interaction["correct"] == 1
    assert last_interaction["skill_id"] == pos_1_skill, (
        f"Expected skill_id={pos_1_skill}, got {last_interaction['skill_id']}"
    )
    assert last_interaction["attempt_count"] == 1
    assert last_interaction["hint_used"] == 0
    assert abs(last_interaction["response_time_ms"] - 5420.0) < 1e-3

    print(f"\n[Task 0 Test 2 PASS] Correct answer verified for set {assigned_set}, position 1 (skill {pos_1_skill}).")


def test_3_submit_wrong_answer_pos_2(manager):
    """
    3. Submit a wrong answer for position 2.
    Assert: recorded interaction has correct=0.
    """
    sid, _ = manager.start_session()
    sess = manager.get_session(sid)
    assigned_set = sess["set_name"]
    pos_2_skill = manager.bank.get_skill_for_position(2)["raw_skill_id"]

    # Submit deliberately wrong answer
    wrong_ans = "obviously_incorrect_answer_xyz_999"

    result = manager.record_answer(
        session_id=sid,
        position=2,
        submitted_answer=wrong_ans,
        response_time_ms=6200.0,
        hint_used=0
    )

    assert result["correct"] == 0, f"Expected correct=0 for wrong answer, got {result['correct']}"

    history = manager.get_history(sid)
    assert len(history) == 1
    last_interaction = history[0]

    assert last_interaction["correct"] == 0
    assert last_interaction["skill_id"] == pos_2_skill
    assert last_interaction["attempt_count"] == 1

    print(f"\n[Task 0 Test 3 PASS] Wrong answer for position 2 recorded with correct=0.")


def test_4_randomization_distribution(manager):
    """
    4. Run 20 repeated session-starts.
    Assert: not all 20 sessions picked the same set (confirms randomization is
    actually happening, not silently always returning 'A').
    Print the distribution of sets chosen.
    """
    chosen_sets = []
    for _ in range(20):
        sid, _ = manager.start_session()
        sess = manager.get_session(sid)
        chosen_sets.append(sess["set_name"])

    counts = Counter(chosen_sets)
    print(f"\n[Task 0 Test 4 Distribution across 20 sessions]:")
    for s in sorted(manager.bank.available_sets):
        print(f"  Set {s}: {counts[s]} times ({counts[s]/20.0*100:.1f}%)")

    # Assert that at least 2 distinct sets were picked across 20 runs
    # Probability of picking the same set 20 times with 5 equal sets is (1/5)^19 ~ 5e-14
    assert len(counts) > 1, f"Randomization failed! All 20 sessions picked set: {chosen_sets[0]}"
    print(f"[Task 0 Test 4 PASS] Multi-set randomization verified across {len(counts)} distinct sets.")


def test_5_full_session_skill_sequence_preservation(manager):
    """
    5. Complete a full 10-question session.
    Assert: the resulting stored history has exactly 10 interactions, with
    skill_id values in the EXACT same order as skill_sequence in question_bank.json,
    regardless of which set (A-E) was used.
    """
    sid, questions = manager.start_session()
    sess = manager.get_session(sid)
    chosen_set = sess["set_name"]

    expected_skill_ids = [s["raw_skill_id"] for s in manager.bank.skill_sequence]

    # Answer all 10 questions with simulated student answers
    for pos in range(1, 11):
        q = manager.bank.get_question(chosen_set, pos)
        # Alternate correct and incorrect to test mixed history
        ans = q["correct_answer"] if (pos % 2 != 0) else "wrong_ans"
        manager.record_answer(
            session_id=sid,
            position=pos,
            submitted_answer=ans,
            response_time_ms=5000.0 + pos * 200,
            hint_used=0
        )

    history = manager.get_history(sid)
    assert len(history) == 10, f"Expected 10 interactions, got {len(history)}"

    actual_skill_ids = [inter["skill_id"] for inter in history]
    assert actual_skill_ids == expected_skill_ids, (
        f"Skill sequence violated!\nExpected: {expected_skill_ids}\nActual:   {actual_skill_ids}"
    )

    print(f"\n[Task 0 Test 5 PASS] Full 10-question sequence on set '{chosen_set}' preserved exact skill order.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
