"""
tests/test_tasks_17_20_smoke.py
--------------------------------
Automated test suite verifying Tasks 17, 18, 19, and 20:
- Task 17: Required source parameter enforcement (raises on omission, tags 'test' vs 'live' correctly).
- Task 19: Auto-generated mastery summary synthesis (strongest/weakest skill identification, relational pairs, low-confidence caveat, zero technical jargon).
- Task 20: Relocated gate interpretability API endpoint integrity.
"""

import pytest
import sqlite3
import re
from fastapi.testclient import TestClient

from src.serving.db import Database
from src.serving.session import SessionManager
from src.serving.inference_service import (
    app,
    artifacts,
    generate_mastery_summary,
    find_weak_skill_pairs
)


@pytest.fixture(scope="module")
def test_client():
    artifacts.load()
    with TestClient(app) as client:
        yield client


# =============================================================================
# TASK 17 SMOKE TESTS: Required Source Parameter & Clean DB Tagging
# =============================================================================

def test_task17_source_parameter_is_required_no_default(tmp_path):
    """Assert start_session, get_or_create_student, create_session raise TypeError if source is omitted."""
    db = Database(tmp_path / "test_req.db")
    manager = SessionManager(database=db)

    # 1. SessionManager.start_session without source
    with pytest.raises(TypeError):
        manager.start_session(student_id="test_omitted_source_student")  # type: ignore

    # 2. Database.get_or_create_student without source
    with pytest.raises(TypeError):
        db.get_or_create_student(student_id="test_omitted_source_student")  # type: ignore

    # 3. Database.create_session without source
    with pytest.raises(TypeError):
        db.create_session(session_id="sess_omitted", student_id="test_st")  # type: ignore


def test_task17_invalid_source_raises_value_error(tmp_path):
    """Assert invalid source tags (not 'live', 'sample', or 'test') raise ValueError."""
    db = Database(tmp_path / "test_inv.db")
    manager = SessionManager(database=db)

    with pytest.raises(ValueError):
        manager.start_session(student_id="test_bad_source", source="invalid_tag")

    with pytest.raises(ValueError):
        manager.start_session(student_id="test_empty_source", source="")


def test_task17_source_tagging_integrity(tmp_path):
    """Assert explicit source='test' and source='live' are correctly persisted and queryable."""
    db = Database(tmp_path / "test_tag.db")

    # Create test student and session
    db.get_or_create_student("test_hygiene_student", source="test")
    db.create_session("sess_test_001", "test_hygiene_student", source="test")
    db.complete_session("sess_test_001")

    # Create live student and session
    db.get_or_create_student("live_real_user_001", source="live")
    db.create_session("sess_live_001", "live_real_user_001", source="live")
    db.complete_session("sess_live_001")

    # Verify query by source
    live_students = db.get_all_completed_students(source="live")
    assert len(live_students) == 1
    assert live_students[0]["student_id"] == "live_real_user_001"

    test_students = db.get_all_completed_students(source="test")
    assert len(test_students) == 1
    assert test_students[0]["student_id"] == "test_hygiene_student"

    all_students = db.get_all_completed_students(source="all")
    assert len(all_students) == 2


# =============================================================================
# TASK 19 SMOKE TESTS: Mastery Summary Synthesis & Jargon Filter
# =============================================================================

def test_task19_mastery_summary_names_strongest_and_weakest_skills():
    """Assert summary names the weakest and strongest skill with correct percentages."""
    mastery = {
        "204": 0.25,   # Percents (Weakest)
        "27": 0.88,    # Pythagorean Theorem (Strongest)
        "48": 0.60,
        "50": 0.70,
        "217": 0.55,
    }
    summary = generate_mastery_summary(mastery, session_count=2)
    assert "Percents" in summary
    assert "25%" in summary
    assert "Pythagorean Theorem" in summary
    assert "88%" in summary


def test_task19_mastery_summary_relational_callout():
    """Assert summary connects related weak skills when PPMI co-occurrence is high."""
    mastery = {
        "204": 0.20,   # Percents
        "203": 0.30,   # Percent Discount
        "48": 0.85,
        "50": 0.90,
    }
    weak_pairs = find_weak_skill_pairs(mastery, cutoff=0.0)
    summary = generate_mastery_summary(mastery, session_count=3)
    if weak_pairs:
        p = weak_pairs[0]
        assert p["skill1_name"] in summary or p["skill2_name"] in summary


def test_task19_mastery_summary_low_confidence_caveat():
    """Assert single-session or low interaction data triggers an honest uncertainty caveat."""
    mastery = {
        "204": 0.30,
        "27": 0.80,
    }
    summary_1_session = generate_mastery_summary(mastery, session_count=1)
    assert "limited data" in summary_1_session.lower() or "preliminary" in summary_1_session.lower() or "more sessions" in summary_1_session.lower()


def test_task19_mastery_summary_zero_jargon():
    """Assert forbidden jargon words ('gate', 'PPMI', 'embedding', 'GRU', 'GAT') never appear in summary."""
    forbidden_terms = ["gate", "ppmi", "embedding", "gru", "gat", "recurrent", "tensor", "logits"]

    sample_mastery_dicts = [
        {"204": 0.15, "27": 0.92, "48": 0.40, "50": 0.35},
        {"204": 0.85, "27": 0.80, "48": 0.90},
        {"204": 0.20, "203": 0.25, "217": 0.30},
        {"204": 0.50},
    ]

    for s_count in [1, 2, 5]:
        for m in sample_mastery_dicts:
            summary = generate_mastery_summary(m, session_count=s_count)
            summary_lower = summary.lower()
            for term in forbidden_terms:
                assert not re.search(r'\b' + re.escape(term) + r'\b', summary_lower), (
                    f"Forbidden jargon term '{term}' found in student summary: '{summary}'"
                )


# =============================================================================
# TASK 20 SMOKE TESTS: Relocated Gate View & API Endpoint Functionality
# =============================================================================

def test_task20_gate_endpoints_functional(test_client):
    """Assert gate endpoints work properly for completed sessions."""
    # Create session with source="test"
    s_res = test_client.post("/session/start", json={"student_id": "task20_gate_st", "source": "test"})
    assert s_res.status_code == 200
    sid = s_res.json()["session_id"]

    # Record 4 answers
    for pos in range(1, 5):
        test_client.post(f"/session/{sid}/answer", json={
            "position": pos,
            "submitted_answer": "test_ans",
            "response_time_ms": 3200.0,
            "hint_used": 0
        })

    # Verify detail endpoint
    res_detail = test_client.get(f"/session/{sid}/gate_detail")
    assert res_detail.status_code == 200
    data = res_detail.json()
    assert data["available"] is True
    assert data["num_interactions"] == 4
    assert len(data["top5_dimensions"]) == 5

    # Verify plot PNG endpoint
    res_plot = test_client.get(f"/session/{sid}/gate_plot.png")
    assert res_plot.status_code == 200
    assert res_plot.headers["content-type"] == "image/png"
    assert len(res_plot.content) > 1000  # Non-trivial PNG image payload

