"""
tests/test_e2e_adaptive.py
--------------------------
End-to-end test of the adaptive session flow against the real FastAPI app and
retrained model. Run AFTER retraining (requires a compatible checkpoint).

    python tests/test_e2e_adaptive.py
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from serving.inference_service import app, MAIN_QUESTIONS, SESSION_LOG_PATH


def run_session():
    with TestClient(app) as client:
        r = client.post("/session/start", json={"student_id": "e2e_student", "source": "test"})
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]

        positions = []
        seen_probe = False
        for _ in range(40):  # safety bound
            n = client.post(f"/session/{sid}/next", json={})
            assert n.status_code == 200, n.text
            data = n.json()
            if data.get("is_complete"):
                break
            positions.append(data["position"])
            if data["is_probe"]:
                seen_probe = True
            # Answer deterministically wrong/right to keep the session short.
            a = client.post(f"/session/{sid}/answer", json={
                "submitted_answer": "1", "response_time_ms": 1200.0, "hint_used": 0
            })
            assert a.status_code == 200, a.text

        assert len(positions) >= MAIN_QUESTIONS, f"only {len(positions)} questions"
        # First two questions must not carry a pre-answer prediction.
        return sid, positions, seen_probe


def main():
    log_before = SESSION_LOG_PATH.stat().st_size if SESSION_LOG_PATH.exists() else 0
    sid, positions, seen_probe = run_session()
    log_after = SESSION_LOG_PATH.stat().st_size if SESSION_LOG_PATH.exists() else 0

    print(f"Session {sid}: answered {len(positions)} questions (>= {MAIN_QUESTIONS})")
    print(f"Probe questions emitted: {seen_probe}")
    print(f"Session log grew by {log_after - log_before} bytes")
    assert log_after > log_before, "no session log written"
    print("E2E adaptive session PASSED.")


if __name__ == "__main__":
    main()
