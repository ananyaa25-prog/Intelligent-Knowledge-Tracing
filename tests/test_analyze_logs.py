"""Tests for scripts/analyze_logs.py (T8d)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_logs import analyze, load_events
from serving.policy_config import PolicyConfig


def _write(tmp_path, n_students=4, per=10):
    lines = []
    for s in range(n_students):
        sid = f"sess{s}"
        stu = f"learner{s}"
        for k in range(per):
            p = 0.2 + 0.06 * k
            lines.append({"event": "question", "session_id": sid, "student_id": stu, "position": k + 1,
                          "topic": 50, "level": "medium", "p_before": p, "is_probe": False, "policy": "model"})
            lines.append({"event": "answer", "session_id": sid, "student_id": stu, "position": k + 1, "topic": 50,
                          "level": "medium", "p_before": p, "outcome": int(k >= 5), "is_probe": 0})
        lines.append({"event": "feedback", "session_id": sid, "student_id": stu, "perceived": "about_right", "helpful": 4})
    lines.append({"event": "answer", "session_id": "x", "student_id": "e2e_student", "position": 1, "topic": 50,
                  "level": "easy", "p_before": 0.9, "outcome": 0, "is_probe": 0})
    lines.append({"event": "answer", "session_id": "sess0", "student_id": "learner0", "position": 99, "topic": 27,
                  "level": "medium", "p_before": 0.4, "outcome": 1, "is_probe": 1})
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return path


def test_analyze_excludes_test_accounts_and_computes_metrics(tmp_path):
    r = analyze(load_events(_write(tmp_path)), PolicyConfig())
    assert r["excluded_answers"] == 1
    assert r["usable"] == 40 and r["students"] == 4
    assert 0.0 <= r["brier"] <= 1.0
    assert len(r["probes"]) == 1
    assert r["feedback"]["model"]["about_right"] == 4


def test_script_flags_small_samples(tmp_path):
    log = _write(tmp_path, n_students=1, per=5)
    out = tmp_path / "pilot.md"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_logs.py"), "--log", str(log), "--out", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0
    assert "Not meaningful yet" in out.read_text(encoding="utf-8")
