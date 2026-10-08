"""Tests for src/serving/policy_config.py (T1)."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving import policy_config as pc

ENV_KEYS = ["POLICY", "TARGET_P", "APPLY_CALIBRATION", "SESSION_MODE", "CORE_TOPICS", "T_PROF",
            "T_READY", "MIN_ANSWERS", "MONOTONE_WARN", "EXCLUDE_STUDENT_REGEX"]


def _clear(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_defaults(monkeypatch):
    _clear(monkeypatch)
    cfg = pc.get_config()
    assert cfg.policy == "staircase"
    assert cfg.target_p == 0.70
    assert cfg.apply_calibration is False
    assert cfg.session_mode == "full10"
    assert cfg.core_topics == [50, 48, 204, 203, 350]
    assert (cfg.t_prof, cfg.t_ready, cfg.min_answers, cfg.monotone_warn) == (0.70, 0.60, 2, 0.10)
    assert cfg.thresholds() == {"T_PROF": 0.70, "T_READY": 0.60, "MIN_ANSWERS": 2}


def test_env_overrides_and_invalid_values(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("POLICY", "model")
    monkeypatch.setenv("APPLY_CALIBRATION", "true")
    monkeypatch.setenv("SESSION_MODE", "core5")
    monkeypatch.setenv("CORE_TOPICS", "50, 48")
    monkeypatch.setenv("T_PROF", "0.8")
    cfg = pc.get_config()
    assert (cfg.policy, cfg.apply_calibration, cfg.session_mode) == ("model", True, "core5")
    assert cfg.core_topics == [50, 48]
    assert cfg.t_prof == 0.8
    monkeypatch.setenv("POLICY", "bogus")
    monkeypatch.setenv("SESSION_MODE", "bogus")
    monkeypatch.setenv("T_PROF", "not-a-number")
    cfg = pc.get_config()
    assert (cfg.policy, cfg.session_mode, cfg.t_prof) == ("staircase", "full10", 0.70)


def test_exclusion_regex(monkeypatch):
    _clear(monkeypatch)
    cfg = pc.get_config()
    for sid in ["e2e_student", "smoke_1", "pred_check", "test_abc", "sim_strong_01"]:
        assert cfg.is_excluded_student(sid), sid
    for sid in ["alex", "student_42"]:
        assert not cfg.is_excluded_student(sid), sid


def test_recommended_actions_cover_all_statuses():
    for st in ["needs_support", "developing", "proficient", "mastered", "not_enough_evidence"]:
        assert pc.RECOMMENDED_ACTIONS[st]


def test_load_env_file(tmp_path):
    f = tmp_path / "demo.env"
    f.write_text("# comment\nPOLICY=model\n\nTARGET_P = 0.7\n", encoding="utf-8")
    assert pc.load_env_file(f) == {"POLICY": "model", "TARGET_P": "0.7"}
    assert pc.load_env_file(tmp_path / "missing.env") == {}


def test_profile_reliability_file_shape():
    import json
    p = pc.PROFILE_RELIABILITY_PATH
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        assert "reliable" in d and "violation_rate" in d
