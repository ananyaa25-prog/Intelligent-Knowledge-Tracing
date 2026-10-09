"""Tests for optional temperature scaling (T8b)."""

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving import calibration as cal


def test_temperature_preserves_ranking_and_midpoint():
    ps = [0.01, 0.2, 0.45, 0.5, 0.7, 0.99]
    for T in (0.5, 1.17, 3.0):
        out = [cal.apply_temperature(p, T) for p in ps]
        assert out == sorted(out)
        assert abs(cal.apply_temperature(0.5, T) - 0.5) < 1e-9
    assert cal.apply_temperature(0.9, 2.0) < 0.9 < cal.apply_temperature(0.9, 0.5)


def test_disabled_by_default_is_identity(monkeypatch):
    monkeypatch.delenv("APPLY_CALIBRATION", raising=False)
    assert cal.maybe_calibrate(0.83) == 0.83
    assert cal.is_active() is False


def test_enabled_uses_file(monkeypatch, tmp_path):
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps({"temperature": 2.0}), encoding="utf-8")
    monkeypatch.setattr(cal, "CALIBRATION_PATH", f)
    monkeypatch.setattr(cal, "load_temperature", lambda path=f: 2.0)
    monkeypatch.setenv("APPLY_CALIBRATION", "true")
    assert abs(cal.maybe_calibrate(0.9) - cal.apply_temperature(0.9, 2.0)) < 1e-12
    assert cal.is_active() is True


def test_calibration_report_auc_unchanged():
    from serving.policy_config import CALIBRATION_PATH
    if CALIBRATION_PATH.exists():
        d = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        assert abs(d["auc_before"] - d["auc_after"]) < 1e-9
