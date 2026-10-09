"""
src/serving/calibration.py
--------------------------
Optional temperature scaling of the predictor's probabilities (T8b).

p_cal = sigmoid(logit(p) / T), with T fitted by scripts/fit_calibration.py and
stored in data/processed/calibration.json. Applied only when
APPLY_CALIBRATION=true AND the file exists. A monotone transform: it never
changes the ranking of predictions (AUC unchanged).
"""

import json
import math
from typing import Optional

from serving.policy_config import CALIBRATION_PATH, get_config

_cache = {"mtime": None, "T": None}


def load_temperature(path=CALIBRATION_PATH) -> Optional[float]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _cache["mtime"] != mtime:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _cache["T"] = float(json.load(f)["temperature"])
        except Exception:
            _cache["T"] = None
        _cache["mtime"] = mtime
    return _cache["T"]


def apply_temperature(p: float, T: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p)) / T
    return 1.0 / (1.0 + math.exp(-z))


def is_active() -> bool:
    return bool(get_config().apply_calibration and load_temperature() is not None)


def maybe_calibrate(p: float) -> float:
    """Calibrated p if enabled and fitted; otherwise p unchanged."""
    if not get_config().apply_calibration:
        return p
    T = load_temperature()
    return apply_temperature(p, T) if T else p
