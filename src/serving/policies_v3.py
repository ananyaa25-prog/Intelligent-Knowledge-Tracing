"""
src/serving/policies_v3.py
--------------------------
Score sources for the v3 difficulty policies (POLICY=elo | maxinfo | uncertainty).
The policies themselves live in adaptive_session.AdaptiveSession._decide_level and
share the same allowed-level guardrails as POLICY=model (D27); they differ only in
where the level scores come from and which P they aim for:

  model        (ZPD)   model P(correct), level nearest TARGET_P (0.70)       [existing]
  elo          (IRT)   Elo/Rasch P from the student's own answers, nearest TARGET_P
  maxinfo      (CAT)   model P, level nearest 0.50 (max Fisher information p(1-p))
  uncertainty          mean P of an ensemble of the 3 existing full_gated checkpoints,
                       nearest TARGET_P; if the ensemble spread (max - min) at the chosen
                       level exceeds UNCERTAINTY_MAX_SPREAD the staircase level is used and
                       the decision is flagged "low-confidence prediction".

Elo item difficulties per level are NOT the simulator's generating values: they are the
mean Elo item difficulty per difficulty bucket learned from the ASSISTments train split
by the Elo baseline (src/baselines/elo.py), centred on medium = 0, cached in
data/processed_v3/elo_level_difficulty.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
ELO_LEVELS_PATH = ROOT / "data" / "processed_v3" / "elo_level_difficulty.json"
MAXINFO_TARGET = 0.50
UNCERTAINTY_MAX_SPREAD = 0.15   # pre-registered before the Stage 9 simulation (DEMO_CONFIG.md)
ELO_A, ELO_B = 1.0, 0.05        # step size A / (1 + B n), same as the Elo baseline
LEVELS = ("easy", "medium", "hard")

_CACHE: Dict[str, Any] = {}


def elo_level_difficulty() -> Dict[str, float]:
    if "elo" in _CACHE:
        return _CACHE["elo"]
    vals = None
    if ELO_LEVELS_PATH.exists():
        try:
            vals = json.loads(ELO_LEVELS_PATH.read_text(encoding="utf-8"))["b"]
        except Exception:
            vals = None
    if vals is None:
        vals = fit_elo_level_difficulty()
    _CACHE["elo"] = vals
    return vals


def fit_elo_level_difficulty(seed: int = 0) -> Dict[str, float]:
    """Mean Elo item difficulty per bucket (1/2/3) on the ASSISTments train split,
    weighted by item usage, centred so medium = 0. Writes the JSON cache."""
    import pickle
    import sys
    import numpy as np
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from baselines.elo import EloBaseline
    with open(ROOT / "data" / "processed" / "windowed_data.pkl", "rb") as f:
        d = pickle.load(f)
    m = EloBaseline(num_skills=int(d["num_skills"]), seed=seed).fit(d["splits"]["train"])
    b = {}
    for bucket, lvl in ((1, "easy"), (2, "medium"), (3, "hard")):
        w = m.nb[:, bucket]
        b[lvl] = float((m.b[:, bucket] * w).sum() / max(w.sum(), 1.0))
    mid = b["medium"]
    b = {k: round(v - mid, 4) for k, v in b.items()}
    ELO_LEVELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ELO_LEVELS_PATH.write_text(json.dumps({"b": b, "seed": seed, "source": "EloBaseline on windowed_data train split"},
                                          indent=2), encoding="utf-8")
    return b


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _level(it: Dict[str, Any]) -> str:
    lvl = it.get("level")
    if lvl in LEVELS:
        return lvl
    return {1: "easy", 2: "medium", 3: "hard"}.get(int(it.get("difficulty_bucket") or 2), "medium")


def elo_scores(history: List[Dict[str, Any]], topic: int, topic_of) -> Dict[str, float]:
    """P(correct) per level for `topic` from an Elo/Rasch estimate built only from the
    student's own answers (global ability + topic ability)."""
    b = elo_level_difficulty()
    theta, n_theta = 0.0, 0
    th_t: Dict[int, float] = {}
    n_t: Dict[int, int] = {}
    for it in history:
        t = topic_of(it)
        if t is None:
            continue
        p = _sig(theta + th_t.get(t, 0.0) - b[_level(it)])
        err = float(int(it["correct"])) - p
        theta += ELO_A / (1 + ELO_B * n_theta) * err
        th_t[t] = th_t.get(t, 0.0) + ELO_A / (1 + ELO_B * n_t.get(t, 0)) * err
        n_theta += 1
        n_t[t] = n_t.get(t, 0) + 1
    return {lvl: _sig(theta + th_t.get(topic, 0.0) - b[lvl]) for lvl in LEVELS}


def ensemble_scores(ensemble, history, cands) -> Optional[List[List[float]]]:
    """ensemble(history, cands) -> one list of P per member, or None if unavailable."""
    if ensemble is None:
        return None
    try:
        out = ensemble(history, cands)
        return [list(map(float, m)) for m in out] if out else None
    except Exception as e:  # never break a session on the optional ensemble
        print(f"[policies_v3] ensemble unavailable ({e})")
        return None
