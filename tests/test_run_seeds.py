"""scripts/run_seeds.py: one seed gives the same result twice (determinism); a different
seed changes the initialisation."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("torch")
from baselines.base import sequences_to_split  # noqa: E402
from run_seeds import train_one  # noqa: E402

S = 5


def _split(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = int(rng.integers(8, 16))
        sk = rng.integers(1, S + 1, L)
        out.append({"skill_seq": sk.astype(np.int64), "correct_seq": (rng.random(L) < 0.2 + 0.15 * sk).astype(np.int64),
                    "difficulty_seq": rng.integers(1, 4, L).astype(np.int64), "response_time_seq": np.full(L, 4000.0),
                    "attempt_seq": np.ones(L), "hint_seq": np.zeros(L)})
    return sequences_to_split(out)


def test_one_seed_is_deterministic():
    tr, va, te = _split(30, 0), _split(8, 1), _split(8, 2)
    a = train_one("dkt_only", 7, tr, va, te, S, epochs=2)
    b = train_one("dkt_only", 7, tr, va, te, S, epochs=2)
    assert a == b
    assert 0.0 <= a["test_auc"] <= 1.0 and 0.0 <= a["brier"] <= 1.0
    c = train_one("dkt_only", 8, tr, va, te, S, epochs=2)
    assert c["test_auc"] != a["test_auc"] or c["brier"] != a["brier"]
