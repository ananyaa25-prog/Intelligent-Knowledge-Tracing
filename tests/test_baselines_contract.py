"""v3 baselines benchmark (scripts/benchmark_baselines_v3.py): the shared interface
fit(train) / predict(history, next_skill, next_difficulty) -> P in [0, 1] for each model
in the table -- output range, shape, seed determinism and empty history."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines import make_classical  # noqa: E402
from baselines.base import sequences_to_split  # noqa: E402

S = 6


def _seqs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = int(rng.integers(8, 20))
        sk = rng.integers(1, S + 1, L)
        out.append({"skill_seq": sk.astype(np.int64), "correct_seq": (rng.random(L) < 0.3 + 0.1 * (sk % 4)).astype(np.int64),
                    "difficulty_seq": rng.integers(1, 4, L).astype(np.int64), "response_time_seq": np.full(L, 5000.0),
                    "attempt_seq": np.ones(L), "hint_seq": np.zeros(L)})
    return out


@pytest.fixture(scope="module")
def data():
    s = _seqs()
    return sequences_to_split(s[:28]), sequences_to_split(s[28:34]), sequences_to_split(s[34:])


def _fit(name, seed, tr, va):
    if name in ("pfa", "bkt"):
        return make_classical(name, S, seed=seed).fit(tr)
    pytest.importorskip("torch")
    from baselines.deep import make_deep
    return make_deep(name, S, seed=seed, epochs=2).fit(tr, va)


@pytest.mark.parametrize("name", ["pfa", "bkt", "dkt_gru", "sakt_lite"])
def test_baseline_contract(name, data):
    tr, va, te = data
    m = _fit(name, 0, tr, va)
    p, y, owner = m.predict_split(te)
    assert p.shape == y.shape == owner.shape and len(p) == int(np.asarray(te["mask"])[:, 1:].sum())
    assert np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p))
    p2, _, _ = _fit(name, 0, tr, va).predict_split(te)
    np.testing.assert_allclose(p, p2, atol=1e-6)                       # same seed -> same predictions
    for hist in ([], [(1, 1, 2)], [{"skill": 2, "correct": 0, "difficulty": 3}, (3, 1, 1)]):
        v = m.predict(hist, 2, 2)
        assert 0.0 <= v <= 1.0                                         # empty history included


@pytest.mark.parametrize("cond", ["full_gated", "graph_no_gate"])
def test_existing_checkpoint_wrapper(cond):
    torch = pytest.importorskip("torch")
    path = ROOT / "data" / "processed" / f"best_{cond}_seed42.pt"
    if not path.exists():
        pytest.skip("checkpoint not present")
    from baselines.ours import OursVariant
    before = path.stat().st_mtime
    a = OursVariant.from_checkpoint(cond, 42, num_skills=123)
    b = OursVariant.from_checkpoint(cond, 42, num_skills=123)
    for hist in ([], [(5, 1, 2), (7, 0, 3)]):
        pa, pb = a.predict(hist, 5, 2), b.predict(hist, 5, 2)
        assert 0.0 <= pa <= 1.0 and abs(pa - pb) < 1e-6
    assert path.stat().st_mtime == before                              # read-only, never rewritten
