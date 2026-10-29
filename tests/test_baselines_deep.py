"""Unit tests for the v3 deep baselines and the wrappers of our GatedKT variants:
output range and shape, seed determinism, empty history, predict() consistent with
the batched path, existing checkpoints loaded read-only. Tiny data, 1 epoch."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.base import sequences_to_split, split_to_sequences  # noqa: E402
from baselines.deep import EXTERNAL_DEEP, OUR_VARIANTS, make_deep  # noqa: E402
from baselines.ours import OursVariant  # noqa: E402

NUM_SKILLS = 123   # our variants need the real skill graph (124 x 124)
PROCESSED = ROOT / "data" / "processed"


def _synthetic(n=24, seed=0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        L = int(rng.integers(3, 18))
        sk = rng.integers(1, 12, L)
        seqs.append({"skill_seq": sk.astype(np.int64), "correct_seq": rng.integers(0, 2, L).astype(np.int64),
                     "difficulty_seq": rng.integers(0, 4, L).astype(np.int64), "problem_seq": sk * 10,
                     "response_time_seq": rng.uniform(2000, 30000, L), "attempt_seq": np.ones(L),
                     "hint_seq": np.zeros(L)})
    return seqs


@pytest.fixture(scope="module")
def data():
    s = _synthetic()
    return sequences_to_split(s[:18]), sequences_to_split(s[18:])


ALL = list(EXTERNAL_DEEP) + OUR_VARIANTS


@pytest.mark.parametrize("name", ALL)
def test_range_shape_determinism_empty(name, data):
    train, test = data
    a = make_deep(name, NUM_SKILLS, seed=5, epochs=1).fit(train, test)
    p, y, owner = a.predict_split(test)
    expected = int(sum(max(int(r.sum()) - 1, 0) for r in test["mask"]))
    assert p.shape == y.shape == owner.shape == (expected,)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    b = make_deep(name, NUM_SKILLS, seed=5, epochs=1).fit(train, test)
    np.testing.assert_allclose(p, b.predict_split(test)[0], atol=1e-6)
    for hist in ([], [(3, 1, 2)], [{"skill": 4, "correct": 0, "difficulty": 1}]):
        v = a.predict(hist, 3, 2)
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("name", ["dkt_gru", "dkvmn_lite", "sakt_lite", "full_gated"])
def test_predict_matches_batched_path(name, data):
    """predict() on a prefix equals the batched prediction for that position
    (metadata fixed to the defaults, which predict_sequence also uses)."""
    train, test = data
    m = make_deep(name, NUM_SKILLS, seed=2, epochs=1).fit(train)
    s = split_to_sequences(test)[0]
    seq = m.predict_sequence(s["skill_seq"], s["correct_seq"], s["difficulty_seq"])
    for t in (1, len(s["skill_seq"]) - 1):
        hist = list(zip(s["skill_seq"][:t], s["correct_seq"][:t], s["difficulty_seq"][:t]))
        v = m.predict(hist, int(s["skill_seq"][t]), int(s["difficulty_seq"][t]))
        assert abs(v - seq[t - 1]) < 1e-4


def test_lite_models_are_labelled():
    assert make_deep("dkvmn_lite").name == "dkvmn_lite"
    assert make_deep("sakt_lite").name == "sakt_lite"


def test_ablation_transforms():
    m = make_deep("no_difficulty", NUM_SKILLS)
    args = (torch.ones(1, 3).long(), torch.ones(1, 3).long(), torch.ones(1, 3), torch.full((1, 3), 9000.0),
            torch.full((1, 3), 3.0), torch.ones(1, 3), torch.full((1, 3), 2).long())
    assert int(m.transform(args)[6].sum()) == 0
    nm = make_deep("no_metadata", NUM_SKILLS, rt_stats=(8.0, 1.0)).transform(args)
    assert torch.allclose(torch.log1p(nm[3]), torch.full((1, 3), 8.0))
    assert float(nm[4].max()) == 1.0 and float(nm[5].max()) == 0.0


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


@pytest.mark.skipif(not (PROCESSED / "best_full_gated_seed42.pt").exists(), reason="checkpoint not present")
def test_existing_checkpoint_loaded_read_only(data):
    _, test = data
    ck = PROCESSED / "best_full_gated_seed42.pt"
    before = _md5(ck)
    m = OursVariant.from_checkpoint("full_gated", 42)
    p, _, _ = m.predict_split(test)
    assert np.all((p >= 0) & (p <= 1))
    assert _md5(ck) == before
    with pytest.raises(ValueError):
        m.save(PROCESSED / "should_not_exist.pt")
    assert not (PROCESSED / "should_not_exist.pt").exists()
