"""Unit tests for the v3 classical baselines (src/baselines): output range, shape,
seed determinism, empty history, and a few model-specific sanity checks.
Uses a tiny synthetic dataset, so it runs in about a second."""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines import CLASSICAL, make_classical  # noqa: E402
from baselines.base import (all_metrics, auc, derive_difficulty_from_sequences, ece,  # noqa: E402
                            mean_ci, resplit, sequences_to_split, split_to_sequences)

NUM_SKILLS = 6


def _synthetic(n_students=60, seed=0, max_len=30):
    """Students with a latent ability; harder buckets are less likely correct."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_students):
        L = int(rng.integers(5, max_len))
        ability = rng.normal()
        sk = rng.integers(1, NUM_SKILLS + 1, L)
        pr = sk * 100 + rng.integers(0, 6, L)
        di = rng.integers(0, 4, L)
        p = 1 / (1 + np.exp(-(ability - 0.6 * (di - 2) + 0.05 * np.arange(L))))
        co = (rng.random(L) < p).astype(np.int64)
        seqs.append({"skill_seq": sk.astype(np.int64), "correct_seq": co, "difficulty_seq": di.astype(np.int64),
                     "problem_seq": pr.astype(np.int64), "response_time_seq": np.full(L, 5000.0),
                     "attempt_seq": np.ones(L), "hint_seq": np.zeros(L)})
    return seqs


@pytest.fixture(scope="module")
def data():
    seqs = _synthetic()
    return sequences_to_split(seqs[:45]), sequences_to_split(seqs[45:])


@pytest.mark.parametrize("name", list(CLASSICAL))
def test_range_shape_and_empty_history(name, data):
    train, test = data
    m = make_classical(name, NUM_SKILLS, seed=3).fit(train)
    p, y, owner = m.predict_split(test)
    expected = int(sum(max(int(r.sum()) - 1, 0) for r in test["mask"]))
    assert p.shape == y.shape == owner.shape == (expected,)
    assert np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p))
    # empty history and unknown skill still give a valid probability
    for hist in ([], [(1, 1, 2)], [{"skill": 2, "correct": 0, "difficulty": 3}]):
        v = m.predict(hist, 1, 2)
        assert 0.0 <= v <= 1.0
    assert 0.0 <= m.predict([], NUM_SKILLS + 50, 0) <= 1.0


@pytest.mark.parametrize("name", list(CLASSICAL))
def test_seed_determinism(name, data):
    train, test = data
    a = make_classical(name, NUM_SKILLS, seed=7).fit(train).predict_split(test)[0]
    b = make_classical(name, NUM_SKILLS, seed=7).fit(train).predict_split(test)[0]
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("name", list(CLASSICAL))
def test_predict_matches_predict_sequence(name, data):
    """The fast online path must equal calling predict() on each prefix."""
    train, test = data
    m = make_classical(name, NUM_SKILLS, seed=1).fit(train)
    s = split_to_sequences(test)[0]
    fast = m.predict_sequence(s["skill_seq"], s["correct_seq"], s["difficulty_seq"])
    slow = [m.predict(list(zip(s["skill_seq"][:t], s["correct_seq"][:t], s["difficulty_seq"][:t])),
                      int(s["skill_seq"][t]), int(s["difficulty_seq"][t])) for t in range(1, len(s["skill_seq"]))]
    np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_majority_is_constant_and_skill_mean_uses_difficulty(data):
    train, test = data
    m = make_classical("majority", NUM_SKILLS).fit(train)
    assert len(set(np.round(m.predict_split(test)[0], 12))) == 1
    sm = make_classical("skill_mean", NUM_SKILLS).fit(train)
    assert sm.predict([], 1, 1) != sm.predict([], 1, 3)


def test_bkt_em_loglik_non_decreasing(data):
    train, _ = data
    for name in ("bkt", "bkt_diff"):
        m = make_classical(name, NUM_SKILLS, seed=2).fit(train)
        tr = np.asarray(m.loglik_trace)
        assert len(tr) >= 2
        assert np.all(np.diff(tr) >= -1e-6 * np.abs(tr[:-1])), tr
        assert np.all(m.G <= 0.3 + 1e-12) and np.all(m.Sl <= 0.3 + 1e-12)


def test_bkt_seed_changes_init():
    a = make_classical("bkt", NUM_SKILLS, seed=1)
    b = make_classical("bkt", NUM_SKILLS, seed=2)
    a._init_params(NUM_SKILLS + 1)
    b._init_params(NUM_SKILLS + 1)
    assert not np.allclose(a.L0, b.L0)


def test_counts_models_learn_signal(data):
    """On the synthetic data (ability-driven), history models beat the constant."""
    train, test = data
    for name in ("logreg_counts", "elo", "bkt_diff"):
        p, y, _ = make_classical(name, NUM_SKILLS, seed=0).fit(train).predict_split(test)
        assert auc(y, p) > 0.55, name


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert ece(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    m = all_metrics(y, np.array([0.1, 0.6, 0.8, 0.9]))
    assert m["acc"] == 0.75 and m["n"] == 4
    ci = mean_ci([0.8, 0.82, 0.81])
    assert ci["ci_low"] < 0.81 < ci["ci_high"] and abs(ci["var"] - ci["std"] ** 2) < 1e-12


def test_resplit_derives_difficulty_from_train_only():
    seqs = _synthetic(80, seed=5)
    tr, va, te = resplit(seqs, seed=11)
    assert len(tr) == 64 and len(va) == 8 and len(te) == 8
    dm = derive_difficulty_from_sequences(tr)
    for s in te:
        for k, p, d in zip(s["skill_seq"], s["problem_seq"], s["difficulty_seq"]):
            assert d == dm.get((int(k), int(p)), 0)
    tr2, _, _ = resplit(seqs, seed=11)
    assert [len(s["skill_seq"]) for s in tr] == [len(s["skill_seq"]) for s in tr2]
