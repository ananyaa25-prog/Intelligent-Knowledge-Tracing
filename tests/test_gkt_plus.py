"""v3 follow-up: GatedKT+ (per-question difficulty, no graph, regularised) and the
GatedKT+ / Elo blend. Tiny synthetic data in which question difficulty matters."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines.base import auc, sequences_to_split  # noqa: E402
from baselines.deep import make_deep  # noqa: E402

NUM_SKILLS = 5


def _synthetic(n=60, seed=0):
    """Each skill has easy (problem id even) and hard (odd) questions."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        L = int(rng.integers(15, 30))
        sk = rng.integers(1, NUM_SKILLS + 1, L)
        pr = sk * 100 + rng.integers(0, 10, L)
        p = np.where(pr % 2 == 0, 0.9, 0.2)
        seqs.append({"skill_seq": sk.astype(np.int64), "correct_seq": (rng.random(L) < p).astype(np.int64),
                     "difficulty_seq": np.zeros(L, dtype=np.int64), "problem_seq": pr.astype(np.int64),
                     "response_time_seq": np.full(L, 5000.0), "attempt_seq": np.ones(L), "hint_seq": np.zeros(L)})
    return seqs


@pytest.fixture(scope="module")
def data():
    s = _synthetic()
    return sequences_to_split(s[:40]), sequences_to_split(s[40:50]), sequences_to_split(s[50:])


def test_gkt_plus_range_determinism_and_item_difficulty(data):
    tr, va, te = data
    a = make_deep("gkt_plus", NUM_SKILLS, seed=3, epochs=8, patience=8, lr=1e-2).fit(tr, va)
    p, y, owner = a.predict_split(te)
    assert p.shape == y.shape == owner.shape and np.all((p >= 0) & (p <= 1))
    b = make_deep("gkt_plus", NUM_SKILLS, seed=3, epochs=8, patience=8, lr=1e-2).fit(tr, va)
    np.testing.assert_allclose(p, b.predict_split(te)[0], atol=1e-6)
    # question difficulty (the only signal in these data) is learned
    assert auc(y, p) > 0.75
    w = a.net.prob_b.weight.detach().numpy().ravel()
    assert w[0] == 0.0                                           # unseen-question slot stays neutral
    assert np.abs(w[1:]).max() > 0.05
    assert 0.0 <= a.predict([], 1, 0) <= 1.0


def test_unseen_questions_map_to_zero(data):
    tr, va, _ = data
    m = make_deep("gkt_plus", NUM_SKILLS, seed=0, epochs=1).fit(tr, va)
    idx = m.map_problems(torch.tensor([[101, 999999, 0]]))
    assert idx[0, 0] > 0 and idx[0, 1] == 0 and idx[0, 2] == 0


def test_blend_fitted_on_validation(data):
    tr, va, te = data
    m = make_deep("gkt_plus_elo", NUM_SKILLS, seed=1, epochs=3).fit(tr, va)
    assert set(m.blend_weights) == {"gkt_plus", "elo", "bias"}
    p, y, _ = m.predict_split(te)
    assert np.all((p >= 0) & (p <= 1)) and len(p) == len(y)
    with pytest.raises(ValueError):
        make_deep("gkt_plus_elo", NUM_SKILLS, seed=1, epochs=1).fit(tr)


def test_pretraining_copies_shared_weights_only(data, tmp_path):
    tr, va, _ = data
    src = make_deep("gkt_plus", NUM_SKILLS, seed=0, epochs=1).fit(tr, va)
    path = src.save(tmp_path / "pre.pt")
    m = make_deep("gkt_plus", NUM_SKILLS + 3, seed=1, epochs=1, init_from=str(path)).fit(tr, va)
    assert any(k.startswith("long_term") for k in m.pretrained_params)
    assert not any(k.startswith(("skill_emb", "prob_")) for k in m.pretrained_params)


def test_existing_models_unchanged_by_trainer_extension(data):
    """Default DeepKT behaviour (no weight decay, no early stopping) is unchanged."""
    tr, va, te = data
    m = make_deep("dkt_gru", NUM_SKILLS, seed=2, epochs=2)
    assert m.patience is None and m.weight_decay == 0.0 and not m.uses_problem
