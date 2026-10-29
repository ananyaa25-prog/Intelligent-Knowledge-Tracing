"""v3 follow-up 2: hints and retries in the session (HINTS_ENABLED, MAX_ATTEMPTS) and the
hint / attempt pathways of GatedKT+ (meta_mode) inside the GRU input and the gate."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving.adaptive_session import AdaptiveSession, AdaptiveSessionManager  # noqa: E402
from serving.policy_config import get_config  # noqa: E402
from questions.hints import HINTS, hint_for  # noqa: E402
from questions.generator import all_topic_keys  # noqa: E402


def _session(**kw):
    random.seed(0)
    s = AdaptiveSession("s1", "stu", **kw)
    q = s.next_question(None)
    return s, q


def test_defaults_unchanged(monkeypatch):
    for k in ("HINTS_ENABLED", "MAX_ATTEMPTS", "SERVING_MODEL"):
        monkeypatch.delenv(k, raising=False)
    cfg = get_config()
    assert (cfg.hints_enabled, cfg.max_attempts, cfg.serving_model) == (False, 1, "gated_kt")
    s, q = _session()
    inter = s.record_answer("definitely wrong", 1234.0, hint_used=1)
    assert not inter.get("retry")
    assert inter["attempt_count"] == 1 and inter["hint_used"] == 1 and inter["correct"] == 0
    assert inter["response_time_ms"] == 1234.0 and s.pending is None
    s.next_question(None)
    with pytest.raises(ValueError):
        s.request_hint()


def test_env_flags(monkeypatch):
    monkeypatch.setenv("HINTS_ENABLED", "true")
    monkeypatch.setenv("MAX_ATTEMPTS", "7")
    monkeypatch.setenv("SERVING_MODEL", "gkt_plus_meta")
    cfg = get_config()
    assert cfg.hints_enabled and cfg.max_attempts == 3 and cfg.serving_model == "gkt_plus_meta"
    monkeypatch.setenv("SERVING_MODEL", "nonsense")
    assert get_config().serving_model == "gated_kt"


def test_retry_then_correct_keeps_first_attempt_label():
    s, q = _session(max_attempts=3)
    level_before = s.levels[q["topic"]]
    r = s.record_answer("definitely wrong", 4000.0)
    assert r["retry"] and r["attempts_left"] == 2 and s.pending is not None and not s.history
    inter = s.record_answer(q["correct_answer"], 2500.0)
    assert inter["correct"] == 0 and inter["solved"] == 1 and inter["attempt_count"] == 2
    assert inter["response_time_ms"] == 4000.0            # first-response time, as in ASSISTments
    assert s.pending is None and len(s.history) == 1
    assert s.levels[q["topic"]] in ("easy", level_before)  # staircase follows the first attempt


def test_attempts_run_out():
    s, q = _session(max_attempts=2)
    assert s.record_answer("nope", 1000.0)["retry"]
    inter = s.record_answer("still nope", 1000.0)
    assert inter["correct"] == 0 and inter["solved"] == 0 and inter["attempt_count"] == 2


def test_correct_first_try():
    s, q = _session(max_attempts=3, hints_enabled=True)
    inter = s.record_answer(q["correct_answer"], 1500.0)
    assert inter["correct"] == 1 and inter["attempt_count"] == 1 and inter["hint_used"] == 0


def test_hint_before_first_attempt_counts_as_not_first_try_correct():
    s, q = _session(hints_enabled=True)
    h = s.request_hint()
    assert h["hint_text"] and h["counts_as_first_try_wrong"] and q["correct_answer"] not in h["hint_text"]
    inter = s.record_answer(q["correct_answer"], 1500.0)
    assert inter["correct"] == 0 and inter["solved"] == 1 and inter["hint_used"] == 1


def test_hint_after_wrong_attempt():
    s, q = _session(hints_enabled=True, max_attempts=2)
    s.record_answer("nope", 1000.0)
    assert not s.request_hint()["counts_as_first_try_wrong"]
    inter = s.record_answer(q["correct_answer"], 900.0)
    assert inter["correct"] == 0 and inter["hint_used"] == 1 and inter["attempt_count"] == 2


def test_manager_does_not_store_retries():
    stored, logged = [], []
    m = AdaptiveSessionManager(store=lambda sid, st, i: stored.append(i), log=logged.append)
    random.seed(1)
    sid, _ = m.start(student_id="stu", max_attempts=2, hints_enabled=True)
    q = m.next_question(sid)
    m.request_hint(sid)
    assert m.record_answer(sid, "nope", 1000.0)["retry"] and not stored
    m.record_answer(sid, q["correct_answer"], 800.0)
    assert len(stored) == 1 and stored[0]["attempt_count"] == 2 and stored[0]["hint_used"] == 1
    assert {"hint", "retry", "answer"} <= {e.get("event") for e in logged}


def test_every_generator_topic_has_a_hint():
    assert set(all_topic_keys()) <= set(HINTS)
    assert hint_for({"raw_skill_id": 204}) == HINTS["d_percents"]


# ---------------------------------------------------------------- model pathways
torch = pytest.importorskip("torch")
from baselines.gkt_plus import (_GKTPlusNet, ServingGKTPlus, help_rates, outcome_class)  # noqa: E402


def test_outcome_classes():
    c = torch.tensor([1, 0, 0, 0, 0, 0, 0, 1])
    a = torch.tensor([1, 1, 2, 5, 1, 2, 3, 4], dtype=torch.float)
    h = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float)
    assert outcome_class(c, a, h).tolist() == [0, 1, 2, 3, 4, 5, 6, 0]


def test_help_rates_are_causal_and_windowed():
    att = torch.tensor([[1, 2, 1, 1, 3.]])
    hint = torch.tensor([[0, 0, 1, 0, 0.]])
    r = help_rates(att, hint, torch.ones(1, 5), window=2)
    np.testing.assert_allclose(r[0, :, 0].numpy(), [0, 0, 0.5, 0.5, 0])
    np.testing.assert_allclose(r[0, :, 1].numpy(), [0, 0.5, 0.5, 0, 0.5])


def _hist(hint_last, att_last):
    torch.manual_seed(0)
    L = 6
    skill = torch.tensor([[1, 2, 1, 3, 2, 1]])
    correct = torch.tensor([[1, 0, 1, 1, 0, 0]])
    att = torch.ones(1, L)
    hint = torch.zeros(1, L)
    att[0, 4], hint[0, 4] = att_last, hint_last
    return skill, correct, torch.ones(1, L), torch.full((1, L), 5000.0), att, hint, torch.ones(1, L, dtype=torch.long)


@pytest.mark.parametrize("mode,changes_gru,changes_gate", [("full", True, True), ("gru", True, False),
                                                           ("gate", False, True), ("none", False, False)])
def test_hints_reach_gru_and_gate_only_where_intended(mode, changes_gru, changes_gate):
    torch.manual_seed(0)
    net = _GKTPlusNet(3, 0, 8.5, 1.2, meta_mode=mode).eval()
    outs = []
    for h, a in ((0, 1), (1, 3)):
        sk, co, mk, rt, at, hi, di = _hist(h, a)
        with torch.no_grad():
            e = net.question(sk, di, torch.zeros_like(sk))
            hs_x = (net.outcome_emb(outcome_class(co, at, hi)) + net.meta_to_x(net.metadata_encoder(rt, at, hi))
                    if mode in ("full", "gru") else net.correct_emb(co))
            hs = net.short_term(e + hs_x, mk)
            _, g = net.states(sk, co, mk, rt, at, hi, di, torch.zeros_like(sk))
        outs.append((hs[0, -1], g[0, -1], g[0, 4]))
    gru_changed = not torch.allclose(outs[0][0], outs[1][0])
    assert gru_changed == changes_gru
    if mode in ("gate", "full"):
        assert not torch.allclose(outs[0][2], outs[1][2])          # gate at the hinted step
    if mode == "none":
        assert torch.allclose(outs[0][1], outs[1][1])


def test_serving_wrapper_interface():
    torch.manual_seed(0)
    srv = ServingGKTPlus(_GKTPlusNet(3, 5, 8.5, 1.2, meta_mode="full"))
    sk, co, mk, rt, at, hi, di = _hist(1, 2)
    pred, g = srv(sk, co, mk, rt, at, hi)                     # difficulty optional, as in GatedKTModel
    assert pred.shape == (1, 5) and g.shape[:2] == (1, 5)
    p = srv.predict_candidates(sk, co, mk, rt, at, hi, di, [(1, 1), (2, 3), (3, 2)])
    assert p.shape == (3,) and bool(((p > 0) & (p < 1)).all())
    with torch.no_grad():
        full, _ = srv(torch.cat([sk, torch.tensor([[2]])], 1), torch.cat([co, torch.tensor([[0]])], 1),
                      torch.ones(1, 7), torch.cat([rt, rt[:, :1]], 1), torch.cat([at, at[:, :1]], 1),
                      torch.cat([hi, hi[:, :1]], 1), torch.cat([di, torch.tensor([[3]])], 1))
    assert abs(float(full[0, -1]) - float(p[1])) < 1e-5     # candidate score = next-step prediction
