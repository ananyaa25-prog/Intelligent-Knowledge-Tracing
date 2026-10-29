"""v3 Stage 9: POLICY=elo | maxinfo | uncertainty in the session engine (fake predictor
and fake ensemble, so the tests are fast and deterministic)."""

import random
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving.adaptive_session import AdaptiveSession, LEVEL_NAMES  # noqa: E402
from serving import policies_v3 as pv  # noqa: E402

LEVEL_P = {1: 0.85, 2: 0.6, 3: 0.35}


def fake_predictor(history, cands):
    return [LEVEL_P[b] for _, b in cands]


def _with_ensemble(spread):
    def ens(history, cands):
        return [[LEVEL_P[b] - spread / 2 for _, b in cands], [LEVEL_P[b] for _, b in cands],
                [LEVEL_P[b] + spread / 2 for _, b in cands]]
    p = lambda h, c: fake_predictor(h, c)
    p.ensemble = ens
    return p


def _play(policy, predictor, n=12, seed=0):
    random.seed(seed)
    s = AdaptiveSession("t", "stud", policy=policy, session_mode="full10")
    qs = []
    for i in range(n):
        q = s.next_question(predictor)
        if q is None:
            break
        qs.append(q)
        s.record_answer(q["correct_answer"] if i % 3 else "__wrong__", 4000)
    return s, qs


def test_elo_level_difficulty_from_data_is_ordered():
    b = pv.elo_level_difficulty()
    assert set(b) == {"easy", "medium", "hard"} and b["medium"] == 0.0
    assert b["easy"] < b["medium"] < b["hard"]


def test_elo_scores_follow_answers():
    topic_of = lambda it: it["topic"]
    up = pv.elo_scores([{"topic": 1, "correct": 1, "level": "medium"}] * 5, 1, topic_of)
    down = pv.elo_scores([{"topic": 1, "correct": 0, "level": "medium"}] * 5, 1, topic_of)
    assert up["medium"] > 0.5 > down["medium"]
    assert up["easy"] > up["medium"] > up["hard"]


@pytest.mark.parametrize("policy", ["elo", "maxinfo", "uncertainty"])
def test_new_policies_run_and_explain(policy):
    s, qs = _play(policy, _with_ensemble(0.02))
    main = [q for q in qs if q["decision"]["reason"] and not q["decision"]["reason"].startswith("Warm-up")]
    assert main and all(q["level"] in LEVEL_NAMES for q in qs)
    assert all("policy_scores" in q["decision"] for q in main)
    assert s.policy_fallbacks == 0


def test_maxinfo_prefers_p_near_half():
    s = AdaptiveSession("t", "stud", policy="maxinfo")
    hist = [{"topic": 204, "skill_id": 204, "correct": 1, "level": "medium"}]
    s.history = list(hist) + [{"topic": 203, "skill_id": 203, "correct": 1, "level": "medium"}]
    # after one correct Medium answer the allowed levels are {medium, hard}; P 0.60 vs 0.35:
    # target 0.5 -> |0.10| vs |0.15| -> medium; ZPD (0.70) -> medium as well, so check the scores used
    lvl, cand, guard, fb = s._decide_level(204, fake_predictor)
    assert not fb and lvl == "medium" and s.last_policy_scores == {"easy": 0.85, "medium": 0.6, "hard": 0.35}


def test_uncertainty_low_confidence_falls_back_to_staircase():
    s, qs = _play("uncertainty", _with_ensemble(0.40))
    flagged = [q for q in qs if q["decision"].get("low_confidence")]
    assert flagged and s.low_confidence_count == len(flagged)
    assert all(q["decision"]["guardrail"] == "low_confidence_prediction" for q in flagged)
    assert all("Low-confidence prediction" in q["decision"]["reason"] for q in flagged)


def test_uncertainty_without_ensemble_falls_back():
    s, qs = _play("uncertainty", fake_predictor)
    assert s.policy_fallbacks > 0
    assert all(q["decision"]["fallback"] for q in qs if "policy_scores" not in q["decision"]
               and not q["decision"]["reason"].startswith("Warm-up"))


def test_default_policy_decision_has_no_v3_fields():
    _, qs = _play(None, fake_predictor)
    assert all("policy_scores" not in q["decision"] for q in qs)
