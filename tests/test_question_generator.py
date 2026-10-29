"""v3 Stage 8: parametric question generator (determinism, answer correctness,
uniqueness, section coverage, difficulty ordering) and the QUESTION_SOURCE flag."""

import ast
import operator
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from questions.generator import (DEFAULT_TOPIC_KEYS, LEVEL_RANGES, QuestionStream, all_topic_keys,  # noqa: E402
                                 base_params, check_answer, generate)
from serving.sections import SECTIONS  # noqa: E402

KEYS = all_topic_keys()
LEVELS = ("easy", "medium", "hard")


def _eval(expr: str) -> Fraction:
    """Independent exact evaluator for '+ - * / ()' expressions with a/b fractions."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def ev(n):
        if isinstance(n, ast.BinOp):
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Constant):
            return Fraction(str(n.value))
        raise ValueError(n)
    return ev(ast.parse(expr.replace("×", "*").replace("÷", "/"), mode="eval").body)


@pytest.mark.parametrize("key", KEYS)
def test_determinism(key):
    for lvl in LEVELS:
        a = [generate(key, lvl, rng=random.Random(42)) for _ in range(3)]
        b = [generate(key, lvl, rng=random.Random(42)) for _ in range(3)]
        assert a == b


@pytest.mark.parametrize("key", KEYS)
def test_own_answer_accepted_and_wrong_rejected(key):
    rng = random.Random(7)
    for lvl in LEVELS:
        for _ in range(60):
            it = generate(key, lvl, rng=rng)
            assert set(it) >= {"question_text", "correct_answer", "answer_type", "difficulty_bucket", "params"}
            assert it["difficulty_bucket"] == {"easy": 1, "medium": 2, "hard": 3}[lvl]
            assert check_answer(it["correct_answer"], it)
            if it["answer_type"] == "mc":
                idx = it["choices"].index(it["correct_answer"])
                assert check_answer("ABCD"[idx], it) and check_answer("abcd"[idx].lower(), it)
                wrong = [i for i in range(len(it["choices"])) if i != idx]
                assert not check_answer("ABCD"[wrong[0]], it)
                assert len(set(it["choices"])) == len(it["choices"]) >= 2
            else:
                assert not check_answer("123456789", it)
                assert not check_answer("", it) and not check_answer(None, it)


def test_independent_recomputation_of_arithmetic_items():
    """Re-derive the answer of 'Calculate ...' items from the question text alone."""
    n = 0
    for key in ("p_add_sub", "p_mul_div", "p_fractions", "p_decimals"):
        rng = random.Random(11)
        for lvl in LEVELS:
            for _ in range(40):
                it = generate(key, lvl, rng=rng)
                m = re.match(r"(?:Calculate|Add) (.+?)\.(?: Give| *$)", it["question_text"])
                assert m, it["question_text"]
                val = _eval(m.group(1))
                assert check_answer(str(val.numerator) + ("" if val.denominator == 1 else f"/{val.denominator}"), it)
                n += 1
    assert n == 480


def test_independent_recomputation_of_other_topics():
    rng = random.Random(5)
    for _ in range(50):
        it = generate("u_linear_algebra", "easy", rng=rng)
        a, b, c, d = map(int, re.findall(r"-?\d+", it["question_text"].split("[[")[1]))
        assert int(it["correct_answer"]) == a * d - b * c
        it = generate("u_discrete", "easy", rng=rng)
        k, n = map(int, re.findall(r"\d+", it["question_text"])[:2])
        from math import comb
        assert int(it["correct_answer"]) == comb(n, k)
        it = generate("s_equations", "medium", rng=rng)
        if it["answer_type"] == "numeric" and "=" in it["question_text"]:
            x = int(it["correct_answer"])
            lhs, rhs = it["question_text"].split(": ", 1)[1].rstrip(".").split(" = ")
            f = lambda e: _eval(re.sub(r"(\d)x", r"\1*X", e).replace("X", f"({x})").replace("x", f"({x})"))
            assert f(lhs) == f(rhs)


def test_tolerant_checking():
    it = {"answer_type": "fraction", "correct_answer": "1/2"}
    for s in ("1/2", "2/4", " 0.5 ", "0.50", "x = 1/2"):
        assert check_answer(s, it), s
    assert not check_answer("0.6", it)
    num = {"answer_type": "numeric", "correct_answer": "12.35", "tolerance": 0.01}
    assert check_answer("12.351", num) and check_answer("12,35".replace(",", "."), num)
    assert not check_answer("12.4", num)
    mixed = {"answer_type": "fraction", "correct_answer": "3/2"}
    assert check_answer("1 1/2", mixed)
    mc = {"answer_type": "mc", "correct_answer": "x - 54", "choices": ["x - 9", "x - 54", "11x - 54"]}
    assert check_answer("B", mc) and check_answer("x-54", mc) and check_answer("(b)", mc) and not check_answer("A", mc)


@pytest.mark.parametrize("key", KEYS)
def test_no_repeat_within_a_stream(key):
    s = QuestionStream(seed=3)
    texts = [s.next(key, LEVELS[i % 3])["question_text"] for i in range(25)]
    assert len(set(texts)) == len(texts)


def test_all_sections_and_default_topics_covered():
    for sec in SECTIONS.values():
        for t in sec["topics"]:
            assert t["key"] in KEYS, t
    for raw in (204, 203, 217, 346, 371, 350, 378, 50, 48, 27):
        assert DEFAULT_TOPIC_KEYS[raw] in KEYS


@pytest.mark.parametrize("key", KEYS)
def test_complexity_strictly_ordered(key):
    means = []
    for lvl in LEVELS:
        rng = random.Random(9)
        cs = [generate(key, lvl, rng=rng)["complexity"] for _ in range(100)]
        means.append(sum(cs) / len(cs))
        lo, hi = LEVEL_RANGES[lvl]
        assert lo <= generate(key, lvl, rng=rng)["difficulty"] <= hi
    assert means[0] < means[1] < means[2], (key, means)


def test_fine_difficulty_parameters_monotone():
    ps = [base_params(d) for d in range(1, 11)]
    for a, b in zip(ps, ps[1:]):
        assert b["digits"] >= a["digits"] and b["steps"] >= a["steps"]
        assert b["negatives"] >= a["negatives"] and b["fractions"] >= a["fractions"]
    it = generate("s_equations", difficulty=9, rng=random.Random(1))
    assert it["difficulty"] == 9 and it["level"] == "hard" and it["difficulty_bucket"] == 3


# ---- QUESTION_SOURCE flag ------------------------------------------------------

def _first_questions(monkeypatch, value, n=4):
    from serving.adaptive_session import AdaptiveSession
    if value is None:
        monkeypatch.delenv("QUESTION_SOURCE", raising=False)
    else:
        monkeypatch.setenv("QUESTION_SOURCE", value)
    random.seed(0)
    s = AdaptiveSession("t", "stud")
    out = []
    for _ in range(n):
        q = s.next_question(None)
        out.append(q)
        s.record_answer(q["correct_answer"], 3000)
    return s, out


def test_question_source_default_is_bank(monkeypatch):
    from serving.policy_config import get_config
    monkeypatch.delenv("QUESTION_SOURCE", raising=False)
    assert get_config().question_source == "bank"
    _, qs = _first_questions(monkeypatch, None)
    assert all("question_source" not in q for q in qs)


def test_question_source_generated_and_mixed(monkeypatch):
    s, qs = _first_questions(monkeypatch, "generated")
    assert all(q.get("question_source") == "generated" for q in qs)
    assert all(h["correct"] == 1 for h in s.history)          # correct answers accepted
    _, qm = _first_questions(monkeypatch, "mixed")
    assert [q.get("question_source", "bank") for q in qm] == ["bank", "generated", "bank", "generated"]


def test_invalid_question_source_falls_back(monkeypatch):
    from serving.policy_config import get_config
    monkeypatch.setenv("QUESTION_SOURCE", "llm")
    assert get_config().question_source == "bank"
