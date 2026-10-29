"""
src/questions/generator.py
--------------------------
Seeded, offline, parametric question generator (v3).

    generate(topic_key, level="medium", difficulty=None, rng=None, exclude_texts=None)
      -> {question_text, correct_answer, answer_type, difficulty_bucket, difficulty,
          params, complexity, topic_key, tolerance, choices?}
    check_answer(submitted, item) -> bool   (tolerant; see below)

Difficulty is explicit. A fine difficulty d in 1..10 (default: drawn from the
level's range, easy 1-3 / medium 4-7 / hard 8-10) sets the base parameters

    digits    operand size in digits        1,1,1,2,2,2,3,3,3,4
    steps     operations / terms / stages   1 for d<=3, 2 for d<=7, 3 for d>=8
    negatives negative operands allowed     d >= 5   (only where the topic allows)
    fractions fractional values allowed     d >= 7   (only where the topic allows)

and each topic generator turns them into an item. `complexity` = digits + steps +
negatives + fractions of the parameters the topic actually used; by construction it
is strictly ordered easy < medium < hard (tested). The bucket is 1/2/3 for
easy/medium/hard, the same buckets the model uses -- an approximation of the
data-derived buckets, never claimed to be exact.

Answer types: "numeric" (|x - answer| <= tolerance; fractions like 3/4 accepted),
"fraction" (any equivalent fraction, mixed number or decimal within 0.005), and
"mc" (letter A-D or the choice text, case/space-insensitive). Multiple-choice
questions also list the choices in the text so a plain text box still works.

Every function draws only from the passed random.Random, so the same seed gives
the same items on every platform. No network, no files.
"""

from __future__ import annotations

import math
import random
import re
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

LEVEL_RANGES = {"easy": (1, 3), "medium": (4, 7), "hard": (8, 10)}
LEVEL_TO_BUCKET = {"easy": 1, "medium": 2, "hard": 3}
DIGITS = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
NUMERIC_TOL = 0.01
FRACTION_TOL = 0.005


def level_of(d: int) -> str:
    return "easy" if d <= 3 else ("medium" if d <= 7 else "hard")


def base_params(d: int) -> Dict[str, Any]:
    d = int(min(max(d, 1), 10))
    return {"d": d, "digits": DIGITS[d - 1], "steps": 1 if d <= 3 else (2 if d <= 7 else 3),
            "negatives": d >= 5, "fractions": d >= 7}


def _int(rng: random.Random, digits: int, neg: bool = False, min_val: int = 1) -> int:
    lo = max(min_val, 10 ** (digits - 1) if digits > 1 else min_val)
    hi = 10 ** digits - 1
    v = rng.randint(lo, max(lo, hi))
    return -v if (neg and rng.random() < 0.5) else v


def _sgn(v: int) -> str:
    return f"+ {v}" if v >= 0 else f"- {abs(v)}"


def _frac_str(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _num_str(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _mc(rng: random.Random, correct: str, distractors: List[str]):
    opts = [correct] + [d for d in dict.fromkeys(distractors) if d != correct][:3]
    rng.shuffle(opts)
    return opts


# ---------------------------------------------------------------------------------
# Topic generators: (rng, p) -> (text, answer, answer_type, extra)
# extra may hold "choices", "tolerance", "used" (list of used flags)
# ---------------------------------------------------------------------------------

def g_add_sub(rng, p):
    nums = [_int(rng, p["digits"]) for _ in range(p["steps"] + 1)]
    ops = [rng.choice("+-") for _ in range(p["steps"])]
    # keep every running total non-negative (primary: no negatives)
    total = nums[0]
    for i, op in enumerate(ops):
        if op == "-" and nums[i + 1] > total:
            ops[i] = "+"
        total = total + nums[i + 1] if ops[i] == "+" else total - nums[i + 1]
    text = f"{nums[0]}" + "".join(f" {o} {n}" for o, n in zip(ops, nums[1:]))
    return f"Calculate {text}.", str(total), "numeric", {}


def g_mul_div(rng, p):
    a = _int(rng, p["digits"], min_val=2)
    b = rng.randint(2, 9) if p["steps"] == 1 else _int(rng, max(1, p["digits"] - 1), min_val=2)
    if p["steps"] == 1:
        return f"Calculate {a} × {b}.", str(a * b), "numeric", {}
    if p["steps"] == 2:
        return f"Calculate {a * b} ÷ {b}.", str(a), "numeric", {}
    c = rng.choice([x for x in range(2, 10) if (a * b) % x == 0] or [1])
    return f"Calculate ({a} × {b}) ÷ {c}.", str(a * b // c), "numeric", {}


def g_fractions(rng, p):
    dmax = 6 if p["digits"] == 1 else (12 if p["digits"] == 2 else 20)
    if p["steps"] == 1:
        d = rng.randint(3, dmax)
        a, b = rng.randint(1, d - 1), rng.randint(1, d - 1)
        ans = Fraction(a, d) + Fraction(b, d)
        return f"Add {a}/{d} + {b}/{d}. Give the answer as a fraction in simplest form.", _frac_str(ans), "fraction", {}
    fr = [Fraction(rng.randint(1, d - 1), d) for d in (rng.randint(2, dmax) for _ in range(p["steps"]))]
    ops = [rng.choice("+-") for _ in range(p["steps"] - 1)]
    total = fr[0]
    for op, f in zip(ops, fr[1:]):
        total = total + f if op == "+" else total - f
    if total < 0:  # primary: keep the result non-negative
        ops = ["+"] * len(ops)
        total = sum(fr, Fraction(0))
    text = f"{fr[0].numerator}/{fr[0].denominator}" + "".join(
        f" {o} {f.numerator}/{f.denominator}" for o, f in zip(ops, fr[1:]))
    return f"Calculate {text}. Give the answer as a fraction in simplest form.", _frac_str(total), "fraction", {"used": ["fractions"]}


def g_decimals(rng, p):
    places = p["steps"]
    whole = p["digits"]
    a = round(rng.randint(10 ** (whole - 1) if whole > 1 else 0, 10 ** whole - 1) + rng.randint(1, 10 ** places - 1) / 10 ** places, places)
    b = round(rng.randint(0, 10 ** whole - 1) + rng.randint(1, 10 ** places - 1) / 10 ** places, places)
    if p["steps"] == 3:
        a1, b1 = round(rng.randint(1, 10 ** whole - 1) / 10, 1), round(rng.randint(1, 99) / 10, 1)
        return f"Calculate {a1} × {b1}.", _num_str(round(a1 * b1, 2)), "numeric", {"tolerance": 0.001}
    op = rng.choice("+-")
    if op == "-" and b > a:
        a, b = b, a
    ans = round(a + b if op == "+" else a - b, places)
    return f"Calculate {a:.{places}f} {op} {b:.{places}f}.", _num_str(ans), "numeric", {"tolerance": 0.001}


def g_geometry_primary(rng, p):
    w, h = _int(rng, p["digits"], min_val=2), _int(rng, p["digits"], min_val=2)
    if p["steps"] == 1:
        return f"A rectangle is {w} cm long and {h} cm wide. What is its perimeter in cm?", str(2 * (w + h)), "numeric", {}
    if p["steps"] == 2:
        if rng.random() < 0.5:
            return f"A rectangle is {w} cm long and {h} cm wide. What is its area in cm²?", str(w * h), "numeric", {}
        b = w if w % 2 == 0 else w + 1
        return f"A triangle has base {b} cm and height {h} cm. What is its area in cm²?", str(b * h // 2), "numeric", {}
    cw, ch = rng.randint(1, max(1, w - 1)), rng.randint(1, max(1, h - 1))
    return (f"An L-shape is a {w} cm by {h} cm rectangle with a {cw} cm by {ch} cm rectangle cut from one corner. "
            f"What is its area in cm²?"), str(w * h - cw * ch), "numeric", {}


def g_algebra(rng, p):
    neg = p["negatives"]
    if p["steps"] < 3:
        a, b = _int(rng, 1, neg, 1), _int(rng, p["digits"], neg)
        x = _int(rng, 1, neg, 1)
        if p["steps"] == 1:
            return f"Evaluate {a}x {_sgn(b)} when x = {x}.", str(a * x + b), "numeric", {"used": ["negatives"]}
        c = _int(rng, 1, neg, 1)
        return (f"Evaluate {a}x² {_sgn(c)}x {_sgn(b)} when x = {x}.", str(a * x * x + c * x + b), "numeric",
                {"used": ["negatives"]})
    a, b, c = rng.randint(2, 6), _int(rng, 1, True, 1), rng.randint(1, 5)
    coef, const = a - c, a * b
    def fmt(cx, k):
        cx_s = "" if cx == 0 else ("x" if cx == 1 else ("-x" if cx == -1 else f"{cx}x"))
        if not cx_s:
            return str(k)
        return cx_s if k == 0 else f"{cx_s} {_sgn(k)}"
    correct = fmt(coef, const)
    distract = [fmt(coef, b), fmt(a + c, const), fmt(coef, const + c), fmt(coef + 1, const)]
    choices = _mc(rng, correct, distract)
    letters = "ABCD"
    return (f"Which expression equals {a}(x {_sgn(b)}) - {c}x? " + "  ".join(f"{letters[i]}) {o}" for i, o in enumerate(choices)),
            correct, "mc", {"choices": choices, "used": ["negatives"]})


def g_equations(rng, p):
    neg = p["negatives"]
    if p["fractions"]:
        a = rng.randint(2, 9)
        x = Fraction(_int(rng, 1, neg, 1), a + rng.randint(1, 3))
        if x.denominator == 1:
            x = Fraction(x.numerator, 7)
        b = _int(rng, p["digits"] - 1 or 1, neg)
        c = a * x + b
        if p["steps"] >= 3:
            k = rng.randint(2, 4)
            return (f"Solve for x: {k}({a}x {_sgn(b)}) = {_frac_str(k * c)}. Give x as a fraction.",
                    _frac_str(x), "fraction", {"used": ["negatives", "fractions"]})
        return f"Solve for x: {a}x {_sgn(b)} = {_frac_str(c)}. Give x as a fraction.", _frac_str(x), "fraction", {"used": ["negatives", "fractions"]}
    x = _int(rng, 1, neg, 1)
    b = _int(rng, p["digits"], neg)
    if p["steps"] == 1:
        return f"Solve for x: x {_sgn(b)} = {x + b}.", str(x), "numeric", {"used": ["negatives"]}
    a = rng.randint(2, 9)
    if p["steps"] == 2:
        return f"Solve for x: {a}x {_sgn(b)} = {a * x + b}.", str(x), "numeric", {"used": ["negatives"]}
    c = rng.randint(1, a - 1) if a > 2 else 1
    rhs_const = a * x + b - c * x
    return f"Solve for x: {a}x {_sgn(b)} = {c}x {_sgn(rhs_const)}.", str(x), "numeric", {"used": ["negatives"]}


def g_ratios(rng, p):
    if p["steps"] == 1:
        pct = rng.choice([10, 20, 25, 50])
        base = rng.choice([20, 40, 60, 80, 100, 200])
        return f"What is {pct}% of {base}?", _num_str(pct * base / 100), "numeric", {}
    if p["steps"] == 2:
        r1, r2 = rng.randint(1, 5), rng.randint(2, 7)
        unit = _int(rng, p["digits"] - 1 or 1, min_val=2)
        total = (r1 + r2) * unit
        return f"Share {total} in the ratio {r1}:{r2}. How big is the larger share?", str(max(r1, r2) * unit), "numeric", {}
    price = _int(rng, p["digits"], min_val=20)
    up, down = rng.choice([10, 20, 25, 30]), rng.choice([10, 15, 20, 25])
    ans = price * (1 + up / 100) * (1 - down / 100)
    return (f"A price of {price} is increased by {up}% and the new price is then reduced by {down}%. "
            f"What is the final price? (2 decimal places)"), _num_str(round(ans, 2)), "numeric", {"tolerance": 0.01}


def g_geometry_secondary(rng, p):
    if p["steps"] == 1:
        k = rng.randint(1, 4)
        a, b, c = (k * v for v in rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]))
        return f"A right triangle has legs {a} and {b}. How long is the hypotenuse?", str(c), "numeric", {}
    if p["steps"] == 2:
        a, b = rng.randint(20, 80), rng.randint(20, 80)
        return f"Two angles of a triangle are {a}° and {b}°. What is the third angle in degrees?", str(180 - a - b), "numeric", {}
    if p["fractions"] and rng.random() < 0.5:
        r = _int(rng, p["digits"] - 1 or 1, min_val=2)
        return (f"Find the area of a circle with radius {r} cm, using π = 3.14. (2 decimal places)",
                _num_str(round(3.14 * r * r, 2)), "numeric", {"tolerance": 0.01, "used": ["fractions"]})
    c, a = rng.randint(10, 40), rng.randint(3, 9)
    return (f"A right triangle has hypotenuse {c} and one leg {a}. How long is the other leg? (2 decimal places)",
            _num_str(round(math.sqrt(c * c - a * a), 2)), "numeric", {"tolerance": 0.01, "used": ["fractions"]})


def g_statistics(rng, p):
    n = 3 + p["steps"] * 2 - 1
    vals = [_int(rng, p["digits"], p["negatives"]) for _ in range(n)]
    kind = rng.choice(["mean", "median", "range"]) if p["steps"] > 1 else rng.choice(["mean", "range"])
    data = ", ".join(str(v) for v in vals)
    if kind == "mean":
        m = sum(vals) / n
        return (f"Find the mean of {data}." + (" (2 decimal places)" if abs(m - round(m)) > 1e-9 else ""),
                _num_str(round(m, 2)), "numeric", {"tolerance": 0.01, "used": ["negatives"]})
    if kind == "median":
        return f"Find the median of {data}.", _num_str(sorted(vals)[n // 2]), "numeric", {"used": ["negatives"]}
    return f"Find the range of {data}.", str(max(vals) - min(vals)), "numeric", {"used": ["negatives"]}


def g_calculus(rng, p):
    terms = []
    for k in range(p["steps"] + 1, 0, -1):
        terms.append((_int(rng, 1, p["negatives"], 1), k))
    if p["steps"] == 1:
        terms.append((_int(rng, 1, p["negatives"], 1), 0))
    x0 = _int(rng, 1, p["negatives"], 1) if p["d"] > 3 else rng.randint(1, 3)
    def term_s(c, k):
        if k == 0:
            return str(c)
        return f"{c}x" + (f"^{k}" if k > 1 else "")
    poly = " + ".join(term_s(c, k) for c, k in terms).replace("+ -", "- ")
    deriv = sum(c * k * x0 ** (k - 1) for c, k in terms if k > 0)
    return f"Let f(x) = {poly}. Find f'({x0}).", str(deriv), "numeric", {"used": ["negatives"]}


def g_linear_algebra(rng, p):
    m = [[_int(rng, p["digits"] if p["digits"] < 3 else 2, p["negatives"], 1) for _ in range(2)] for _ in range(2)]
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if p["steps"] < 3 or det == 0:
        return (f"Find the determinant of the matrix [[{m[0][0]}, {m[0][1]}], [{m[1][0]}, {m[1][1]}]].",
                str(det), "numeric", {"used": ["negatives"]})
    e, f = _int(rng, 1, True, 1), _int(rng, 1, True, 1)
    x = Fraction(e * m[1][1] - m[0][1] * f, det)
    return (f"Solve {m[0][0]}x {_sgn(m[0][1])}y = {e} and {m[1][0]}x {_sgn(m[1][1])}y = {f}. Give x as a fraction.",
            _frac_str(x), "fraction", {"used": ["negatives", "fractions"]})


def g_probability(rng, p):
    if p["steps"] == 1:
        s = rng.randint(2, 12)
        ways = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == s)
        return f"Two fair dice are rolled. What is the probability that the sum is {s}? Give a fraction.", _frac_str(Fraction(ways, 36)), "fraction", {}
    if p["steps"] == 2:
        r, b = rng.randint(2, 4 + p["digits"] * 2), rng.randint(2, 4 + p["digits"] * 2)
        ans = Fraction(r, r + b) * Fraction(r - 1, r + b - 1)
        return (f"A bag has {r} red and {b} blue balls. Two are drawn without replacement. "
                f"What is the probability both are red? Give a fraction."), _frac_str(ans), "fraction", {}
    n = rng.randint(3, 6)
    k = rng.randint(1, n - 1)
    q = Fraction(1, rng.choice([2, 3, 4]))
    ans = math.comb(n, k) * q ** k * (1 - q) ** (n - k)
    return (f"A trial succeeds with probability {_frac_str(q)}. In {n} independent trials, what is the probability "
            f"of exactly {k} successes? Give a fraction."), _frac_str(ans), "fraction", {"used": ["fractions"]}


def g_discrete(rng, p):
    if p["steps"] == 1:
        n, k = rng.randint(4, 8), rng.randint(2, 3)
        return f"How many ways can {k} items be chosen from {n} (order does not matter)?", str(math.comb(n, k)), "numeric", {}
    if p["steps"] == 2:
        n, k = rng.randint(5, 9 + p["digits"]), rng.randint(2, 4)
        return f"How many ordered arrangements of {k} items can be made from {n} distinct items?", str(math.perm(n, k)), "numeric", {}
    a, b, m = _int(rng, 2, False, 2), rng.randint(5, 40), rng.randint(5, 13)
    return f"Compute {a}^{b} mod {m}.", str(pow(a, b, m)), "numeric", {}


# ---- the 10 original ASSISTments topics (QUESTION_SOURCE=generated|mixed) --------

def g_percents(rng, p):
    pct = rng.choice([10, 20, 25, 50, 75]) if p["steps"] == 1 else rng.randint(1, 99)
    base = rng.choice([20, 40, 60, 80, 100, 200]) if p["steps"] == 1 else _int(rng, p["digits"], min_val=10)
    tol = {"tolerance": 0.01}
    if p["steps"] == 3:
        part = round(pct * base / 100, 2)
        return f"{_num_str(part)} is what percent of {base}? (2 decimal places)", _num_str(pct), "numeric", tol
    return f"What is {pct}% of {base}?" + ("" if p["steps"] == 1 else " (2 decimal places)"), _num_str(round(pct * base / 100, 2)), "numeric", tol


def g_percent_discount(rng, p):
    price = _int(rng, p["digits"] + 1, min_val=10)
    disc = rng.choice([10, 20, 25, 50]) if p["steps"] == 1 else rng.randint(5, 45)
    ans = price * (1 - disc / 100)
    if p["steps"] == 3:
        tax = rng.choice([5, 8, 10])
        ans = ans * (1 + tax / 100)
        return (f"An item costs {price}. It is discounted by {disc}% and then {tax}% tax is added. "
                f"What is the final price? (2 decimal places)"), _num_str(round(ans, 2)), "numeric", {"tolerance": 0.01}
    return (f"An item costs {price} and is discounted by {disc}%. What is the sale price?"
            + ("" if p["steps"] == 1 else " (2 decimal places)")), _num_str(round(ans, 2)), "numeric", {"tolerance": 0.01}


def g_rate(rng, p):
    speed = _int(rng, max(1, p["digits"] - 1), min_val=2) * 5
    hours = rng.randint(2, 6)
    if p["steps"] == 1:
        return f"A car travels {speed * hours} km in {hours} hours. What is its speed in km/h?", str(speed), "numeric", {}
    if p["steps"] == 2:
        return f"A car travels at {speed} km/h for {hours} hours. How far does it go in km?", str(speed * hours), "numeric", {}
    mins = rng.choice([15, 30, 45])
    return (f"A cyclist rides at {speed} km/h for {hours} hours and {mins} minutes. How far in km? (2 decimal places)",
            _num_str(round(speed * (hours + mins / 60), 2)), "numeric", {"tolerance": 0.01, "used": ["fractions"]})


def g_poly_factors(rng, p):
    r1, r2 = _int(rng, 1, p["negatives"], 1), _int(rng, 1, p["negatives"], 1)
    lead = 1 if p["steps"] < 3 else rng.randint(2, 3)
    b, c = lead * (r1 + r2), lead * r1 * r2
    def fac(a1, a2):
        return f"(x {_sgn(a1)})(x {_sgn(a2)})"
    pre = "" if lead == 1 else str(lead)
    correct = pre + fac(r1, r2)
    distract = [pre + fac(-r1, -r2), pre + fac(r1, -r2), pre + fac(r1 + 1, r2 - 1 if r2 - 1 else r2 + 1)]
    choices = _mc(rng, correct, distract)
    poly = f"{'' if lead == 1 else lead}x² {_sgn(b)}x {_sgn(c)}"
    return ("Which is a factorisation of " + poly + "? " + "  ".join(f"{'ABCD'[i]}) {o}" for i, o in enumerate(choices)),
            correct, "mc", {"choices": choices, "used": ["negatives"]})


def g_exponents(rng, p):
    a, b = rng.randint(2, 4 + p["digits"]), rng.randint(2, 4 + p["digits"])
    if p["steps"] == 1:
        return f"Simplify x^{a} · x^{b} = x^n. What is n?", str(a + b), "numeric", {}
    c = rng.randint(2, 4)
    if p["steps"] == 2:
        return f"Simplify (x^{a})^{c} = x^n. What is n?", str(a * c), "numeric", {}
    d = rng.randint(1, a + b - 1)
    return f"Simplify (x^{a} · x^{b})^{c} / x^{d} = x^n. What is n?", str((a + b) * c - d), "numeric", {}


def g_systems(rng, p):
    x, y = _int(rng, 1, p["negatives"], 1), _int(rng, 1, p["negatives"], 1)
    if p["steps"] == 1:
        return f"Solve x + y = {x + y} and x - y = {x - y}. What is x?", str(x), "numeric", {"used": ["negatives"]}
    a, b, c, d = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5)
    while a * d - b * c == 0:
        d += 1
    return (f"Solve {a}x {_sgn(b)}y = {a * x + b * y} and {c}x {_sgn(d)}y = {c * x + d * y}. What is x?",
            str(x), "numeric", {"used": ["negatives"]})


def g_order_fractions(rng, p):
    dmax = 6 if p["digits"] == 1 else 12
    k = 2 + p["steps"]
    fr = []
    while len(fr) < k:
        f = Fraction(rng.randint(1, dmax - 1), rng.randint(2, dmax))
        if 0 < f < 1 and f not in fr:
            fr.append(f)
    correct = _frac_str(max(fr))
    opts = [_frac_str(f) for f in fr][:4]
    if correct not in opts:
        opts[-1] = correct
    rng.shuffle(opts)
    return ("Which fraction is the largest? " + "  ".join(f"{'ABCD'[i]}) {o}" for i, o in enumerate(opts)),
            correct, "mc", {"choices": opts})


def g_equiv_fractions(rng, p):
    a, b = rng.randint(1, 5 + p["digits"]), rng.randint(6 + p["digits"], 9 + 3 * p["digits"])
    f = Fraction(a, b)
    m = rng.randint(2, 3 + p["steps"] * 2)
    return f"Fill in the blank: {f.numerator}/{f.denominator} = ?/{f.denominator * m}", str(f.numerator * m), "numeric", {}


def g_pythagoras(rng, p):
    return g_geometry_secondary(rng, dict(p, steps=1 if p["steps"] == 1 else 3, fractions=p["steps"] > 1))


# topic_key -> (generator, section, allow_negatives, allow_fractions)
TOPIC_GENERATORS: Dict[str, Callable] = {
    "p_add_sub": g_add_sub, "p_mul_div": g_mul_div, "p_fractions": g_fractions, "p_decimals": g_decimals,
    "p_geometry": g_geometry_primary,
    "s_algebra": g_algebra, "s_equations": g_equations, "s_ratios": g_ratios, "s_geometry": g_geometry_secondary,
    "s_statistics": g_statistics,
    "u_calculus": g_calculus, "u_linear_algebra": g_linear_algebra, "u_probability": g_probability,
    "u_discrete": g_discrete,
    "d_percents": g_percents, "d_percent_discount": g_percent_discount, "d_rate": g_rate,
    "d_poly_factors": g_poly_factors, "d_exponents": g_exponents, "d_systems": g_systems,
    "d_order_fractions": g_order_fractions, "d_equiv_fractions": g_equiv_fractions, "d_pythagoras": g_pythagoras,
}
NO_NEGATIVES = {"p_add_sub", "p_mul_div", "p_fractions", "p_decimals", "p_geometry", "s_ratios", "s_geometry",
                "u_probability", "u_discrete", "d_percents", "d_percent_discount", "d_rate", "d_exponents",
                "d_order_fractions", "d_equiv_fractions", "d_pythagoras"}
DEFAULT_TOPIC_KEYS = {204: "d_percents", 203: "d_percent_discount", 217: "d_rate", 346: "d_poly_factors",
                      371: "d_exponents", 350: "d_systems", 378: "d_systems", 50: "d_order_fractions",
                      48: "d_equiv_fractions", 27: "d_pythagoras"}


def topic_key_for_default(raw_skill_id: int) -> str:
    return DEFAULT_TOPIC_KEYS[int(raw_skill_id)]


def all_topic_keys() -> List[str]:
    return list(TOPIC_GENERATORS)


def _complexity(p: Dict[str, Any], used_flags: List[str]) -> int:
    return int(p["digits"] + p["steps"] + (p["negatives"] and "negatives" in used_flags)
               + (p["fractions"] and "fractions" in used_flags))


def generate(topic_key: str, level: str = "medium", difficulty: Optional[int] = None,
             rng: Optional[random.Random] = None, exclude_texts=None, max_tries: int = 40) -> Dict[str, Any]:
    if topic_key not in TOPIC_GENERATORS:
        raise KeyError(f"unknown topic '{topic_key}'")
    rng = rng if rng is not None else random.Random()
    if difficulty is None:
        if level not in LEVEL_RANGES:
            raise ValueError(f"level must be one of {list(LEVEL_RANGES)}")
        lo, hi = LEVEL_RANGES[level]
        difficulty = rng.randint(lo, hi)
    difficulty = int(min(max(int(difficulty), 1), 10))
    level = level_of(difficulty)
    p = base_params(difficulty)
    if topic_key in NO_NEGATIVES:
        p["negatives"] = False
    exclude = set(exclude_texts or ())
    item = None
    for _ in range(max_tries):
        text, ans, atype, extra = TOPIC_GENERATORS[topic_key](rng, p)
        item = {"question_text": text, "correct_answer": ans, "answer_type": atype,
                "difficulty_bucket": LEVEL_TO_BUCKET[level], "difficulty": difficulty, "level": level,
                "params": dict(p), "complexity": _complexity(p, extra.get("used", [])),
                "topic_key": topic_key,
                "tolerance": extra.get("tolerance", NUMERIC_TOL if atype == "numeric" else FRACTION_TOL)}
        if extra.get("choices"):
            item["choices"] = list(extra["choices"])
        if text not in exclude:
            return item
    item["repeated"] = True   # parameter space exhausted (reported, never silent)
    return item


class QuestionStream:
    """Generates items for one session and never repeats an item text."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.used: set = set()

    def next(self, topic_key: str, level: str = "medium", difficulty: Optional[int] = None) -> Dict[str, Any]:
        it = generate(topic_key, level, difficulty, self.rng, self.used)
        self.used.add(it["question_text"])
        return it


# ---- tolerant answer checking --------------------------------------------------

def _parse_number(s: str) -> Optional[Fraction]:
    s = str(s).strip().lower().replace(",", "").replace("−", "-").replace(" ", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".").replace("x =", "").replace("x=", "").strip()
    m = re.fullmatch(r"(-?\d+) (\d+)/(\d+)", s)        # mixed number 1 1/2
    if m:
        whole, n, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d == 0:
            return None
        return Fraction(whole) + (Fraction(n, d) if whole >= 0 else -Fraction(n, d))
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        d = int(m.group(2))
        return None if d == 0 else Fraction(int(m.group(1)), d)
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def check_answer(submitted: Any, item: Dict[str, Any]) -> bool:
    if submitted is None:
        return False
    atype = item.get("answer_type", "numeric")
    correct = item["correct_answer"]
    if atype == "mc":
        sub = str(submitted).strip()
        choices = item.get("choices") or []
        m = re.fullmatch(r"\(?([a-dA-D])\)?\.?", sub)
        if m and choices:
            idx = "abcd".index(m.group(1).lower())
            return idx < len(choices) and _norm_text(choices[idx]) == _norm_text(correct)
        return _norm_text(sub) == _norm_text(correct)
    sub = _parse_number(submitted)
    cor = _parse_number(correct)
    if sub is None or cor is None:
        return _norm_text(submitted) == _norm_text(correct)
    if atype == "fraction":
        return sub == cor or abs(float(sub) - float(cor)) <= FRACTION_TOL
    return abs(float(sub) - float(cor)) <= float(item.get("tolerance", NUMERIC_TOL)) + 1e-9
