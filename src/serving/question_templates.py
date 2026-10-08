"""
src/serving/question_templates.py
---------------------------------
Task 13: Procedural Question Generation for 10 Fixed Skill Positions,
extended with per-position difficulty LEVELS (easy / medium / hard).

Replaces the static 5-set lookup with parameterized templates and deterministic
correct answer computation. Preserves exact position-to-skill mapping, and adds
a `level` parameter that maps onto the model's difficulty buckets:

  level "easy"   -> difficulty bucket 1
  level "medium" -> difficulty bucket 2
  level "hard"   -> difficulty bucket 3

Template levels are an APPROXIMATION of the data-derived difficulty buckets;
the calibration step on logged answers reconciles the two. Do not claim the
buckets are exact.

  Position 1:  Percents (raw_skill_id: 204)
  Position 2:  Percent Discount (raw_skill_id: 203)
  Position 3:  Rate (raw_skill_id: 217)
  Position 4:  Polynomial Factors (raw_skill_id: 346)
  Position 5:  Simplifying Expressions positive exponents (raw_skill_id: 371)
  Position 6:  Solving Systems of Linear Equations (raw_skill_id: 350)
  Position 7:  Solving Systems of Linear Equations by Graphing (raw_skill_id: 378)
  Position 8:  Ordering Fractions (raw_skill_id: 50)
  Position 9:  Equivalent Fractions (raw_skill_id: 48)
  Position 10: Pythagorean Theorem (raw_skill_id: 27)
"""

import math
import random
from typing import Dict, Any, List, Tuple


LEVEL_TO_BUCKET = {"easy": 1, "medium": 2, "hard": 3}
VALID_LEVELS = ("easy", "medium", "hard")


def _validate_level(level: str) -> str:
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level '{level}': must be one of {VALID_LEVELS}")
    return level


def _format_linear_term(m: int, var: str = "x") -> str:
    """Helper to format coefficient and variable: e.g. 1 -> 'x', -1 -> '-x', 2 -> '2x'."""
    if m == 1:
        return var
    elif m == -1:
        return f"-{var}"
    else:
        return f"{m}{var}"


def _format_line_eq(m: int, c: int, var: str = "x") -> str:
    """Helper to format y = mx + c line expression nicely."""
    mx = _format_linear_term(m, var)
    if c == 0:
        return mx
    elif c > 0:
        return f"{mx} + {c}"
    else:
        return f"{mx} - {abs(c)}"


# -------------------------------------------------------------
# Generators for Positions 1 to 10
# -------------------------------------------------------------

def generate_position_1(level: str = "medium") -> Dict[str, Any]:
    """
    Position 1: Percents (raw_skill_id: 204)
    Pattern: "What is {pct}% of {base}?"
    """
    _validate_level(level)
    if level == "easy":
        pct_options = [10, 20, 25, 50, 75]
        base_options = [20, 40, 60, 80, 100]
    elif level == "hard":
        pct_options = [15, 35, 45, 65, 85]
        base_options = [120, 140, 160, 180, 240, 300]
    else:
        pct_options = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 85, 90]
        base_options = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200, 240, 250, 300]

    # Filter valid pairs with whole-integer answers
    valid_pairs = [(p, b) for p in pct_options for b in base_options if (p * b) % 100 == 0]
    pct, base = random.choice(valid_pairs)
    ans = (pct * base) // 100

    return {
        "position": 1,
        "skill_name": "Percents",
        "raw_skill_id": 204,
        "level": level,
        "question_text": f"What is {pct}% of {base}?",
        "correct_answer": str(ans),
        "params": {"pct": pct, "base": base, "ans": ans}
    }


def generate_position_2(level: str = "medium") -> Dict[str, Any]:
    """
    Position 2: Percent Discount (raw_skill_id: 203)
    Pattern: "A {item} costs ${price}. It is on sale for {discount_pct}% off. What is the sale price?"
    """
    _validate_level(level)
    items = [
        "jacket", "shirt", "pair of shoes", "backpack", "sweater",
        "coat", "book", "watch", "camera", "hat", "skateboard",
        "pair of headphones", "gaming mouse", "tennis racket", "desk lamp"
    ]
    if level == "easy":
        prices = [20, 40, 50, 100]
        discounts = [10, 20, 25, 50]
    elif level == "hard":
        prices = [120, 140, 150, 160, 180, 200]
        discounts = [15, 35, 45]
    else:
        prices = [20, 30, 40, 45, 50, 60, 70, 75, 80, 90, 100, 120, 140, 150, 160, 180, 200]
        discounts = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60]

    valid_pairs = [(pr, d) for pr in prices for d in discounts if (pr * d) % 100 == 0]
    price, discount = random.choice(valid_pairs)
    item = random.choice(items)
    sale_price = price - ((price * discount) // 100)

    return {
        "position": 2,
        "skill_name": "Percent Discount",
        "raw_skill_id": 203,
        "level": level,
        "question_text": f"A {item} costs ${price}. It is on sale for {discount}% off. What is the sale price?",
        "correct_answer": str(sale_price),
        "params": {"item": item, "price": price, "discount": discount, "sale_price": sale_price}
    }


def generate_position_3(level: str = "medium") -> Dict[str, Any]:
    """
    Position 3: Rate (raw_skill_id: 217)
    Pattern: "{subject} travels {distance} {unit} in {time} hours. What is its rate in {unit} per hour?"
    """
    _validate_level(level)
    subjects = [
        ("A car", "miles", "hours", "hour"),
        ("A runner", "miles", "hours", "hour"),
        ("A cyclist", "miles", "hours", "hour"),
        ("A train", "miles", "hours", "hour"),
        ("A swimmer", "meters", "minutes", "minute"),
        ("A boat", "miles", "hours", "hour"),
        ("A bus", "miles", "hours", "hour"),
        ("A plane", "miles", "hours", "hour")
    ]
    if level == "easy":
        rates = [4, 5, 10, 20]
        times = [2, 3, 4]
    elif level == "hard":
        rates = [55, 60, 65, 70, 75, 80]
        times = [6, 8]
    else:
        rates = [4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        times = [2, 3, 4, 5, 6, 8]

    rate = random.choice(rates)
    time = random.choice(times)
    subj, dist_u, time_u, time_sing = random.choice(subjects)
    distance = rate * time

    return {
        "position": 3,
        "skill_name": "Rate",
        "raw_skill_id": 217,
        "level": level,
        "question_text": f"{subj} travels {distance} {dist_u} in {time} {time_u}. What is its rate in {dist_u} per {time_sing}?",
        "correct_answer": str(rate),
        "params": {"subject": subj, "distance": distance, "time": time, "rate": rate}
    }


def generate_position_4(level: str = "medium") -> Dict[str, Any]:
    """
    Position 4: Polynomial Factors (raw_skill_id: 346)
    Pattern: "Factor completely: {var}^2 - {a_sq}" -> "({var}-{a})({var}+{a})"
    """
    _validate_level(level)
    variables = ["x", "y", "a", "b", "m", "n", "p", "k", "t"]
    if level == "easy":
        a_values = [2, 3, 4, 5]
    elif level == "hard":
        a_values = [11, 12, 13, 14, 15]
    else:
        a_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    var = random.choice(variables)
    a = random.choice(a_values)
    a_sq = a * a

    return {
        "position": 4,
        "skill_name": "Polynomial Factors",
        "raw_skill_id": 346,
        "level": level,
        "question_text": f"Factor completely: {var}^2 - {a_sq}",
        "correct_answer": f"({var}-{a})({var}+{a})",
        "params": {"var": var, "a": a, "a_sq": a_sq}
    }


def generate_position_5(level: str = "medium") -> Dict[str, Any]:
    """
    Position 5: Simplifying Expressions positive exponents (raw_skill_id: 371)
    Pattern: "Simplify: {var}^{p1} * {var}^{p2}" -> "{var}^{p1+p2}"
    """
    _validate_level(level)
    variables = ["x", "y", "a", "b", "c", "m", "n", "p", "k", "z", "w"]
    if level == "easy":
        powers = [2, 3]
    elif level == "hard":
        powers = [5, 6, 7]
    else:
        powers = [2, 3, 4, 5, 6, 7]

    var = random.choice(variables)
    p1 = random.choice(powers)
    p2 = random.choice(powers)
    total_power = p1 + p2

    return {
        "position": 5,
        "skill_name": "Simplifying Expressions positive exponents",
        "raw_skill_id": 371,
        "level": level,
        "question_text": f"Simplify: {var}^{p1} * {var}^{p2}",
        "correct_answer": f"{var}^{total_power}",
        "params": {"var": var, "p1": p1, "p2": p2, "total_power": total_power}
    }


def generate_position_6(level: str = "medium") -> Dict[str, Any]:
    """
    Position 6: Solving Systems of Linear Equations (raw_skill_id: 350)
    Pattern: "Solve the system: x + y = {sum}, x - y = {diff}. What is x?" -> "{x}"
    """
    _validate_level(level)
    if level == "easy":
        x_choices = [4, 5, 6, 7, 8]
        y_choices = [1, 2, 3]
    elif level == "hard":
        x_choices = [15, 16, 18, 20]
        y_choices = [8, 9, 10]
    else:
        x_choices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
        y_choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    valid_pairs = [(x, y) for x in x_choices for y in y_choices if x > y]
    x, y = random.choice(valid_pairs)
    sum_val = x + y
    diff_val = x - y

    return {
        "position": 6,
        "skill_name": "Solving Systems of Linear Equations",
        "raw_skill_id": 350,
        "level": level,
        "question_text": f"Solve the system: x + y = {sum_val}, x - y = {diff_val}. What is x?",
        "correct_answer": str(x),
        "params": {"x": x, "y": y, "sum": sum_val, "diff": diff_val}
    }


def generate_position_7(level: str = "medium") -> Dict[str, Any]:
    """
    Position 7: Solving Systems of Linear Equations by Graphing (raw_skill_id: 378)
    Pattern: "The lines y = {line1} and y = {line2} intersect at what x-value?" -> "{x}"
    """
    _validate_level(level)
    if level == "easy":
        x_choices = [1, 2, 3]
        m1_choices = [1, 2]
        m2_choices = [-1, -2]
        c1_choices = [-2, -1, 0, 1, 2]
    elif level == "hard":
        x_choices = [5, 6]
        m1_choices = [3]
        m2_choices = [-3]
        c1_choices = [-5, -4, 4, 5]
    else:
        x_choices = [1, 2, 3, 4, 5, 6]
        m1_choices = [1, 2, 3]
        m2_choices = [-1, -2, -3]
        c1_choices = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    x = random.choice(x_choices)
    m1 = random.choice(m1_choices)
    m2 = random.choice(m2_choices)
    c1 = random.choice(c1_choices)

    y_val = m1 * x + c1
    c2 = y_val - (m2 * x)

    line1 = _format_line_eq(m1, c1, "x")
    line2 = _format_line_eq(m2, c2, "x")

    return {
        "position": 7,
        "skill_name": "Solving Systems of Linear Equations by Graphing",
        "raw_skill_id": 378,
        "level": level,
        "question_text": f"The lines y = {line1} and y = {line2} intersect at what x-value?",
        "correct_answer": str(x),
        "params": {"x": x, "m1": m1, "c1": c1, "m2": m2, "c2": c2, "line1": line1, "line2": line2}
    }


def generate_position_8(level: str = "medium") -> Dict[str, Any]:
    """
    Position 8: Ordering Fractions (raw_skill_id: 50)
    Pattern: "Which is larger: {n1}/{d1} or {n2}/{d2}?" -> larger fraction string
    """
    _validate_level(level)
    if level == "easy":
        dens = [3, 4, 5]
    elif level == "hard":
        dens = [7, 8, 9, 10, 12]
    else:
        dens = [3, 4, 5, 6, 7, 8, 9, 10, 12]
    all_fractions = [(n, d) for d in dens for n in range(1, d) if math.gcd(n, d) == 1]

    # Pick two distinct non-equal fractions
    f1, f2 = random.sample(all_fractions, 2)
    while f1[0] * f2[1] == f2[0] * f1[1]:
        f1, f2 = random.sample(all_fractions, 2)

    n1, d1 = f1
    n2, d2 = f2

    larger = f"{n1}/{d1}" if (n1 * d2 > n2 * d1) else f"{n2}/{d2}"

    return {
        "position": 8,
        "skill_name": "Ordering Fractions",
        "raw_skill_id": 50,
        "level": level,
        "question_text": f"Which is larger: {n1}/{d1} or {n2}/{d2}?",
        "correct_answer": larger,
        "params": {"f1": f"{n1}/{d1}", "f2": f"{n2}/{d2}", "larger": larger}
    }


def generate_position_9(level: str = "medium") -> Dict[str, Any]:
    """
    Position 9: Equivalent Fractions (raw_skill_id: 48)
    Pattern: "Which fraction is equivalent to {n}/{d}?" -> "{n*k}/{d*k}"
    """
    _validate_level(level)
    if level == "easy":
        base_fractions = [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4)]
        multipliers = [2, 3]
    elif level == "hard":
        base_fractions = [(3, 8), (5, 8), (7, 8)]
        multipliers = [5, 6]
    else:
        base_fractions = [
            (1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5),
            (3, 5), (4, 5), (1, 6), (5, 6), (3, 8), (5, 8), (7, 8)
        ]
        multipliers = [2, 3, 4, 5, 6]

    n, d = random.choice(base_fractions)
    k = random.choice(multipliers)
    eq_n = n * k
    eq_d = d * k

    return {
        "position": 9,
        "skill_name": "Equivalent Fractions",
        "raw_skill_id": 48,
        "level": level,
        "question_text": f"Which fraction is equivalent to {n}/{d}?",
        "correct_answer": f"{eq_n}/{eq_d}",
        "params": {"base_n": n, "base_d": d, "multiplier": k, "ans_n": eq_n, "ans_d": eq_d}
    }


def generate_position_10(level: str = "medium") -> Dict[str, Any]:
    """
    Position 10: Pythagorean Theorem (raw_skill_id: 27)
    Pattern: "A right triangle has legs of length {leg_a} and {leg_b}. What is the length of the hypotenuse?" -> "{hyp}"
    """
    _validate_level(level)
    easy_triples = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17)]
    hard_triples = [(9, 40, 41), (12, 35, 37), (20, 21, 29), (16, 30, 34)]
    all_triples = easy_triples + [
        (12, 16, 20), (7, 24, 25), (15, 20, 25), (10, 24, 26), (18, 24, 30),
    ] + hard_triples

    if level == "easy":
        triples = easy_triples
    elif level == "hard":
        triples = hard_triples
    else:
        triples = all_triples

    a, b, hyp = random.choice(triples)
    # Randomize leg order
    if random.random() < 0.5:
        leg_a, leg_b = a, b
    else:
        leg_a, leg_b = b, a

    return {
        "position": 10,
        "skill_name": "Pythagorean Theorem",
        "raw_skill_id": 27,
        "level": level,
        "question_text": f"A right triangle has legs of length {leg_a} and {leg_b}. What is the length of the hypotenuse?",
        "correct_answer": str(hyp),
        "params": {"leg_a": leg_a, "leg_b": leg_b, "hypotenuse": hyp}
    }


# Map of position (1-10) to generator function
GENERATOR_MAP = {
    1: generate_position_1,
    2: generate_position_2,
    3: generate_position_3,
    4: generate_position_4,
    5: generate_position_5,
    6: generate_position_6,
    7: generate_position_7,
    8: generate_position_8,
    9: generate_position_9,
    10: generate_position_10,
}


def generate_question(position: int, level: str = "medium") -> Dict[str, Any]:
    """Generates a fresh question with parameters and correct answer for given position and level."""
    if position not in GENERATOR_MAP:
        raise ValueError(f"Invalid position {position}: must be 1..10")
    return GENERATOR_MAP[position](level)
