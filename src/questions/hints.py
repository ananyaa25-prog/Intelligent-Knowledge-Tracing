"""
src/questions/hints.py
----------------------
One strategy hint per question topic (v3, HINTS_ENABLED). Hints describe the method,
never the answer of the current question. Written for this project; not reviewed by a
teacher (docs/LIMITATIONS.md).
"""

from typing import Any, Dict

HINTS: Dict[str, str] = {
    "p_add_sub": "Line the numbers up by place value (ones under ones) and work from the right, carrying or borrowing as you go.",
    "p_mul_div": "Split the bigger number into tens and ones, multiply (or share) each part, then combine the results.",
    "p_fractions": "To add or subtract fractions, first rewrite them with the same denominator; then work with the numerators only.",
    "p_decimals": "Line up the decimal points before adding or subtracting; for multiplying, ignore the point, then count decimal places.",
    "p_geometry": "Write down the formula first (perimeter = sum of the sides, area of a rectangle = length x width), then substitute.",
    "s_algebra": "Substitute the given value for the letter everywhere it appears, then follow the order of operations.",
    "s_equations": "Do the same operation to both sides to get the unknown on its own: undo addition/subtraction first, then multiplication/division.",
    "s_ratios": "Add the parts of the ratio to find the total number of parts, then find the size of one part.",
    "s_geometry": "Identify which formula the shape needs (angles in a triangle add to 180, circle area = pi r^2) and substitute carefully.",
    "s_statistics": "Mean = total / count; median = middle value after sorting; mode = most frequent value; range = largest - smallest.",
    "u_calculus": "Use the power rule: the derivative of a x^n is n a x^(n-1); differentiate each term separately.",
    "u_linear_algebra": "For a 2x2 matrix [[a, b], [c, d]] the determinant is ad - bc; for products, multiply rows by columns.",
    "u_probability": "Probability = favourable outcomes / all equally likely outcomes; for independent events multiply the probabilities.",
    "u_discrete": "Decide whether order matters: if it does, use permutations; if not, use combinations n! / (r! (n - r)!).",
    "d_percents": "x% of a number = (x / 100) x the number. Find 10% or 1% first if that is easier.",
    "d_percent_discount": "Find the discount (percent x price / 100), then subtract it from the original price.",
    "d_rate": "Unit rate = total amount / number of units; check which quantity goes on top.",
    "d_poly_factors": "Look for two numbers that multiply to the constant term and add to the middle coefficient.",
    "d_exponents": "Same base: multiply -> add the exponents, divide -> subtract them, power of a power -> multiply them.",
    "d_systems": "Add or subtract the equations so one variable cancels, solve for the other, then substitute back.",
    "d_order_fractions": "Rewrite the fractions with a common denominator (or as decimals) and then compare the numerators.",
    "d_equiv_fractions": "Multiply or divide the numerator and the denominator by the same number.",
    "d_pythagoras": "In a right triangle a^2 + b^2 = c^2, where c is the side opposite the right angle (the longest side).",
}

GENERIC_HINT = "Write down what you know and what you need to find, then try one small step at a time."


def hint_for(topic_meta: Dict[str, Any]) -> str:
    """Hint for a session topic (section topics carry their generator key; default topics
    map through the ASSISTments skill id)."""
    key = topic_meta.get("key")
    if not key:
        try:
            from questions.generator import topic_key_for_default
            key = topic_key_for_default(topic_meta["raw_skill_id"])
        except (KeyError, ValueError, TypeError):
            key = None
    return HINTS.get(key, GENERIC_HINT)
