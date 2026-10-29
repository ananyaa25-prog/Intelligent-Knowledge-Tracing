# Parametric question generator (v3)

Code: `src/questions/generator.py`. Tests: `tests/test_question_generator.py`.
Measured check: `python scripts/check_generator.py` -> `reports/v3/generator_check.md`.

## What it does

`generate(topic_key, level="medium", difficulty=None, rng=None, exclude_texts=None)` returns

```json
{"question_text": "...", "correct_answer": "3/4", "answer_type": "fraction",
 "difficulty_bucket": 2, "difficulty": 5, "level": "medium",
 "params": {"d": 5, "digits": 2, "steps": 2, "negatives": true, "fractions": false},
 "complexity": 5, "topic_key": "s_equations", "tolerance": 0.005, "choices": ["..."]}
```

- **Offline and seeded**: every random draw comes from the `random.Random` passed in, so the same seed
  gives the same items on every platform. No network, no model, no files.
- **Parametric**: numbers, structure and answer are produced together from explicit parameters; the
  answer is computed (exact `Fraction` arithmetic where relevant), never looked up.
- **Topics**: all 14 section topics (5 primary, 5 secondary, 4 undergraduate, see
  `src/serving/sections.py`) and the 10 original ASSISTments topics (`d_*` keys), 23 generators in total.

## How difficulty is set

A fine difficulty `d` in 1..10 (if not given, drawn from the level: easy 1-3, medium 4-7, hard 8-10)
fixes the base parameters:

| d | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| digits (operand size) | 1 | 1 | 1 | 2 | 2 | 2 | 3 | 3 | 3 | 4 |
| steps (operations / terms / stages) | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 3 | 3 | 3 |
| negatives allowed | no | no | no | no | yes | yes | yes | yes | yes | yes |
| fractions allowed | no | no | no | no | no | no | yes | yes | yes | yes |

Each topic interprets them, for example: addition/subtraction uses `steps + 1` operands of `digits`
digits; equations go from `x + b = c` (1 step) to `ax + b = c` (2) to `ax + b = cx + d` (3), with
negative or fractional solutions when allowed; calculus differentiates a polynomial with `steps + 1`
terms; probability goes from one dice event to draws without replacement to a binomial probability.
Topics where negatives make no sense at that level (all primary topics, percents, geometry, counting)
ignore the `negatives` flag.

`complexity = digits + steps + negatives + fractions` (flags counted only where the topic uses them).
It is therefore strictly ordered easy < medium < hard **by construction**; the test only guards
against regressions. The bucket (1/2/3) uses the same numbers as the model's difficulty buckets, but
the generator's levels are an approximation of the data-derived ASSISTments buckets, not a calibrated
match.

## Answer types and checking (`check_answer(submitted, item)`)

| Type | Accepted |
|---|---|
| `numeric` | any number within `tolerance` (default 0.01, as in the existing templates); `3/4` style input also parsed |
| `fraction` | any equivalent fraction (`2/4` for `1/2`), mixed numbers (`1 1/2`), decimals within 0.005, optional `x =` prefix |
| `mc` | the letter A-D (also `(b)`, lower case) or the choice text, ignoring case and spaces |

Multiple-choice items also list `A) ... B) ...` in `question_text`, so the plain text box of the
existing quiz still works (type the letter). The correct answer and the checker data stay on the
server; `/session/{id}/next` exposes `answer_type` and `choices` only.

## No repeats

`generate(..., exclude_texts=used)` re-draws until the text is new (up to 40 tries; if the parameter
space is exhausted the item is flagged `repeated: true`, never silently repeated). The session engine
passes all texts already asked in the session; `QuestionStream(seed)` does the same for scripts.

## How it is switched on

- Section sessions (`/session/start` with `section`) always use the generator: those topics have no
  bank.
- For the original 10 topics, `QUESTION_SOURCE` in `src/serving/policy_config.py` decides:
  `bank` (default, existing templates - original behaviour), `generated`, or `mixed`
  (alternates bank / generated question by question). `question_bank.json` and
  `src/serving/question_templates.py` are unchanged and remain the fallback.

## What has and has not been validated

- Validated by tests: determinism, every generated answer accepted by the checker, wrong answers
  rejected, independent re-computation of the answer from the question text for arithmetic,
  fraction, decimal, determinant, combination and linear-equation items, no repeats in a stream of
  25, all section topics covered, complexity ordering, fine-difficulty monotonicity, the
  `QUESTION_SOURCE` flag.
- Measured (reports/v3/generator_check.md): a surface proxy (digits shown in the question) that is
  NOT built into the generator is ordered easy < medium < hard for most but not all topics.
- **Not validated**: that students find "hard" items harder than "easy" ones, that items are
  curriculum-appropriate for each grade, and that wording is clear. That needs a teacher review of a
  sample of items and logged answers from real students (docs/PILOT_GUIDE.md).
