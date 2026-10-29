# v3 demo configuration: pre-registration and gate outcomes

## Part 1 - Pre-registration (written 2026-10-29, before scripts/simulate_v3.py was run)

Fixed before any v3 simulation result existed; not changed afterwards except bug fixes (any bug fix is
listed in Part 2 with its reason).

**Simulation protocol (scripts/simulate_v3.py).** Primary setting = the current demo setting:
SESSION_MODE=core5, TOPIC_SELECTION=evidence_blend, APPLY_CALIBRATION=true. 100 paired synthetic
students per policy, seed 2026, student model of v2 (g ~ N(0, 0.8); a_t = g + N(0, 0.7);
P_true = 0.05 + 0.95 sigmoid(a_t - b), b = -1 / 0 / +1). Policies: random, staircase, elo, maxinfo,
model (ZPD, TARGET_P 0.70), uncertainty (3-checkpoint ensemble, spread limit 0.15). All share the topic
selection and the D27 allowed-level guardrails. Elo level difficulties come from the ASSISTments train
split (data/processed_v3/elo_level_difficulty.json), not from the simulator.

**Gate P (difficulty policy for the demo).** Candidates in this order: `uncertainty`, `model`.
A candidate passes if Gate M passes (profile_reliable = true) AND its in-band share >= staircase in-band
share - 0.02 AND its pooled status accuracy >= staircase pooled status accuracy - 0.03 (same margins
as v2 Gate S). If both pass, the one with the higher pooled status accuracy is used (tie: `uncertainty`).
If none passes, the demo keeps POLICY=staircase. `elo` and `maxinfo` are comparison policies only and
cannot be selected for the demo.

**Gate Q (question source for the demo).** QUESTION_SOURCE=mixed only if (a) tests/test_question_generator.py
passes completely and (b) scripts/e2e_http_check.py passes through the Node proxy with QUESTION_SOURCE=mixed;
otherwise QUESTION_SOURCE=bank.

Other demo values (TARGET_P, APPLY_CALIBRATION, SESSION_MODE, TOPIC_SELECTION) stay as decided in v2
(reports/DEMO_CONFIG.md). Sections are chosen by the user in the UI; the default stays the original topics.

## Part 2 - Outcomes

(filled in by scripts/apply_gates_v3.py after the runs)
