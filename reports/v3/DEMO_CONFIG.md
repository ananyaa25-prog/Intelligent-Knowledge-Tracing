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

Applied 2026-10-29 11:10 UTC by `python scripts/apply_gates_v3.py`. Numbers from reports/v3/policy_comparison.md (primary setting) and reports/v3/e2e_http_check.md.

| Gate | Rule | Numbers | Result |
|---|---|---|---|
| P (uncertainty) | Gate M and in-band >= staircase - 0.02 and pooled status accuracy >= staircase - 0.03 | in-band 45.9% vs need >= 41.2% (True); status accuracy 50.4% vs need >= 45.3% (True); Gate M True | PASS |
| P (model) | Gate M and in-band >= staircase - 0.02 and pooled status accuracy >= staircase - 0.03 | in-band 50.3% vs need >= 41.2% (True); status accuracy 47.3% vs need >= 45.3% (True); Gate M True | PASS |
| Q | generator tests pass and e2e HTTP check passes with QUESTION_SOURCE=mixed | generator tests: 100 passed in 0.32s; e2e: mixed PASS (306 checks), demo_mixed PASS (324 checks) | PASS |

Staircase reference: in-band 43.2%, pooled status accuracy 48.3%.

**Demo POLICY = uncertainty; QUESTION_SOURCE = mixed.**

Note on the selected policy: in the primary simulation the uncertainty-aware policy fell back to the staircase level on 7.1 of 12.0 level decisions per session (ensemble spread > 0.15), so in practice it behaves mostly like the staircase and flags those questions as low-confidence predictions.

Resulting `demo.env`:

```
POLICY=uncertainty
TARGET_P=0.70
APPLY_CALIBRATION=true
SESSION_MODE=core5
TOPIC_SELECTION=evidence_blend
QUESTION_SOURCE=mixed
```

Bug fixes after pre-registration: (1) the first simulate_v3.py run crashed while writing the markdown report (a str.format KeyError) before any result was shown or saved; the format bug was fixed, the JSON is now saved before the report, and the run was repeated unchanged otherwise. (2) Section sessions in evidence-only mode no longer print neutral 50% placeholders in the decision text (affects sections only, not this simulation). No policy, threshold or student parameter was changed.

Significance: the gate is a non-inferiority rule. In the primary setting the in-band gain over the staircase is significant for both candidates, but no status-accuracy difference is (Holm-corrected p = 1.0 for every policy; reports/v3/policy_comparison.md). Passing the gate means 'not worse by more than the margin', not 'better'.

Caveats: the simulation uses synthetic students and validates decision logic only; the Elo level difficulties learned from ASSISTments (-1.12 / 0 / +1.16) happen to be close to the simulator's generating values (-1 / 0 / +1), which favours the Elo policy in this simulation; Elo and max-information were comparison policies and could not be selected for the demo by the pre-registered rule.
