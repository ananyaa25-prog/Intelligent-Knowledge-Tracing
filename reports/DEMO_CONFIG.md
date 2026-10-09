# Demo configuration (decision gates)

Generated 2026-10-09 06:10 UTC by `scripts/apply_gates.py`. Gates were fixed before the results existed and are applied mechanically.

| Gate | Rule | Numbers | Result | Demo setting |
|---|---|---|---|---|
| M | violation_rate <= MONOTONE_WARN | violation_rate = 0.0% (MONOTONE_WARN 10%) | PASS | profile_reliable = true |
| S | Gate M and model in-band >= staircase - 0.02 and model status acc >= staircase - 0.03 | model in-band 40.1% vs staircase 32.6% (need >= 30.6%: True); model status accuracy 48.1% vs staircase 51.5% (need >= 48.5%: False); Gate M True | FAIL | POLICY = staircase |
| C | ECE improves >= 10% relative and AUC unchanged | ECE 0.0249 -> 0.0117 (53% relative); AUC 0.8946 -> 0.8946 (unchanged: True); T = 1.16696 | PASS | APPLY_CALIBRATION = true |
| K | T8c DONE and core5 end-to-end check passes | T8c DONE in BUILD_LOG: True; core5 end-to-end HTTP check passed: True | PASS | SESSION_MODE = core5 |
| T (post hoc, after user testing) | blended topic selection not worse than the original in simulation | staircase: status accuracy 51.5% (original) vs 52.8% (blended); in-band 32.6% vs 41.6% | PASS | TOPIC_SELECTION = evidence_blend |

Resulting `demo.env`:

```
POLICY=staircase
TARGET_P=0.70
APPLY_CALIBRATION=true
SESSION_MODE=core5
TOPIC_SELECTION=evidence_blend
```

**Model-guided difficulty did not beat the staircase in simulation; the demo uses the staircase. The model still drives topic selection.**
Revision note: the level rules of the model policy were revised after user testing (D27) and the simulation was re-run; the earlier run (archived in reports/archive_v1/) passed Gate S, the re-run with the revised rules does not, so the demo now uses the staircase. Rule T was added after user testing and is labelled as such.

Caveats:
- Gate S was evaluated in simulation with full10 mode and uncalibrated predictions; the combination POLICY=staircase + APPLY_CALIBRATION=true + SESSION_MODE=core5 was checked end-to-end for correctness (reports/e2e_http_check.md) but not re-simulated.
- The simulation uses synthetic students; it validates decision logic only (see reports/simulation.md).
- The default configuration (no demo.env) is the original behaviour: staircase, full10, no calibration.
