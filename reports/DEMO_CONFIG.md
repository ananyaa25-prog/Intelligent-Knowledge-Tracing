# Demo configuration (decision gates)

Generated 2026-10-09 05:01 UTC by `scripts/apply_gates.py`. Gates were fixed before the results existed and are applied mechanically.

| Gate | Rule | Numbers | Result | Demo setting |
|---|---|---|---|---|
| M | violation_rate <= MONOTONE_WARN | violation_rate = 0.0% (MONOTONE_WARN 10%) | PASS | profile_reliable = true |
| S | Gate M and model in-band >= staircase - 0.02 and model status acc >= staircase - 0.03 | model in-band 57.6% vs staircase 32.6% (need >= 30.6%: True); model status accuracy 45.6% vs staircase 40.3% (need >= 37.3%: True); Gate M True | PASS | POLICY = model |
| C | ECE improves >= 10% relative and AUC unchanged | ECE 0.0249 -> 0.0117 (53% relative); AUC 0.8946 -> 0.8946 (unchanged: True); T = 1.16696 | PASS | APPLY_CALIBRATION = true |
| K | T8c DONE and core5 end-to-end check passes | T8c DONE in BUILD_LOG: True; core5 end-to-end HTTP check passed: True | PASS | SESSION_MODE = core5 |

Resulting `demo.env`:

```
POLICY=model
TARGET_P=0.70
APPLY_CALIBRATION=true
SESSION_MODE=core5
```

Caveats:
- Gate S was evaluated in simulation with full10 mode and uncalibrated predictions; the combination POLICY=model + APPLY_CALIBRATION=true + SESSION_MODE=core5 was checked end-to-end for correctness (reports/e2e_http_check.md) but not re-simulated.
- The simulation uses synthetic students; it validates decision logic only (see reports/simulation.md).
- The default configuration (no demo.env) is the original behaviour: staircase, full10, no calibration.
