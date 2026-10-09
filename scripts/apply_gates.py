"""
scripts/apply_gates.py
----------------------
Applies the pre-registered decision gates MECHANICALLY from the generated
reports and writes demo.env + reports/DEMO_CONFIG.md.

  Gate M: profile_reliable = violation_rate <= MONOTONE_WARN   (data/processed/profile_reliability.json)
  Gate S: POLICY=model iff Gate M AND model in-band >= staircase in-band - 0.02
          AND model status accuracy >= staircase status accuracy - 0.03     (reports/simulation_summary.json)
  Gate C: APPLY_CALIBRATION=true iff ECE improves >= 10% relative and AUC unchanged (data/processed/calibration.json)
  Gate K: SESSION_MODE=core5 iff T8c is DONE and the core5 end-to-end HTTP check passed (reports/e2e_http_check.json)
  Rule T (added after user testing, D31; NOT pre-registered): TOPIC_SELECTION=evidence_blend iff, in the
          full10 simulation with the policy chosen by Gate S, blended topic selection gives status accuracy
          >= the original rule - 0.03 and in-band share >= the original - 0.02
          (reports/simulation_summary.json vs reports/simulation_summary_blend_full10.json)

Usage: python scripts/apply_gates.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    rel = load(PROCESSED_DIR / "profile_reliability.json")
    sim = load(REPORTS_DIR / "simulation_summary.json")
    cal = load(PROCESSED_DIR / "calibration.json")
    e2e = load(REPORTS_DIR / "e2e_http_check.json")
    build_log = (PROJECT_ROOT / "docs" / "BUILD_LOG.md").read_text(encoding="utf-8") if (PROJECT_ROOT / "docs" / "BUILD_LOG.md").exists() else ""

    # Gate M
    gate_m = bool(rel and rel.get("violation_rate") is not None and rel.get("reliable"))
    m_txt = (f"violation_rate = {rel['violation_rate']:.1%} (MONOTONE_WARN {rel.get('monotone_warn', 0.10):.0%})"
             if rel and rel.get("violation_rate") is not None else "monotonicity check did not run")

    # Gate S
    gate_s = False
    s_txt = "simulation summary missing"
    if sim:
        sm, ss = sim["summary"]["model"], sim["summary"]["staircase"]
        c1 = sm["inband_share"] >= ss["inband_share"] - 0.02
        c2 = sm["status_accuracy"] >= ss["status_accuracy"] - 0.03
        gate_s = bool(gate_m and c1 and c2)
        s_txt = (f"model in-band {sm['inband_share']:.1%} vs staircase {ss['inband_share']:.1%} (need >= {ss['inband_share'] - 0.02:.1%}: {c1}); "
                 f"model status accuracy {sm['status_accuracy']:.1%} vs staircase {ss['status_accuracy']:.1%} "
                 f"(need >= {ss['status_accuracy'] - 0.03:.1%}: {c2}); Gate M {gate_m}")

    # Gate C
    gate_c = False
    c_txt = "calibration report missing"
    if cal:
        auc_same = abs(cal["auc_before"] - cal["auc_after"]) < 1e-9
        gate_c = bool(cal["ece_relative_improvement"] >= 0.10 and auc_same)
        c_txt = (f"ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f} ({cal['ece_relative_improvement']:.0%} relative); "
                 f"AUC {cal['auc_before']:.4f} -> {cal['auc_after']:.4f} (unchanged: {auc_same}); T = {cal['temperature']}")

    # Gate K
    t8c_done = "| T8c" in build_log and "| T8c session mode core5 | DONE" in build_log
    core5_pass = False
    if e2e:
        for r in e2e:
            if r.get("env", {}).get("SESSION_MODE") == "core5" and r.get("passed"):
                core5_pass = True
    gate_k = bool(t8c_done and core5_pass)
    k_txt = f"T8c DONE in BUILD_LOG: {t8c_done}; core5 end-to-end HTTP check passed: {core5_pass}"

    # Rule T (post hoc, after user testing)
    blend = load(REPORTS_DIR / "simulation_summary_blend_full10.json")
    pol = "model" if gate_s else "staircase"
    rule_t = False
    t_txt = "blended-selection simulation missing"
    if sim and blend:
        a, b = sim["summary"][pol], blend["summary"][pol]
        rule_t = bool(b["status_accuracy"] >= a["status_accuracy"] - 0.03 and b["inband_share"] >= a["inband_share"] - 0.02)
        t_txt = (f"{pol}: status accuracy {a['status_accuracy']:.1%} (original) vs {b['status_accuracy']:.1%} (blended); "
                 f"in-band {a['inband_share']:.1%} vs {b['inband_share']:.1%}")

    cfg = {
        "POLICY": pol,
        "TARGET_P": "0.70",
        "APPLY_CALIBRATION": "true" if gate_c else "false",
        "SESSION_MODE": "core5" if gate_k else "full10",
        "TOPIC_SELECTION": "evidence_blend" if rule_t else "model",
    }
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    env_lines = [f"# demo.env - generated {ts} by scripts/apply_gates.py from the pre-registered gates.",
                 "# See reports/DEMO_CONFIG.md for the numbers. Read by scripts/start_demo.py.",
                 "# Delete this file (or set values to the defaults) to get the original behaviour."]
    env_lines += [f"{k}={v}" for k, v in cfg.items()]
    (PROJECT_ROOT / "demo.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    def yn(b):
        return "PASS" if b else "FAIL"

    L = ["# Demo configuration (decision gates)", "",
         f"Generated {ts} by `scripts/apply_gates.py`. Gates were fixed before the results existed and are applied mechanically.", "",
         "| Gate | Rule | Numbers | Result | Demo setting |", "|---|---|---|---|---|",
         f"| M | violation_rate <= MONOTONE_WARN | {m_txt} | {yn(gate_m)} | profile_reliable = {str(gate_m).lower()} |",
         f"| S | Gate M and model in-band >= staircase - 0.02 and model status acc >= staircase - 0.03 | {s_txt} | {yn(gate_s)} | POLICY = {cfg['POLICY']} |",
         f"| C | ECE improves >= 10% relative and AUC unchanged | {c_txt} | {yn(gate_c)} | APPLY_CALIBRATION = {cfg['APPLY_CALIBRATION']} |",
         f"| K | T8c DONE and core5 end-to-end check passes | {k_txt} | {yn(gate_k)} | SESSION_MODE = {cfg['SESSION_MODE']} |",
         f"| T (post hoc, after user testing) | blended topic selection not worse than the original in simulation | {t_txt} | {yn(rule_t)} | TOPIC_SELECTION = {cfg['TOPIC_SELECTION']} |",
         "", "Resulting `demo.env`:", "", "```"] + [f"{k}={v}" for k, v in cfg.items()] + ["```", ""]
    if not gate_s:
        L.append("**Model-guided difficulty did not beat the staircase in simulation; the demo uses the staircase. "
                 "The model still drives topic selection.**")
    L += ["Revision note: the level rules of the model policy were revised after user testing (D27) and the simulation "
          "was re-run; the earlier run (archived in reports/archive_v1/) passed Gate S, the re-run with the revised rules "
          "does not, so the demo now uses the staircase. Rule T was added after user testing and is labelled as such.", "",
          "Caveats:",
          "- Gate S was evaluated in simulation with full10 mode and uncalibrated predictions; the combination "
          f"POLICY={cfg['POLICY']} + APPLY_CALIBRATION={cfg['APPLY_CALIBRATION']} + SESSION_MODE={cfg['SESSION_MODE']} "
          "was checked end-to-end for correctness (reports/e2e_http_check.md) but not re-simulated.",
          "- The simulation uses synthetic students; it validates decision logic only (see reports/simulation.md).",
          "- The default configuration (no demo.env) is the original behaviour: staircase, full10, no calibration."]
    (REPORTS_DIR / "DEMO_CONFIG.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
