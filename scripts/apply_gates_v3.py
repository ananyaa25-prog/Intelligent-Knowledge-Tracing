"""
scripts/apply_gates_v3.py
-------------------------
Applies the v3 gates pre-registered in reports/v3/DEMO_CONFIG.md (Part 1) mechanically
and writes demo.env + Part 2 of reports/v3/DEMO_CONFIG.md.

Gate P (difficulty policy): candidates uncertainty, model; a candidate passes if Gate M
  (profile_reliable) passes AND in-band >= staircase - 0.02 AND pooled status accuracy >=
  staircase - 0.03 in the primary simulation setting (reports/v3/policy_comparison_summary.json).
  If both pass: higher pooled status accuracy (tie: uncertainty). None: staircase.
Gate Q (question source): mixed only if tests/test_question_generator.py passes AND the
  end-to-end HTTP check passes with QUESTION_SOURCE=mixed (reports/v3/e2e_http_check.json,
  configurations "mixed" and "demo_mixed"); else bank.
Other demo.env values are kept from the existing demo.env (v2 gates).

Usage: python scripts/apply_gates_v3.py
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from serving.policy_config import load_env_file  # noqa: E402

OUT = ROOT / "reports" / "v3"
DEMO_ENV = ROOT / "demo.env"
CONFIG_MD = OUT / "DEMO_CONFIG.md"
PART2 = "## Part 2 - Outcomes"


def main():
    summ = json.loads((OUT / "policy_comparison_summary.json").read_text(encoding="utf-8"))["primary"]
    s = summ["summary"]
    gate_m = bool(summ["reliable"])
    st_in, st_acc = s["staircase"]["inband"]["mean"], s["staircase"]["status_acc_pooled"]
    rows, passing = [], []
    for cand in ("uncertainty", "model"):
        c_in, c_acc = s[cand]["inband"]["mean"], s[cand]["status_acc_pooled"]
        ok_in, ok_acc = c_in >= st_in - 0.02, c_acc >= st_acc - 0.03
        ok = gate_m and ok_in and ok_acc
        rows.append(f"| P ({cand}) | Gate M and in-band >= staircase - 0.02 and pooled status accuracy >= staircase - 0.03 | "
                    f"in-band {c_in:.1%} vs need >= {st_in - 0.02:.1%} ({ok_in}); status accuracy {c_acc:.1%} vs need >= "
                    f"{st_acc - 0.03:.1%} ({ok_acc}); Gate M {gate_m} | {'PASS' if ok else 'FAIL'} |")
        if ok:
            passing.append((c_acc, cand == "uncertainty", cand))
    policy = max(passing)[2] if passing else "staircase"

    t = subprocess.run([sys.executable, "-m", "pytest", "tests/test_question_generator.py", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    gen_ok = t.returncode == 0
    gen_line = (t.stdout.strip().splitlines() or ["no output"])[-1]
    e2e_path = OUT / "e2e_http_check.json"
    e2e = json.loads(e2e_path.read_text(encoding="utf-8")) if e2e_path.exists() else []
    mixed = [r for r in e2e if r["config"] in ("mixed", "demo_mixed")]
    e2e_ok = bool(mixed) and all(r["passed"] for r in mixed)
    qsrc = "mixed" if (gen_ok and e2e_ok) else "bank"
    e2e_txt = ", ".join(f"{r['config']} {'PASS' if r['passed'] else 'FAIL'} ({r['n_checks']} checks)" for r in mixed)
    rows.append(f"| Q | generator tests pass and e2e HTTP check passes with QUESTION_SOURCE=mixed | "
                f"generator tests: {gen_line}; e2e: {e2e_txt or 'not run'} | {'PASS' if qsrc == 'mixed' else 'FAIL'} |")

    env = load_env_file(DEMO_ENV)
    env["POLICY"] = policy
    env["QUESTION_SOURCE"] = qsrc
    order = ["POLICY", "TARGET_P", "APPLY_CALIBRATION", "SESSION_MODE", "TOPIC_SELECTION", "QUESTION_SOURCE"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# demo.env - generated {stamp} by scripts/apply_gates_v3.py from the pre-registered v3 gates.",
             "# See reports/v3/DEMO_CONFIG.md (v3) and reports/DEMO_CONFIG.md (v2) for the numbers. Read by scripts/start_demo.py.",
             "# Delete this file (or set values to the defaults) to get the original behaviour."]
    lines += [f"{k}={env[k]}" for k in order if k in env] + [f"{k}={v}" for k, v in env.items() if k not in order]
    DEMO_ENV.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))   # demo.env is CRLF

    md = CONFIG_MD.read_text(encoding="utf-8")
    md = md[: md.index(PART2)] if PART2 in md else md
    md += "\n".join([PART2, "", f"Applied {stamp} by `python scripts/apply_gates_v3.py`. Numbers from "
                     "reports/v3/policy_comparison.md (primary setting) and reports/v3/e2e_http_check.md.", "",
                     "| Gate | Rule | Numbers | Result |", "|---|---|---|---|"] + rows + [
        "", f"Staircase reference: in-band {st_in:.1%}, pooled status accuracy {st_acc:.1%}.", "",
        f"**Demo POLICY = {policy}; QUESTION_SOURCE = {qsrc}.**", "",
        (f"Note on the selected policy: in the primary simulation the uncertainty-aware policy fell back to the staircase "
         f"level on {s['uncertainty']['low_conf_per_session']:.1f} of {s['uncertainty']['main_decisions_per_session']:.1f} "
         "level decisions per session (ensemble spread > 0.15), so in practice it behaves mostly like the staircase and "
         "flags those questions as low-confidence predictions." if policy == "uncertainty" else ""), "",
        "Resulting `demo.env`:", "", "```"] + [l for l in lines if not l.startswith("#")] + ["```", "",
        "Bug fixes after pre-registration: (1) the first simulate_v3.py run crashed while writing the markdown "
        "report (a str.format KeyError) before any result was shown or saved; the format bug was fixed, the JSON is "
        "now saved before the report, and the run was repeated unchanged otherwise. (2) Section sessions in "
        "evidence-only mode no longer print neutral 50% placeholders in the decision text (affects sections only, "
        "not this simulation). No policy, threshold or student parameter was changed.", "",
        "Significance: the gate is a non-inferiority rule. In the primary setting the in-band gain over the "
        "staircase is significant for both candidates, but no status-accuracy difference is (Holm-corrected "
        "p = 1.0 for every policy; reports/v3/policy_comparison.md). Passing the gate means 'not worse by more than "
        "the margin', not 'better'.", "",
        "Caveats: the simulation uses synthetic students and validates decision logic only; the Elo level "
        "difficulties learned from ASSISTments (-1.12 / 0 / +1.16) happen to be close to the simulator's "
        "generating values (-1 / 0 / +1), which favours the Elo policy in this simulation; Elo and max-information "
        "were comparison policies and could not be selected for the demo by the pre-registered rule.", ""])
    CONFIG_MD.write_text(md, encoding="utf-8")
    print(f"[gates] POLICY={policy} QUESTION_SOURCE={qsrc}; demo.env and reports/v3/DEMO_CONFIG.md written")


if __name__ == "__main__":
    main()
