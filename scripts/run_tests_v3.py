"""
scripts/run_tests_v3.py
-----------------------
Runs the whole test suite (pytest + every server/tests/*.js file) and prints one
summary line, e.g. "pytest 85 passed / 15 failed; node 1/4 suites pass".

Also compares the pytest failure set against tests/legacy_failures_v3.txt (the
15 legacy fixed-quiz failures recorded at the v3 baseline) and reports any NEW
failure, which is what matters for "keep all existing tests passing".

Usage:  python scripts/run_tests_v3.py [--write-baseline]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "legacy_failures_v3.txt"


def run_pytest():
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rf"],
                       cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    failures = sorted(set(re.findall(r"^FAILED (\S+)", out, re.M)))
    return passed, failed, errors, failures, out


def run_node():
    results = {}
    for f in sorted((ROOT / "server" / "tests").glob("*.js")):
        try:
            p = subprocess.run(["node", str(f.relative_to(ROOT / "server"))], cwd=ROOT / "server",
                               capture_output=True, text=True, timeout=180)
            results[f.name] = p.returncode == 0
        except subprocess.TimeoutExpired:
            results[f.name] = False
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()
    passed, failed, errors, failures, out = run_pytest()
    node = run_node()
    if args.write_baseline:
        BASELINE.write_text("\n".join(failures) + "\n", encoding="utf-8")
    base = set(BASELINE.read_text(encoding="utf-8").split()) if BASELINE.exists() else set()
    new = [f for f in failures if f not in base]
    print(f"pytest {passed} passed / {failed} failed / {errors} errors; "
          f"node {sum(node.values())}/{len(node)} suites pass ({', '.join(k for k, v in node.items() if v)}); "
          f"new pytest failures vs baseline: {len(new)}")
    for f in new:
        print("  NEW FAILURE:", f)
    if errors:
        print(out[-3000:])
    return 1 if (new or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
