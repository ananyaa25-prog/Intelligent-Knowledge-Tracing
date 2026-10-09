"""
scripts/visual_qa.py
--------------------
Starts the real services on a TEMPORARY copy of the DB (never the real one),
optionally with a demo configuration and seeded simulated students, then runs
scripts/visual_qa.js in headless Chrome/Edge. Screenshots -> reports/screenshots/.

Prerequisite (once):  npm install --prefix .tools/qa puppeteer-core@23
Usage: python scripts/visual_qa.py [--env-file demo.env] [--seed-sim]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, REPORTS_DIR, find_node
from _services import Services, temp_db_copy

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--seed-sim", action="store_true", help="seed simulated sim_ students into the temp DB first")
    args = ap.parse_args()
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        print("No Chrome/Edge found; visual QA cannot run.")
        return 2
    if not (PROJECT_ROOT / ".tools" / "qa" / "node_modules" / "puppeteer-core").exists():
        print("puppeteer-core missing: npm install --prefix .tools/qa puppeteer-core@23")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="kt_vqa_"))
    db = temp_db_copy(tmp)
    env = {"KT_DB_PATH": str(db), "KT_LOG_PATH": str(tmp / "logs.jsonl")}
    if args.env_file:
        from serving.policy_config import load_env_file
        env.update(load_env_file(PROJECT_ROOT / args.env_file))
    if args.seed_sim:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "seed_demo_data.py")],
                       env={**os.environ, **env}, check=True)
    out = REPORTS_DIR / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    with Services(env, log_dir=tmp) as svc:
        r = subprocess.run([find_node(), str(PROJECT_ROOT / "scripts" / "visual_qa.js"), svc.base, str(out), browser],
                           capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-6000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
