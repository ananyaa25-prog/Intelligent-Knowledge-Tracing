"""
scripts/start_demo.py
---------------------
Starts the demo: loads demo.env (gate-selected configuration), starts the
FastAPI inference service and the Node proxy, prints the URLs and keeps
running until Ctrl+C. Uses the real database (data/processed/kt_sessions.db)
unless KT_DB_PATH is set.

Usage:
  python scripts/start_demo.py                  # demo.env configuration
  python scripts/start_demo.py --defaults       # original behaviour (ignore demo.env)
  python scripts/start_demo.py --py-port 8000 --node-port 3000
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from serving.policy_config import load_env_file
from _services import Services


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", action="store_true", help="ignore demo.env")
    ap.add_argument("--env-file", default=str(PROJECT_ROOT / "demo.env"))
    ap.add_argument("--py-port", type=int, default=8000)
    ap.add_argument("--node-port", type=int, default=3000)
    args = ap.parse_args()
    env = {} if args.defaults else load_env_file(args.env_file)
    print("Configuration:", env or "defaults (staircase, full10, no calibration)")
    svc = Services(env, py_port=args.py_port, node_port=args.node_port, quiet=False)
    try:
        svc.start()
    except RuntimeError as e:
        print(f"Could not start: {e}. Are ports {args.py_port}/{args.node_port} free? Is .venv / node installed?")
        return 1
    base = svc.base
    print("\n" + "=" * 60)
    print(f"  Student quiz        : {base}/")
    print(f"  Teacher dashboard   : {base}/dashboard.html")
    print(f"  Model internals     : {base}/model-internals.html")
    print(f"  FastAPI docs        : http://127.0.0.1:{args.py_port}/docs")
    print("=" * 60)
    print("Press Ctrl+C to stop.")
    try:
        while all(p.poll() is None for p in svc.procs):
            time.sleep(1)
        print("A service exited unexpectedly.")
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
