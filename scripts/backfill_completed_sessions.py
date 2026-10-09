"""
scripts/backfill_completed_sessions.py
--------------------------------------
Before the fix of D28, finished adaptive sessions were never marked completed,
so they were missing from "completed sessions" counts and from the
Student x Skill Mastery Matrix. This marks such sessions as completed:
set_name = 'adaptive', completed_at IS NULL and at least --min-answers answers
(default 15 = the shortest finished session). completed_at is set to the time
of the session's last answer. Unfinished sessions are left untouched.

Default is a dry run (prints what would change). With --apply it first backs the
DB up with the SQLite backup API to data/backups/, then writes.

Usage:
  python scripts/backfill_completed_sessions.py            # dry run
  python scripts/backfill_completed_sessions.py --apply    # back up, then update
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.environ["KT_DB_PATH"]) if os.environ.get("KT_DB_PATH") else PROJECT_ROOT / "data" / "processed" / "kt_sessions.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-answers", type=int, default=15)
    args = ap.parse_args()
    conn = sqlite3.connect(str(DB), timeout=10)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT s.session_id, s.student_id, COUNT(i.id) AS n, MAX(i.recorded_at) AS last_at
           FROM sessions s JOIN interactions i ON i.session_id = s.session_id
           WHERE s.set_name = 'adaptive' AND s.completed_at IS NULL
           GROUP BY s.session_id HAVING COUNT(i.id) >= ?
           ORDER BY s.student_id, last_at""", (args.min_answers,)).fetchall()
    print(f"DB: {DB}")
    print(f"{len(rows)} finished adaptive session(s) not marked completed:")
    for r in rows:
        print(f"  student {r['student_id']!r:24} session {r['session_id'][:8]}  answers {r['n']}  last answer {r['last_at']}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to update (a backup is made first).")
        return 0
    if not rows:
        return 0
    backups = PROJECT_ROOT / "data" / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    dst_path = backups / f"kt_sessions_pre_backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    dst = sqlite3.connect(str(dst_path))
    conn.backup(dst)
    dst.close()
    print(f"Backup written: {dst_path}")
    with conn:
        for r in rows:
            conn.execute("UPDATE sessions SET completed_at = ? WHERE session_id = ? AND completed_at IS NULL",
                         (r["last_at"], r["session_id"]))
    print(f"Marked {len(rows)} session(s) as completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
