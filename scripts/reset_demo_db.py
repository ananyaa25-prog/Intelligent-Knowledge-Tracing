"""
scripts/reset_demo_db.py
------------------------
Task 11: Demo Hygiene & Database Maintenance Script.

Purges all students, quiz sessions, and interaction logs from the SQLite database,
returning it to a clean state for live demos.

SAFETY: Requires the explicit `--confirm` flag. Without `--confirm`, the script
strictly refuses to run and exits with code 1 to prevent accidental data loss.
"""

import sys
import argparse
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.db import Database, DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Safely reset the GatedKT demo database (students, sessions, interactions)."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required confirmation flag to execute the database wipe."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to the SQLite database file (default: {DEFAULT_DB_PATH})"
    )

    args = parser.parse_args()

    if not args.confirm:
        print("\n" + "=" * 60)
        print("ERROR: Database reset aborted -- missing confirmation flag!")
        print("=" * 60)
        print("This operation permanently purges all students, sessions, and interactions.")
        print("To proceed, you MUST pass the explicit flag:\n")
        print("    python scripts/reset_demo_db.py --confirm")
        print("=" * 60 + "\n")
        sys.exit(1)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"[ResetDB] Database file does not exist at {db_path}. Initializing clean database...")
        db = Database(db_path)
        print("[ResetDB] Database initialized successfully.")
        return

    print(f"[ResetDB] Connecting to database at {db_path}...")
    db = Database(db_path)
    stats = db.reset_database()

    print("\n" + "=" * 60)
    print("DATABASE RESET SUCCESSFUL")
    print("=" * 60)
    print(f"  • Deleted Interactions: {stats['deleted_interactions']}")
    print(f"  • Deleted Sessions:     {stats['deleted_sessions']}")
    print(f"  • Deleted Students:     {stats['deleted_students']}")
    print(f"  • Database File:        {db_path}")
    print("  • Status:               Clean & Ready for Live Usage")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
