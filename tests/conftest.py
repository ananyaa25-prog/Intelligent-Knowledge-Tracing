"""
tests/conftest.py
-----------------
Test isolation: before any test module imports the serving code, point the
SQLite DB (KT_DB_PATH) and the JSONL session log (KT_LOG_PATH) at temporary
copies, so the test suite never writes to data/processed/kt_sessions.db or
data/processed/session_logs.jsonl. The copy of the DB is made with the SQLite
backup API (safe with WAL). Set KT_TESTS_USE_REAL_DB=1 to opt out.
"""

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DB = PROJECT_ROOT / "data" / "processed" / "kt_sessions.db"
REAL_LOG = PROJECT_ROOT / "data" / "processed" / "session_logs.jsonl"

if os.environ.get("KT_TESTS_USE_REAL_DB") != "1" and not os.environ.get("KT_DB_PATH"):
    _tmp = Path(tempfile.mkdtemp(prefix="kt_tests_"))
    _db = _tmp / "kt_sessions.db"
    if REAL_DB.exists():
        src = sqlite3.connect(str(REAL_DB))
        dst = sqlite3.connect(str(_db))
        src.backup(dst)
        dst.close()
        src.close()
    _log = _tmp / "session_logs.jsonl"
    if REAL_LOG.exists():
        shutil.copy2(REAL_LOG, _log)
    os.environ["KT_DB_PATH"] = str(_db)
    os.environ["KT_LOG_PATH"] = str(_log)
