"""
src/serving/db.py
-----------------
Task 4: SQLite persistence layer for students, sessions, and interactions.

Stores:
- students: persistent student identities and creation timestamps.
- sessions: quiz sessions linked to students, assigned set (A-E), start and completion times.
- interactions: step-by-step quiz interaction logs with raw_skill_id, correctness,
  response time, attempt count, and hint usage.

Uses Python's standard sqlite3 library with synchronous writes and WAL journal mode.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Default database path in data/processed/
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "kt_sessions.db"
# Optional override (policy_config KT_DB_PATH), e.g. tests or demo seeding on a copy.
if os.environ.get("KT_DB_PATH"):
    DEFAULT_DB_PATH = Path(os.environ["KT_DB_PATH"])


class Database:
    """Manages SQLite database connections, schema initialization, and queries."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def init_db(self):
        """Creates schema tables and indexes if they do not already exist, and applies migrations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'live'
                );
            """)

            # Migration: Ensure 'source' column exists if table was created in an earlier version
            cursor.execute("PRAGMA table_info(students);")
            columns = [col["name"] for col in cursor.fetchall()]
            if "source" not in columns:
                cursor.execute("ALTER TABLE students ADD COLUMN source TEXT DEFAULT 'live';")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    set_name TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    skill_id INTEGER NOT NULL,
                    correct INTEGER NOT NULL,
                    response_time_ms REAL NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    hint_used INTEGER NOT NULL,
                    difficulty_bucket INTEGER,
                    level TEXT,
                    p_before REAL,
                    is_probe INTEGER,
                    question_text TEXT,
                    correct_answer TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                );
            """)

            # Migration: Ensure question_text and correct_answer columns exist
            cursor.execute("PRAGMA table_info(interactions);")
            i_columns = [col["name"] for col in cursor.fetchall()]
            if "question_text" not in i_columns:
                cursor.execute("ALTER TABLE interactions ADD COLUMN question_text TEXT;")
            if "correct_answer" not in i_columns:
                cursor.execute("ALTER TABLE interactions ADD COLUMN correct_answer TEXT;")
            if "difficulty_bucket" not in i_columns:
                cursor.execute("ALTER TABLE interactions ADD COLUMN difficulty_bucket INTEGER;")
            if "level" not in i_columns:
                cursor.execute("ALTER TABLE interactions ADD COLUMN level TEXT;")
            if "p_before" not in i_columns:
                cursor.execute("ALTER TABLE interactions ADD COLUMN p_before REAL;")
            if "is_probe" not in i_columns:
                cursor.execute("ALTER TABLE interactions ADD COLUMN is_probe INTEGER;")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_student ON interactions(student_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id);")
            conn.commit()

    def get_or_create_student(self, student_id: str, *, source: str) -> Dict[str, Any]:
        """Gets existing student or registers a new student_id with appropriate source tag."""
        clean_id = student_id.strip()
        if not clean_id:
            clean_id = "guest_student"

        clean_source = str(source).strip().lower() if source else ""
        if not clean_source or clean_source not in {"live", "sample", "test"}:
            raise ValueError(f"Invalid source '{source}': 'source' is required and must be one of {{'live', 'sample', 'test'}}.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT student_id, created_at, source FROM students WHERE student_id = ?", (clean_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)

            cursor.execute("INSERT INTO students (student_id, source) VALUES (?, ?)", (clean_id, clean_source))
            conn.commit()
            cursor.execute("SELECT student_id, created_at, source FROM students WHERE student_id = ?", (clean_id,))
            return dict(cursor.fetchone())

    def get_student_summary(self, student_id: str) -> Dict[str, Any]:
        """Returns summary metadata for a student: total sessions, completed sessions, and interaction count."""
        clean_id = student_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT student_id, created_at, source FROM students WHERE student_id = ?", (clean_id,))
            student_row = cursor.fetchone()
            if not student_row:
                return {
                    "student_id": clean_id,
                    "exists": False,
                    "source": "live",
                    "session_count": 0,
                    "completed_session_count": 0,
                    "total_interactions": 0,
                    "created_at": None,
                }

            cursor.execute("SELECT COUNT(*) as total, COUNT(completed_at) as completed FROM sessions WHERE student_id = ?", (clean_id,))
            sess_stats = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) as total_inter FROM interactions WHERE student_id = ?", (clean_id,))
            inter_stats = cursor.fetchone()

            return {
                "student_id": clean_id,
                "exists": True,
                "source": student_row["source"] or "live",
                "session_count": sess_stats["total"] if sess_stats else 0,
                "completed_session_count": sess_stats["completed"] if sess_stats else 0,
                "total_interactions": inter_stats["total_inter"] if inter_stats else 0,
                "created_at": student_row["created_at"],
            }

    def create_session(self, session_id: str, student_id: str, set_name: str = "procedural", *, source: str) -> Dict[str, Any]:
        """Creates a new session for student_id."""
        clean_source = str(source).strip().lower() if source else ""
        if not clean_source or clean_source not in {"live", "sample", "test"}:
            raise ValueError(f"Invalid source '{source}': 'source' is required and must be one of {{'live', 'sample', 'test'}}.")
        self.get_or_create_student(student_id, source=clean_source)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, student_id, set_name) VALUES (?, ?, ?)",
                (session_id, student_id, set_name)
            )
            conn.commit()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            return dict(cursor.fetchone())

    def record_interaction(
        self,
        session_id: str,
        student_id: str,
        position: int,
        skill_id: int,
        correct: int,
        response_time_ms: float,
        attempt_count: int,
        hint_used: int = 0,
        question_text: Optional[str] = None,
        correct_answer: Optional[str] = None
    ) -> int:
        """Synchronously records an interaction row into the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO interactions (
                    session_id, student_id, position, skill_id,
                    correct, response_time_ms, attempt_count, hint_used,
                    question_text, correct_answer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, student_id, position, skill_id, correct, response_time_ms, attempt_count, hint_used, question_text, correct_answer)
            )
            conn.commit()
            return cursor.lastrowid

    def record_adaptive_interaction(
        self,
        session_id: str,
        student_id: str,
        interaction: Dict[str, Any],
    ) -> int:
        """Records an adaptive-session interaction with difficulty/level/p_before/probe metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO interactions (
                    session_id, student_id, position, skill_id,
                    correct, response_time_ms, attempt_count, hint_used,
                    difficulty_bucket, level, p_before, is_probe,
                    question_text, correct_answer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, student_id, interaction["position"], interaction["skill_id"],
                    interaction["correct"], interaction["response_time_ms"],
                    interaction.get("attempt_count", 1), interaction.get("hint_used", 0),
                    interaction.get("difficulty_bucket"), interaction.get("level"),
                    interaction.get("p_before"), interaction.get("is_probe", 0),
                    interaction.get("question_text"), interaction.get("correct_answer"),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_last_question_text_for_position(self, student_id: str, position: int) -> Optional[str]:
        """Returns the question_text from the most recent interaction for this student at this position."""
        clean_id = student_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT question_text
                FROM interactions
                WHERE student_id = ? AND position = ? AND question_text IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (clean_id, position)
            )
            row = cursor.fetchone()
            return row["question_text"] if row else None

    def complete_session(self, session_id: str):
        """Marks a session as completed by setting completed_at to the current timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET completed_at = CURRENT_TIMESTAMP WHERE session_id = ? AND completed_at IS NULL",
                (session_id,)
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session metadata by session_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_session_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns all interaction records for a specific session in position order."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT skill_id, correct, response_time_ms, attempt_count, hint_used, position,
                       difficulty_bucket, level, p_before, is_probe, recorded_at
                FROM interactions
                WHERE session_id = ?
                ORDER BY position ASC, id ASC
                """,
                (session_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "skill_id": int(r["skill_id"]),
                    "topic": int(r["skill_id"]),
                    "correct": int(r["correct"]),
                    "response_time_ms": float(r["response_time_ms"]),
                    "attempt_count": int(r["attempt_count"]),
                    "hint_used": int(r["hint_used"]),
                    "position": int(r["position"]),
                    "difficulty_bucket": int(r["difficulty_bucket"]) if r["difficulty_bucket"] is not None else 0,
                    "level": r["level"],
                    "p_before": float(r["p_before"]) if r["p_before"] is not None else None,
                    "is_probe": int(r["is_probe"]) if r["is_probe"] is not None else 0,
                    "recorded_at": r["recorded_at"],
                }
                for r in rows
            ]

    def get_full_history(self, student_id: str) -> List[Dict[str, Any]]:
        """
        Returns ALL interactions for a student across all their sessions,
        ordered strictly chronologically by interaction insertion sequence (i.id ASC).
        """
        clean_id = student_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT i.skill_id, i.correct, i.response_time_ms, i.attempt_count, i.hint_used,
                       i.position, i.session_id, i.difficulty_bucket, i.level, i.p_before,
                       i.is_probe, s.started_at, i.recorded_at
                FROM interactions i
                JOIN sessions s ON i.session_id = s.session_id
                WHERE i.student_id = ?
                ORDER BY i.id ASC
                """,
                (clean_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "skill_id": int(r["skill_id"]),
                    "correct": int(r["correct"]),
                    "response_time_ms": float(r["response_time_ms"]),
                    "attempt_count": int(r["attempt_count"]),
                    "hint_used": int(r["hint_used"]),
                    "session_id": str(r["session_id"]),
                    "difficulty_bucket": int(r["difficulty_bucket"]) if r["difficulty_bucket"] is not None else 0,
                    "level": r["level"],
                    "p_before": float(r["p_before"]) if r["p_before"] is not None else None,
                    "is_probe": int(r["is_probe"]) if r["is_probe"] is not None else 0,
                    "recorded_at": r["recorded_at"],
                }
                for r in rows
            ]

    def get_student_session_boundaries(self, student_id: str) -> List[Dict[str, Any]]:
        """
        Returns ordered completed sessions for student_id with each session's interactions
        and completion timestamp for historical trend analysis.
        """
        clean_id = student_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, set_name, started_at, completed_at
                FROM sessions
                WHERE student_id = ? AND completed_at IS NOT NULL
                ORDER BY completed_at ASC, started_at ASC
                """,
                (clean_id,)
            )
            sess_rows = cursor.fetchall()

            sessions_list = []
            for s_row in sess_rows:
                sid = s_row["session_id"]
                interactions = self.get_session_interactions(sid)
                sessions_list.append({
                    "session_id": sid,
                    "set_name": s_row["set_name"],
                    "started_at": s_row["started_at"],
                    "completed_at": s_row["completed_at"],
                    "interactions": interactions,
                })
            return sessions_list

    def get_student_skill_interaction_counts(self, student_id: str) -> Dict[str, int]:
        """
        Returns a dictionary mapping str(skill_id) -> total interaction count for that student.
        """
        clean_id = student_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT skill_id, COUNT(*) as cnt
                FROM interactions
                WHERE student_id = ?
                GROUP BY skill_id
                """,
                (clean_id,)
            )
            rows = cursor.fetchall()
            return {str(r["skill_id"]): int(r["cnt"]) for r in rows}

    def get_all_completed_students(self, source: Optional[str] = "live") -> List[Dict[str, Any]]:
        """
        Returns a list of students who have completed at least one quiz session,
        optionally filtered by source ('live', 'test', or None/'all' for all).
        """
        query = """
            SELECT s.student_id,
                   s.source,
                   COUNT(DISTINCT sess.session_id) as completed_sessions,
                   COUNT(i.id) as total_interactions,
                   MIN(sess.started_at) as first_session_at,
                   MAX(sess.completed_at) as last_completed_at
            FROM students s
            JOIN sessions sess ON s.student_id = sess.student_id
            LEFT JOIN interactions i ON s.student_id = i.student_id
            WHERE sess.completed_at IS NOT NULL
        """
        params = []

        if source == "live":
            query += " AND (s.source = 'live' OR s.source IS NULL) AND s.student_id NOT LIKE 'test_%' AND s.student_id NOT LIKE 'synthetic_%'"
        elif source == "test":
            query += " AND (s.source = 'test' OR s.student_id LIKE 'test_%' OR s.student_id LIKE 'synthetic_%')"
        # If source is "all" or None, no additional filtering

        query += """
            GROUP BY s.student_id
            ORDER BY last_completed_at DESC
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_attempt_count_for_position(self, session_id: str, position: int) -> int:
        """Returns the number of recorded attempts for a given session position."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM interactions WHERE session_id = ? AND position = ?",
                (session_id, position)
            )
            row = cursor.fetchone()
            return row["count"] if row else 0

    def reset_database(self) -> Dict[str, int]:
        """
        Purges all rows from interactions, sessions, and students tables,
        and performs a database vacuum.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM interactions;")
            i_cnt = cursor.fetchone()["cnt"]
            cursor.execute("SELECT COUNT(*) as cnt FROM sessions;")
            s_cnt = cursor.fetchone()["cnt"]
            cursor.execute("SELECT COUNT(*) as cnt FROM students;")
            st_cnt = cursor.fetchone()["cnt"]

            cursor.execute("DELETE FROM interactions;")
            cursor.execute("DELETE FROM sessions;")
            cursor.execute("DELETE FROM students;")
            conn.commit()

        # VACUUM outside transaction
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("VACUUM;")
        conn.close()

        return {
            "deleted_interactions": i_cnt,
            "deleted_sessions": s_cnt,
            "deleted_students": st_cnt,
        }


# Default global database instance
db = Database()

