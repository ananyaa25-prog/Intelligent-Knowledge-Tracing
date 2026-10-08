"""
src/serving/session.py
----------------------
Task 0 & Task 4: Question bank serving, live session management, and SQLite persistence.

Maintains persistent session states for live quiz participants using SQLite.
Strictly serves questions in fixed skill order (positions 1-10)
from a randomly selected parallel set (A-E), verifies answers
server-side, and records interaction histories for downstream
knowledge tracing inference.
"""

import sys
import json
import uuid
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Base paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
QUESTION_BANK_PATH = PROJECT_ROOT / "question_bank.json"

if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from serving.db import Database, db as default_db
from serving.question_templates import generate_question


class QuestionBank:
    """Loads and indexes question_bank.json once at startup."""
    def __init__(self, path: Path = QUESTION_BANK_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Question bank not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.meta = data.get("_meta", {})
        # List of { position, skill_name, raw_skill_id }
        self.skill_sequence: List[Dict[str, Any]] = sorted(
            data.get("skill_sequence", []), key=lambda x: x["position"]
        )
        self.position_to_skill: Dict[int, Dict[str, Any]] = {
            s["position"]: s for s in self.skill_sequence
        }

        # Sets dict: "A", "B", "C", "D", "E"
        # Each is a list of { position, question_text, correct_answer }
        self.sets: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for set_name, q_list in data.get("sets", {}).items():
            self.sets[set_name] = {
                q["position"]: q for q in q_list
            }

        self.available_sets: List[str] = sorted(list(self.sets.keys()))
        if not self.available_sets:
            raise ValueError("No question sets found in question bank")

    def get_skill_for_position(self, position: int) -> Dict[str, Any]:
        if position not in self.position_to_skill:
            raise ValueError(f"Invalid position {position}: valid positions are 1..{len(self.skill_sequence)}")
        return self.position_to_skill[position]

    def get_question(self, set_name: str, position: int) -> Dict[str, Any]:
        if set_name in self.sets and position in self.sets[set_name]:
            return self.sets[set_name][position]
        if position in self.position_to_skill:
            return generate_question(position)
        raise ValueError(f"Invalid position {position}: valid positions are 1..{len(self.skill_sequence)}")


# Load once at module import
_question_bank = QuestionBank()


def check_answer(submitted: Any, correct: str) -> bool:
    """
    Evaluates whether the submitted answer matches the correct answer:
    1. Exact trimmed, case-insensitive string equality.
    2. Numeric equality within 0.01 tolerance for decimal/integer numbers.
    """
    if submitted is None:
        return False

    sub_str = str(submitted).strip().lower()
    corr_str = str(correct).strip().lower()

    if sub_str == corr_str:
        return True

    # Numeric comparison fallback
    try:
        sub_num = float(sub_str)
        corr_num = float(corr_str)
        return abs(sub_num - corr_num) <= 0.01
    except (ValueError, TypeError):
        pass

    return False


class SessionManager:
    """Session manager backed by SQLite database for persistence."""
    def __init__(self, bank: QuestionBank = _question_bank, database: Optional[Database] = None):
        self.bank = bank
        self.db = database or default_db
        # In-memory cache of active sessions
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(
        self,
        student_id: Optional[str] = None,
        session_id: Optional[str] = None,
        set_name: Optional[str] = None,
        *,
        source: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Starts a new session:
        - Associates with student_id (registers in SQLite if new).
        - Generates procedural questions for positions 1-10 with anti-repeat safeguard.
        - Inserts session into SQLite with explicit source tag ('live' or 'test').
        - Returns (session_id, questions) without revealing correct_answer.
        """
        clean_source = str(source).strip().lower() if source else ""
        if not clean_source or clean_source not in {"live", "sample", "test"}:
            raise ValueError(f"Invalid source '{source}': 'source' is required and must be one of {{'live', 'sample', 'test'}}.")

        sid = session_id or str(uuid.uuid4())
        st_id = (student_id or str(uuid.uuid4())[:8]).strip()
        if not st_id:
            st_id = "guest_student"

        sess_name = set_name or random.choice(self.bank.available_sets)
        self.db.create_session(session_id=sid, student_id=st_id, set_name=sess_name, source=clean_source)

        generated_questions: Dict[int, Dict[str, Any]] = {}
        client_questions: List[Dict[str, Any]] = []

        for pos_info in self.bank.skill_sequence:
            pos = pos_info["position"]
            skill_name = pos_info["skill_name"]
            raw_skill_id = pos_info["raw_skill_id"]

            if set_name and set_name in self.bank.sets:
                # If explicit legacy set requested (e.g. A-E)
                q_data = self.bank.get_question(set_name, pos)
            else:
                # Task 13: Procedural generation with anti-immediate-repeat safeguard
                last_q_text = self.db.get_last_question_text_for_position(st_id, pos)
                q_data = generate_question(pos)
                resample_attempts = 0
                while resample_attempts < 3 and last_q_text is not None and q_data["question_text"] == last_q_text:
                    q_data = generate_question(pos)
                    resample_attempts += 1

            generated_questions[pos] = {
                "position": pos,
                "skill_name": skill_name,
                "raw_skill_id": raw_skill_id,
                "question_text": q_data["question_text"],
                "correct_answer": q_data["correct_answer"],
            }

            client_questions.append({
                "position": pos,
                "question_text": q_data["question_text"],
                "skill_name": skill_name,
                "raw_skill_id": raw_skill_id,
            })

        self.sessions[sid] = {
            "session_id": sid,
            "student_id": st_id,
            "set_name": sess_name,
            "current_position": 1,
            "attempts_per_pos": {},
            "history": [],
            "questions": generated_questions,
        }

        return sid, client_questions

    def _ensure_session_loaded(self, session_id: str) -> Dict[str, Any]:
        """Ensures session exists in memory, loading from SQLite if needed."""
        if session_id in self.sessions:
            return self.sessions[session_id]

        # Fetch from DB
        sess_row = self.db.get_session(session_id)
        if not sess_row:
            raise KeyError(f"Session '{session_id}' not found")

        interactions = self.db.get_session_interactions(session_id)
        attempts = {}
        for inter in interactions:
            pos = inter.get("position", len(attempts) + 1)
            attempts[pos] = inter.get("attempt_count", 1)

        self.sessions[session_id] = {
            "session_id": session_id,
            "student_id": sess_row["student_id"],
            "set_name": sess_row["set_name"],
            "current_position": len(interactions) + 1,
            "attempts_per_pos": attempts,
            "history": interactions,
            "questions": {},
        }
        return self.sessions[session_id]

    def record_answer(
        self,
        session_id: str,
        position: int,
        submitted_answer: Any,
        response_time_ms: float,
        hint_used: int = 0
    ) -> Dict[str, Any]:
        """
        Records a student's answer for a given position:
        - Looks up correct answer server-side for this session's generated questions.
        - Determines correctness (1/0).
        - Tracks attempt_count per position.
        - Synchronously writes interaction into SQLite database including question_text & correct_answer.
        - Marks session completed if 10th question is recorded.
        """
        sess = self._ensure_session_loaded(session_id)
        set_name = sess["set_name"]
        student_id = sess["student_id"]

        q_data = sess.get("questions", {}).get(position)
        if not q_data:
            if set_name in self.bank.sets:
                q_data = self.bank.get_question(set_name, position)
            else:
                q_data = generate_question(position)

        skill_info = self.bank.get_skill_for_position(position)
        is_correct = check_answer(submitted_answer, q_data.get("correct_answer"))
        if not is_correct and set_name in self.bank.sets:
            bank_q = self.bank.sets[set_name].get(position)
            if bank_q and check_answer(submitted_answer, bank_q.get("correct_answer")):
                is_correct = True

        correct_val = 1 if is_correct else 0

        # Track attempt count for this position
        attempts = sess["attempts_per_pos"].get(position, 0) + 1
        sess["attempts_per_pos"][position] = attempts

        hint_flag = 1 if hint_used else 0
        raw_skill_id = int(skill_info["raw_skill_id"])

        # Synchronous SQLite write with question text and correct answer
        self.db.record_interaction(
            session_id=session_id,
            student_id=student_id,
            position=position,
            skill_id=raw_skill_id,
            correct=int(correct_val),
            response_time_ms=float(response_time_ms),
            attempt_count=int(attempts),
            hint_used=int(hint_flag),
            question_text=q_data.get("question_text"),
            correct_answer=q_data.get("correct_answer"),
        )

        interaction = {
            "skill_id": raw_skill_id,
            "correct": int(correct_val),
            "response_time_ms": float(response_time_ms),
            "attempt_count": int(attempts),
            "hint_used": int(hint_flag),
            "position": position,
            "question_text": q_data.get("question_text"),
        }
        sess["history"].append(interaction)
        sess["current_position"] = position + 1

        is_complete = len(sess["history"]) >= len(self.bank.skill_sequence)
        if is_complete:
            self.db.complete_session(session_id)

        return {
            "session_id": session_id,
            "student_id": student_id,
            "position": position,
            "correct": correct_val,
            "attempt_count": attempts,
            "interactions_count": len(sess["history"]),
            "is_complete": is_complete,
        }

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns the interaction history recorded so far for session_id."""
        return self.db.get_session_interactions(session_id)

    def get_full_history(self, student_id: str) -> List[Dict[str, Any]]:
        """Returns ALL interactions for a student across all their sessions."""
        return self.db.get_full_history(student_id)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieves session metadata."""
        return self._ensure_session_loaded(session_id)

    def get_student_summary(self, student_id: str) -> Dict[str, Any]:
        """Returns student summary metadata (session count, interaction count)."""
        return self.db.get_student_summary(student_id)


# Module singleton
session_manager = SessionManager(_question_bank, default_db)
