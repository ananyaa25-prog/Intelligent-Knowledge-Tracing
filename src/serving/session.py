"""
src/serving/session.py
----------------------
Task 0: Question bank serving and live session management.

Maintains in-memory session states for live quiz participants.
Strictly serves questions in fixed skill order (positions 1-10)
from a randomly selected parallel set (A-E), verifies answers
server-side, and records interaction histories for downstream
knowledge tracing inference.
"""

import json
import uuid
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Base paths
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
QUESTION_BANK_PATH = PROJECT_ROOT / "question_bank.json"


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
        if set_name not in self.sets:
            raise ValueError(f"Unknown set '{set_name}'")
        if position not in self.sets[set_name]:
            raise ValueError(f"Position {position} not found in set '{set_name}'")
        return self.sets[set_name][position]


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
    """In-memory session manager for live quiz sessions."""
    def __init__(self, bank: QuestionBank = _question_bank):
        self.bank = bank
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(self, session_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Starts a new session:
        - Randomly selects one of the 5 sets (A-E).
        - Returns the full 10-question sequence in position order (1..10)
          WITHOUT revealing correct_answer.
        - Stores the assigned set and initialized history in memory.
        """
        sid = session_id or str(uuid.uuid4())
        chosen_set = random.choice(self.bank.available_sets)

        # Build client payload (NEVER includes correct_answer)
        client_questions = []
        for pos_info in self.bank.skill_sequence:
            pos = pos_info["position"]
            q_data = self.bank.get_question(chosen_set, pos)
            client_questions.append({
                "position": pos,
                "question_text": q_data["question_text"],
                "skill_name": pos_info["skill_name"],
                "raw_skill_id": pos_info["raw_skill_id"],
            })

        self.sessions[sid] = {
            "session_id": sid,
            "set_name": chosen_set,
            "current_position": 1,
            "attempts_per_pos": {},
            "history": [],  # List of interaction records
        }

        return sid, client_questions

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
        - Looks up correct answer server-side for this session's set.
        - Determines correctness (1/0).
        - Genuinely tracks attempt_count per position.
        - Appends interaction record to session history.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")

        sess = self.sessions[session_id]
        set_name = sess["set_name"]
        q_data = self.bank.get_question(set_name, position)
        skill_info = self.bank.get_skill_for_position(position)

        correct_val = 1 if check_answer(submitted_answer, q_data["correct_answer"]) else 0

        # Track attempt count for this position
        attempts = sess["attempts_per_pos"].get(position, 0) + 1
        sess["attempts_per_pos"][position] = attempts

        # Ensure hint_used is 0 (no hint UI built in frontend)
        hint_flag = 1 if hint_used else 0

        interaction = {
            "skill_id": int(skill_info["raw_skill_id"]),
            "correct": int(correct_val),
            "response_time_ms": float(response_time_ms),
            "attempt_count": int(attempts),
            "hint_used": int(hint_flag),
        }

        sess["history"].append(interaction)
        sess["current_position"] = position + 1

        return {
            "session_id": session_id,
            "position": position,
            "correct": correct_val,
            "attempt_count": attempts,
            "interactions_count": len(sess["history"]),
            "is_complete": len(sess["history"]) >= len(self.bank.skill_sequence)
        }

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns the interaction history recorded so far for session_id."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")
        return list(self.sessions[session_id]["history"])

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")
        return self.sessions[session_id]


# Module singleton
session_manager = SessionManager(_question_bank)
