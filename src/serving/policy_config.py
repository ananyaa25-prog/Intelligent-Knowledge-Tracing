"""
src/serving/policy_config.py
----------------------------
Environment-variable driven configuration for the adaptive assessment policy,
the mastery profile and the evaluation scripts.

Every value has a safe default. With ALL defaults the system behaves exactly as
before this module existed (staircase difficulty, full 10-topic session, no
calibration). Values are read from the environment each time `get_config()` is
called, so tests and scripts can change them without re-importing modules.

Environment variables (all optional):
  POLICY               staircase | model | random   (difficulty rule; default staircase)
                       v3 adds elo | maxinfo | uncertainty (src/serving/policies_v3.py)
  TARGET_P             float, target P(correct) for POLICY=model (default 0.70)
  APPLY_CALIBRATION    true|false, apply temperature scaling in the predictor (default false)
  SESSION_MODE         full10 | core5 (default full10)
  TOPIC_SELECTION      model | evidence_blend (default model = lowest model prediction;
                       evidence_blend = lowest (EVIDENCE_PRIOR_WEIGHT*P_model + correct) /
                       (EVIDENCE_PRIOR_WEIGHT + answered) on that topic, docs/DECISIONS.md D31)
  CORE_TOPICS          comma-separated raw skill ids for core5 (default 50,48,204,203,350)
  T_PROF               float, P threshold for "can do this level" (default 0.70)
  T_READY              float, P threshold for "ready for the next level" (default 0.60)
  MIN_ANSWERS          int, answers per topic before any status is given (default 2)
  MONOTONE_WARN        float, max violation rate of easy>=medium>=hard for the
                       model-based profile to be trusted (default 0.10)
  EXCLUDE_STUDENT_REGEX  regex of student ids excluded from live analytics
                       (default (smoke|e2e|pred_check|test|^sim_))
  QUESTION_SOURCE      bank | generated | mixed (v3; default bank = the existing templates for
                       the 10 original topics; generated = src/questions/generator.py;
                       mixed = alternate bank / generated). Section sessions always use
                       the generator (they have no bank).
  KT_DB_PATH           optional override of the SQLite DB path
  KT_LOG_PATH          optional override of the JSONL session-log path
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Output of scripts/check_monotonicity.py; read by the mastery module.
PROFILE_RELIABILITY_PATH = PROCESSED_DIR / "profile_reliability.json"
# Output of scripts/fit_calibration.py; read by the predictor when enabled.
CALIBRATION_PATH = PROCESSED_DIR / "calibration.json"

VALID_POLICIES = ("staircase", "model", "random", "elo", "maxinfo", "uncertainty")
VALID_SESSION_MODES = ("full10", "core5")
VALID_TOPIC_SELECTIONS = ("model", "evidence_blend")
VALID_QUESTION_SOURCES = ("bank", "generated", "mixed")

# Defaults (documented above). Changing them changes the system's behaviour.
DEFAULT_POLICY = "staircase"           # existing staircase rule
DEFAULT_TARGET_P = 0.70                # model policy aims for ~70% predicted success
DEFAULT_APPLY_CALIBRATION = False      # raw model probabilities, as before
DEFAULT_SESSION_MODE = "full10"        # existing 10-topic session
DEFAULT_TOPIC_SELECTION = "model"      # existing rule: lowest predicted P
DEFAULT_QUESTION_SOURCE = "bank"       # existing template bank (v3 flag)
EVIDENCE_PRIOR_WEIGHT = 2.0            # evidence_blend: model prediction counts like 2 answers
DEFAULT_CORE_TOPICS = [50, 48, 204, 203, 350]  # fractions, percents, linear systems
DEFAULT_T_PROF = 0.70                  # "proficient at a level" threshold
DEFAULT_T_READY = 0.60                 # "ready to try the next level" threshold
DEFAULT_MIN_ANSWERS = 2                # below this a topic is "not enough evidence"
DEFAULT_MONOTONE_WARN = 0.10           # >10% ordering violations -> model profile not trusted
DEFAULT_EXCLUDE_STUDENT_REGEX = r"(smoke|e2e|pred_check|test|^sim_)"

# core5 mode session shape (see docs/DECISIONS.md): 14 main questions on the 5
# core topics (2 of them warm-up) + 2 probes on non-core topics = 16 in total.
CORE5_MAX_PER_TOPIC = 4
CORE5_TOTAL_QUESTIONS = 16

# Rule-based teacher/student suggestions per status (shown as suggestions,
# never as a verdict).
RECOMMENDED_ACTIONS: Dict[str, str] = {
    "needs_support": "Revisit basics, assign Easy practice",
    "developing": "Practise at Medium",
    "proficient": "Try Hard questions",
    "mastered": "Move to a new topic",
    "not_enough_evidence": "Ask 2-3 more questions",
}


def _env_str(name: str, default: str, allowed=None) -> str:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return default
    val = str(val).strip().lower()
    if allowed is not None and val not in allowed:
        print(f"[policy_config] Ignoring invalid {name}={val!r}; using {default!r}")
        return default
    return val


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _env_int_list(name: str, default: List[int]) -> List[int]:
    val = os.environ.get(name)
    if not val:
        return list(default)
    try:
        out = [int(x) for x in re.split(r"[,\s]+", val.strip().strip("[]")) if x]
        return out or list(default)
    except ValueError:
        return list(default)


@dataclass
class PolicyConfig:
    policy: str = DEFAULT_POLICY
    target_p: float = DEFAULT_TARGET_P
    apply_calibration: bool = DEFAULT_APPLY_CALIBRATION
    session_mode: str = DEFAULT_SESSION_MODE
    topic_selection: str = DEFAULT_TOPIC_SELECTION
    core_topics: List[int] = field(default_factory=lambda: list(DEFAULT_CORE_TOPICS))
    t_prof: float = DEFAULT_T_PROF
    t_ready: float = DEFAULT_T_READY
    min_answers: int = DEFAULT_MIN_ANSWERS
    monotone_warn: float = DEFAULT_MONOTONE_WARN
    exclude_student_regex: str = DEFAULT_EXCLUDE_STUDENT_REGEX
    question_source: str = DEFAULT_QUESTION_SOURCE
    db_path: Optional[str] = None
    log_path: Optional[str] = None

    def thresholds(self) -> Dict[str, float]:
        return {"T_PROF": self.t_prof, "T_READY": self.t_ready, "MIN_ANSWERS": self.min_answers}

    def is_excluded_student(self, student_id: str) -> bool:
        try:
            return re.search(self.exclude_student_regex, str(student_id or ""), re.IGNORECASE) is not None
        except re.error:
            return False


def get_config() -> PolicyConfig:
    """Read the configuration from the environment (cheap; call per use)."""
    return PolicyConfig(
        policy=_env_str("POLICY", DEFAULT_POLICY, VALID_POLICIES),
        target_p=_env_float("TARGET_P", DEFAULT_TARGET_P),
        apply_calibration=_env_bool("APPLY_CALIBRATION", DEFAULT_APPLY_CALIBRATION),
        session_mode=_env_str("SESSION_MODE", DEFAULT_SESSION_MODE, VALID_SESSION_MODES),
        topic_selection=_env_str("TOPIC_SELECTION", DEFAULT_TOPIC_SELECTION, VALID_TOPIC_SELECTIONS),
        core_topics=_env_int_list("CORE_TOPICS", DEFAULT_CORE_TOPICS),
        t_prof=_env_float("T_PROF", DEFAULT_T_PROF),
        t_ready=_env_float("T_READY", DEFAULT_T_READY),
        min_answers=_env_int("MIN_ANSWERS", DEFAULT_MIN_ANSWERS),
        monotone_warn=_env_float("MONOTONE_WARN", DEFAULT_MONOTONE_WARN),
        exclude_student_regex=os.environ.get("EXCLUDE_STUDENT_REGEX") or DEFAULT_EXCLUDE_STUDENT_REGEX,
        question_source=_env_str("QUESTION_SOURCE", DEFAULT_QUESTION_SOURCE, VALID_QUESTION_SOURCES),
        db_path=os.environ.get("KT_DB_PATH") or None,
        log_path=os.environ.get("KT_LOG_PATH") or None,
    )


def load_env_file(path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE file (used for demo.env). Comments start with #."""
    out: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out
