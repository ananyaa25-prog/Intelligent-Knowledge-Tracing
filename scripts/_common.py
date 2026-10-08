"""
scripts/_common.py
------------------
Shared helpers for the evaluation scripts (cross-platform, no bash needed).

- Puts src/ on sys.path.
- Unless KT_DB_PATH is already set, points the serving code at a throw-away
  temporary SQLite DB so importing the service never touches the real
  data/processed/kt_sessions.db (and the JSONL log likewise).
- Loads the model exactly as the FastAPI service does (artifacts.load()).
"""

import os
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WINDOWED_DATA_PATH = PROCESSED_DIR / "windowed_data.pkl"
SAMPLE_STUDENTS_PATH = PROCESSED_DIR / "sample_test_students.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def isolate_storage():
    """Use temp DB/log paths unless the caller set them explicitly."""
    if not os.environ.get("KT_DB_PATH"):
        tmp = Path(tempfile.mkdtemp(prefix="kt_script_"))
        os.environ["KT_DB_PATH"] = str(tmp / "scratch.db")
        os.environ.setdefault("KT_LOG_PATH", str(tmp / "scratch_logs.jsonl"))


def seed_everything(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    print(f"[seed] {seed}")


def load_service():
    """Import the FastAPI module and load artifacts like the service lifespan does."""
    isolate_storage()
    import serving.inference_service as svc
    svc.artifacts.load()
    return svc


def load_validation_split():
    """Return the validation split dict from windowed_data.pkl or None."""
    try:
        import pickle
        with open(WINDOWED_DATA_PATH, "rb") as f:
            data = pickle.load(f)
        return data["splits"]["val"]
    except Exception as e:
        print(f"[data] windowed_data.pkl validation split unavailable: {e}")
        return None


def val_student_history(val, svc, i, upto=None):
    """Interaction dicts (raw skill ids) for validation student i, first `upto` steps."""
    n = int(val["mask"][i].sum().item())
    if upto is not None:
        n = min(n, upto)
    inv = svc.artifacts.inv_skill_map
    out = []
    for t in range(n):
        out.append({
            "skill_id": inv.get(int(val["skill_seq"][i, t].item()), 1),
            "correct": int(val["correct_seq"][i, t].item()),
            "response_time_ms": float(val["response_time_seq"][i, t].item()),
            "attempt_count": float(val["attempt_seq"][i, t].item()),
            "hint_used": float(val["hint_seq"][i, t].item()),
            "difficulty_bucket": int(val["difficulty_seq"][i, t].item()),
        })
    return out


def find_node():
    """Locate node.exe/node (PATH first, then the WinGet install location)."""
    import shutil
    exe = shutil.which("node")
    if exe:
        return exe
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.exists():
        for p in base.glob("OpenJS.NodeJS*/*/node.exe"):
            return str(p)
    return None


def find_npm():
    import shutil
    exe = shutil.which("npm") or shutil.which("npm.cmd")
    if exe:
        return exe
    node = find_node()
    if node:
        cand = Path(node).parent / ("npm.cmd" if os.name == "nt" else "npm")
        if cand.exists():
            return str(cand)
    return None
