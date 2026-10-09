"""
scripts/seed_demo_data.py
-------------------------
Seeds SIMULATED demo students (ids prefixed `sim_`, students.source='sample')
for the teacher dashboard's "Simulated demo" view. Each student plays one full
adaptive session through the REAL engine (AdaptiveSession + served predictor);
answers are drawn from a synthetic ability model (same as scripts/simulate.py).
Nothing here is real student data; the dashboard labels it as simulated.

Personas (3 students each, 12 in total by default):
  strong            high general ability
  struggling        low general ability
  mixed             average ability, large topic-to-topic differences
  hint_heavy_slow   below-average ability, slow answers, frequent hints

Safety: by default this refuses to write to the real DB
(data/processed/kt_sessions.db). Use a copy via KT_DB_PATH, or pass
--yes-real-db explicitly. No JSONL log events are written.

Usage:
  python scripts/seed_demo_data.py --yes-real-db          # seed the real DB (demo)
  python scripts/seed_demo_data.py --remove --yes-real-db # remove all sim_ students
  KT_DB_PATH=/tmp/copy.db python scripts/seed_demo_data.py
"""

import argparse
import math
import os
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, SRC_DIR, seed_everything

REAL_DB = (PROJECT_ROOT / "data" / "processed" / "kt_sessions.db").resolve()
PERSONAS = {
    "strong": {"g": 1.3, "spread": 0.5, "rt": 6000, "hint": 0.0},
    "struggling": {"g": -1.3, "spread": 0.5, "rt": 14000, "hint": 0.1},
    "mixed": {"g": 0.0, "spread": 1.4, "rt": 9000, "hint": 0.05},
    "hint_heavy_slow": {"g": -0.4, "spread": 0.7, "rt": 30000, "hint": 0.6},
}
B_LEVEL = {"easy": -1.0, "medium": 0.0, "hard": 1.0}


def p_true(a, level):
    return 0.05 + 0.95 / (1.0 + math.exp(-(a - B_LEVEL[level])))


def target_db_path() -> Path:
    return Path(os.environ["KT_DB_PATH"]).resolve() if os.environ.get("KT_DB_PATH") else REAL_DB


def remove_sim(db) -> int:
    with db._get_connection() as conn:
        ids = [r["student_id"] for r in conn.execute("SELECT student_id FROM students WHERE student_id LIKE 'sim\\_%' ESCAPE '\\'")]
        for sid in ids:
            conn.execute("DELETE FROM interactions WHERE student_id = ?", (sid,))
            conn.execute("DELETE FROM sessions WHERE student_id = ?", (sid,))
            conn.execute("DELETE FROM students WHERE student_id = ?", (sid,))
        conn.commit()
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-persona", type=int, default=3)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--remove", action="store_true", help="delete all sim_ students and their data")
    ap.add_argument("--yes-real-db", action="store_true", help="allow writing to data/processed/kt_sessions.db")
    args = ap.parse_args()

    db_path = target_db_path()
    if db_path == REAL_DB and not args.yes_real_db:
        print(f"Refusing to modify the real DB ({REAL_DB}) without --yes-real-db. "
              "Set KT_DB_PATH to a copy, or pass --yes-real-db.")
        return 2
    os.environ["KT_DB_PATH"] = str(db_path)
    os.environ.setdefault("KT_LOG_PATH", str(Path(os.environ.get("TEMP", "/tmp")) / "kt_seed_unused.jsonl"))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    if args.remove:
        from serving.db import Database
        n = remove_sim(Database(db_path))
        print(f"Removed {n} simulated students (sim_*) from {db_path}")
        return 0

    seed_everything(args.seed)
    import serving.inference_service as svc
    svc.artifacts.load()
    from serving.adaptive_session import AdaptiveSession, TOPIC_IDS
    db = svc.session_manager.db
    removed = remove_sim(db)
    if removed:
        print(f"Replaced {removed} existing simulated students")
    rng = random.Random(args.seed)
    created = []
    for persona, cfg in PERSONAS.items():
        for k in range(1, args.per_persona + 1):
            sid_student = f"sim_{persona}_{k:02d}"
            ability = {t: cfg["g"] + rng.gauss(0, cfg["spread"]) for t in TOPIC_IDS}
            session_id = str(uuid.uuid4())
            db.create_session(session_id=session_id, student_id=sid_student, set_name="adaptive", source="sample")
            sess = AdaptiveSession(session_id, sid_student)
            while True:
                q = sess.next_question(svc._predictor)
                if q is None:
                    break
                correct = rng.random() < p_true(ability[q["topic"]], q["level"])
                hint = 1 if rng.random() < cfg["hint"] else 0
                rt = max(1500.0, rng.gauss(cfg["rt"], cfg["rt"] * 0.3))
                inter = sess.record_answer(q["correct_answer"] if correct else "__wrong__", rt, hint)
                db.record_adaptive_interaction(session_id, sid_student, inter)
            db.complete_session(session_id)
            created.append((sid_student, len(sess.history)))
    print(f"Seeded {len(created)} simulated students into {db_path}:")
    for s, n in created:
        print(f"  {s}: {n} answers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
