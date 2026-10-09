"""
scripts/dry_run.py
------------------
Rehearsal of the three demo scenarios against the real FastAPI app (in-process
TestClient) with the demo configuration, on a TEMPORARY copy of the DB and log.

Because answers are free text, the dry run reads the pending question's correct
answer from the in-memory session (test harness only; the API never exposes it)
to play a strong student. Scenarios:
  strong      always correct
  struggling  always wrong
  mixed       correct on fractions/percent topics, wrong on the others

Prints the question-by-question trace (prediction, level, reason) and the final
profile for each scenario, then a list of sanity findings.

Usage: python scripts/dry_run.py [--env-file demo.env] [--defaults]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MIXED_STRONG_TOPICS = {50, 48, 204, 203}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=str(PROJECT_ROOT / "demo.env"))
    ap.add_argument("--defaults", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="seed for reproducible warm-up/question choices")
    args = ap.parse_args()
    from serving.policy_config import load_env_file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _services import temp_db_copy
    tmp = Path(tempfile.mkdtemp(prefix="kt_dry_"))
    os.environ["KT_DB_PATH"] = str(temp_db_copy(tmp))
    os.environ["KT_LOG_PATH"] = str(tmp / "logs.jsonl")
    if not args.defaults:
        os.environ.update(load_env_file(args.env_file))
    if args.seed is not None:
        import random
        random.seed(args.seed)
        print(f"[seed] {args.seed}")
    from fastapi.testclient import TestClient
    import serving.inference_service as svc

    findings = []
    summary = {}
    with TestClient(svc.app) as client:
        for scenario in ["strong", "struggling", "mixed"]:
            student = f"dry_{scenario}"
            sid = client.post("/session/start", json={"student_id": student, "source": "live"}).json()["session_id"]
            rows = []
            while True:
                r = client.post(f"/session/{sid}/next", json={})
                if r.status_code != 200:
                    findings.append(f"{scenario}: /next HTTP {r.status_code} {r.text[:200]}")
                    break
                q = r.json()
                if q["is_complete"]:
                    break
                correct_answer = svc.adaptive_manager.get(sid).pending["correct_answer"]
                good = scenario == "strong" or (scenario == "mixed" and q["topic"] in MIXED_STRONG_TOPICS)
                a = client.post(f"/session/{sid}/answer", json={
                    "submitted_answer": str(correct_answer) if good else "__wrong__", "response_time_ms": 6000.0})
                if a.status_code != 200:
                    findings.append(f"{scenario}: /answer HTTP {a.status_code} {a.text[:200]}")
                    break
                if good and a.json()["correct"] != 1:
                    findings.append(f"{scenario}: correct answer {correct_answer!r} graded wrong ({q['question_text']})")
                rows.append((q, a.json()["correct"]))
            prof = client.get(f"/session/{sid}/profile").json()
            fb = client.post(f"/session/{sid}/feedback", json={"perceived": "about_right", "helpful": 4}).status_code
            if fb != 200:
                findings.append(f"{scenario}: feedback HTTP {fb}")
            if not args.quiet:
                print(f"\n=== {scenario} ({len(rows)} questions, policy={rows[0][0]['policy']}, mode={rows[0][0]['session_mode']}) ===")
                for q, c in rows:
                    p = q["predicted_p"]
                    print(f"Q{q['position']:>2} {q['skill_name'][:28]:28} {q['level']:6} "
                          f"P={'  - ' if p is None else f'{p:4.0%}'} {'probe' if q['is_probe'] else '     '} ans={c} | {q['decision']['reason'][:110]}")
                print("profile_reliable:", prof["profile_reliable"])
                for t in prof["topics"]:
                    p = t["p"] or {}
                    print(f"  {t['skill_name'][:30]:30} n={t['n_answers']} {t['status']:20} ready={t['ready_for_next']!s:5} "
                          f"p(e/m/h)={'/'.join(f'{p[k]:.2f}' for k in ['easy','medium','hard']) if p else '-'} | {t['evidence_note'] or ''}")
                for u in prof["unasked"]:
                    print(f"  (inferred) {u['skill_name'][:30]:30} predicted={u['predicted_p']}")
            statuses = [t["status"] for t in prof["topics"]]
            levels = [q["level"] for q, _ in rows]
            ps = [q["predicted_p"] for q, _ in rows if q["predicted_p"] is not None]
            summary[scenario] = {"statuses": statuses, "levels": levels, "max_p": max(ps) if ps else None, "min_p": min(ps) if ps else None}

        # Returning student: second session of the strong student must predict from
        # question 1 (no blind warm-up) and resume topics at their earlier level.
        sid2 = client.post("/session/start", json={"student_id": "dry_strong", "source": "live"}).json()["session_id"]
        q1 = client.post(f"/session/{sid2}/next", json={}).json()
        if q1.get("predicted_p") is None:
            findings.append("returning: first question of session 2 has no prediction")
        if q1.get("level") == "easy":
            findings.append("returning: strong student restarted at Easy in session 2")
        print(f"=== returning strong student, session 2, Q1: {q1.get('skill_name')} {q1.get('level')} "
              f"P={q1.get('predicted_p')} | {(q1.get('decision') or {}).get('reason', '')[:120]}")
        done = svc.session_manager.db.get_session(sid)["completed_at"]
        if not done:
            findings.append("completed sessions are not marked completed")

    s, w, m = summary["strong"], summary["struggling"], summary["mixed"]
    if "hard" not in s["levels"]:
        findings.append("strong: never served a Hard question")
    if not any(x in ("proficient", "mastered") for x in s["statuses"]):
        findings.append(f"strong: no topic reached Proficient/Mastered (statuses {s['statuses']})")
    if s["max_p"] is not None and s["max_p"] < 0.70:
        findings.append(f"strong: prediction never reached the green band (max {s['max_p']:.2f})")
    if any(x in ("proficient", "mastered") for x in w["statuses"]):
        findings.append(f"struggling: a topic was rated Proficient/Mastered ({w['statuses']})")
    if "hard" in w["levels"]:
        findings.append("struggling: served a Hard question")
    if not (any(x in ("proficient", "mastered", "developing") for x in m["statuses"]) and "needs_support" in m["statuses"]):
        findings.append(f"mixed: expected both strong and weak topics (statuses {m['statuses']})")
    print("\n=== Findings ===")
    print("\n".join(f"- {f}" for f in findings) if findings else "- none: all scenario sanity checks passed")
    print(json.dumps({k: {"max_p": v["max_p"], "levels": {l: v["levels"].count(l) for l in set(v["levels"])}} for k, v in summary.items()}))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
