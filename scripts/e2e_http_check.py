"""
scripts/e2e_http_check.py
-------------------------
End-to-end check through the REAL services: starts FastAPI + the Node proxy on
free ports (temporary DB copy and temporary log; the real DB is never touched),
plays complete scripted sessions via the Node proxy (/api/...) and asserts the
adaptive-v2 contracts:

  A  /next fields (total_questions, session_mode, policy, decision{...}),
     predicted_p present for every non-probe question after the 2 warm-ups,
     no verdict / answer key in the question payload
  B  /session/{id}/profile and /student/{id}/profile shapes
  C  /class/overview shape
  D  /session/{id}/feedback -> "feedback" event in the log
  core5 topic set and total_questions=16 when SESSION_MODE=core5
  calibration_applied logged as configured

Configurations: "staircase" (all defaults), "candidate" (POLICY=model,
APPLY_CALIBRATION=true, SESSION_MODE=core5) and "demo" (demo.env, if present).
Output: reports/e2e_http_check.md and reports/e2e_http_check.json. Exit 1 on failure.

Usage: python scripts/e2e_http_check.py [--only staircase,candidate,demo]
"""

import argparse
import json
import random
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, REPORTS_DIR, SRC_DIR
from _services import Services, http_json, temp_db_copy

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from serving.policy_config import load_env_file

CORE = [50, 48, 204, 203, 350]
DECISION_KEYS = {"policy", "topic_scores", "level_candidates", "chosen_level", "reason"}
PROFILE_KEYS = {"session_id", "student_id", "profile_reliable", "thresholds", "topics", "unasked"}
TOPIC_KEYS = {"topic", "skill_name", "n_answers", "n_this_session", "levels_attempted", "correct_by_level", "p",
              "status", "ready_for_next", "monotone_adjusted", "evidence_note", "recommended_action"}


class Check:
    def __init__(self):
        self.items = []

    def __call__(self, name, cond, detail=""):
        self.items.append({"check": name, "ok": bool(cond), "detail": str(detail)[:300]})
        return bool(cond)


def play(base, chk, student, expect, rng):
    st, d = http_json("POST", f"{base}/api/session/start", {"student_id": student, "source": "live"})
    chk("session start 200", st == 200, st)
    sid = d["session_id"]
    served = []
    for _ in range(40):
        st, q = http_json("POST", f"{base}/api/session/{sid}/next", {})
        chk("next 200", st == 200, st)
        if q.get("is_complete"):
            chk("completion carries total_questions", isinstance(q.get("total_questions"), int), q)
            break
        for k in ["position", "topic", "skill_name", "level", "difficulty_bucket", "question_text", "predicted_p", "is_probe", "is_complete"]:
            chk(f"A: old field {k}", k in q)
        chk("A: total_questions int", isinstance(q.get("total_questions"), int), q.get("total_questions"))
        chk("A: session_mode", q.get("session_mode") == expect["mode"], q.get("session_mode"))
        chk("A: policy", q.get("policy") == expect["policy"], q.get("policy"))
        chk("A: decision keys", DECISION_KEYS <= set((q.get("decision") or {}).keys()), q.get("decision"))
        chk("A: decision.chosen_level == level", (q.get("decision") or {}).get("chosen_level") == q["level"])
        chk("A: no verdict/answer key in question", not ({"correct", "correct_answer", "is_correct"} & set(q.keys())))
        if not q["is_probe"] and q["position"] > 2:
            chk("A: predicted_p before every non-probe question (after warm-up)", q["predicted_p"] is not None, q["position"])
        served.append(q)
        ans = rng.choice(["1", "2", "5", "10", "0.5"])
        st, a = http_json("POST", f"{base}/api/session/{sid}/answer",
                          {"submitted_answer": ans, "response_time_ms": 3000 + rng.random() * 4000, "hint_used": 0})
        chk("answer 200", st == 200, st)
    main = [q for q in served if not q["is_probe"]]
    probes = [q for q in served if q["is_probe"]]
    if expect["mode"] == "core5":
        chk("core5: main topics within CORE_TOPICS", all(q["topic"] in CORE for q in main), [q["topic"] for q in main])
        chk("core5: probes on non-core topics", all(q["topic"] not in CORE for q in probes))
        chk("core5: total_questions == 16", all(q["total_questions"] == 16 for q in served), {q["total_questions"] for q in served})
        chk("core5: 16 questions served", len(served) == 16, len(served))
    else:
        chk("full10: >= 15 questions", len(served) >= 15, len(served))
    return sid, served


def run_config(name, env_extra, expect):
    chk = Check()
    tmp = Path(tempfile.mkdtemp(prefix=f"kt_e2e_{name}_"))
    log = tmp / "logs.jsonl"
    env = {"KT_DB_PATH": str(temp_db_copy(tmp)), "KT_LOG_PATH": str(log)}
    env.update(env_extra)
    rng = random.Random(42)
    summary = {"config": name, "env": env_extra, "questions": 0}
    try:
        with Services(env, log_dir=tmp) as svc:
            base = svc.base
            student = f"e2e_{name}_{rng.randint(1000, 9999)}"
            sid, served = play(base, chk, student, expect, rng)
            summary["questions"] = len(served)
            summary["levels"] = [q["level"] for q in served]
            summary["example_reason"] = next((q["decision"]["reason"] for q in served if q["position"] > 2), "")
            st, p = http_json("GET", f"{base}/api/session/{sid}/profile")
            chk("B: session profile 200", st == 200, st)
            chk("B: profile keys", PROFILE_KEYS <= set(p.keys()), list(p.keys()))
            chk("B: thresholds keys", set(p.get("thresholds", {}).keys()) == {"T_PROF", "T_READY", "MIN_ANSWERS"})
            chk("B: topic keys", all(TOPIC_KEYS <= set(t.keys()) for t in p.get("topics", [])))
            chk("B: unasked measured=false", all(u.get("measured") is False for u in p.get("unasked", [])))
            chk("B: 10 topics covered", len(p.get("topics", [])) + len(p.get("unasked", [])) == 10)
            summary["profile_reliable"] = p.get("profile_reliable")
            summary["statuses"] = {t["skill_name"]: t["status"] for t in p.get("topics", [])}
            st, p2 = http_json("GET", f"{base}/api/student/{student}/profile")
            chk("B: student profile 200", st == 200 and p2.get("student_id") == student, st)
            st, o = http_json("GET", f"{base}/api/class/overview?source=live")
            chk("C: overview 200 + keys", st == 200 and {"students", "status_counts", "needs_attention"} <= set(o.keys()), st)
            chk("C: e2e_ account excluded from live overview", all(s["student_id"] != student for s in o.get("students", [])))
            st, f = http_json("POST", f"{base}/api/session/{sid}/feedback", {"perceived": "about_right", "helpful": 4})
            chk("D: feedback 200", st == 200, st)
            st, _ = http_json("POST", f"{base}/api/session/{sid}/feedback", {"perceived": "nope"})
            chk("D: invalid feedback 400", st == 400, st)
        events = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        chk("log: feedback event written", any(e.get("event") == "feedback" and e.get("session_id") == sid for e in events))
        qev = [e for e in events if e.get("event") == "question" and e.get("session_id") == sid]
        chk("log: question events carry policy + decision", qev and all("policy" in e and "decision" in e for e in qev))
        want_cal = env_extra.get("APPLY_CALIBRATION", "false").lower() == "true"
        chk("log: calibration_applied as configured", qev and all(e.get("calibration_applied") == want_cal for e in qev),
            {e.get("calibration_applied") for e in qev})
        probe_ev = [e for e in qev if e.get("is_probe")]
        chk("log: probes log predicted_before_probe", all("predicted_before_probe" in e for e in probe_ev))
    except Exception as e:
        chk("services ran without exception", False, traceback.format_exc()[-500:])
    summary["passed"] = all(i["ok"] for i in chk.items)
    summary["n_checks"] = len(chk.items)
    summary["failed_checks"] = [i for i in chk.items if not i["ok"]]
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="staircase,candidate,demo")
    args = ap.parse_args()
    wanted = args.only.split(",")
    configs = []
    if "staircase" in wanted:
        configs.append(("staircase", {}, {"policy": "staircase", "mode": "full10"}))
    if "candidate" in wanted:
        configs.append(("candidate", {"POLICY": "model", "APPLY_CALIBRATION": "true", "SESSION_MODE": "core5"},
                        {"policy": "model", "mode": "core5"}))
    demo_env = PROJECT_ROOT / "demo.env"
    if "demo" in wanted and demo_env.exists():
        d = load_env_file(demo_env)
        d = {k: v for k, v in d.items() if k in ("POLICY", "APPLY_CALIBRATION", "SESSION_MODE", "TARGET_P", "TOPIC_SELECTION")}
        configs.append(("demo", d, {"policy": d.get("POLICY", "staircase"), "mode": d.get("SESSION_MODE", "full10")}))
    results = [run_config(*c) for c in configs]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "e2e_http_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    L = ["# End-to-end HTTP check", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/e2e_http_check.py`: real FastAPI + Node proxy "
         "on free ports, temporary DB copy and log, one scripted session per configuration through `/api/*`.", "",
         "| Configuration | env | questions | checks | result |", "|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['config']} | `{json.dumps(r['env'])}` | {r['questions']} | {r['n_checks']} | {'PASS' if r['passed'] else 'FAIL'} |")
    for r in results:
        if r["failed_checks"]:
            L += ["", f"## Failures in {r['config']}", ""] + [f"- {c['check']}: {c['detail']}" for c in r["failed_checks"]]
    L += ["", "Note: warm-up questions 1-2 carry no prediction by design (the model needs 2 answers). The `/answer` "
          "response includes `correct` (pre-existing API field); the UI stores it only for the results screen and never "
          "shows it during the quiz (SHOW_VERDICT=false).", ""]
    for r in results:
        L.append(f"- {r['config']}: example decision reason: \"{r.get('example_reason', '')}\"")
    (REPORTS_DIR / "e2e_http_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
