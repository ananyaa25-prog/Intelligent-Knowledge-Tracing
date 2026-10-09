"""
scripts/analyze_logs.py
-----------------------
T8d: Pilot analysis of data/processed/session_logs.jsonl (read-only).

Excludes student ids matching EXCLUDE_STUDENT_REGEX (test/smoke/e2e/sim_ accounts).
Reports: sessions, answers, Brier score, accuracy of (p >= 0.5), a reliability
table, predicted vs actual by level, probe validation (graph prediction made
before a probe vs the probe outcome), and perceived difficulty by policy from
feedback events. With fewer than 30 usable answers it states that the numbers
are not meaningful yet.

Usage: python scripts/analyze_logs.py [--log path] [--out reports/pilot.md] [--include-all]
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROCESSED_DIR, REPORTS_DIR, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from serving.policy_config import get_config

MIN_USABLE = 30
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def brier(rows):
    return sum((p - y) ** 2 for p, y in rows) / len(rows) if rows else None


def acc05(rows):
    return sum(1 for p, y in rows if (p >= 0.5) == (y == 1)) / len(rows) if rows else None


def analyze(events, cfg, include_all=False):
    sess_student = {}
    for e in events:
        if e.get("student_id"):
            sess_student[e["session_id"]] = e["student_id"]
    keep = lambda e: include_all or not cfg.is_excluded_student(sess_student.get(e.get("session_id"), ""))
    answers = [e for e in events if e.get("event") == "answer" and keep(e)]
    excluded = sum(1 for e in events if e.get("event") == "answer") - len(answers)
    questions = {(e["session_id"], e["position"]): e for e in events if e.get("event") == "question" and keep(e)}
    feedback = [e for e in events if e.get("event") == "feedback" and keep(e)]
    sessions = {e["session_id"] for e in answers}
    students = {e.get("student_id") for e in answers}

    main = [(float(e["p_before"]), int(e["outcome"]), e.get("level")) for e in answers
            if e.get("p_before") is not None and not e.get("is_probe")]
    probes = [(float(e["p_before"]), int(e["outcome"])) for e in answers
              if e.get("p_before") is not None and e.get("is_probe")]
    rows = [(p, y) for p, y, _ in main]

    rel = []
    for lo, hi in BINS:
        b = [(p, y) for p, y in rows if lo <= p < hi]
        rel.append((lo, min(hi, 1.0), len(b), (sum(p for p, _ in b) / len(b)) if b else None,
                    (sum(y for _, y in b) / len(b)) if b else None))
    by_level = {}
    for lvl in ("easy", "medium", "hard"):
        b = [(p, y) for p, y, l in main if l == lvl]
        by_level[lvl] = (len(b), (sum(p for p, _ in b) / len(b)) if b else None, (sum(y for _, y in b) / len(b)) if b else None)

    policy_of = {}
    for (sid, _), q in questions.items():
        if q.get("policy"):
            policy_of[sid] = q["policy"]
    fb = defaultdict(lambda: defaultdict(int))
    helpful = defaultdict(list)
    for e in feedback:
        pol = e.get("policy") or policy_of.get(e["session_id"]) or "unknown"
        fb[pol][e.get("perceived")] += 1
        if e.get("helpful"):
            helpful[pol].append(int(e["helpful"]))

    return {"n_events": len(events), "excluded_answers": excluded, "sessions": len(sessions),
            "students": len(students), "answers": len(answers), "usable": len(rows),
            "brier": brier(rows), "acc05": acc05(rows), "mean_p": (sum(p for p, _ in rows) / len(rows)) if rows else None,
            "mean_y": (sum(y for _, y in rows) / len(rows)) if rows else None,
            "reliability": rel, "by_level": by_level, "probes": probes,
            "probe_brier": brier(probes), "feedback": {k: dict(v) for k, v in fb.items()},
            "helpful": {k: sum(v) / len(v) for k, v in helpful.items() if v}, "n_feedback": len(feedback)}


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(PROCESSED_DIR / "session_logs.jsonl"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "pilot.md"))
    ap.add_argument("--include-all", action="store_true", help="do not exclude test/sim accounts")
    args = ap.parse_args()
    cfg = get_config()
    path = Path(args.log)
    if not path.exists():
        print(f"No log at {path}")
        return 1
    r = analyze(load_events(path), cfg, args.include_all)
    meaningful = r["usable"] >= MIN_USABLE
    L = ["# Pilot log analysis", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from `{path.name}` by `scripts/analyze_logs.py`.",
         f"Excluded ids matching `{cfg.exclude_student_regex}`" + (" (disabled with --include-all)" if args.include_all else "")
         + f": {r['excluded_answers']} answer events excluded.", ""]
    if not meaningful:
        L += [f"> **Not meaningful yet:** only {r['usable']} usable answers with a pre-answer prediction "
              f"(minimum {MIN_USABLE}). The numbers below are shown for completeness only and must not be used as evidence.", ""]
    L += ["| Quantity | Value |", "|---|---|",
          f"| Students / sessions | {r['students']} / {r['sessions']} |",
          f"| Answers (all) / usable (non-probe, with prediction) | {r['answers']} / {r['usable']} |",
          f"| Brier score (lower is better; 0.25 = always 50%) | {fmt(r['brier'])} |",
          f"| Accuracy of (p >= 0.5) | {fmt(r['acc05'], True)} |",
          f"| Mean predicted vs observed correct | {fmt(r['mean_p'], True)} vs {fmt(r['mean_y'], True)} |",
          "", "## Reliability table (non-probe answers)", "",
          "| predicted p | n | mean predicted | observed correct |", "|---|---|---|---|"]
    for lo, hi, n, mp, my in r["reliability"]:
        L.append(f"| {lo:.1f}-{hi:.1f} | {n} | {fmt(mp, True)} | {fmt(my, True)} |")
    L += ["", "## Predicted vs actual by level", "", "| level | n | mean predicted | observed correct |", "|---|---|---|---|"]
    for lvl, (n, mp, my) in r["by_level"].items():
        L.append(f"| {lvl} | {n} | {fmt(mp, True)} | {fmt(my, True)} |")
    pr = r["probes"]
    L += ["", "## Probe validation (graph prediction before a never-asked topic)", "",
          f"{len(pr)} probe answers with a prediction; mean predicted {fmt(sum(p for p, _ in pr) / len(pr), True) if pr else 'n/a'}, "
          f"observed correct {fmt(sum(y for _, y in pr) / len(pr), True) if pr else 'n/a'}, Brier {fmt(r['probe_brier'])}.",
          "", "## Perceived difficulty by policy (feedback events)", ""]
    if r["n_feedback"]:
        L += ["| policy | too easy | about right | too hard | mean helpfulness (1-5) |", "|---|---|---|---|---|"]
        for pol, c in r["feedback"].items():
            L.append(f"| {pol} | {c.get('too_easy', 0)} | {c.get('about_right', 0)} | {c.get('too_hard', 0)} | "
                     f"{fmt(r['helpful'].get(pol)) if pol in r['helpful'] else 'n/a'} |")
    else:
        L.append("No feedback events yet.")
    L += ["", "What this can and cannot show: see docs/PILOT_GUIDE.md. A small pilot can show whether predictions are "
          "roughly calibrated for these quiz questions and how students perceive the difficulty; it cannot show learning gains."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:20]))
    if not meaningful:
        print(f"\nnot meaningful yet ({r['usable']} usable answers < {MIN_USABLE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
