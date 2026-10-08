"""
scripts/evaluate_sessions.py
-----------------------------
Phase C: evaluate logged adaptive sessions.

Reads data/processed/session_logs.jsonl (written by the adaptive engine) and
reports:
  - AUC, accuracy at 0.5, Brier score
  - calibration table (overall and by level)
  - confident misses (p_before > 0.8 but answered incorrectly)
  - probe-question predicted-vs-actual comparison (graph propagation test)

Run:
    python scripts/evaluate_sessions.py [path/to/session_logs.jsonl]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

DEFAULT_LOG_PATH = "data/processed/session_logs.jsonl"


def load_answers(log_path: Path):
    answers = []
    if not log_path.exists():
        return answers
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "answer" and rec.get("p_before") is not None:
                answers.append(rec)
    return answers


def auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def brier(y, p):
    return float(np.mean((np.array(p) - np.array(y)) ** 2))


def accuracy_at_half(y, p):
    return float(np.mean((np.array(p) >= 0.5) == np.array(y)))


def calibration_table(rows, label):
    """Bin predictions into 0.1-width buckets; report count, mean p, observed rate."""
    bins = defaultdict(list)
    for r in rows:
        b = min(int(r["p_before"] * 10), 9)
        bins[b].append(r)

    print(f"\n  Calibration ({label}):")
    print(f"  {'bin':>12} {'n':>5} {'mean_p':>7} {'obs':>7}")
    for b in range(10):
        group = bins.get(b, [])
        if not group:
            continue
        p = np.mean([r["p_before"] for r in group])
        o = np.mean([r["outcome"] for r in group])
        print(f"  [{b/10:.1f},{b/10+0.1:.1f}) {len(group):>5} {p:>7.3f} {o:>7.3f}")


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_LOG_PATH)
    answers = load_answers(log_path)

    if not answers:
        print(f"No logged answers with predictions found in {log_path}.")
        print("Run an adaptive session first, then re-run this script.")
        return

    y = np.array([r["outcome"] for r in answers])
    p = np.array([r["p_before"] for r in answers])

    print(f"Logged questions with pre-answer predictions: {len(answers)}")
    print("=" * 60)
    print("Overall metrics")
    print(f"  AUC:                {auc(y, p):.4f}")
    print(f"  Accuracy @ 0.5:     {accuracy_at_half(y, p):.4f}")
    print(f"  Brier score:        {brier(y, p):.4f}")
    print(f"  Mean predicted:     {p.mean():.4f}   Observed correct: {y.mean():.4f}")

    calibration_table(answers, "overall")

    # By level
    for level in ["easy", "medium", "hard"]:
        subset = [r for r in answers if r.get("level") == level]
        if subset:
            yl = np.array([r["outcome"] for r in subset])
            pl = np.array([r["p_before"] for r in subset])
            print(f"\n  Level '{level}' (n={len(subset)}): AUC={auc(yl, pl):.4f} "
                  f"acc@0.5={accuracy_at_half(yl, pl):.4f} Brier={brier(yl, pl):.4f} "
                  f"mean_p={pl.mean():.3f} obs={yl.mean():.3f}")

    # Confident misses
    confident_misses = [r for r in answers if r["p_before"] > 0.8 and r["outcome"] == 0]
    print(f"\nConfident misses (p_before > 0.8 but wrong): {len(confident_misses)}")
    for r in confident_misses[:20]:
        print(f"  session={r['session_id'][:8]} pos={r['position']} "
              f"topic={r['topic']} level={r['level']} p_before={r['p_before']:.3f}")
    if len(confident_misses) > 20:
        print(f"  ... and {len(confident_misses) - 20} more")

    # Probe comparison (graph propagation test)
    probes = [r for r in answers if r.get("is_probe") in (1, True)]
    if probes:
        print("\n" + "=" * 60)
        print("End-of-session probe questions (graph propagation test)")
        pp = np.array([r["p_before"] for r in probes])
        po = np.array([r["outcome"] for r in probes])
        print(f"  n probes: {len(probes)}")
        print(f"  mean predicted P(correct): {pp.mean():.4f}")
        print(f"  observed correct rate:     {po.mean():.4f}")
        print(f"  AUC on probes:             {auc(po, pp):.4f}")
        by_topic = defaultdict(list)
        for r in probes:
            by_topic[r["topic"]].append(r)
        print(f"  {'topic':>6} {'n':>3} {'mean_p':>7} {'obs':>7}")
        for t, rows in sorted(by_topic.items()):
            tp = np.mean([r["p_before"] for r in rows])
            to = np.mean([r["outcome"] for r in rows])
            print(f"  {t:>6} {len(rows):>3} {tp:>7.3f} {to:>7.3f}")
        print("  (If observed roughly tracks predicted, the graph is propagating")
        print("   knowledge to topics the student never answered directly.)")
    else:
        print("\nNo probe questions logged yet.")


if __name__ == "__main__":
    main()
