"""
scripts/check_answer_response.py
--------------------------------
Cross-check of how the served model's prediction REACTS TO ANSWERS (added after
user testing reported "even after correct answers my chance goes down").

1. Answer sensitivity (validation split): take a history prefix, flip ONLY the
   last answer (correct vs wrong) and compare P(next question). A sensible
   model gives a higher P after a correct answer. Reported separately for
   "next question on the same skill" and "on a different skill".
2. Repeated practice: P(skill) after k = 0..6 consecutive correct answers on that
   skill (from validation prefixes), compared with the EMPIRICAL success rate in
   the same data after k consecutive correct same-skill answers.
3. First attempt vs after one correct answer on the quiz topics.

Usage: python scripts/check_answer_response.py [--n 400] [--seed 5]
Outputs: reports/answer_response.md, reports/answer_response.png
"""

import argparse
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPORTS_DIR, load_service, load_validation_split, seed_everything, val_student_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    seed_everything(args.seed)
    svc = load_service()
    val = load_validation_split()
    P = svc._predictor
    rng = random.Random(args.seed)
    lengths = val["mask"].sum(dim=1).tolist()
    elig = [i for i, n in enumerate(lengths) if n >= 6]

    # 1. answer sensitivity
    same, diff = [], []
    for _ in range(args.n):
        i = rng.choice(elig)
        n = int(lengths[i])
        t = rng.randint(3, min(n - 1, 150))
        h = val_student_history(val, svc, i, upto=t + 1)
        prefix, nxt = h[:t], h[t]
        cand = [(nxt["skill_id"], nxt["difficulty_bucket"] or 2)]
        hc = prefix[:-1] + [dict(prefix[-1], correct=1)]
        hw = prefix[:-1] + [dict(prefix[-1], correct=0)]
        pc, pw = P(hc, cand)[0], P(hw, cand)[0]
        (same if prefix[-1]["skill_id"] == nxt["skill_id"] else diff).append(pc - pw)
    viol_same = sum(1 for d in same if d < -0.005) / max(1, len(same))
    viol_diff = sum(1 for d in diff if d < -0.005) / max(1, len(diff))

    # 2a. empirical: success after k consecutive correct same-skill answers
    emp = defaultdict(lambda: [0, 0])
    sk, cs, mk = val["skill_seq"], val["correct_seq"], val["mask"]
    for i in range(sk.shape[0]):
        n = int(mk[i].sum())
        run_skill, run = None, 0
        for t in range(n):
            s, c = int(sk[i, t]), int(cs[i, t])
            k = run if s == run_skill else 0
            if t > 0 and s == run_skill:
                emp[min(k, 6)][0] += c
                emp[min(k, 6)][1] += 1
            if s == run_skill:
                run = run + 1 if c else 0
            else:
                run_skill, run = s, (1 if c else 0)
    # 2b. model: P(skill) after k consecutive correct answers appended to real prefixes
    model_k = defaultdict(list)
    for _ in range(max(60, args.n // 5)):
        i = rng.choice(elig)
        t = rng.randint(3, min(int(lengths[i]) - 1, 100))
        h = val_student_history(val, svc, i, upto=t + 1)
        prefix, target = h[:t], h[t]
        s = target["skill_id"]
        b = target["difficulty_bucket"] or 2
        cur = list(prefix)
        for k in range(7):
            model_k[k].append(P(cur, [(s, b)])[0])
            cur.append(dict(target, correct=1))

    # 3. quiz topics: first attempt vs after one correct answer (short neutral history)
    from serving.adaptive_session import TOPICS
    base = [{"skill_id": 50, "correct": 1, "response_time_ms": 8000, "attempt_count": 1, "hint_used": 0, "difficulty_bucket": 2},
            {"skill_id": 48, "correct": 1, "response_time_ms": 8000, "attempt_count": 1, "hint_used": 0, "difficulty_bucket": 2}]
    quiz_rows = []
    for tp in TOPICS:
        t = tp["raw_skill_id"]
        if t in (50, 48):
            continue
        before = P(base, [(t, 1)])[0]
        after = P(base + [{"skill_id": t, "correct": 1, "response_time_ms": 8000, "attempt_count": 1, "hint_used": 0,
                           "difficulty_bucket": 1}], [(t, 1)])[0]
        after_wrong = P(base + [{"skill_id": t, "correct": 0, "response_time_ms": 8000, "attempt_count": 1, "hint_used": 0,
                                 "difficulty_bucket": 1}], [(t, 1)])[0]
        quiz_rows.append((tp["skill_name"], before, after, after_wrong))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = list(range(7))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, [sum(model_k[k]) / len(model_k[k]) for k in ks], "o-", color="#2563eb", label="model P(next on same skill)")
    ax.plot(ks, [emp[k][0] / emp[k][1] if emp[k][1] else None for k in ks], "s--", color="#d97706",
            label="observed success in ASSISTments validation data")
    ax.set_xlabel("consecutive correct answers on the skill just before (6 = 6 or more)")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Repeated correct practice on one skill")
    plt.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    plt.savefig(REPORTS_DIR / "answer_response.png", dpi=130)
    plt.close(fig)

    L = ["# How the prediction reacts to answers (cross-check)", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/check_answer_response.py` (seed {args.seed}). "
         "Served model, uncalibrated predictor, ASSISTments validation split.", "",
         "## 1. Correct vs wrong last answer (everything else identical)", "",
         "| next question | cases | mean P(after correct) - P(after wrong) | share where a correct answer gives a LOWER P |", "|---|---|---|---|",
         f"| same skill as the last answer | {len(same)} | {sum(same) / max(1, len(same)):+.3f} | {viol_same:.1%} |",
         f"| different skill | {len(diff)} | {sum(diff) / max(1, len(diff)):+.3f} | {viol_diff:.1%} |", "",
         "## 2. Repeated correct answers on one skill", "",
         "| consecutive correct before | model P (mean over prefixes) | observed success in data | observed n |", "|---|---|---|---|"]
    for k in ks:
        m = sum(model_k[k]) / len(model_k[k])
        e = emp[k][0] / emp[k][1] if emp[k][1] else float("nan")
        L.append(f"| {k}{'+' if k == 6 else ''} | {m:.3f} | {e:.3f} | {emp[k][1]} |")
    L += ["", "## 3. Quiz topics: Easy prediction before any answer on the topic vs after one answer", "",
          "| topic | before | after 1 correct | after 1 wrong |", "|---|---|---|---|"]
    for name, b, a, w in quiz_rows:
        L.append(f"| {name} | {b:.2f} | {a:.2f} | {w:.2f} |")
    L += ["", "![answer response](answer_response.png)", ""]
    (REPORTS_DIR / "answer_response.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
