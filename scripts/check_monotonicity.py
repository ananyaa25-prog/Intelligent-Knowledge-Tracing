"""
scripts/check_monotonicity.py
-----------------------------
T1: Does the served model order difficulty sensibly, i.e. is
P(correct | easy) >= P(correct | medium) >= P(correct | hard)
for the same history and skill?

The mastery profile's counterfactual P(easy/medium/hard) is only meaningful if
this ordering mostly holds. Outputs:
  reports/monotonicity.md, reports/monotonicity.png
  data/processed/profile_reliability.json {violation_rate, reliable, ...}

Pre-registered rule: reliable = (violation_rate <= MONOTONE_WARN), where
violation_rate is the WORSE of two samples: (A) validation (history prefix,
actual next skill) cases and (B) the same prefixes scored on the 10 quiz topics.

Usage:  python scripts/check_monotonicity.py [--n 600] [--seed 7] [--max-prefix 200]
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (REPORTS_DIR, SAMPLE_STUDENTS_PATH, load_service, load_validation_split,
                     seed_everything, val_student_history)

TOL = 0.005
BUCKETS = (1, 2, 3)


def is_violation(pe, pm, ph, tol=TOL):
    return not (pe >= pm - tol and pm >= ph - tol)


def write_unavailable(reason, cfg):
    from serving.policy_config import PROFILE_RELIABILITY_PATH
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "monotonicity.md").write_text(
        "# Monotonicity check\n\nThe check could not run: " + reason +
        "\n\nprofile_reliable defaults to **false** (evidence-only profile).\n", encoding="utf-8")
    PROFILE_RELIABILITY_PATH.write_text(json.dumps({
        "violation_rate": None, "reliable": False, "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print("[monotonicity] could not run:", reason)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="number of (prefix, skill) cases (>=500)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-prefix", type=int, default=200, help="cap on history prefix length (speed)")
    args = ap.parse_args()
    seed_everything(args.seed)
    t0 = time.time()

    svc = load_service()
    from serving.policy_config import get_config, PROFILE_RELIABILITY_PATH
    from serving.adaptive_session import TOPICS
    cfg = get_config()
    quiz_ids = [t["raw_skill_id"] for t in TOPICS]
    quiz_names = {t["raw_skill_id"]: t["skill_name"] for t in TOPICS}

    # ---- build cases ------------------------------------------------------
    source = None
    prefixes = []  # (history_list, actual_next_raw_skill)
    val = load_validation_split()
    if val is not None:
        source = "windowed_data.pkl validation split"
        lengths = val["mask"].sum(dim=1).tolist()
        eligible = [i for i, n in enumerate(lengths) if n >= 4]
        rng = random.Random(args.seed)
        for _ in range(args.n):
            i = rng.choice(eligible)
            n = int(lengths[i])
            t = rng.randint(2, min(n - 1, args.max_prefix))
            hist = val_student_history(val, svc, i, upto=t + 1)
            prefixes.append((hist[:t], hist[t]["skill_id"]))
    else:
        try:
            students = json.loads(SAMPLE_STUDENTS_PATH.read_text(encoding="utf-8"))
            source = "sample_test_students.json"
            rng = random.Random(args.seed)
            for _ in range(args.n):
                s = rng.choice([s for s in students if len(s["history"]) >= 4])
                t = rng.randint(2, len(s["history"]) - 1)
                prefixes.append((s["history"][:t], s["history"][t]["skill_id"]))
        except Exception as e:
            write_unavailable(f"no data source could be loaded ({e})", cfg)
            return

    # ---- score ------------------------------------------------------------
    rows_a = []  # sample A: actual next skill
    rows_b = []  # sample B: quiz topics
    for hist, nxt in prefixes:
        cands = [(nxt, b) for b in BUCKETS] + [(q, b) for q in quiz_ids for b in BUCKETS]
        ps = svc._predictor(hist, cands)
        pe, pm, ph = ps[0:3]
        rows_a.append((nxt, pe, pm, ph))
        for k, q in enumerate(quiz_ids):
            off = 3 + 3 * k
            rows_b.append((q, ps[off], ps[off + 1], ps[off + 2]))

    def summarize(rows):
        n = len(rows)
        viol = sum(1 for _, pe, pm, ph in rows if is_violation(pe, pm, ph))
        em = sum(1 for _, pe, pm, ph in rows if pe < pm - TOL)
        mh = sum(1 for _, pe, pm, ph in rows if pm < ph - TOL)
        gap_em = sum(pe - pm for _, pe, pm, ph in rows) / n
        gap_mh = sum(pm - ph for _, pe, pm, ph in rows) / n
        by_skill = defaultdict(lambda: [0, 0, 0.0, 0.0])
        for s, pe, pm, ph in rows:
            b = by_skill[s]
            b[0] += 1
            b[1] += int(is_violation(pe, pm, ph))
            b[2] += pe - pm
            b[3] += pm - ph
        return {"n": n, "violations": viol, "violation_rate": viol / n,
                "easy_lt_medium": em / n, "medium_lt_hard": mh / n,
                "mean_gap_easy_medium": gap_em, "mean_gap_medium_hard": gap_mh,
                "by_skill": {k: {"n": v[0], "violation_rate": v[1] / v[0],
                                 "gap_em": v[2] / v[0], "gap_mh": v[3] / v[0]} for k, v in by_skill.items()}}

    A = summarize(rows_a)
    B = summarize(rows_b)
    violation_rate = max(A["violation_rate"], B["violation_rate"])
    reliable = violation_rate <= cfg.monotone_warn
    elapsed = time.time() - t0

    PROFILE_RELIABILITY_PATH.write_text(json.dumps({
        "violation_rate": round(violation_rate, 4),
        "violation_rate_next_skill": round(A["violation_rate"], 4),
        "violation_rate_quiz_topics": round(B["violation_rate"], 4),
        "tolerance": TOL, "monotone_warn": cfg.monotone_warn,
        "reliable": bool(reliable), "n_cases": A["n"], "n_quiz_topic_cases": B["n"],
        "data_source": source, "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    # ---- plot -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.hist([pe - pm for _, pe, pm, ph in rows_a], bins=40, alpha=0.7, label="P(easy) - P(medium)", color="#16a34a")
    ax.hist([pm - ph for _, pe, pm, ph in rows_a], bins=40, alpha=0.7, label="P(medium) - P(hard)", color="#d97706")
    ax.axvline(0, color="#111", lw=1)
    ax.set_title(f"Sample A: gaps for the actual next skill (n={A['n']})")
    ax.set_xlabel("probability gap (positive = ordered as expected)")
    ax.legend(fontsize=8)
    ax = axes[1]
    names = [quiz_names[q][:22] for q in quiz_ids]
    vr = [B["by_skill"][q]["violation_rate"] * 100 for q in quiz_ids]
    ax.barh(names, vr, color="#2563eb")
    ax.axvline(cfg.monotone_warn * 100, color="#dc2626", ls="--", lw=1, label=f"MONOTONE_WARN {cfg.monotone_warn:.0%}")
    ax.set_xlabel("violation rate (%)")
    ax.set_title("Sample B: violation rate per quiz topic")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "monotonicity.png", dpi=130)
    plt.close(fig)

    # ---- report -----------------------------------------------------------
    top = sorted(A["by_skill"].items(), key=lambda kv: -kv[1]["n"])[:10]
    lines = [
        "# Monotonicity check (T1)",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/check_monotonicity.py` "
        f"(seed {args.seed}, {elapsed:.0f}s). Model loaded exactly as the service loads it (`best_full_gated.pt`).",
        "",
        f"Data source: **{source}**. {A['n']} (history prefix, skill) cases; prefixes of length 2..{args.max_prefix}.",
        f"Check: P(easy) >= P(medium) >= P(hard) with tolerance {TOL}.",
        "",
        "| Sample | cases | violation rate | P(easy)<P(medium) | P(medium)<P(hard) | mean gap easy-medium | mean gap medium-hard |",
        "|---|---|---|---|---|---|---|",
        f"| A: actual next skill | {A['n']} | {A['violation_rate']:.1%} | {A['easy_lt_medium']:.1%} | {A['medium_lt_hard']:.1%} | {A['mean_gap_easy_medium']:+.4f} | {A['mean_gap_medium_hard']:+.4f} |",
        f"| B: 10 quiz topics | {B['n']} | {B['violation_rate']:.1%} | {B['easy_lt_medium']:.1%} | {B['medium_lt_hard']:.1%} | {B['mean_gap_easy_medium']:+.4f} | {B['mean_gap_medium_hard']:+.4f} |",
        "",
        f"**Decision (Gate M, pre-registered):** violation_rate = max(A, B) = {violation_rate:.1%}; "
        f"MONOTONE_WARN = {cfg.monotone_warn:.0%} -> profile_reliable = **{str(reliable).lower()}**.",
        "",
        "## Sample B per quiz topic",
        "",
        "| Topic | id | cases | violation rate | mean gap easy-medium | mean gap medium-hard |",
        "|---|---|---|---|---|---|",
    ]
    for q in quiz_ids:
        b = B["by_skill"][q]
        lines.append(f"| {quiz_names[q]} | {q} | {b['n']} | {b['violation_rate']:.1%} | {b['gap_em']:+.4f} | {b['gap_mh']:+.4f} |")
    lines += ["", "## Sample A: 10 most frequent next skills", "",
              "| skill id | cases | violation rate | mean gap easy-medium | mean gap medium-hard |", "|---|---|---|---|---|"]
    for s, b in top:
        lines.append(f"| {s} | {b['n']} | {b['violation_rate']:.1%} | {b['gap_em']:+.4f} | {b['gap_mh']:+.4f} |")
    lines += ["", "![monotonicity](monotonicity.png)", "",
              "Notes: difficulty buckets are per-problem tertiles of training-set success within each skill "
              "(1 = easy, 2 = medium, 3 = hard). The quiz templates' Easy/Medium/Hard levels only approximate "
              "these buckets, so a well-ordered model does not prove the quiz levels are equally spaced.",
              "",
              "Interpretation caveat: the model adds a learned difficulty embedding to the target-skill representation, "
              "so a consistent ordering is expected by construction once training learned it. A low violation rate shows "
              "the ordering is stable across histories; it does NOT show that the size of the gaps is calibrated for "
              "this quiz's questions."]
    (REPORTS_DIR / "monotonicity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[monotonicity] A={A['violation_rate']:.3f} B={B['violation_rate']:.3f} -> reliable={reliable} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
