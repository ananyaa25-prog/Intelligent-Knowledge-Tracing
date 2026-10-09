"""
scripts/simulate.py
-------------------
T6: Policy comparison with synthetic students (decision-logic validation only).

Drives AdaptiveSession directly (no HTTP) with the REAL served predictor.
Synthetic students: general ability g ~ N(0, 0.8); topic ability a_t = g + N(0, 0.7);
P_true(level) = guess + (1 - guess) * sigmoid(a_t - b_level), b = {easy: -1, medium: 0, hard: 1},
guess = 0.05. The student answers correctly with probability P_true.
The SAME students (paired: same abilities, same random streams) are run under
POLICY = random, staircase, model. Topic selection code is identical for all.

Metrics
 1. in-band share: share of questions whose TRUE success probability is in [0.55, 0.85]
 2. status accuracy of final profile statuses vs ground truth derived from P_true with the
    same T_PROF (topics with >= MIN_ANSWERS answers, i.e. a non-"not_enough_evidence" status),
    plus coverage (share of the 10 topics that received such a status)
 3. answers per topic needed before a non-"not_enough_evidence" status
 4. share of ready_for_next flags that are truly ready (P_true(next level) >= T_READY)

IMPORTANT: synthetic students are not drawn from the model's training distribution. This
validates the decision logic only - not real learning and not knowledge-tracing accuracy.

Usage: python scripts/simulate.py [--students 150] [--seed 2026]
Outputs: reports/simulation.md, reports/simulation.csv, reports/simulation_inband.png,
         reports/simulation_status.png, reports/simulation_summary.json
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPORTS_DIR, load_service, seed_everything

import numpy as np

POLICIES = ["random", "staircase", "model"]
B_LEVEL = {"easy": -1.0, "medium": 0.0, "hard": 1.0}
GUESS = 0.05
BAND = (0.55, 0.85)
NEXT_LEVEL = {"developing": "medium", "proficient": "hard"}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def p_true(a, level):
    return GUESS + (1 - GUESS) * sigmoid(a - B_LEVEL[level])


def true_status(a, t_prof):
    pe, pm, ph = p_true(a, "easy"), p_true(a, "medium"), p_true(a, "hard")
    if ph >= t_prof:
        return "mastered"
    if pm >= t_prof:
        return "proficient"
    if pe >= t_prof:
        return "developing"
    return "needs_support"


def make_students(n, seed, topic_ids):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = float(rng.normal(0.0, 0.8))
        abil = {t: g + float(rng.normal(0.0, 0.7)) for t in topic_ids}
        out.append({"idx": i, "g": g, "ability": abil})
    return out


def run_one(svc, student, policy, seed, cfg, topics, reliable, session_mode="full10", topic_selection="model"):
    from serving.adaptive_session import AdaptiveSession
    from serving.mastery import build_profile, STATUS_ORDER
    random.seed(seed * 100003 + student["idx"])           # same topic randomness across policies
    ans_rng = random.Random(seed * 7919 + student["idx"])  # same answer stream across policies
    sess = AdaptiveSession(f"sim-{policy}-{student['idx']}", f"simstu_{student['idx']}",
                           policy=policy, session_mode=session_mode, topic_selection=topic_selection)
    served = []
    while True:
        q = sess.next_question(svc._predictor)
        if q is None:
            break
        pt = p_true(student["ability"][q["topic"]], q["level"])
        correct = ans_rng.random() < pt
        sess.record_answer(q["correct_answer"] if correct else "__wrong__", 5000.0)
        served.append({"topic": q["topic"], "level": q["level"], "p_true": pt, "p_pred": q["predicted_p"],
                       "is_probe": q["is_probe"], "correct": int(correct)})
    prof = build_profile(sess.history, topics, predictor=svc._predictor, session_history=sess.history,
                         cfg=cfg, reliable=reliable)
    evals = []
    for t in prof["topics"]:
        a = student["ability"][t["topic"]]
        truth = true_status(a, cfg.t_prof)
        row = {"topic": t["topic"], "status": t["status"], "truth": truth, "n": t["n_answers"],
               "ready": t["ready_for_next"], "truly_ready": None}
        if t["status"] in NEXT_LEVEL and t["ready_for_next"]:
            row["truly_ready"] = p_true(a, NEXT_LEVEL[t["status"]]) >= cfg.t_ready
        evals.append(row)
    main = [s for s in served if not s["is_probe"]]
    return {"served": served, "main": main, "evals": evals, "fallbacks": sess.policy_fallbacks}


def bootstrap_ci(values, seed, n_boot=2000):
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--session-mode", default="full10", choices=["full10", "core5"])
    ap.add_argument("--topic-selection", default="model", choices=["model", "evidence_blend"])
    ap.add_argument("--tag", default="", help="suffix for output files (main report when empty)")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    seed_everything(args.seed)
    t0 = time.time()

    svc = load_service()
    from serving.adaptive_session import TOPICS, TOPIC_IDS
    from serving.mastery import load_profile_reliability, STATUS_ORDER
    from serving.policy_config import get_config
    cfg = get_config()
    reliable = load_profile_reliability()["reliable"]
    topics = [{"raw_skill_id": t["raw_skill_id"], "skill_name": t["skill_name"]} for t in TOPICS]
    students = make_students(args.students, args.seed, TOPIC_IDS)

    per = {p: [] for p in POLICIES}
    for s in students:
        for p in POLICIES:
            per[p].append(run_one(svc, s, p, args.seed, cfg, topics, reliable, args.session_mode, args.topic_selection))
    elapsed = time.time() - t0

    # ---- aggregate --------------------------------------------------------
    summary = {}
    csv_rows = []
    for p in POLICIES:
        inband_q, inband_stu, acc_stu, cov_stu, within1 = [], [], [], [], []
        status_hits = status_n = 0
        ready_flags = ready_true = 0
        first_status_answers = []
        fallbacks = 0
        level_counts = {"easy": 0, "medium": 0, "hard": 0}
        for s, r in zip(students, per[p]):
            fallbacks += r["fallbacks"]
            m = r["main"][2:]  # exclude the 2 warm-ups (Medium for every policy)
            flags = [BAND[0] <= q["p_true"] <= BAND[1] for q in m]
            inband_q.extend(flags)
            inband_stu.append(float(np.mean(flags)) if flags else float("nan"))
            for q in m:
                level_counts[q["level"]] += 1
            ev = [e for e in r["evals"] if e["status"] != "not_enough_evidence"]
            hits = sum(1 for e in ev if e["status"] == e["truth"])
            near = sum(1 for e in ev if abs(STATUS_ORDER.index(e["status"]) - STATUS_ORDER.index(e["truth"])) <= 1)
            status_hits += hits
            status_n += len(ev)
            acc_stu.append(hits / len(ev) if ev else float("nan"))
            within1.append(near / len(ev) if ev else float("nan"))
            cov_stu.append(len(ev) / len(TOPIC_IDS))
            for e in r["evals"]:
                if e["n"] >= cfg.min_answers:
                    first_status_answers.append(cfg.min_answers)
                if e["truly_ready"] is not None:
                    ready_flags += 1
                    ready_true += int(e["truly_ready"])
            csv_rows.append({"policy": p, "student": s["idx"], "g": round(s["g"], 4),
                             "n_questions": len(r["served"]), "inband_share": round(inband_stu[-1], 4),
                             "status_accuracy": None if math.isnan(acc_stu[-1]) else round(acc_stu[-1], 4),
                             "coverage": round(cov_stu[-1], 4),
                             "ready_flags": sum(1 for e in r["evals"] if e["truly_ready"] is not None),
                             "ready_true": sum(1 for e in r["evals"] if e["truly_ready"])})
        total_levels = sum(level_counts.values())
        summary[p] = {
            "inband_share": float(np.mean(inband_q)),
            "inband_ci": bootstrap_ci([x for x in inband_stu if not math.isnan(x)], args.seed),
            "status_accuracy": status_hits / status_n if status_n else float("nan"),
            "status_accuracy_ci": bootstrap_ci([x for x in acc_stu if not math.isnan(x)], args.seed),
            "status_within_one": float(np.nanmean(within1)),
            "coverage": float(np.mean(cov_stu)),
            "answers_before_status": float(np.mean(first_status_answers)) if first_status_answers else float("nan"),
            "topics_with_status": len(first_status_answers),
            "ready_flags": ready_flags,
            "ready_precision": ready_true / ready_flags if ready_flags else float("nan"),
            "level_mix": {k: v / total_levels for k, v in level_counts.items()} if total_levels else {},
            "policy_fallbacks": fallbacks,
            "n_questions_scored": len(inband_q),
        }

    # Paired differences (model - staircase) per student.
    def paired(metric_idx_key):
        d = []
        for a, b in zip([r for r in csv_rows if r["policy"] == "model"], [r for r in csv_rows if r["policy"] == "staircase"]):
            if a[metric_idx_key] is not None and b[metric_idx_key] is not None:
                d.append(a[metric_idx_key] - b[metric_idx_key])
        return float(np.mean(d)) if d else float("nan"), bootstrap_ci(d, args.seed + 1)

    diff_inband = paired("inband_share")
    diff_acc = paired("status_accuracy")

    # ---- Gate S (pre-registered) --------------------------------------------
    from serving.mastery import load_profile_reliability as _lpr
    gate_m = _lpr()["reliable"]
    gate_s = bool(gate_m
                  and summary["model"]["inband_share"] >= summary["staircase"]["inband_share"] - 0.02
                  and summary["model"]["status_accuracy"] >= summary["staircase"]["status_accuracy"] - 0.03)

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / f"simulation{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    out = {"seed": args.seed, "students_per_policy": args.students, "session_mode": args.session_mode,
           "topic_selection": args.topic_selection, "elapsed_s": round(elapsed, 1),
           "profile_reliable": reliable, "summary": summary,
           "paired_model_minus_staircase": {"inband_share": diff_inband, "status_accuracy": diff_acc},
           "gate_s_model_policy": gate_s, "generated_at": datetime.now(timezone.utc).isoformat()}
    (REPORTS_DIR / f"simulation_summary{sfx}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- charts -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"random": "#94a3b8", "staircase": "#d97706", "model": "#2563eb"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 26)
    for p in POLICIES:
        vals = [q["p_true"] for r in per[p] for q in r["main"][2:]]
        ax.hist(vals, bins=bins, histtype="step", lw=2, color=colors[p],
                label=f"{p} (in band {summary[p]['inband_share']:.1%})")
    ax.axvspan(BAND[0], BAND[1], color="#16a34a", alpha=0.08, label="target band 0.55-0.85")
    ax.set_xlabel("TRUE success probability of the question served (synthetic student)")
    ax.set_ylabel("questions")
    ax.set_title(f"Difficulty fit by policy ({args.students} paired synthetic students each)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"simulation_inband{sfx}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    metrics = [("inband_share", "in-band share"), ("status_accuracy", "status accuracy"),
               ("coverage", "coverage"), ("ready_precision", "ready precision")]
    x = np.arange(len(metrics))
    wdt = 0.26
    for k, p in enumerate(POLICIES):
        vals = [summary[p][m] for m, _ in metrics]
        ax.bar(x + (k - 1) * wdt, vals, wdt, color=colors[p], label=p)
        for xi, v in zip(x + (k - 1) * wdt, vals):
            if not math.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_title("Simulation metrics by policy (synthetic students; decision logic only)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"simulation_status{sfx}.png", dpi=130)
    plt.close(fig)

    # ---- report ---------------------------------------------------------------
    def f(v, pct=True):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{v:.1%}" if pct else f"{v:.2f}"

    L = ["# Policy simulation (T6)", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/simulate.py` "
         f"(seed {args.seed}, {args.students} paired synthetic students per policy, {elapsed:.0f}s). "
         f"Session mode {args.session_mode}, topic selection {args.topic_selection}; level rules as in "
         "adaptive_session.py (revised D27).",
         "",
         "> **Scope.** Synthetic students are NOT drawn from the model's training distribution (ASSISTments). "
         "This validates the decision logic (which level gets served, how statuses are assigned) only. It says "
         "nothing about real learning gains and is not a test of knowledge-tracing accuracy.",
         "",
         "Student model: g ~ N(0, 0.8); a_t = g + N(0, 0.7); P_true = 0.05 + 0.95 * sigmoid(a_t - b), "
         "b = easy -1 / medium 0 / hard +1. The real served predictor (`best_full_gated.pt`) drives topic choice "
         f"and (for POLICY=model) the level. Profile mode: profile_reliable = {str(reliable).lower()}. "
         "Policy parameters were fixed before running (TARGET_P = "
         f"{cfg.target_p}, T_PROF = {cfg.t_prof}, T_READY = {cfg.t_ready}, MIN_ANSWERS = {cfg.min_answers}).",
         "",
         "| Metric | random | staircase | model |", "|---|---|---|---|",
         "| 1. In-band share (true P in [0.55, 0.85]; excl. 2 warm-ups) | "
         + " | ".join(f"{f(summary[p]['inband_share'])} (95% CI {f(summary[p]['inband_ci'][0])}-{f(summary[p]['inband_ci'][1])})" for p in POLICIES) + " |",
         "| 2. Status accuracy (exact) | "
         + " | ".join(f"{f(summary[p]['status_accuracy'])} (95% CI {f(summary[p]['status_accuracy_ci'][0])}-{f(summary[p]['status_accuracy_ci'][1])})" for p in POLICIES) + " |",
         "| 2b. Status within one step | " + " | ".join(f(summary[p]["status_within_one"]) for p in POLICIES) + " |",
         "| 2c. Coverage (topics with a status / 10) | " + " | ".join(f(summary[p]["coverage"]) for p in POLICIES) + " |",
         "| 3. Answers per topic before first status | " + " | ".join(f(summary[p]["answers_before_status"], False) for p in POLICIES) + " |",
         "| 4. Ready-for-next precision (flags truly ready) | " + " | ".join(f"{f(summary[p]['ready_precision'])} of {summary[p]['ready_flags']} flags" for p in POLICIES) + " |",
         "| Level mix served (easy / medium / hard) | " + " | ".join(
             "/".join(f"{summary[p]['level_mix'].get(k, 0):.0%}" for k in ["easy", "medium", "hard"]) for p in POLICIES) + " |",
         "| Model-policy fallbacks to staircase | - | - | " + str(summary["model"]["policy_fallbacks"]) + " |",
         "",
         f"Paired difference model - staircase (per student, 95% bootstrap CI): in-band share "
         f"{diff_inband[0]:+.3f} ({diff_inband[1][0]:+.3f} to {diff_inband[1][1]:+.3f}); status accuracy "
         f"{diff_acc[0]:+.3f} ({diff_acc[1][0]:+.3f} to {diff_acc[1][1]:+.3f}).",
         "",
         "## Gate S (pre-registered)",
         "",
         f"Model policy is used for the demo only if Gate M passes (profile_reliable = {str(gate_m).lower()}) AND "
         f"model in-band share >= staircase - 0.02 ({f(summary['model']['inband_share'])} vs "
         f"{f(summary['staircase']['inband_share'] - 0.02)}) AND model status accuracy >= staircase - 0.03 "
         f"({f(summary['model']['status_accuracy'])} vs {f(summary['staircase']['status_accuracy'] - 0.03)}).",
         "",
         f"**Result: Gate S {'PASSES' if gate_s else 'FAILS'}.**",
         "",
         "Metric 3 note: with the pre-registered rule a topic receives a status as soon as it has MIN_ANSWERS = "
         f"{cfg.min_answers} answers, so metric 3 equals {cfg.min_answers} by construction; what differs between "
         "policies is coverage (2c), i.e. how many topics reach that many answers.",
         "",
         f"![in-band](simulation_inband{sfx}.png)", "", f"![metrics](simulation_status{sfx}.png)", ""]
    (REPORTS_DIR / f"simulation{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({p: {k: summary[p][k] for k in ["inband_share", "status_accuracy", "coverage", "ready_precision"]} for p in POLICIES}, indent=1))
    print(f"[simulate] Gate S = {gate_s}; elapsed {elapsed:.0f}s")


if __name__ == "__main__":
    main()
