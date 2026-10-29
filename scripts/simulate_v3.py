"""
scripts/simulate_v3.py
----------------------
v3 Stage 9: policy comparison with paired synthetic students (decision logic only).

Policies (POLICY values of src/serving/adaptive_session.py; all share the same topic
selection and the same allowed-level guardrails; they differ only in the level rule):
  random       uniform random level (baseline)
  staircase    one step per answer on the topic (current demo default)
  elo          Elo/IRT estimate from the student's own answers, level nearest TARGET_P
  maxinfo      CAT-style maximum information: model P nearest 0.50
  model        ZPD target: model P nearest TARGET_P = 0.70 (existing POLICY=model)
  uncertainty  ensemble of the 3 full_gated checkpoints, nearest 0.70; staircase level and a
               "low-confidence prediction" flag when the ensemble spread > 0.15

Pre-registered protocol (reports/v3/DEMO_CONFIG.md, written BEFORE the run):
  primary setting = the current demo setting (SESSION_MODE=core5, TOPIC_SELECTION=
  evidence_blend, APPLY_CALIBRATION=true); 100 paired students per policy; seed 2026;
  student model as in v2 (g ~ N(0, 0.8); a_t = g + N(0, 0.7);
  P_true = 0.05 + 0.95 sigmoid(a_t - b), b = -1 / 0 / +1 for easy / medium / hard).
  Exploratory (not used by any gate): the full10 + model-topic-selection setting of v2.

Metrics per student (then mean, std, 95% t-CI; paired differences vs staircase with
paired t-test, Wilcoxon, bootstrap CI, Holm correction over policies within a metric):
  inband      share of main questions (after the 2 warm-ups) with true P in [0.55, 0.85]
  status_acc  final profile status == true status (topics with a status)
  coverage    topics with a status / topics of the session
  to_stable   answers on a topic until its evidence-only status stops changing
  ready_prec  share of ready_for_next flags that are truly ready (P_true(next) >= T_READY)
Synthetic students are not drawn from the model's training data: this validates the
decision logic only, not learning and not knowledge-tracing accuracy.

Usage: python scripts/simulate_v3.py [--students 100] [--seed 2026] [--exploratory]
Outputs: reports/v3/policy_comparison.md / .csv / _summary.json / _*.png
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, seed_everything  # noqa: E402

import numpy as np  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "v3"
POLICIES = ["random", "staircase", "elo", "maxinfo", "model", "uncertainty"]
LABEL = {"random": "random", "staircase": "staircase", "elo": "Elo/IRT target", "maxinfo": "max-information (CAT)",
         "model": "ZPD target 0.70 (model)", "uncertainty": "uncertainty-aware (ensemble)"}
B_LEVEL = {"easy": -1.0, "medium": 0.0, "hard": 1.0}
GUESS = 0.05
BAND = (0.55, 0.85)
NEXT_LEVEL = {"developing": "medium", "proficient": "hard"}
METRICS = ["inband", "status_acc", "coverage", "to_stable", "ready_prec"]
PRIMARY_ENV = {"SESSION_MODE": "core5", "TOPIC_SELECTION": "evidence_blend", "APPLY_CALIBRATION": "true"}
EXPLORATORY_ENV = {"SESSION_MODE": "full10", "TOPIC_SELECTION": "model", "APPLY_CALIBRATION": "false"}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def p_true(a, level):
    return GUESS + (1 - GUESS) * sigmoid(a - B_LEVEL[level])


def true_status(a, t_prof):
    if p_true(a, "hard") >= t_prof:
        return "mastered"
    if p_true(a, "medium") >= t_prof:
        return "proficient"
    if p_true(a, "easy") >= t_prof:
        return "developing"
    return "needs_support"


def make_students(n, seed, topic_ids):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = float(rng.normal(0.0, 0.8))
        out.append({"idx": i, "g": g, "ability": {t: g + float(rng.normal(0.0, 0.7)) for t in topic_ids}})
    return out


def answers_to_stable(history, topic, cfg):
    from serving.mastery import fallback_status, topic_evidence
    th = [h for h in history if int(h["topic"]) == topic]
    if len(th) < cfg.min_answers:
        return None
    statuses = []
    for k in range(1, len(th) + 1):
        prefix = [h for h in history if int(h["topic"]) == topic][:k]
        st = fallback_status(topic_evidence(prefix, topic), cfg)
        statuses.append(st[0] if isinstance(st, tuple) else st)
    last = statuses[-1]
    k = len(statuses)
    while k > 1 and statuses[k - 2] == last:
        k -= 1
    return k


def run_one(svc, student, policy, seed, cfg, topics, reliable, env):
    from serving.adaptive_session import AdaptiveSession
    from serving.mastery import build_profile
    random.seed(seed * 100003 + student["idx"])            # same topic randomness across policies
    ans_rng = random.Random(seed * 7919 + student["idx"])   # same answer stream across policies
    sess = AdaptiveSession(f"sim-{policy}-{student['idx']}", f"simv3_{student['idx']}", policy=policy,
                           session_mode=env["SESSION_MODE"], topic_selection=env["TOPIC_SELECTION"])
    served, low_conf = [], 0
    while True:
        q = sess.next_question(svc._predictor)
        if q is None:
            break
        if (q.get("decision") or {}).get("low_confidence"):
            low_conf += 1
        pt = p_true(student["ability"][q["topic"]], q["level"])
        correct = ans_rng.random() < pt
        sess.record_answer(q["correct_answer"] if correct else "__wrong__", 5000.0)
        served.append({"topic": q["topic"], "level": q["level"], "p_true": pt, "is_probe": q["is_probe"]})
    prof = build_profile(sess.history, topics, predictor=svc._predictor, session_history=sess.history,
                         cfg=cfg, reliable=reliable)
    main = [s for s in served if not s["is_probe"]][2:]
    inband = float(np.mean([BAND[0] <= s["p_true"] <= BAND[1] for s in main])) if main else float("nan")
    ev = [t for t in prof["topics"] if t["status"] != "not_enough_evidence"]
    hits = [int(t["status"] == true_status(student["ability"][t["topic"]], cfg.t_prof)) for t in ev]
    flags = [p_true(student["ability"][t["topic"]], NEXT_LEVEL[t["status"]]) >= cfg.t_ready
             for t in prof["topics"] if t["status"] in NEXT_LEVEL and t["ready_for_next"]]
    stab = [answers_to_stable(sess.history, t["topic"], cfg) for t in ev]
    stab = [s for s in stab if s is not None]
    return {"inband": inband, "status_acc": float(np.mean(hits)) if hits else float("nan"),
            "status_hits": sum(hits), "status_n": len(hits),
            "coverage": len(ev) / len(sess.main_topics), "to_stable": float(np.mean(stab)) if stab else float("nan"),
            "ready_prec": float(np.mean(flags)) if flags else float("nan"), "ready_flags": len(flags),
            "ready_true": int(sum(flags)), "low_conf": low_conf, "n_q": len(served),
            "fallbacks": sess.policy_fallbacks,
            "levels": [s["level"] for s in main]}


def summarise(per, students, args):
    from scipy import stats
    summ, comps = {}, []
    for p in POLICIES:
        rs = per[p]
        rec = {}
        for m in METRICS:
            v = np.asarray([r[m] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            h = float(stats.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v))) if len(v) > 1 else 0.0
            rec[m] = {"mean": float(v.mean()) if len(v) else float("nan"), "std": sd, "ci": [float(v.mean()) - h, float(v.mean()) + h] if len(v) else [float("nan")] * 2, "n": int(len(v))}
        rec["status_acc_pooled"] = sum(r["status_hits"] for r in rs) / max(1, sum(r["status_n"] for r in rs))
        rec["ready_prec_pooled"] = sum(r["ready_true"] for r in rs) / max(1, sum(r["ready_flags"] for r in rs))
        rec["ready_flags"] = sum(r["ready_flags"] for r in rs)
        rec["low_conf_per_session"] = float(np.mean([r["low_conf"] for r in rs]))
        rec["main_decisions_per_session"] = float(np.mean([len(r["levels"]) for r in rs]))
        rec["fallbacks"] = int(sum(r["fallbacks"] for r in rs))
        lv = [l for r in rs for l in r["levels"]]
        rec["level_mix"] = {k: lv.count(k) / max(1, len(lv)) for k in ("easy", "medium", "hard")}
        summ[p] = rec
    rng = np.random.default_rng(args.seed + 1)
    for m in METRICS:
        block = []
        for p in POLICIES:
            if p == "staircase":
                continue
            a = np.asarray([r[m] for r in per[p]], dtype=float)
            b = np.asarray([r[m] for r in per["staircase"]], dtype=float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            d = a[ok] - b[ok]
            if len(d) > 1 and d.std(ddof=1) > 0:
                tp = float(stats.ttest_rel(a[ok], b[ok]).pvalue)
                h = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
            else:
                tp, h = 1.0, 0.0
            try:
                wp = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                wp = float("nan")
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)] if len(d) else [float("nan")]
            block.append({"metric": m, "policy": p, "n_pairs": int(len(d)), "diff_mean": float(d.mean()) if len(d) else float("nan"),
                          "t_ci_low": float(d.mean() - h) if len(d) else float("nan"), "t_ci_high": float(d.mean() + h) if len(d) else float("nan"),
                          "boot_low": float(np.percentile(boots, 2.5)), "boot_high": float(np.percentile(boots, 97.5)),
                          "t_p": tp, "wilcoxon_p": wp})
        ps = np.asarray([c["t_p"] for c in block])
        order = np.argsort(ps)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
            block[i]["t_p_holm"] = running
        comps += block
    return summ, comps


def plots(summ, per, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, title) in zip(axes, [("inband", "In-band share"), ("status_acc", "Status accuracy"), ("coverage", "Coverage")]):
        means = [summ[p][m]["mean"] for p in POLICIES]
        lo = [summ[p][m]["mean"] - summ[p][m]["ci"][0] for p in POLICIES]
        hi = [summ[p][m]["ci"][1] - summ[p][m]["mean"] for p in POLICIES]
        ax.bar(range(len(POLICIES)), means, color=colors, width=0.6, edgecolor="#fcfcfb", linewidth=2)
        ax.errorbar(range(len(POLICIES)), means, yerr=[lo, hi], fmt="none", ecolor="#0b0b0b", capsize=3, linewidth=1)
        ax.set_xticks(range(len(POLICIES)))
        ax.set_xticklabels(POLICIES, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title + " (mean, 95% CI)", loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    fig.suptitle("Policy comparison, paired synthetic students (decision logic only)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"policy_comparison{tag}.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, p in enumerate(POLICIES):
        v = [r["inband"] for r in per[p] if not math.isnan(r["inband"])]
        ax.boxplot([v], positions=[i], widths=0.5, patch_artist=True,
                   boxprops={"facecolor": colors[i], "alpha": 0.8, "edgecolor": "#fcfcfb"},
                   medianprops={"color": "#0b0b0b"}, flierprops={"markersize": 3})
    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels(POLICIES)
    ax.set_ylabel("In-band share per student")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_title("Distribution of per-student in-band share", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"policy_inband_dist{tag}.png", dpi=150)
    plt.close(fig)


def run_setting(env, args, tag):
    for k, v in env.items():
        os.environ[k] = v
    from _common import load_service
    svc = load_service()
    from serving.adaptive_session import AdaptiveSession, TOPICS
    from serving.mastery import load_profile_reliability
    from serving.policy_config import get_config
    cfg = get_config()
    reliable = load_profile_reliability()["reliable"]
    topics = [{"raw_skill_id": t["raw_skill_id"], "skill_name": t["skill_name"]} for t in TOPICS]
    students = make_students(args.students, args.seed, [t["raw_skill_id"] for t in TOPICS])
    t0 = time.time()
    per = {p: [] for p in POLICIES}
    for s in students:
        for p in POLICIES:
            per[p].append(run_one(svc, s, p, args.seed, cfg, topics, reliable, env))
    elapsed = time.time() - t0
    summ, comps = summarise(per, students, args)
    rows = []
    for p in POLICIES:
        for s, r in zip(students, per[p]):
            rows.append({"setting": tag or "primary", "policy": p, "student": s["idx"], "g": round(s["g"], 4),
                         **{m: (None if (isinstance(r[m], float) and math.isnan(r[m])) else round(r[m], 4)) for m in METRICS},
                         "low_conf_flags": r["low_conf"], "n_questions": r["n_q"]})
    with open(OUT / f"policy_comparison{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plots(summ, per, tag)
    return {"env": env, "elapsed_s": round(elapsed, 1), "reliable": reliable, "summary": summ,
            "paired_vs_staircase": comps, "students": args.students, "seed": args.seed,
            "cfg": {"target_p": cfg.target_p, "t_prof": cfg.t_prof, "t_ready": cfg.t_ready, "min_answers": cfg.min_answers}}


def _p4(v):
    return "n/a" if v != v else f"{v:.4f}"


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.2f}"


def write_md(res, exp):
    s = res["summary"]
    L = ["# Policy comparison (v3 Stage 9)", "",
         f"Generated by `python scripts/simulate_v3.py --students {res['students']} --seed {res['seed']}"
         f"{' --exploratory' if exp else ''}` in {res['elapsed_s']:.0f}s. Pre-registration: reports/v3/DEMO_CONFIG.md.", "",
         "> **Scope.** Synthetic students validate the decision logic only (which level is served, how statuses "
         "are assigned). They are not drawn from real data, say nothing about learning gains, and are not a test of "
         "knowledge-tracing accuracy.", "",
         f"Primary setting (= current demo setting): {res['env']}. {res['students']} paired students per policy: the "
         "same abilities, the same topic randomness and the same answer random stream for every policy. Policy "
         f"parameters fixed before the run: TARGET_P {res['cfg']['target_p']}, max-information target 0.50, "
         f"uncertainty spread limit 0.15, T_PROF {res['cfg']['t_prof']}, T_READY {res['cfg']['t_ready']}, "
         f"MIN_ANSWERS {res['cfg']['min_answers']}. Profile reliable: {str(res['reliable']).lower()}.", "",
         "| Policy | In-band share | Status accuracy | Coverage | Answers to stable status | Ready-flag precision | Level mix E/M/H | Low-confidence flags / session |",
         "|---|---|---|---|---|---|---|---|"]
    for p in POLICIES:
        r = s[p]
        cell = lambda m, pct=True: f"{fmt(r[m]['mean'], pct)} ± {fmt(r[m]['std'], pct)} [{fmt(r[m]['ci'][0], pct)}, {fmt(r[m]['ci'][1], pct)}]"
        L.append(f"| {LABEL[p]} | {cell('inband')} | {cell('status_acc')} (pooled {fmt(r['status_acc_pooled'])}) | "
                 f"{cell('coverage')} | {cell('to_stable', False)} | {fmt(r['ready_prec_pooled'])} of {r['ready_flags']} flags | "
                 + "/".join(f"{r['level_mix'][k]:.0%}" for k in ("easy", "medium", "hard"))
                 + f" | {r['low_conf_per_session']:.2f} |")
    L += ["", "Cells: mean ± std over students [95% t-CI of the mean]. Status accuracy per student is averaged over "
          "the topics that received a status; 'pooled' = all hits / all statuses (the v2 gate statistic).", "",
          "## Paired differences vs staircase", "",
          "| Metric | Policy | Pairs | Mean diff | 95% CI (t) | 95% CI (bootstrap) | t-test p (Holm) | Wilcoxon p |",
          "|---|---|---|---|---|---|---|---|"]
    for c in res["paired_vs_staircase"]:
        pct = c["metric"] != "to_stable"
        f2 = (lambda v: f"{v:+.1%}") if pct else (lambda v: f"{v:+.2f}")
        L.append(f"| {c['metric']} | {c['policy']} | {c['n_pairs']} | {f2(c['diff_mean'])} | [{f2(c['t_ci_low'])}, "
                 f"{f2(c['t_ci_high'])}] | [{f2(c['boot_low'])}, {f2(c['boot_high'])}] | {c['t_p_holm']:.4f} | "
                 f"{_p4(c['wilcoxon_p'])} |")
    u = s["uncertainty"]
    share = u["low_conf_per_session"] / max(u["main_decisions_per_session"], 1e-9)
    L += ["", f"**Uncertainty-aware policy:** the ensemble spread exceeded the pre-registered limit (0.15) on "
          f"{u['low_conf_per_session']:.2f} of {u['main_decisions_per_session']:.2f} level decisions per session "
          f"({share:.0%}); those decisions used the staircase level. Its results therefore largely reflect the "
          "staircase rule, not the ensemble prediction."]
    L += ["", f"![metrics](policy_comparison{'_exploratory' if exp else ''}.png)", "",
          f"![in-band distribution](policy_inband_dist{'_exploratory' if exp else ''}.png)", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--exploratory", action="store_true", help="also run the v2 full10 / model-topic setting")
    args = ap.parse_args()
    seed_everything(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    res = run_setting(PRIMARY_ENV, args, "")
    out = {"primary": res}
    summary_path = OUT / "policy_comparison_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")   # saved before the report
    L = write_md(res, False)
    if args.exploratory:
        ex = run_setting(EXPLORATORY_ENV, args, "_exploratory")
        out["exploratory"] = ex
        summary_path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        L += ["## Exploratory (not used by any gate)", "",
              "Same students and policies in the v2 simulation setting (full10, model topic selection, no calibration).", ""]
        L_ex = write_md(ex, True)
        L += L_ex[next(i for i, l in enumerate(L_ex) if l.startswith("| Policy |")):]
    (OUT / "policy_comparison_summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (OUT / "policy_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("[simulate_v3] " + "; ".join(f"{p}: in-band {res['summary'][p]['inband']['mean']:.3f}, status "
                                      f"{res['summary'][p]['status_acc_pooled']:.3f}" for p in POLICIES))


if __name__ == "__main__":
    main()
