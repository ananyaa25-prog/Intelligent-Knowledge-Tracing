"""
scripts/benchmark_baselines_v3.py
---------------------------------
Our model against four baselines on the existing ASSISTments split (official
train / val / test split by student, windowed_data.pkl).

Models
  pfa            Performance Factors Analysis            src/baselines/pfa.py      --classical-seeds seeds
  bkt            Bayesian Knowledge Tracing (own EM)      src/baselines/bkt.py      --classical-seeds seeds
  dkt_gru        DKT with a GRU (our DKT-only code path)  src/baselines/dkt.py      --seeds seeds
  sakt_lite      one self-attention layer ("-lite")       src/baselines/sakt.py     --seeds seeds
  graph_no_gate  our existing checkpoints, read-only      src/baselines/ours.py     seeds 42 / 123 / 2024
  full_gated     our existing checkpoints, read-only      src/baselines/ours.py     seeds 42 / 123 / 2024

Inputs: skill, correct, difficulty bucket for every trained model. The existing GatedKT
checkpoints are wrapped as trained (they also read response time / attempts / hints);
they are not retrained and no checkpoint is written (docs/DECISIONS.md D41).
Deep baselines: Adam 1e-3, batch 32, --epochs epochs (no early stop), best-validation
weights. Every seed is printed; each finished (model, seed) is appended to
reports/v3/benchmark_runs.csv at once and its predictions cached, so a rerun resumes.

Variance: (a) seeds -- spread of per-seed test AUC (std, t-CI); (b) bootstrap -- 1000
resamples of TEST STUDENTS of the seed-averaged AUC. Paired test of full_gated against
each baseline: Wilcoxon signed-rank over test students on the per-student Brier score
(seed-averaged), Holm-corrected; "improves over X" only if the corrected p < 0.05 and the
bootstrap 95% CI of the AUC difference excludes 0 (seeds cannot be matched: the
checkpoints use seeds 42/123/2024).

Usage: python scripts/benchmark_baselines_v3.py [--seeds 10] [--classical-seeds 50] [--epochs 10]
       [--max_students 0] [--workers 6] [--bootstrap 1000] [--report-only]
Outputs: reports/v3/benchmark.md, benchmark.csv, benchmark_runs.csv, benchmark_boxplot.png
"""

import argparse
import csv
import os
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baselines.base import all_metrics, mean_ci  # noqa: E402

OUT = ROOT / "reports" / "v3"
PRED = ROOT / "data" / "processed_v3" / "baselines_v3_preds"
RUNS = OUT / "benchmark_runs.csv"
CLASSICAL = ["pfa", "bkt"]
DEEP = ["dkt_gru", "sakt_lite"]
OURS = ["graph_no_gate", "full_gated"]
CKPT_SEEDS = [42, 123, 2024]
ORDER = OURS[::-1] + DEEP + CLASSICAL
LABEL = {"full_gated": "full gated (ours)", "graph_no_gate": "graph, no gate (ours)", "dkt_gru": "DKT (GRU)",
         "sakt_lite": "SAKT-lite", "pfa": "PFA", "bkt": "BKT"}
RUN_FIELDS = ["model", "seed", "val_auc", "auc", "acc", "brier", "ece", "n", "seconds"]

_D = {}


def _data():
    if not _D:
        with open(ROOT / "data" / "processed" / "windowed_data.pkl", "rb") as f:
            d = pickle.load(f)
        _D.update(d["splits"])
        _D["num_skills"] = int(d["num_skills"])
    return _D


def _subset(split, n):
    if not n or split["mask"].shape[0] <= n:
        return split
    idx = np.sort(np.random.default_rng(0).permutation(split["mask"].shape[0])[:n])
    return {k: v[idx] for k, v in split.items()}


def _pred_path(model, seed, tag):
    return PRED / f"{model}__s{seed}__{tag}.npz"


def _task(t):
    model, seed, epochs, max_students, tag = t
    import random
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"  start {model} seed {seed}", flush=True)
    from baselines.ours import rt_stats_from_split
    d = _data()
    t0 = time.time()
    val_auc = float("nan")
    if model in CLASSICAL:
        from baselines import make_classical
        m = make_classical(model, d["num_skills"], seed=seed).fit(_subset(d["train"], max_students))
        pv, yv, _ = m.predict_split(d["val"])
        val_auc = all_metrics(yv, pv)["auc"]
    elif model in OURS:
        from baselines.ours import OursVariant
        m = OursVariant.from_checkpoint(model, seed, num_skills=d["num_skills"], rt_stats=rt_stats_from_split(d["train"]))
        pv, yv, _ = m.predict_split(d["val"])
        val_auc = all_metrics(yv, pv)["auc"]
    else:
        from baselines.deep import make_deep
        m = make_deep(model, d["num_skills"], seed=seed, epochs=epochs, rt_stats=rt_stats_from_split(d["train"]))
        m.fit(_subset(d["train"], max_students), d["val"])
        val_auc = float(m.best_val_auc)
    p, y, owner = m.predict_split(d["test"])
    secs = time.time() - t0
    PRED.mkdir(parents=True, exist_ok=True)
    tmp = _pred_path(model, seed, tag).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, p=p.astype(np.float32), y=y.astype(np.int8), owner=owner.astype(np.int32))
    os.replace(tmp, _pred_path(model, seed, tag))
    return dict(all_metrics(y, p), model=model, seed=seed, val_auc=val_auc, seconds=round(secs, 1))


def _done():
    if not RUNS.exists():
        return set()
    with open(RUNS, newline="", encoding="utf-8") as f:
        return {(r["model"], int(r["seed"])) for r in csv.DictReader(f)}


def _append(row):
    new = not RUNS.exists()
    with open(RUNS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def plan(args):
    seeds = {m: list(range(args.classical_seeds)) for m in CLASSICAL}
    seeds.update({m: list(range(args.seeds)) for m in DEEP})
    seeds.update({m: list(CKPT_SEEDS) for m in OURS})
    return seeds


def run(args):
    tag = f"e{args.epochs}_m{args.max_students}"
    seeds = plan(args)
    for m in ORDER:
        print(f"[baselines] {m}: seeds {seeds[m]}", flush=True)
    done = _done()
    tasks = [(m, s, args.epochs, args.max_students, tag) for m in ORDER for s in seeds[m]
             if (m, s) not in done or not _pred_path(m, s, tag).exists()]
    tasks.sort(key=lambda t: t[0] not in ("sakt_lite", "dkt_gru"))
    print(f"[baselines] {len(tasks)} runs to do ({len(done)} already in {RUNS.name})", flush=True)
    t0 = time.time()
    if tasks:
        with get_context("spawn").Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_task, tasks)):
                _append(row)
                print(f"  done {row['model']} seed {row['seed']}: test AUC {row['auc']:.4f} "
                      f"({i + 1}/{len(tasks)}, {(time.time() - t0) / 60:.1f} min)", flush=True)
    report(args, (time.time() - t0) / 60)


def report(args, minutes):
    from scipy import stats
    from benchmark_deep import _FastAUC, _holm
    tag = f"e{args.epochs}_m{args.max_students}"
    seeds = plan(args)
    with open(RUNS, newline="", encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if int(r["seed"]) in seeds.get(r["model"], [])]
    preds = {m: {} for m in ORDER}
    for m in ORDER:
        for s in seeds[m]:
            path = _pred_path(m, s, tag)
            if path.exists():
                z = np.load(path)
                preds[m][s] = (z["p"].astype(float), z["y"].astype(int), z["owner"].astype(int))
    models = [m for m in ORDER if preds[m]]
    n_st = int(next(iter(preds[models[0]].values()))[2].max()) + 1
    summary = {}
    for m in models:
        rs = [r for r in runs if r["model"] == m]
        c = mean_ci([float(r["auc"]) for r in rs])
        summary[m] = {"model": m, "seeds": len(rs), "seed_list": " ".join(str(s) for s in sorted(int(r["seed"]) for r in rs)),
                      "auc_mean": c["mean"], "auc_std": c["std"], "auc_ci_low": c["ci_low"], "auc_ci_high": c["ci_high"],
                      **{k: float(np.mean([float(r[k]) for r in rs])) for k in ("acc", "brier", "ece", "val_auc")}}
    # (b) bootstrap over test students of the seed-averaged AUC
    rng = np.random.default_rng(12345)
    fast = {m: [_FastAUC(*preds[m][s]) for s in preds[m]] for m in models}
    boot = {m: np.empty(args.bootstrap) for m in models}
    for b in range(args.bootstrap):
        cnt = np.bincount(rng.integers(0, n_st, n_st), minlength=n_st)
        for m in models:
            boot[m][b] = np.mean([f(cnt) for f in fast[m]])
    for m in models:
        summary[m]["boot_ci_low"], summary[m]["boot_ci_high"] = (float(x) for x in np.percentile(boot[m], [2.5, 97.5]))

    def student_brier(m):
        acc = np.zeros(n_st)
        for p, y, o in preds[m].values():
            se = np.bincount(o, (p - y) ** 2, n_st) / np.maximum(np.bincount(o, minlength=n_st), 1)
            acc += se
        return acc / len(preds[m])

    comps = []
    if "full_gated" in models:
        ours_b = student_brier("full_gated")
        for m in [x for x in models if x != "full_gated"]:
            diff = student_brier(m) - ours_b                   # > 0: ours has lower (better) Brier
            wp = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            d_auc = boot["full_gated"] - boot[m]
            lo, hi = np.percentile(d_auc, [2.5, 97.5])
            comps.append({"vs": m, "auc_diff": summary["full_gated"]["auc_mean"] - summary[m]["auc_mean"],
                          "boot_lo": float(lo), "boot_hi": float(hi), "brier_diff_median": float(np.median(diff)),
                          "rank_biserial": _rank_biserial(diff), "wilcoxon_p": wp})
        for c, h in zip(comps, _holm(np.asarray([c["wilcoxon_p"] for c in comps]))):
            c["p_holm"] = float(h)
            if h < 0.05 and c["boot_lo"] > 0 and c["brier_diff_median"] > 0:
                c["verdict"] = f"improves over {c['vs']}"
            elif h < 0.05 and c["boot_hi"] < 0 and c["brier_diff_median"] < 0:
                c["verdict"] = f"{c['vs']} is better"
            else:
                c["verdict"] = "comparable"
    with open(OUT / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(next(iter(summary.values())).keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in models:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary[m].items()})
    _plot(runs, models)
    _write_md(args, summary, comps, models, n_st, minutes)
    print("[baselines] reports/v3/benchmark.md written", flush=True)


def _rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def _plot(runs, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [[float(r["auc"]) for r in runs if r["model"] == m] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5, medianprops={"color": "#1f2937", "linewidth": 2})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor("#2a78d6" if m in OURS else "#c9d6e3")
        patch.set_edgecolor("#475569")
    for i, d in enumerate(data, start=1):
        ax.scatter(d, np.random.default_rng(i).uniform(i - 0.12, i + 0.12, len(d)), s=10, color="#1f2937", zorder=3)
    ax.set_yticks(range(1, len(models) + 1), [f"{LABEL[m]} (n={len(d)})" for m, d in zip(models, data)])
    ax.set_xlabel("Test AUC per seed (ASSISTments official split)")
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_boxplot.png", dpi=150)
    plt.close(fig)


def _write_md(args, summary, comps, models, n_st, minutes):
    f4 = lambda v: "n/a" if v is None or v != v else f"{v:.4f}"
    L = ["# Benchmark: our model against four baselines (ASSISTments 2009, official split)", "",
         f"Generated by `python scripts/benchmark_baselines_v3.py --seeds {args.seeds} --classical-seeds "
         f"{args.classical_seeds} --epochs {args.epochs} --max_students {args.max_students} --bootstrap {args.bootstrap}` "
         f"(this run: {minutes:.1f} min of new work). Per-seed rows: reports/v3/benchmark_runs.csv. Test students: {n_st}.",
         "", "## Setup", "",
         "- Same train / val / test split by student (data/processed/windowed_data.pkl), same targets (every position "
         "t >= 1 of a test student, predicted from the history before t), same metrics.",
         f"- PFA and BKT: {args.classical_seeds} seeds (0-{args.classical_seeds - 1}). BKT seeds change the EM starting "
         "point; PFA has no random component (a seed std of 0 means identical results).",
         f"- DKT (GRU) and SAKT-lite: {args.seeds} seeds (0-{args.seeds - 1}); inputs skill, correct, difficulty "
         f"bucket; Adam 1e-3, batch 32, {args.epochs} epochs, best-validation weights.",
         "- graph_no_gate and full_gated: the existing checkpoints (seeds 42, 123, 2024), wrapped read-only, not "
         "retrained. They were trained earlier with their own budget and also read response time / attempts / hints, "
         "so they are not under exactly the same budget or inputs (docs/DECISIONS.md D41).",
         f"- Max training students: {args.max_students or 'all'}.", "",
         "## Results", "",
         "Seed variance = spread of per-seed test AUC (std and t-based 95% CI over seeds). Bootstrap = 95% interval "
         f"of the seed-averaged test AUC over {args.bootstrap} resamples of test students. The two are different "
         "sources of uncertainty.", "",
         "| Model | Seeds | Test AUC mean | Std (seeds) | 95% CI (seeds) | 95% CI (bootstrap, students) | Accuracy | Brier | ECE |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(models, key=lambda m: -summary[m]["auc_mean"]):
        s = summary[m]
        L.append(f"| {LABEL[m]} | {s['seeds']} | {f4(s['auc_mean'])} | {f4(s['auc_std'])} | [{f4(s['auc_ci_low'])}, "
                 f"{f4(s['auc_ci_high'])}] | [{s['boot_ci_low']:.4f}, {s['boot_ci_high']:.4f}] | {s['acc']:.4f} | "
                 f"{s['brier']:.4f} | {s['ece']:.4f} |")
    L += ["", "Seeds used: " + "; ".join(f"{LABEL[m]}: {summary[m]['seed_list']}" for m in models), "",
          "## Full gated model against each baseline", "",
          "Paired test: Wilcoxon signed-rank over test students on the per-student Brier score (seed-averaged "
          "predictions), Holm-corrected over the comparisons below; effect size = rank-biserial correlation (positive = "
          "full gated has the lower Brier score). AUC difference CI: student bootstrap.", "",
          "| vs | AUC diff (full gated - X) | 95% CI (bootstrap) | Median Brier diff (X - full gated) | Rank-biserial | Wilcoxon p (Holm) | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for c in comps:
        L.append(f"| {LABEL[c['vs']]} | {c['auc_diff']:+.4f} | [{c['boot_lo']:+.4f}, {c['boot_hi']:+.4f}] | "
                 f"{c['brier_diff_median']:+.5f} | {c['rank_biserial']:+.3f} | {c['p_holm']:.2g} | **{c['verdict']}** |")
    L += ["", "Rule: \"improves over X\" only if the Holm-corrected p < 0.05, the Brier difference favours the full gated "
          "model and the bootstrap CI of the AUC difference excludes 0; \"X is better\" symmetrically; otherwise "
          "\"comparable\".", "", "![Per-seed test AUC](benchmark_boxplot.png)", "", "## Interpretation", ""]
    L += _interpret(summary, comps)
    L.append("")
    (OUT / "benchmark.md").write_text("\n".join(L), encoding="utf-8")


def _interpret(summary, comps):
    rank = sorted(summary, key=lambda m: -summary[m]["auc_mean"])
    fg = summary.get("full_gated")
    out = [f"1. Highest mean test AUC: {LABEL[rank[0]]} ({summary[rank[0]]['auc_mean']:.4f}); lowest: "
           f"{LABEL[rank[-1]]} ({summary[rank[-1]]['auc_mean']:.4f})."]
    if fg and comps:
        imp = [LABEL[c["vs"]] for c in comps if c["verdict"].startswith("improves")]
        worse = [LABEL[c["vs"]] for c in comps if c["verdict"].endswith("is better")]
        comp = [LABEL[c["vs"]] for c in comps if c["verdict"] == "comparable"]
        out.append(f"2. The full gated model improves over: {', '.join(imp) or 'none'}; comparable to: "
                   f"{', '.join(comp) or 'none'}; baselines that beat it: {', '.join(worse) or 'none'}.")
        deep = [c for c in comps if c["vs"] in DEEP]
        if deep:
            best = max(deep, key=lambda c: -c["auc_diff"])
            out.append(f"3. Against the strongest trained deep baseline ({LABEL[best['vs']]}) the AUC difference is "
                       f"{best['auc_diff']:+.4f} (bootstrap CI [{best['boot_lo']:+.4f}, {best['boot_hi']:+.4f}]).")
        seed_w = fg["auc_ci_high"] - fg["auc_ci_low"]
        boot_w = fg["boot_ci_high"] - fg["boot_ci_low"]
        out.append(f"4. Full gated: seed 95% CI width {seed_w:.4f} ({fg['seeds']} checkpoints) vs student-bootstrap "
                   f"width {boot_w:.4f}; the {'bootstrap' if boot_w > seed_w else 'seed'} interval is wider, so "
                   f"{'the choice of test students' if boot_w > seed_w else 'the training seed'} is the larger "
                   "source of uncertainty here.")
    out.append("5. Caveat: our checkpoints were trained earlier with a longer budget and extra inputs (response time, "
               "attempts, hints), while the deep baselines got 10 epochs on skill, correct and difficulty only; the "
               "comparison is therefore not perfectly budget-matched.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10, help="seeds for DKT (GRU) and SAKT-lite")
    ap.add_argument("--classical-seeds", type=int, default=50, help="seeds for PFA and BKT")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_students", type=int, default=0, help="cap on training students (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        report(args, 0.0)
    else:
        run(args)


if __name__ == "__main__":
    main()
