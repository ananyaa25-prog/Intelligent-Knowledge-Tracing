"""
scripts/benchmark_deep.py
-------------------------
Stage 5: deep models, our GatedKT variants and the classical baselines on the
OFFICIAL split (the main model's split), many seeds, plus significance tests.
Called from scripts/benchmark_seeds.py (parts "deep" and "report").

Protocol
- Every (model, seed) is trained from scratch with the same reduced budget
  (--epochs, Adam 1e-3, batch 32, best-val-AUC weights) and scored on the same test
  students. Seeds 0..N-1 fix init and batch order. Classical models are refitted
  per seed (deterministic ones are seed-independent).
- Reference rows: the existing v2 checkpoints (15 epochs, seeds 42/123/2024) of
  full_gated / graph_no_gate / dkt_only are scored read-only on the same test set.
- Per-(model, seed) test predictions are cached in data/processed_v3/benchmark_preds/
  so the run can resume and the report can be rebuilt without retraining.

Uncertainty
- Seeds: mean, std, variance and t-based 95% CI of the per-seed test AUC.
- Test students: paired bootstrap (default 1000 resamples of test students with
  replacement) of the seed-averaged AUC and of the difference full_gated - X.
Significance (full_gated vs every other model): paired t-test over seeds (primary),
Wilcoxon signed-rank (secondary), Cohen's d_z, Holm correction over all comparisons.
"improves over X" only if the Holm-corrected t-test p < 0.05 AND the seed CI and the
bootstrap CI of the difference both exclude 0 (and the difference is positive);
"worse than X" symmetrically; otherwise "comparable".
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines import CLASSICAL, make_classical  # noqa: E402
from baselines.base import all_metrics, mean_ci, split_to_sequences  # noqa: E402

WINDOWED = ROOT / "data" / "processed" / "windowed_data.pkl"
OUT = ROOT / "reports" / "v3"
PRED_DIR = ROOT / "data" / "processed_v3" / "benchmark_preds"
REF = "full_gated"

EXTERNAL = ["dkt_classic", "dkt_gru", "dkt_lstm", "dkvmn_lite", "sakt_lite"]
OURS = ["full_gated", "graph_no_gate", "dkt_only", "no_graph", "no_metadata", "no_difficulty"]
ABLATIONS = {"no_metadata": "no metadata", "no_difficulty": "no difficulty", "no_graph": "no graph",
             "graph_no_gate": "no gate"}
# rough relative cost per epoch (1 thread), used only to schedule long tasks first
COST = {"dkvmn_lite": 22, "sakt_lite": 14, "dkt_classic": 4, "dkt_gru": 4, "dkt_lstm": 5}

_D = {}


def _data():
    if not _D:
        with open(WINDOWED, "rb") as f:
            d = pickle.load(f)
        _D.update(d["splits"])
        _D["num_skills"] = int(d["num_skills"])
    return _D


def _subset(split, n, seed):
    if not n or split["mask"].shape[0] <= n:
        return split
    idx = np.sort(np.random.default_rng(seed).permutation(split["mask"].shape[0])[:n])
    return {k: v[idx] for k, v in split.items()}


def _cfg_tag(args):
    return f"e{args.epochs}_m{args.max_students or 0}"


def _pred_path(model, seed, tag):
    return PRED_DIR / f"{model}__s{seed}__{tag}.npz"


def _task(t):
    kind, model, seed, epochs, max_students, tag = t
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    path = _pred_path(model, seed, tag)
    if path.exists():
        return {"model": model, "seed": seed, "cached": True}
    d = _data()
    t0 = time.time()
    info = {}
    if kind == "classical":
        tr = _subset(d["train"], max_students, 0)
        m = make_classical(model, d["num_skills"], seed=seed).fit(tr)
        p, y, owner = m.predict_split(d["test"])
    elif kind == "ref":
        from baselines.ours import OursVariant, rt_stats_from_split
        cond = model.replace("v2ckpt_", "")
        m = OursVariant.from_checkpoint(cond, seed, num_skills=d["num_skills"], rt_stats=rt_stats_from_split(d["train"]))
        p, y, owner = m.predict_split(d["test"])
    else:
        from baselines.deep import make_deep
        from baselines.ours import rt_stats_from_split
        tr = _subset(d["train"], max_students, 0)
        m = make_deep(model, d["num_skills"], seed=seed, epochs=epochs, rt_stats=rt_stats_from_split(d["train"]))
        m.fit(tr, d["val"])
        info["best_val_auc"] = m.best_val_auc
        p, y, owner = m.predict_split(d["test"])
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, p=p.astype(np.float32), y=y.astype(np.int8), owner=owner.astype(np.int32),
                        seconds=time.time() - t0, best_val_auc=info.get("best_val_auc", float("nan")))
    os.replace(tmp, path)
    return {"model": model, "seed": seed, "seconds": time.time() - t0}


def run_deep(args):
    seeds = list(range(args.seeds))
    tag = _cfg_tag(args)
    deep = EXTERNAL + OURS
    if args.models:
        keep = set(args.models.split(","))
        deep = [m for m in deep if m in keep]
    print(f"[deep] seeds = {seeds[0]}..{seeds[-1]} ({len(seeds)}); epochs = {args.epochs}; "
          f"max_students = {args.max_students}; workers = {args.workers}; models = {deep}", flush=True)
    tasks = [("deep", m, s, args.epochs, args.max_students, tag) for s in seeds for m in deep]
    tasks.sort(key=lambda t: (-COST.get(t[1], 25), t[2]))
    tasks += [("ref", f"v2ckpt_{c}", s, 0, None, "ref") for c in ("full_gated", "graph_no_gate", "dkt_only")
              for s in (42, 123, 2024)]
    for name, cls in CLASSICAL.items():
        for s in (seeds if not cls.deterministic else seeds[:1]):
            tasks.append(("classical", name, s, 0, args.max_students, tag))
    todo = [t for t in tasks if not _pred_path(t[1], t[2], t[5]).exists()]
    print(f"[deep] {len(tasks)} tasks, {len(tasks) - len(todo)} cached, {len(todo)} to run", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, todo)):
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    meta = {"seeds": seeds, "epochs": args.epochs, "max_students": args.max_students, "tag": tag,
            "workers": args.workers, "wall_minutes": round((time.time() - t0) / 60, 1),
            "deep_models": deep, "bootstrap": args.bootstrap}
    (OUT / "stage5").mkdir(parents=True, exist_ok=True)
    (OUT / "stage5" / "benchmark_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    build_report(args)


# ---- statistics -------------------------------------------------------------------

def _load_preds(meta):
    tag = meta["tag"]
    res = {}
    for path in sorted(PRED_DIR.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        model, s, t = path.stem.split("__")
        if t not in (tag, "ref"):
            continue
        z = np.load(path)
        res.setdefault(model, {})[int(s[1:])] = {"p": z["p"].astype(np.float64), "y": z["y"].astype(np.int64),
                                                 "owner": z["owner"], "seconds": float(z["seconds"]),
                                                 "best_val_auc": float(z["best_val_auc"])}
    # deterministic classical models: one fit, same predictions for every seed
    for name, cls in CLASSICAL.items():
        if cls.deterministic and name in res and len(res[name]) == 1:
            only = next(iter(res[name].values()))
            res[name] = {s: only for s in meta["seeds"]}
    return res


class _FastAUC:
    """Weighted AUC for bootstrap resamples of students (ties handled by grouping)."""

    def __init__(self, p, y, owner):
        order = np.argsort(p, kind="mergesort")
        ps = p[order]
        self.y = y[order].astype(np.float64)
        self.owner = owner[order]
        self.grp = np.concatenate([[0], np.cumsum(ps[1:] != ps[:-1])])
        self.G = int(self.grp[-1]) + 1

    def __call__(self, counts):
        w = counts[self.owner].astype(np.float64)
        pos = np.bincount(self.grp, w * self.y, self.G)
        neg = np.bincount(self.grp, w * (1 - self.y), self.G)
        P, N = pos.sum(), neg.sum()
        if P == 0 or N == 0:
            return float("nan")
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / (P * N))


def _holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def build_report(args):
    from scipy import stats
    meta = json.loads((OUT / "stage5" / "benchmark_run_meta.json").read_text(encoding="utf-8"))
    seeds = meta["seeds"]
    preds = _load_preds(meta)
    if REF not in preds:
        raise SystemExit("no predictions for full_gated yet; run the deep part first")
    rows = []
    for model, by_seed in preds.items():
        for s, r in by_seed.items():
            m = all_metrics(r["y"], r["p"])
            m.update({"model": model, "seed": s, "seconds": r["seconds"], "best_val_auc": r["best_val_auc"]})
            rows.append(m)
    rows.sort(key=lambda r: (r["model"], r["seed"]))
    _csv(OUT / "stage5" / "benchmark_seeds.csv", rows, ["model", "seed", "auc", "acc", "brier", "ece", "n", "best_val_auc", "seconds"])

    def family(m):
        return "ours" if m in OURS else ("external deep" if m in EXTERNAL else
                                         ("v2 checkpoint" if m.startswith("v2ckpt_") else "classical"))

    summary = {}
    for model in preds:
        rs = [r for r in rows if r["model"] == model]
        a = mean_ci([r["auc"] for r in rs])
        rec = {"model": model, "family": family(model), "n_seeds": a["n"], "auc_mean": a["mean"],
               "auc_std": a["std"], "auc_var": a["var"], "auc_ci_low": a["ci_low"], "auc_ci_high": a["ci_high"]}
        for k in ("acc", "brier", "ece"):
            rec[f"{k}_mean"] = float(np.mean([r[k] for r in rs]))
        rec["train_seconds_mean"] = float(np.mean([r["seconds"] for r in rs]))
        summary[model] = rec

    # paired bootstrap over test students (seed-averaged AUC)
    rng = np.random.default_rng(12345)
    any_r = next(iter(preds[REF].values()))
    n_students = int(any_r["owner"].max()) + 1
    B = int(getattr(args, "bootstrap", None) or meta.get("bootstrap", 1000))
    comp_models = [m for m in preds if not m.startswith("v2ckpt_")]
    common_seeds = {m: [s for s in seeds if s in preds[m]] for m in comp_models}
    fast = {m: [_FastAUC(preds[m][s]["p"], preds[m][s]["y"], preds[m][s]["owner"]) for s in common_seeds[m]]
            for m in comp_models}
    boot = {m: np.empty(B) for m in comp_models}
    t0 = time.time()
    for b in range(B):
        counts = np.bincount(rng.integers(0, n_students, n_students), minlength=n_students)
        for m in comp_models:
            boot[m][b] = np.mean([f(counts) for f in fast[m]])
    print(f"[report] bootstrap {B} resamples x {len(comp_models)} models in {time.time() - t0:.0f}s", flush=True)
    for m in comp_models:
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        summary[m]["boot_ci_low"], summary[m]["boot_ci_high"] = float(lo), float(hi)

    # paired tests vs full_gated (paired by seed)
    ref_auc = {r["seed"]: r["auc"] for r in rows if r["model"] == REF}
    comps = []
    for m in comp_models:
        if m == REF:
            continue
        other = {r["seed"]: r["auc"] for r in rows if r["model"] == m}
        ss = [s for s in seeds if s in ref_auc and s in other]
        d = np.asarray([ref_auc[s] - other[s] for s in ss])
        ci = mean_ci(d)
        if len(d) >= 2 and np.std(d, ddof=1) > 0:
            t_p = float(stats.ttest_rel([ref_auc[s] for s in ss], [other[s] for s in ss]).pvalue)
            dz = float(d.mean() / d.std(ddof=1))
        else:
            t_p, dz = (0.0 if len(d) and abs(d.mean()) > 0 else 1.0), float("inf") if len(d) and d.mean() else 0.0
        try:
            w_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            w_p = float("nan")
        bd = boot[REF] - boot[m]
        blo, bhi = np.percentile(bd, [2.5, 97.5])
        comps.append({"model": m, "family": family(m), "n_pairs": len(d), "diff_mean": float(d.mean()),
                      "diff_ci_low": ci["ci_low"], "diff_ci_high": ci["ci_high"], "boot_diff_low": float(blo),
                      "boot_diff_high": float(bhi), "t_p": t_p, "wilcoxon_p": w_p, "cohen_dz": dz})
    adj = _holm(np.asarray([c["t_p"] for c in comps]))
    adj_w = _holm(np.asarray([c["wilcoxon_p"] if c["wilcoxon_p"] == c["wilcoxon_p"] else 1.0 for c in comps]))
    for c, a, aw in zip(comps, adj, adj_w):
        c["t_p_holm"], c["wilcoxon_p_holm"] = float(a), float(aw)
        sig = a < 0.05
        pos = c["diff_ci_low"] > 0 and c["boot_diff_low"] > 0
        neg = c["diff_ci_high"] < 0 and c["boot_diff_high"] < 0
        c["verdict"] = (f"improves over {c['model']}" if sig and pos else
                        (f"worse than {c['model']}" if sig and neg else "comparable"))
    comps.sort(key=lambda c: -c["diff_mean"])
    _csv(OUT / "stage5" / "benchmark.csv", list(summary.values()), list(next(iter(summary.values())).keys()) +
         [k for k in ("boot_ci_low", "boot_ci_high") if k not in next(iter(summary.values()))])
    _csv(OUT / "stage5" / "benchmark_significance.csv", comps, list(comps[0].keys()))
    _plots(rows, summary, seeds)
    _write_md(meta, summary, comps, B, n_students)
    print("[report] reports/v3/stage5/benchmark.md written", flush=True)


def _csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k in fields})


FAM_COLOR = {"ours": "#2a78d6", "external deep": "#eb6834", "classical": "#1baf7a", "v2 checkpoint": "#6250d6"}


def _plots(rows, summary, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    models = [m for m in sorted(summary, key=lambda m: summary[m]["auc_mean"]) if summary[m]["auc_mean"] > 0.55]
    data = [[r["auc"] for r in rows if r["model"] == m] for m in models]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(models) + 1.4))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, medianprops={"color": "#0b0b0b", "linewidth": 1.5},
                    whiskerprops={"color": "#52514e"}, capprops={"color": "#52514e"},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#52514e", "markeredgecolor": "none"})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(FAM_COLOR[summary[m]["family"]])
        patch.set_alpha(0.8)
        patch.set_edgecolor("#fcfcfb")
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels([f"{m} (n={summary[m]['n_seeds']})" for m in models])
    ax.set_xlabel("Test AUC (official split), one value per seed")
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    from matplotlib.patches import Patch
    fams = [f for f in FAM_COLOR if any(summary[m]["family"] == f for m in models)]
    ax.legend(handles=[Patch(facecolor=FAM_COLOR[f], label=f) for f in fams], loc="lower right", frameon=False)
    ax.set_title("Per-seed test AUC by model (majority baseline, AUC 0.5, omitted)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "stage5" / "benchmark_boxplot.png", dpi=150)
    plt.close(fig)

    # seed vs AUC: our variants and the external deep models (lines), one panel each family
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, fam, title in ((axes[0], "ours", "Our GatedKT variants"), (axes[1], "external deep", "External deep baselines")):
        ms = [m for m in summary if summary[m]["family"] == fam]
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
        for i, m in enumerate(sorted(ms, key=lambda m: -summary[m]["auc_mean"])):
            pts = sorted((r["seed"], r["auc"]) for r in rows if r["model"] == m)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=4, linewidth=1.5,
                    color=cols[i % len(cols)], label=m)
        ax.set_xlabel("Seed")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    axes[0].set_ylabel("Test AUC")
    fig.suptitle("Seed-to-seed variation of test AUC (same split, same budget)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "stage5" / "benchmark_seed_auc.png", dpi=150)
    plt.close(fig)


def _f(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f"{v:.{nd}f}"


def _p(v):
    return "n/a" if v != v else ("<1e-4" if v < 1e-4 else f"{v:.4f}")


def _write_md(meta, summary, comps, B, n_students):
    L = ["# Benchmark: deep models, our variants and classical baselines (Stage 5)", "",
         f"Generated by `python scripts/benchmark_seeds.py deep --seeds {len(meta['seeds'])} --epochs {meta['epochs']}"
         + (f" --max_students {meta['max_students']}" if meta["max_students"] else "") + f" --workers {meta['workers']}`"
         f" (report rebuilt with `python scripts/benchmark_seeds.py report`).",
         f"Seeds: {meta['seeds'][0]}..{meta['seeds'][-1]} ({len(meta['seeds'])}). Training wall time (cumulative over both runs: seeds 0-9, then 10-19 added): "
         f"{meta['wall_minutes']} min"
         + (" - above the 30-40 min target. The only size reduction applied is 15 -> "
            f"{meta['epochs']} epochs (all models alike); it was not reduced further so that every model still gets "
            "the same multi-epoch budget. Use --seeds/--epochs/--max_students for a shorter run."
            if meta["wall_minutes"] > 40 else "."), "",
         "## Protocol", "",
         "- Data: ASSISTments 2009 skill-builder, official student split of the main model (train 2495 / val 311 / "
         f"test {n_students} students), sequences truncated at 500, train-derived difficulty buckets.",
         f"- Every deep model (external baselines and our variants) is trained from scratch for each seed with the SAME "
         f"reduced budget: {meta['epochs']} epochs, Adam 1e-3, batch 32 (length-bucketed), best-validation-AUC weights, "
         "1 CPU thread per run. This is a reduction from the 15 epochs of the v2 training, made so the run fits the "
         "time budget; absolute AUCs are therefore lower than fully trained models and the ranking could change "
         "with longer training. All models get the same budget.",
         "- Reference rows `v2ckpt_*`: the existing v2 checkpoints (15 epochs, seeds 42/123/2024), scored read-only.",
         "- Classical models are fitted on the same train students; deterministic ones have one value for all seeds.", "",
         "Two separate sources of uncertainty are reported:",
         "- **Seed uncertainty**: mean, std (ddof=1), variance and t-based 95% CI of the per-seed test AUC.",
         f"- **Test-sample uncertainty**: paired bootstrap over the {n_students} test students ({B} resamples, "
         "students drawn with replacement, all of a student's predictions kept together), applied to the "
         "seed-averaged AUC. Columns marked 'boot'.", "",
         "## Results (sorted by mean test AUC)", "",
         "| Model | Family | Seeds | AUC mean | AUC std | AUC var | 95% CI (seeds) | 95% CI (boot) | ACC | Brier | ECE |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(summary, key=lambda m: -summary[m]["auc_mean"]):
        r = summary[m]
        boot = f"[{_f(r.get('boot_ci_low'))}, {_f(r.get('boot_ci_high'))}]" if "boot_ci_low" in r else "-"
        L.append(f"| {m} | {r['family']} | {r['n_seeds']} | {_f(r['auc_mean'])} | {_f(r['auc_std'])} | "
                 f"{r['auc_var']:.2e} | [{_f(r['auc_ci_low'])}, {_f(r['auc_ci_high'])}] | {boot} | "
                 f"{_f(r['acc_mean'])} | {_f(r['brier_mean'])} | {_f(r['ece_mean'])} |")
    L += ["", "![boxplot](benchmark_boxplot.png)", "", "![seed vs AUC](benchmark_seed_auc.png)", "",
          "## Significance: full_gated vs every other model", "",
          "Difference = AUC(full_gated) - AUC(model), paired by seed. Primary test: paired t-test over seeds with "
          "Holm correction over all comparisons; Wilcoxon signed-rank (Holm-corrected) as a check; effect size "
          "Cohen's d_z = mean difference / std of differences. Verdict rule (fixed before the run): 'improves over X' "
          "only if the Holm-corrected t-test p < 0.05 and both the seed CI and the bootstrap CI of the difference "
          "exclude 0; otherwise 'comparable' (or 'worse than X' if significant in the other direction). "
          "For deterministic classical models the per-seed difference varies only through full_gated's seeds.", "",
          "| Model | Family | Pairs | Diff mean | 95% CI (seeds) | 95% CI (boot) | t p (Holm) | Wilcoxon p (Holm) | d_z | Verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        L.append(f"| {c['model']} | {c['family']} | {c['n_pairs']} | {c['diff_mean']:+.4f} | "
                 f"[{c['diff_ci_low']:+.4f}, {c['diff_ci_high']:+.4f}] | [{c['boot_diff_low']:+.4f}, {c['boot_diff_high']:+.4f}] | "
                 f"{_p(c['t_p_holm'])} | {_p(c['wilcoxon_p_holm'])} | {_f(c['cohen_dz'], 2)} | {c['verdict']} |")
    L += ["", "## Input ablations (same protocol)", "",
          "| Ablation | Model key | AUC mean (seeds) | Diff full - ablation | 95% CI (seeds) | 95% CI (boot) | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for key, label in ABLATIONS.items():
        c = next((c for c in comps if c["model"] == key), None)
        if c is None:
            L.append(f"| {label} | {key} | not run | | | | |")
            continue
        L.append(f"| {label} | {key} | {_f(summary[key]['auc_mean'])} | {c['diff_mean']:+.4f} | "
                 f"[{c['diff_ci_low']:+.4f}, {c['diff_ci_high']:+.4f}] | [{c['boot_diff_low']:+.4f}, {c['boot_diff_high']:+.4f}] | "
                 f"{c['verdict'].replace('improves over', 'full model better than').replace('worse than', 'full model worse than')} |")
    L += ["", "## How to read this", "",
          "- A verdict 'comparable' means the data do not show a difference under this protocol; it is not "
          "evidence that the models are equal.",
          "- All deep models were trained with a reduced budget (see Protocol); compare the `v2ckpt_*` rows for the "
          "fully trained v2 checkpoints (3 seeds only).",
          "- Files: benchmark.csv (summary), benchmark_seeds.csv (per model x seed), benchmark_significance.csv, "
          "benchmark_run_meta.json; cached predictions in data/processed_v3/benchmark_preds/.", ""]
    (OUT / "stage5" / "benchmark.md").write_text("\n".join(L), encoding="utf-8")
