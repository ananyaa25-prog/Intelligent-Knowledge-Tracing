"""
scripts/run_seeds.py
--------------------
Our models over many random seeds, with the variance reported (v3).

Models (--models): dkt_only, graph_no_gate, full_gated, trained with our existing
training path (src/baselines/ours.py: src/model.py GatedKTModel, masked BCE, Adam 1e-3,
batch 32, best-validation weights) on the fixed official split of
data/processed/windowed_data.pkl. A seed changes initialisation and batch order only;
the train / val / test students never change.

Plan (unless --no-plan): one timing wave (one epoch of each model, run in parallel as
the real runs are) gives the time per epoch; the script then keeps the largest setting
that fits --budget-hours, reducing in this order and recording each reduction:
  1. --epochs (default 15, the epochs of the existing v2 checkpoints) down to 10, then 5,
     with early stopping (patience 3) once epochs are reduced;
  2. a capped training subset (2000, then 1500 students; val and test unchanged);
  3. fewer seeds (down to 20).
100 seeds are used when they fit in 75% of the budget.

Every (model, seed) is seeded (Python, numpy, torch), printed, and appended to
reports/v3/seeds.csv as soon as it finishes; a rerun skips finished pairs.
Separate experiment (--resplit-seeds, default 10): the student split itself changes
with the seed (80/10/10 re-split, difficulty buckets and skill graph from that seed's
train students), rows in reports/v3/seeds_resplit.csv, reported separately.

Usage: python scripts/run_seeds.py [--seeds 50] [--epochs 15] [--max_students 0]
       [--models dkt_only,graph_no_gate,full_gated] [--workers 6] [--budget-hours 3.5]
       [--resplit-seeds 10] [--report-only]
Outputs: reports/v3/seeds.md, seeds.csv, seeds_resplit.csv, seeds_plan.json,
seeds_boxplot.png, seeds_hist.png. No checkpoint is saved.
"""

import argparse
import csv
import json
import math
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

OUT = ROOT / "reports" / "v3"
CSV_FIXED = OUT / "seeds.csv"
CSV_RESPLIT = OUT / "seeds_resplit.csv"
PLAN = OUT / "seeds_plan.json"
FIELDS = ["model", "seed", "val_auc", "test_auc", "brier", "ece", "acc", "n", "epochs", "max_students",
          "epochs_run", "seconds"]
MODELS = ["dkt_only", "graph_no_gate", "full_gated"]
LABEL = {"dkt_only": "DKT-only", "graph_no_gate": "graph, no gate", "full_gated": "full gated"}
REF = "full_gated"

_D = {}


def _data():
    if not _D:
        with open(ROOT / "data" / "processed" / "windowed_data.pkl", "rb") as f:
            d = pickle.load(f)
        _D.update(d["splits"])
        _D["num_skills"] = int(d["num_skills"])
    return _D


def _subset(split, n):
    """Fixed training subset (same students for every seed) when n is set."""
    if not n or split["mask"].shape[0] <= n:
        return split
    idx = np.sort(np.random.default_rng(0).permutation(split["mask"].shape[0])[:n])
    return {k: v[idx] for k, v in split.items()}


def seed_everything(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(model, seed, tr, va, te, num_skills, epochs, patience=None, graph=None):
    """One seeded training run -> metrics dict (no checkpoint written)."""
    from baselines.base import all_metrics
    from baselines.deep import make_deep
    from baselines.ours import rt_stats_from_split
    seed_everything(seed)
    kw = {"epochs": epochs, "rt_stats": rt_stats_from_split(tr)}
    if patience:
        kw["patience"] = patience
    if graph is not None:
        kw["graph"] = graph
    m = make_deep(model, num_skills, seed=seed, **kw).fit(tr, va)
    p, y, _ = m.predict_split(te)
    met = all_metrics(y, p)
    return {"val_auc": float(m.best_val_auc), "test_auc": met["auc"], "brier": met["brier"], "ece": met["ece"],
            "acc": met["acc"], "n": met["n"], "epochs_run": len(getattr(m, "history_log", []) or [])}


def _task(t):
    kind, model, seed, epochs, patience, max_students = t
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    print(f"  seed {seed}: {model} ({kind}) started", flush=True)
    t0 = time.time()
    if kind == "resplit":
        from baselines.base import resplit, sequences_to_split, split_to_sequences
        from data.adapters.common import graph_from_train
        d = _data()
        pool = []
        for part in ("train", "val", "test"):
            seqs = split_to_sequences(d[part])
            win = d[part]["window_id_seq"].numpy()
            for i, s in enumerate(seqs):
                s["window_id_seq"] = win[i, :len(s["skill_seq"])]  # needed by the train-only graph
            pool += seqs
        tr, va, te = resplit(pool, seed)
        graph = graph_from_train(tr, d["num_skills"]) if model != "dkt_only" else None
        r = train_one(model, seed, sequences_to_split(tr), sequences_to_split(va), sequences_to_split(te),
                      d["num_skills"], epochs, patience, graph)
    else:
        d = _data()
        r = train_one(model, seed, _subset(d["train"], max_students), d["val"], d["test"], d["num_skills"],
                      epochs, patience)
    return dict(r, kind=kind, model=model, seed=seed, epochs=epochs, max_students=max_students or 0,
                seconds=round(time.time() - t0, 1))


def _read(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def _run_pool(tasks, workers, label):
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks)):
            _append(CSV_RESPLIT if r["kind"] == "resplit" else CSV_FIXED, r)
            print(f"  seed {r['seed']}: {r['model']} test AUC {r['test_auc']:.4f} ({label} {i + 1}/{len(tasks)}, "
                  f"{(time.time() - t0) / 60:.1f} min)", flush=True)


def make_plan(args, models):
    """Timing wave -> chosen (epochs, patience, max_students, seeds) and the reductions made."""
    if PLAN.exists():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        print(f"[seeds] reusing plan {PLAN.name}: {plan['chosen']}", flush=True)
        return plan
    print("[seeds] timing one epoch of each model (in parallel, as in the real runs) ...", flush=True)
    t0 = time.time()
    wave = [("timing", m, 10_000 + i, 1, None, 0) for i, m in enumerate((models * args.workers)[:args.workers])]
    with get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_task, wave)
    per_epoch = max(r["seconds"] for r in res)               # slowest model sets the pace
    n_train = int(_data()["train"]["mask"].shape[0])
    budget = args.budget_hours * 3600

    def est(epochs, cap, seeds):
        frac = (min(cap, n_train) / n_train) if cap else 1.0
        runs = len(models) * seeds + len(models) * args.resplit_seeds
        return per_epoch * epochs * frac * math.ceil(runs / args.workers)

    reductions, chosen = [], None
    epochs_opts = [args.epochs] + [e for e in (10, 5) if e < args.epochs]
    for e in epochs_opts:
        if est(e, args.max_students, args.seeds) <= budget:
            chosen = (e, args.max_students, args.seeds)
            break
    if chosen is None:
        e = epochs_opts[-1]
        for cap in (2000, 1500):
            if est(e, cap, args.seeds) <= budget:
                chosen = (e, cap, args.seeds)
                break
    if chosen is None:
        e, cap = epochs_opts[-1], 1500
        s = args.seeds
        while s > 20 and est(e, cap, s) > budget:
            s -= 5
        chosen = (e, cap, max(s, 20))
    e, cap, s = chosen
    if s == args.seeds and e == args.epochs and not cap and args.seeds < 100 and est(e, cap, 100) <= 0.75 * budget:
        s = 100
        reductions.append(f"100 seeds instead of {args.seeds}: estimated {est(e, cap, 100) / 3600:.1f} h fits easily")
    if e != args.epochs:
        reductions.append(f"epochs reduced from {args.epochs} to {e}, early stopping with patience 3")
    if cap and cap != args.max_students:
        reductions.append(f"training subset capped at {cap} of {n_train} students (val and test unchanged)")
    if s < args.seeds:
        reductions.append(f"seeds reduced from {args.seeds} to {s}")
    plan = {"per_epoch_seconds": per_epoch, "timing_wall_seconds": round(time.time() - t0, 1), "budget_hours":
            args.budget_hours, "workers": args.workers, "requested": {"epochs": args.epochs, "max_students":
            args.max_students, "seeds": args.seeds},
            "chosen": {"epochs": e, "patience": 3 if e != args.epochs else None, "max_students": cap or 0, "seeds": s},
            "estimate_hours": round(est(e, cap, s) / 3600, 2), "estimate_full_hours":
            round(est(args.epochs, args.max_students, args.seeds) / 3600, 2), "reductions": reductions}
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[seeds] one epoch ~{per_epoch:.0f} s per run; {args.seeds} seeds x {len(models)} models at {args.epochs} "
          f"epochs ~{plan['estimate_full_hours']} h; chosen {plan['chosen']} ~{plan['estimate_hours']} h", flush=True)
    for r in reductions:
        print(f"[seeds] reduction: {r}", flush=True)
    return plan


def run(args):
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    plan = make_plan(args, models) if not args.no_plan else {
        "chosen": {"epochs": args.epochs, "patience": None, "max_students": args.max_students, "seeds": args.seeds},
        "reductions": []}
    c = plan["chosen"]
    seeds = list(range(c["seeds"]))
    print(f"[seeds] fixed split: seeds {seeds}", flush=True)
    done = {(r["model"], int(r["seed"])) for r in _read(CSV_FIXED)}
    tasks = [("fixed", m, s, c["epochs"], c["patience"], c["max_students"]) for s in seeds for m in models
             if (m, s) not in done]
    if tasks:
        _run_pool(tasks, args.workers, "fixed split")
    rs = list(range(args.resplit_seeds))
    print(f"[seeds] student re-split: seeds {rs}", flush=True)
    done = {(r["model"], int(r["seed"])) for r in _read(CSV_RESPLIT)}
    tasks = [("resplit", m, s, c["epochs"], c["patience"], 0) for s in rs for m in models if (m, s) not in done]
    if tasks:
        _run_pool(tasks, args.workers, "re-split")
    report(models, plan, args)


def _stats(rows, models):
    from baselines.base import mean_ci
    out = {}
    for m in models:
        v = [float(r["test_auc"]) for r in rows if r["model"] == m]
        if v:
            c = mean_ci(v)
            out[m] = dict(c, min=float(min(v)), max=float(max(v)),
                          val=float(np.mean([float(r["val_auc"]) for r in rows if r["model"] == m])),
                          brier=float(np.mean([float(r["brier"]) for r in rows if r["model"] == m])),
                          ece=float(np.mean([float(r["ece"]) for r in rows if r["model"] == m])))
    return out


def _paired(rows, models):
    from scipy import stats
    from baselines.base import mean_ci
    from benchmark_deep import _holm
    by = {m: {int(r["seed"]): float(r["test_auc"]) for r in rows if r["model"] == m} for m in models}
    comps = []
    for m in models:
        if m == REF or REF not in by:
            continue
        ss = sorted(set(by[REF]) & set(by[m]))
        if len(ss) < 2:
            continue
        d = np.asarray([by[REF][s] - by[m][s] for s in ss])
        ci = mean_ci(d)
        sd = d.std(ddof=1)
        tp = float(stats.ttest_rel([by[REF][s] for s in ss], [by[m][s] for s in ss]).pvalue) if sd > 0 else 1.0
        wp = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        comps.append({"vs": m, "n": len(ss), "diff": float(d.mean()), "ci_lo": ci["ci_low"], "ci_hi": ci["ci_high"],
                      "d_z": float(d.mean() / sd) if sd > 0 else float("nan"), "t_p": tp, "w_p": wp,
                      "wins": int((d > 0).sum())})
    if comps:
        for c, h in zip(comps, _holm(np.asarray([c["t_p"] for c in comps]))):
            c["t_p_holm"] = float(h)
            c["verdict"] = ("improves" if h < 0.05 and c["ci_lo"] > 0 else
                            ("worse" if h < 0.05 and c["ci_hi"] < 0 else "comparable"))
    return comps


def _plots(rows, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"dkt_only": "#9aa9bb", "graph_no_gate": "#7aa5d6", "full_gated": "#2a78d6"}
    data = [[float(r["test_auc"]) for r in rows if r["model"] == m] for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5, medianprops={"color": "#1f2937", "linewidth": 2})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(colors.get(m, "#c9d6e3"))
        patch.set_edgecolor("#475569")
    for i, d in enumerate(data, start=1):
        ax.scatter(d, np.random.default_rng(i).uniform(i - 0.12, i + 0.12, len(d)), s=9, color="#1f2937", zorder=3)
    ax.set_yticks(range(1, len(models) + 1), [f"{LABEL.get(m, m)} (n={len(d)})" for m, d in zip(models, data)])
    ax.set_xlabel("Test AUC per seed (fixed split)")
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "seeds_boxplot.png", dpi=150)
    plt.close(fig)
    lo, hi = min(min(d) for d in data if d), max(max(d) for d in data if d)
    bins = np.linspace(lo, hi, 21) if hi > lo else 10
    fig, axes = plt.subplots(len(models), 1, figsize=(7.5, 1.9 * len(models)), sharex=True)
    for ax, m, d in zip(np.atleast_1d(axes), models, data):
        ax.hist(d, bins=bins, color=colors.get(m, "#c9d6e3"), edgecolor="white", linewidth=1)
        ax.set_ylabel("seeds")
        ax.set_title(f"{LABEL.get(m, m)}", loc="left", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("Test AUC (fixed split)")
    fig.tight_layout()
    fig.savefig(OUT / "seeds_hist.png", dpi=150)
    plt.close(fig)


def report(models, plan, args):
    fixed = [r for r in _read(CSV_FIXED) if r["model"] in models and int(r["seed"]) < plan["chosen"]["seeds"]]
    resp = [r for r in _read(CSV_RESPLIT) if r["model"] in models and int(r["seed"]) < args.resplit_seeds]
    sf, sr = _stats(fixed, models), _stats(resp, models)
    cf, cr = _paired(fixed, models), _paired(resp, models)
    _plots(fixed, models)
    c = plan["chosen"]
    f4 = lambda v: "n/a" if v is None or v != v else f"{v:.4f}"
    L = ["# Our models over many seeds: variance of test AUC", "",
         "Generated by `python scripts/run_seeds.py` from reports/v3/seeds.csv (fixed split) and "
         "reports/v3/seeds_resplit.csv (student re-split). Plan and timing: reports/v3/seeds_plan.json.", "",
         "## Setup", "",
         "- Models: " + ", ".join(LABEL.get(m, m) for m in models) + "; our training path (src/baselines/ours.py on "
         "src/model.py), Adam 1e-3, batch 32, best-validation weights; inputs as in the existing model (skill, correct, "
         "difficulty bucket, response time, attempts, hints). No checkpoint saved.",
         f"- Fixed split (official ASSISTments split): a seed changes initialisation and batch order only; the test set "
         f"({fixed[0]['n'] if fixed else 'n/a'} predictions) is identical for every seed.",
         f"- Budget per run: {c['epochs']} epochs" + (f", early stopping patience {c['patience']}" if c.get("patience")
                                                       else "") +
         f"; training students: {c['max_students'] or 'all'}.",
         f"- Seeds requested: {plan.get('requested', {}).get('seeds', c['seeds'])}; seeds used: "
         + ", ".join(f"{LABEL.get(m, m)} {sf[m]['n'] if m in sf else 0}" for m in models) + ".",
         ]
    if plan.get("per_epoch_seconds"):
        L.append(f"- Timing: one epoch took up to {plan['per_epoch_seconds']:.0f} s per run with {plan['workers']} "
                 f"runs in parallel; estimated {plan['estimate_full_hours']} h for the requested setting, "
                 f"{plan['estimate_hours']} h for the chosen one (budget {plan['budget_hours']} h).")
    L.append("- Reductions: " + ("; ".join(plan["reductions"]) if plan.get("reductions") else "none") + ".")
    L += ["", "## Fixed split: test AUC over seeds", "",
          "| Model | Seeds | Mean | Std | Variance | Min | Max | 95% CI (mean) | Mean val AUC | Brier | ECE |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        if m in sf:
            s = sf[m]
            L.append(f"| {LABEL.get(m, m)} | {s['n']} | {f4(s['mean'])} | {f4(s['std'])} | {s['var']:.2e} | "
                     f"{f4(s['min'])} | {f4(s['max'])} | [{f4(s['ci_low'])}, {f4(s['ci_high'])}] | {f4(s['val'])} | "
                     f"{s['brier']:.4f} | {s['ece']:.4f} |")
    L += ["", "## Full gated against each other model (matched seeds)", "",
          "Paired t-test over matched seeds (Holm-corrected), Wilcoxon signed-rank as a check, effect size Cohen's d_z "
          "(mean paired difference / its std). \"improves\" only if the corrected p < 0.05 and the 95% CI of the paired "
          "difference excludes 0; \"worse\" symmetrically; otherwise \"comparable\".", "",
          "| vs | Pairs | Mean diff (full gated - X) | 95% CI | d_z | Seeds full gated higher | t p (Holm) | Wilcoxon p | Verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in cf:
        L.append(f"| {LABEL.get(x['vs'], x['vs'])} | {x['n']} | {x['diff']:+.4f} | [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | "
                 f"{x['d_z']:+.2f} | {x['wins']}/{x['n']} | {x['t_p_holm']:.2g} | {x['w_p']:.2g} | **{x['verdict']}** |")
    L += ["", "![Test AUC per seed](seeds_boxplot.png)", "", "![Histogram of test AUC per model](seeds_hist.png)", "",
          "## Variance compared with the gain", ""]
    if REF in sf:
        for x in cf:
            ratio = abs(x["diff"]) / sf[REF]["std"] if sf[REF]["std"] > 0 else float("nan")
            L.append(f"- vs {LABEL.get(x['vs'], x['vs'])}: the mean difference ({x['diff']:+.4f}) is {ratio:.1f}x the "
                     f"seed-to-seed std of the full gated model ({sf[REF]['std']:.4f}); the full gated model scored "
                     f"higher on {x['wins']} of {x['n']} matched seeds. "
                     + ("The difference is larger than the seed noise." if ratio > 1 else
                        "The difference is within the seed noise, so a single run could show either order."))
    L += ["", "## Separate experiment: the student split changes with the seed", "",
          f"{args.resplit_seeds} seeds; each seed draws a new 80/10/10 student split (difficulty buckets and skill graph "
          "from that seed's train students only), same budget as above. This variance includes which students are in "
          "the test set, so it is not comparable to the fixed-split variance above.", "",
          "| Model | Splits | Mean | Std | Variance | Min | Max | 95% CI (mean) |", "|---|---|---|---|---|---|---|---|"]
    for m in models:
        if m in sr:
            s = sr[m]
            L.append(f"| {LABEL.get(m, m)} | {s['n']} | {f4(s['mean'])} | {f4(s['std'])} | {s['var']:.2e} | {f4(s['min'])} | "
                     f"{f4(s['max'])} | [{f4(s['ci_low'])}, {f4(s['ci_high'])}] |")
    if cr:
        L += ["", "| vs (re-split, matched splits) | Pairs | Mean diff | 95% CI | d_z | t p (Holm) | Verdict |",
              "|---|---|---|---|---|---|---|"]
        for x in cr:
            L.append(f"| {LABEL.get(x['vs'], x['vs'])} | {x['n']} | {x['diff']:+.4f} | [{x['ci_lo']:+.4f}, "
                     f"{x['ci_hi']:+.4f}] | {x['d_z']:+.2f} | {x['t_p_holm']:.2g} | **{x['verdict']}** |")
    if REF in sf and REF in sr:
        L += ["", f"Std of the full gated test AUC: {sf[REF]['std']:.4f} over seeds on the fixed split vs "
              f"{sr[REF]['std']:.4f} over student re-splits."]
    L.append("")
    (OUT / "seeds.md").write_text("\n".join(L), encoding="utf-8")
    print("[seeds] reports/v3/seeds.md written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max_students", type=int, default=0, help="cap on training students (0 = all)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-hours", type=float, default=3.5)
    ap.add_argument("--resplit-seeds", type=int, default=10)
    ap.add_argument("--no-plan", action="store_true", help="use --epochs/--max_students/--seeds as given")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.report_only:
        plan = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else {
            "chosen": {"epochs": args.epochs, "patience": None, "max_students": args.max_students, "seeds": args.seeds},
            "reductions": []}
        report(models, plan, args)
    else:
        run(args)


if __name__ == "__main__":
    main()
