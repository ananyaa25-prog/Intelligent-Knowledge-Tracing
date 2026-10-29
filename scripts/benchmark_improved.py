"""
scripts/benchmark_improved.py
-----------------------------
Benchmark of the improved model (GatedKT+, src/baselines/gkt_plus.py) against our
original model and the strongest baselines, under the protocol pre-registered in
reports/v3/IMPROVED_PREREG.md.

Every deep model gets the SAME budget: up to --epochs epochs, early stopping on
validation AUC (patience --patience), Adam 1e-3, batch 32, best-validation weights.
GatedKT+ uses the dropout / weight decay chosen on the validation split by
scripts/tune_gkt_plus.py (reports/v3/gkt_plus_tuning.json).

Datasets:
  assist           ASSISTments 2009 official split (as in Stage 5)
  eedi_primary     Eedi learners < 11 (data/processed_v3/eedi.pkl, seeded 80/10/10)
  eedi_secondary   Eedi learners 11-16.5
  statics          Statics2011 undergraduates; also gkt_plus_pt = GatedKT+ initialised
                   from the ASSISTments-trained GatedKT+ (seed 0) and fine-tuned
Models: gkt_plus, gkt_plus_elo (same training run, blend fitted on validation),
full_gated, dkt_classic, dkt_gru, elo, logreg_counts.

Statistics as in Stage 5: per-seed AUC (mean, std, t-CI), paired t-test over seeds
with Holm correction across all comparisons of the report, Wilcoxon, student
bootstrap (--bootstrap resamples). "better than X" only if Holm p < 0.05 AND the seed
CI and the bootstrap CI of the difference both exclude 0; "worse than X" likewise;
otherwise "comparable".

Usage: python scripts/benchmark_improved.py --dataset assist --seeds 20 --epochs 15 --patience 3 --workers 6
Outputs: reports/v3/improved_<dataset>.md / .csv, cached predictions in
data/processed_v3/improved_preds/
"""

from __future__ import annotations

import argparse
import csv
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
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baselines.base import all_metrics, mean_ci, sequences_to_split  # noqa: E402

OUT = ROOT / "reports" / "v3"
PRED = ROOT / "data" / "processed_v3" / "improved_preds"
MODELS_V3 = ROOT / "models_v3"
DEEP = ["gkt_plus_elo", "full_gated", "dkt_classic", "dkt_gru"]   # gkt_plus_elo also yields gkt_plus
CLASSICAL = ["elo", "logreg_counts"]
PRETRAIN_PATH = MODELS_V3 / "gkt_plus_assist_seed0.pt"
PRIMARY = "gkt_plus_elo"
COMPARE = [("gkt_plus_elo", m) for m in ("dkt_classic", "elo", "full_gated", "dkt_gru", "logreg_counts", "gkt_plus")] + \
          [("gkt_plus", m) for m in ("dkt_classic", "elo", "full_gated")]


def tuned():
    p = OUT / "gkt_plus_tuning.json"
    b = json.loads(p.read_text(encoding="utf-8"))["best"] if p.exists() else {"dropout": 0.2, "weight_decay": 1e-5}
    return {"dropout": b["dropout"], "weight_decay": b["weight_decay"]}


def load_dataset(name):
    """-> (train, val, test, num_skills, graph-or-None) as split dicts."""
    if name == "assist":
        with open(ROOT / "data" / "processed" / "windowed_data.pkl", "rb") as f:
            d = pickle.load(f)
        s = d["splits"]
        return s["train"], s["val"], s["test"], int(d["num_skills"]), None
    from run_external_benchmark import _sections
    ds, sec = {"eedi_primary": ("eedi", "primary"), "eedi_secondary": ("eedi", "secondary"),
               "statics": ("statics2011", "undergraduate")}[name]
    tr, va, te, S, _, _ = _sections(ds, 500 if ds == "statics2011" else 200)[sec]
    from data.adapters.common import graph_from_train
    return (sequences_to_split(tr), sequences_to_split(va), sequences_to_split(te), S, graph_from_train(tr, S))


def _path(dataset, model, seed, tag):
    return PRED / f"{dataset}__{model}__s{seed}__{tag}.npz"


def _save(dataset, model, seed, tag, p, y, owner, secs, extra=None):
    PRED.mkdir(parents=True, exist_ok=True)
    tmp = _path(dataset, model, seed, tag).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, p=p.astype(np.float32), y=y.astype(np.int8), owner=owner.astype(np.int32), seconds=secs,
                        extra=json.dumps(extra or {}))
    os.replace(tmp, _path(dataset, model, seed, tag))


def _task(t):
    dataset, model, seed, epochs, patience, tag = t
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from baselines.ours import rt_stats_from_split
    tr, va, te, S, graph = load_dataset(dataset)
    t0 = time.time()
    if model in CLASSICAL:
        from baselines import make_classical
        m = make_classical(model, S, seed=seed).fit(tr)
        p, y, o = m.predict_split(te)
        _save(dataset, model, seed, tag, p, y, o, time.time() - t0)
        return model
    from baselines.deep import make_deep
    kw = {"epochs": epochs, "patience": patience, "rt_stats": rt_stats_from_split(tr)}
    if model.startswith("gkt_plus"):
        kw.update(tuned())
    if model == "gkt_plus_pt":
        kw["init_from"] = str(PRETRAIN_PATH)
    if model == "full_gated" and graph is not None:
        kw["graph"] = graph
    name = "gkt_plus" if model == "gkt_plus_pt" else model
    m = make_deep(name, S, seed=seed, **kw).fit(tr, va)
    p, y, o = m.predict_split(te)
    extra = {"best_val_auc": getattr(m, "best_val_auc", None)}
    if model == "gkt_plus_elo":
        extra["blend"] = m.blend_weights
        pg, yg, og = m.gkt.predict_split(te)
        _save(dataset, "gkt_plus", seed, tag, pg, yg, og, time.time() - t0, {"from": "gkt_plus_elo run"})
        if dataset == "assist" and seed == 0:
            m.gkt.save(PRETRAIN_PATH)        # used to pretrain gkt_plus_pt on Statics2011
    _save(dataset, model, seed, tag, p, y, o, time.time() - t0, extra)
    return model


def run(args):
    seeds = list(range(args.seeds))
    tag = f"e{args.epochs}_p{args.patience}"
    models = DEEP + CLASSICAL + (["gkt_plus_pt"] if args.dataset == "statics" else [])
    if args.dataset == "statics" and not PRETRAIN_PATH.exists():
        raise SystemExit("run --dataset assist first (it saves the pretrained GatedKT+ for Statics)")
    tasks = [(args.dataset, m, s, args.epochs, args.patience, tag) for s in seeds for m in models
             if not (m in ("logreg_counts",) and s > 0)]
    tasks = [t for t in tasks if not _path(t[0], t[1], t[2], t[5]).exists()]
    tasks.sort(key=lambda t: (t[1] not in ("gkt_plus_elo", "full_gated", "gkt_plus_pt"), t[2]))
    print(f"[improved] dataset {args.dataset}; seeds {seeds[0]}..{seeds[-1]}; epochs {args.epochs}, patience "
          f"{args.patience}; tuned {tuned()}; {len(tasks)} tasks to run", flush=True)
    t0 = time.time()
    if tasks:
        with get_context("spawn").Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_task, tasks)):
                if (i + 1) % 5 == 0 or i + 1 == len(tasks):
                    print(f"  {i + 1}/{len(tasks)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    report(args, seeds, tag, (time.time() - t0) / 60)


def report(args, seeds, tag, minutes):
    from scipy import stats
    from benchmark_deep import _FastAUC, _holm
    preds = {}
    for f in sorted(PRED.glob(f"{args.dataset}__*__{tag}.npz")):
        _, model, s, _ = f.stem.split("__")
        z = np.load(f)
        preds.setdefault(model, {})[int(s[1:])] = {"p": z["p"].astype(float), "y": z["y"].astype(int),
                                                   "owner": z["owner"], "extra": json.loads(str(z["extra"]))}
    if "logreg_counts" in preds:                       # deterministic: one fit for all seeds
        only = preds["logreg_counts"][0]
        preds["logreg_counts"] = {s: only for s in seeds}
    rows = []
    for m, by in preds.items():
        for s, r in by.items():
            if s in seeds:
                rows.append(dict(all_metrics(r["y"], r["p"]), model=m, seed=s))
    summ = {}
    for m in preds:
        c = mean_ci([r["auc"] for r in rows if r["model"] == m])
        summ[m] = {"auc": c, "brier": float(np.mean([r["brier"] for r in rows if r["model"] == m])),
                   "ece": float(np.mean([r["ece"] for r in rows if r["model"] == m]))}
    n_st = int(next(iter(preds[PRIMARY].values()))["owner"].max()) + 1
    rng = np.random.default_rng(12345)
    fast = {m: [_FastAUC(preds[m][s]["p"], preds[m][s]["y"], preds[m][s]["owner"]) for s in seeds if s in preds[m]]
            for m in preds}
    boot = {m: np.empty(args.bootstrap) for m in preds}
    for b in range(args.bootstrap):
        cnt = np.bincount(rng.integers(0, n_st, n_st), minlength=n_st)
        for m in preds:
            boot[m][b] = np.mean([f(cnt) for f in fast[m]])
    comps = []
    pairs = list(COMPARE) + ([("gkt_plus_pt", "gkt_plus"), ("gkt_plus_pt", "elo")] if "gkt_plus_pt" in preds else [])
    for a, bm in pairs:
        if a not in preds or bm not in preds:
            continue
        A = {r["seed"]: r["auc"] for r in rows if r["model"] == a}
        B = {r["seed"]: r["auc"] for r in rows if r["model"] == bm}
        ss = [s for s in seeds if s in A and s in B]
        d = np.asarray([A[s] - B[s] for s in ss])
        ci = mean_ci(d)
        tp = float(stats.ttest_rel([A[s] for s in ss], [B[s] for s in ss]).pvalue) if len(d) > 1 and d.std(ddof=1) > 0 else 1.0
        wp = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) and len(d) > 1 else 1.0
        lo, hi = np.percentile(boot[a] - boot[bm], [2.5, 97.5])
        comps.append({"model": a, "vs": bm, "n": len(d), "diff": float(d.mean()), "ci_lo": ci["ci_low"],
                      "ci_hi": ci["ci_high"], "boot_lo": float(lo), "boot_hi": float(hi), "t_p": tp, "w_p": wp})
    for c, h in zip(comps, _holm(np.asarray([c["t_p"] for c in comps]))):
        c["t_p_holm"] = float(h)
        pos = c["ci_lo"] > 0 and c["boot_lo"] > 0
        neg = c["ci_hi"] < 0 and c["boot_hi"] < 0
        c["verdict"] = ("better" if h < 0.05 and pos else ("worse" if h < 0.05 and neg else "comparable"))
    with open(OUT / f"improved_{args.dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "auc", "acc", "brier", "ece", "n"], extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["seed"])):
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    blends = [preds[PRIMARY][s]["extra"].get("blend") for s in seeds if s in preds[PRIMARY]]
    bw = {k: float(np.mean([b[k] for b in blends if b])) for k in ("gkt_plus", "elo", "bias")} if any(blends) else {}
    f4 = lambda v: "n/a" if v is None or v != v else f"{v:.4f}"
    L = [f"# Improved model benchmark: {args.dataset}", "",
         f"Generated by `python scripts/benchmark_improved.py --dataset {args.dataset} --seeds {args.seeds} --epochs "
         f"{args.epochs} --patience {args.patience} --workers {args.workers}` (this run: {minutes:.1f} min of new work). "
         f"Pre-registration: reports/v3/IMPROVED_PREREG.md. Seeds {seeds[0]}..{seeds[-1]}; test students {n_st}.", "",
         f"Same budget for every deep model: up to {args.epochs} epochs, early stopping (patience {args.patience}) on "
         f"validation AUC. GatedKT+ settings chosen on validation only: {tuned()}. Mean Elo-blend weights (logit scale, "
         f"fitted on validation): {', '.join(f'{k} {v:+.3f}' for k, v in bw.items()) or 'n/a'}.", "",
         "| Model | Seeds | Test AUC mean ± std | 95% CI (seeds) | 95% CI (bootstrap) | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    for m in sorted(summ, key=lambda m: -summ[m]["auc"]["mean"]):
        c = summ[m]["auc"]
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        L.append(f"| {m} | {c['n']} | {f4(c['mean'])} ± {f4(c['std'])} | [{f4(c['ci_low'])}, {f4(c['ci_high'])}] | "
                 f"[{lo:.4f}, {hi:.4f}] | {summ[m]['brier']:.4f} | {summ[m]['ece']:.4f} |")
    L += ["", "## Paired comparisons (pre-registered)", "",
          "| Model | vs | Pairs | Diff | 95% CI (seeds) | 95% CI (bootstrap) | t p (Holm) | Wilcoxon p | Verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        L.append(f"| {c['model']} | {c['vs']} | {c['n']} | {c['diff']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
                 f"[{c['boot_lo']:+.4f}, {c['boot_hi']:+.4f}] | {c['t_p_holm']:.4f} | {c['w_p']:.4f} | **{c['verdict']}** |")
    L += ["", "Verdict rule: 'better' only if the Holm-corrected paired t-test p < 0.05 and both CIs of the difference "
          "exclude 0; 'worse' symmetrically; otherwise 'comparable'. Deterministic logreg_counts is fitted once.", ""]
    (OUT / f"improved_{args.dataset}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[improved] reports/v3/improved_{args.dataset}.md written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["assist", "eedi_primary", "eedi_secondary", "statics"], default="assist")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        report(args, list(range(args.seeds)), f"e{args.epochs}_p{args.patience}", 0.0)
    else:
        run(args)


if __name__ == "__main__":
    main()
