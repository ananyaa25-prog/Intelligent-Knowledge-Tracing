"""
scripts/benchmark_seeds.py
--------------------------
Multi-seed benchmark of the v3 baselines and our GatedKT variants.

Part "classical" (Stage 4): the classical baselines with many seeds (default 100).
  Protocol A (primary, "resplit"): seed s draws a new random student-level 80/10/10
    split; difficulty buckets are re-derived from that seed's train students only;
    stochastic models (Elo order, BKT EM init) also use s. Variance = split + init.
  Protocol B ("official"): the official split of windowed_data.pkl (seed 42, stored
    buckets) for every seed; only the stochastic models vary. Deterministic models
    are fitted once and their single value is repeated (std 0 by construction).
  Outputs: reports/v3/benchmark_classical.md, benchmark_classical.csv (summary) and
  benchmark_classical_seeds.csv (one row per model x seed x protocol).

Part "deep" (Stage 5): see run_deep(); outputs reports/v3/stage5/benchmark.md etc.

Usage:
  python scripts/benchmark_seeds.py classical --seeds 100 --workers 5
  python scripts/benchmark_seeds.py deep --seeds 10 --epochs 6 --workers 5
  python scripts/benchmark_seeds.py report      # rebuild benchmark.md from saved results
Seeds are 0..N-1 (printed); --max_students caps the number of students per split.
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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines import CLASSICAL, make_classical  # noqa: E402
from baselines.base import (all_metrics, mean_ci, resplit, sequences_to_split,  # noqa: E402
                            split_to_sequences)

WINDOWED = ROOT / "data" / "processed" / "windowed_data.pkl"
OUT = ROOT / "reports" / "v3"
PRED_DIR = ROOT / "data" / "processed_v3" / "benchmark_preds"
NUM_SKILLS = 123

_DATA = {}


def _load():
    if not _DATA:
        with open(WINDOWED, "rb") as f:
            d = pickle.load(f)
        _DATA["num_skills"] = int(d["num_skills"])
        _DATA["official"] = {k: split_to_sequences(d["splits"][k]) for k in ("train", "val", "test")}
        _DATA["all"] = _DATA["official"]["train"] + _DATA["official"]["val"] + _DATA["official"]["test"]
    return _DATA


def _cap(seqs, n, seed):
    if not n or len(seqs) <= n:
        return seqs
    idx = np.random.default_rng(seed).permutation(len(seqs))[:n]
    return [seqs[i] for i in sorted(idx)]


# ---- classical ------------------------------------------------------------------

def _classical_task(task):
    protocol, seed, models, max_students = task
    import warnings
    warnings.filterwarnings("ignore")
    d = _load()
    if protocol == "resplit":
        tr, _, te = resplit(d["all"], seed, max_students=max_students)
    else:
        tr = _cap(d["official"]["train"], int(0.8 * max_students) if max_students else None, 0)
        te = d["official"]["test"]
    rows = []
    for name in models:
        t0 = time.time()
        m = make_classical(name, d["num_skills"], seed=seed).fit(tr)
        p, y, _ = m.predict_split(te)
        r = all_metrics(y, p)
        r.update({"protocol": protocol, "model": name, "seed": seed, "seconds": round(time.time() - t0, 2)})
        rows.append(r)
    return rows


def run_classical(args):
    seeds = list(range(args.seeds))
    print(f"[classical] seeds = {seeds[0]}..{seeds[-1]} ({len(seeds)}); workers = {args.workers}; "
          f"max_students = {args.max_students}")
    models = list(CLASSICAL)
    stochastic = [m for m in models if not CLASSICAL[m].deterministic]
    deterministic = [m for m in models if CLASSICAL[m].deterministic]
    tasks = [("resplit", s, models, args.max_students) for s in seeds]
    tasks += [("official", s, stochastic + (deterministic if s == seeds[0] else []), args.max_students)
              for s in seeds]
    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_classical_task, tasks)):
            rows.extend(res)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    # deterministic models on the official split: one fit, repeated per seed (std 0)
    for m in deterministic:
        base = [r for r in rows if r["protocol"] == "official" and r["model"] == m][0]
        for s in seeds[1:]:
            rows.append(dict(base, seed=s, seconds=0.0))
    elapsed = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["protocol"], r["model"], r["seed"]))
    _write_csv(OUT / "benchmark_classical_seeds.csv", rows,
               ["protocol", "model", "seed", "auc", "acc", "brier", "ece", "n", "seconds"])
    summary = summarise(rows, ["protocol", "model"])
    _write_csv(OUT / "benchmark_classical.csv", summary, list(summary[0].keys()))
    write_classical_md(summary, seeds, args, elapsed, models)
    print(f"[classical] done in {elapsed / 60:.1f} min -> reports/v3/benchmark_classical.md")


def summarise(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        rec = dict(zip(keys, g))
        a = mean_ci([r["auc"] for r in rs])
        rec.update({"n_seeds": a["n"], "n_nan": sum(1 for r in rs if r["auc"] != r["auc"]), "auc_mean": a["mean"], "auc_std": a["std"], "auc_var": a["var"],
                    "auc_ci_low": a["ci_low"], "auc_ci_high": a["ci_high"],
                    "auc_min": float(np.min([r["auc"] for r in rs])), "auc_max": float(np.max([r["auc"] for r in rs]))})
        for k in ("acc", "brier", "ece"):
            c = mean_ci([r[k] for r in rs])
            rec[f"{k}_mean"], rec[f"{k}_std"] = c["mean"], c["std"]
        out.append(rec)
    return out


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k in fields})


def _f(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def write_classical_md(summary, seeds, args, elapsed, models):
    L = ["# Classical baselines: multi-seed benchmark (Stage 4)", "",
         f"Generated by `python scripts/benchmark_seeds.py classical --seeds {args.seeds} --workers {args.workers}"
         + (f" --max_students {args.max_students}" if args.max_students else "") + "`.",
         f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds). Run time {elapsed / 60:.1f} min.",
         "Data: ASSISTments 2009 skill-builder as preprocessed in data/processed/windowed_data.pkl "
         "(3119 students, sequences truncated at 500). Targets: every position t >= 1 of each test "
         "student predicted from that student's history before t (same targets as the main model).",
         "", "Metrics: AUC; ACC at threshold 0.5; Brier score (lower is better); ECE with 10 equal-width bins "
         "(lower is better). 95% CI = t-interval of the mean over seeds; std uses ddof=1.", ""]
    for proto, title, note in (
            ("resplit", "Protocol A (primary): new random 80/10/10 student split per seed",
             "Variance includes the choice of test students and the train set (buckets re-derived from each "
             "seed's train split), plus model initialisation where the model is stochastic."),
            ("official", "Protocol B: official split (same as the main model), seed varies the model only",
             "Deterministic models (majority, skill_mean, logreg_counts, pfa) have one value (std 0 by "
             "construction); their spread in Protocol A comes only from the data split.")):
        L += [f"## {title}", "", note, "",
              "| Model | AUC mean | AUC std | AUC var | 95% CI | min-max | ACC | Brier | ECE |",
              "|---|---|---|---|---|---|---|---|---|"]
        rows = sorted([r for r in summary if r["protocol"] == proto], key=lambda r: -r["auc_mean"])
        for r in rows:
            L.append(f"| {r['model']} | {_f(r['auc_mean'])} | {_f(r['auc_std'])} | {r['auc_var']:.2e} | "
                     f"[{_f(r['auc_ci_low'])}, {_f(r['auc_ci_high'])}] | {_f(r['auc_min'])}-{_f(r['auc_max'])} | "
                     f"{_f(r['acc_mean'])} | {_f(r['brier_mean'])} | {_f(r['ece_mean'])} |")
        L.append("")
    nan_rows = [r for r in summary if r.get("n_nan")]
    L += ["## Failed runs", "",
          ("None: every model produced a finite AUC for every seed." if not nan_rows else
           "; ".join(f"{r['protocol']}/{r['model']}: {r['n_nan']} seed(s) with undefined AUC, excluded from the mean"
                     for r in nan_rows)), ""]
    L += ["## Reading the numbers", "",
          "- `majority` predicts a constant, so its AUC is exactly 0.5; it is listed for ACC/Brier/ECE.",
          "- These are classical baselines only; the deep models and our GatedKT variants are compared in "
          "reports/v3/stage5/benchmark.md (Stage 5) on the official split.",
          "- Model definitions: docs/BASELINES.md. Per-seed values: benchmark_classical_seeds.csv.", ""]
    (OUT / "benchmark_classical.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["classical", "deep", "report"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max_students", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(5, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--models", default=None, help="comma-separated subset (deep part)")
    args = ap.parse_args()
    if args.part == "classical":
        args.seeds = args.seeds or 100
        run_classical(args)
    elif args.part == "deep":
        args.seeds = args.seeds or 10
        from benchmark_deep import run_deep
        run_deep(args)
    else:
        from benchmark_deep import build_report
        build_report(args)


if __name__ == "__main__":
    main()
