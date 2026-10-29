"""
scripts/run_external_benchmark.py
---------------------------------
v3 Stage 7: benchmark on real external datasets, per learner-level section, plus a
cross-section transfer table.

For every adapter in ADAPTERS whose raw files exist in data/external/ (see
scripts/fetch_datasets.py / docs/DATASETS_TODO.md) the data are converted to our
schema (data/processed_v3/<name>.pkl). For each section present in the data:
  student split 80/10/10 (seed 42), train-only difficulty buckets, a skill graph
  from train windows only (for full_gated), then
  - classical baselines (deterministic ones once, stochastic ones per seed),
  - deep models dkt_gru, sakt_lite, dkt_only (ours, no graph) and full_gated (ours,
    graph from that dataset), seeds 0..N-1, --epochs each.
Transfer: train on section A, test on section B's test students (A's difficulty
buckets and graph), for elo, bkt_diff, dkt_gru, full_gated.
No data are invented: if no dataset is available, the report says so and no
section is marked validated.

Outputs: reports/v3/datasets.md, reports/v3/external_benchmark.md / .csv,
         reports/v3/transfer.csv
Usage: python scripts/run_external_benchmark.py --seeds 3 --epochs 4 --max_students 3000 --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines import CLASSICAL  # noqa: E402
from baselines.base import all_metrics, mean_ci  # noqa: E402

OUT = ROOT / "reports" / "v3"
ADAPTERS = {"eedi": "data.adapters.eedi"}
DEEP = ["dkt_gru", "sakt_lite", "dkt_only", "full_gated"]
TRANSFER_MODELS = ["elo", "bkt_diff", "dkt_gru", "full_gated"]

_CACHE = {}


def _sections(name, max_len):
    """(section -> (train, val, test, num_skills, flags, diff_map)) from the processed file."""
    key = (name, max_len)
    if key in _CACHE:
        return _CACHE[key]
    from data.adapters.common import load_processed, split_and_bucket
    from baselines.base import derive_difficulty_from_sequences
    d = load_processed(name)
    out = {}
    for sec in sorted({s.get("section") for s in d["sequences"] if s.get("section")}):
        seqs = [s for s in d["sequences"] if s.get("section") == sec]
        tr, va, te = split_and_bucket(seqs, seed=42)
        out[sec] = (tr, va, te, d["num_skills"], d["flags"], derive_difficulty_from_sequences(tr))
    _CACHE[key] = out
    return out


def _task(t):
    name, train_sec, test_sec, model, seed, epochs, max_len = t
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    from baselines.base import apply_difficulty, sequences_to_split
    secs = _sections(name, max_len)
    tr, va, _, S, _, dmap = secs[train_sec]
    te = secs[test_sec][2]
    if test_sec != train_sec:   # model trained on A: score B with A's buckets
        te = apply_difficulty([{k: v.copy() if hasattr(v, "copy") else v for k, v in s.items()} for s in te], dmap)
    t0 = time.time()
    if model in CLASSICAL:
        from baselines import make_classical
        m = make_classical(model, S, seed=seed).fit(tr)
        p, y, _ = m.predict_split(te)
    else:
        from baselines.deep import make_deep
        from baselines.ours import rt_stats_from_split
        from data.adapters.common import graph_from_train
        trs = sequences_to_split(tr)
        kw = {"rt_stats": rt_stats_from_split(trs)}
        if model == "full_gated":
            kw["graph"] = graph_from_train(tr, S)
        m = make_deep(model, S, seed=seed, epochs=epochs, **kw)
        m.fit(trs, sequences_to_split(va))
        p, y, _ = m.predict_split(sequences_to_split(te))
    r = all_metrics(y, p)
    r.update({"dataset": name, "train_section": train_sec, "test_section": test_sec, "model": model,
              "seed": seed, "seconds": round(time.time() - t0, 1)})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_students", type=int, default=3000, help="per section")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    print(f"[external] seeds = {seeds}; epochs = {args.epochs}; max_students/section = {args.max_students}")
    import importlib
    from data.adapters.common import PROCESSED_V3, load_processed
    status = {}
    for name, mod in ADAPTERS.items():
        ad = importlib.import_module(mod)
        if (PROCESSED_V3 / f"{name}.pkl").exists() and not args.rebuild:
            status[name] = "processed file present"
        elif ad.available():
            t0 = time.time()
            ad.build(max_students_per_section=args.max_students, max_len=args.max_len)
            status[name] = f"converted from data/external in {time.time() - t0:.0f}s"
        else:
            status[name] = "raw files not available"
        print(f"[external] {name}: {status[name]}", flush=True)
    names = [n for n in ADAPTERS if (PROCESSED_V3 / f"{n}.pkl").exists()]
    write_datasets_md(status, names)
    if not names:
        (OUT / "external_benchmark.md").write_text(
            "# External datasets benchmark (Stage 7)\n\nNo external dataset was available, so nothing was run and "
            "no section is validated beyond ASSISTments. See docs/DATASETS_TODO.md.\n", encoding="utf-8")
        print("[external] no dataset available; report written")
        return 0
    tasks = []
    for name in names:
        secs = sorted({s.get("section") for s in load_processed(name)["sequences"] if s.get("section")})
        for sec in secs:
            for model in CLASSICAL:
                for s in (seeds if not CLASSICAL[model].deterministic else seeds[:1]):
                    tasks.append((name, sec, sec, model, s, args.epochs, args.max_len))
            for model in DEEP:
                for s in seeds:
                    tasks.append((name, sec, sec, model, s, args.epochs, args.max_len))
        for a in secs:
            for b in secs:
                if a != b:
                    for model in TRANSFER_MODELS:
                        for s in (seeds if model not in CLASSICAL or not CLASSICAL[model].deterministic else seeds[:1]):
                            tasks.append((name, a, b, model, s, args.epochs, args.max_len))
    tasks.sort(key=lambda t: t[3] not in ("full_gated", "dkt_only", "sakt_lite"))
    print(f"[external] {len(tasks)} tasks", flush=True)
    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks)):
            rows.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    elapsed = (time.time() - t0) / 60
    rows.sort(key=lambda r: (r["dataset"], r["train_section"], r["test_section"], r["model"], r["seed"]))
    fields = ["dataset", "train_section", "test_section", "model", "seed", "auc", "acc", "brier", "ece", "n", "seconds"]
    with open(OUT / "external_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    write_report(rows, names, args, elapsed, seeds)
    print(f"[external] done in {elapsed:.1f} min -> reports/v3/external_benchmark.md")
    return 0


def write_datasets_md(status, names):
    from data.adapters.common import load_processed
    fetch = {}
    p = OUT / "datasets_status.json"
    if p.exists():
        fetch = json.loads(p.read_text(encoding="utf-8"))
    L = ["# External datasets (Stage 7)", "", "Generated by `python scripts/run_external_benchmark.py` "
         "(download attempts by `python scripts/fetch_datasets.py`, manual steps in docs/DATASETS_TODO.md).", "",
         "| Dataset | Download / local status | Adapter status |", "|---|---|---|"]
    for k, v in fetch.items():
        L.append(f"| {k} | {v['status']} | {status.get(k, 'no adapter (dataset not available)')} |")
    for name in names:
        d = load_processed(name)
        by = {}
        for s in d["sequences"]:
            by.setdefault(s.get("section"), []).append(len(s["skill_seq"]))
        L += ["", f"## {name}", "", f"Source: {d['source']}. Rows in source (tasks 1-2 train file, after joining "
              f"answer dates): {d['n_rows_source']}. Section rule: {d['section_rule']}. Students in the source by "
              f"section rule: {d['section_counts_all_students']}. Kept: a seeded sample of at most "
              f"{d['max_students_per_section']} students per section with >= 10 answers, first {d['max_len']} answers "
              f"each. Skills: {d['num_skills']} (most specific subject of each question).",
              f"Metadata flags: {', '.join(d['flags']) or 'none'} (neutral values used).", "",
              "| Section | Students | Interactions | Mean length |", "|---|---|---|---|"]
        for sec, lens in sorted(by.items()):
            L.append(f"| {sec} | {len(lens)} | {sum(lens)} | {np.mean(lens):.1f} |")
    L.append("")
    (OUT / "datasets.md").write_text("\n".join(L), encoding="utf-8")


def _f(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def write_report(rows, names, args, elapsed, seeds):
    L = ["# External datasets benchmark by section (Stage 7)", "",
         f"Generated by `python scripts/run_external_benchmark.py --seeds {args.seeds} --epochs {args.epochs} "
         f"--max_students {args.max_students} --max_len {args.max_len} --workers {args.workers}` in {elapsed:.1f} min. "
         f"Seeds {seeds}. Same protocol as Stage 5 (best-val weights, batch 32, Adam 1e-3) with a smaller budget; "
         "deterministic classical models are fitted once. Missing metadata (response time, attempts, hints) are "
         "neutral constants, so the metadata branch of our model carries no information here.", ""]
    for name in names:
        secs = sorted({r["train_section"] for r in rows if r["dataset"] == name})
        L += [f"## {name}: in-section test AUC (mean ± std over seeds)", "",
              "| Model | " + " | ".join(secs) + " |", "|---|" + "---|" * len(secs)]
        models = sorted({r["model"] for r in rows if r["dataset"] == name and r["train_section"] == r["test_section"]})
        def cell(m, a, b):
            v = [r["auc"] for r in rows if r["dataset"] == name and r["model"] == m and r["train_section"] == a
                 and r["test_section"] == b]
            c = mean_ci(v)
            return f"{_f(c['mean'])} ± {_f(c['std'])} (n={c['n']})" if v else "-"
        models.sort(key=lambda m: -np.nanmean([r["auc"] for r in rows if r["dataset"] == name and r["model"] == m
                                               and r["train_section"] == r["test_section"]]))
        for m in models:
            L.append(f"| {m} | " + " | ".join(cell(m, s, s) for s in secs) + " |")
        L += ["", f"## {name}: cross-section transfer (train on row section, test on column section)", "",
              "Rows = training section, columns = test section; the diagonal repeats the in-section result. "
              "A drop off the diagonal shows how much a model trained on one learner level loses on another.", ""]
        for m in TRANSFER_MODELS:
            L += [f"**{m}**", "", "| train \\ test | " + " | ".join(secs) + " |", "|---|" + "---|" * len(secs)]
            for a in secs:
                L.append(f"| {a} | " + " | ".join(cell(m, a, b) for b in secs) + " |")
            L.append("")
    L += ["## Notes", "",
          "- Transfer between ASSISTments and these data is not possible with these models: the skill ids differ "
          "(no shared skill vocabulary), so every external result comes from a model trained on that dataset.",
          "- A section is labelled \"validated on real data\" in the app only if a real dataset for that level was "
          "loaded and evaluated here; see src/serving/sections.py and docs/DECISIONS.md.", ""]
    (OUT / "external_benchmark.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
