"""
src/data/adapters/common.py
---------------------------
Shared helpers that turn an external interaction table into the project's schema
(the per-student sequences used by src/baselines and the main model):

    skill_seq (1..num_skills), correct_seq, difficulty_seq (0 unknown, 1 easy,
    2 medium, 3 hard), problem_seq, response_time_seq, attempt_seq, hint_seq,
    window_id_seq

Rules (same as src/dataset.py wherever the data allow it):
- students with fewer than MIN_INTERACTIONS rows are dropped; sequences are kept in
  time order and truncated to max_len (first max_len answers kept);
- student-level 80/10/10 split with a fixed seed;
- difficulty buckets are derived from the TRAIN split only
  (baselines.base.derive_difficulty_from_sequences: per-skill tertiles of per-problem
  correctness, problems with < 5 train answers -> 0 unknown);
- missing metadata gets NEUTRAL values and a flag: response time = 5000 ms for every
  answer (so the normalised value is the same for all answers), 1 attempt, no hint.
  The flag is stored in the processed file's "flags" and printed in every report.
Nothing here invents interactions: every row comes from the source file.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_V3 = ROOT / "data" / "processed_v3"
MIN_INTERACTIONS = 10
NEUTRAL_RT_MS = 5000.0
WINDOW_GROUP = 3          # consecutive quizzes/assignments per co-occurrence window (as in dataset.py)


def frame_to_sequences(df, max_len: int = 200, min_interactions: int = MIN_INTERACTIONS,
                       skill_map: Optional[Dict[Any, int]] = None) -> Dict[str, Any]:
    """df columns: student, skill (raw id), problem, correct (0/1), order (sortable),
    optional: group (quiz/assignment id), section, response_time, attempts, hint.
    Returns {"sequences": [...], "skill_map": raw->1..S, "num_skills": S, "flags": [...]}."""
    import pandas as pd  # noqa: F401  (callers pass a DataFrame)
    df = df.sort_values(["student", "order"], kind="mergesort")
    counts = df.groupby("student").size()
    df = df[df["student"].isin(counts[counts >= min_interactions].index)]
    if skill_map is None:
        skill_map = {raw: i + 1 for i, raw in enumerate(sorted(df["skill"].unique()))}
    flags = [c for c in ("response_time", "attempts", "hint") if c not in df.columns]
    seqs = []
    for sid, g in df.groupby("student", sort=False):
        g = g.iloc[:max_len]
        L = len(g)
        sk = g["skill"].map(skill_map).fillna(0).astype(np.int64).to_numpy()
        if "group" in g.columns:
            change = (g["group"].to_numpy()[1:] != g["group"].to_numpy()[:-1]).cumsum()
            win = np.concatenate([[0], change]) // WINDOW_GROUP
        else:
            win = np.arange(L) // 15
        seq = {
            "student": sid,
            "skill_seq": sk,
            "correct_seq": g["correct"].astype(np.int64).to_numpy(),
            "problem_seq": g["problem"].astype(np.int64).to_numpy(),
            "difficulty_seq": np.zeros(L, dtype=np.int64),
            "response_time_seq": (g["response_time"].astype(float).to_numpy() if "response_time" in g.columns
                                  else np.full(L, NEUTRAL_RT_MS)),
            "attempt_seq": (g["attempts"].astype(float).to_numpy() if "attempts" in g.columns else np.ones(L)),
            "hint_seq": (g["hint"].astype(float).to_numpy() if "hint" in g.columns else np.zeros(L)),
            "window_id_seq": win.astype(np.int64),
        }
        if "section" in g.columns:
            seq["section"] = str(g["section"].iloc[0])
        seqs.append(seq)
    return {"sequences": seqs, "skill_map": skill_map, "num_skills": len(skill_map),
            "flags": [f"{f}_missing_neutral_value" for f in flags]}


def split_and_bucket(seqs: List[Dict[str, Any]], seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    """Student-level split + train-only difficulty buckets. Returns (train, val, test)."""
    from baselines.base import apply_difficulty, derive_difficulty_from_sequences
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(seqs))
    a = int(train_frac * len(seqs))
    b = a + int(val_frac * len(seqs))
    train = [seqs[i] for i in idx[:a]]
    val = [seqs[i] for i in idx[a:b]]
    test = [seqs[i] for i in idx[b:]]
    dm = derive_difficulty_from_sequences(train)
    for part in (train, val, test):
        apply_difficulty(part, dm)
    return train, val, test


def save_processed(name: str, payload: Dict[str, Any]) -> Path:
    PROCESSED_V3.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_V3 / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_processed(name: str) -> Optional[Dict[str, Any]]:
    path = PROCESSED_V3 / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_from_train(train: List[Dict[str, Any]], num_skills: int):
    """PPMI skill graph from TRAIN windows only, with src/graph_builder.py's functions
    (same min-count filter and PPMI). Returns (edge_index, edge_weight)."""
    import torch
    from graph_builder import apply_min_count_filter, build_adjacency_matrix, collect_counts, compute_ppmi
    from graph_encoder import adjacency_to_edge_index
    from baselines.base import sequences_to_split
    sp = sequences_to_split(train)
    win = np.full(sp["mask"].shape, -1, dtype=np.int64)
    for i, s in enumerate(train):
        win[i, :len(s["window_id_seq"])] = s["window_id_seq"]
    splits = {"train": {"skill_seq": torch.as_tensor(sp["skill_seq"]), "window_id_seq": torch.as_tensor(win),
                        "mask": torch.as_tensor(sp["mask"])}}
    sc, pc, tw = collect_counts(splits, num_skills)
    ppmi = compute_ppmi(sc, apply_min_count_filter(pc), tw)
    adj = build_adjacency_matrix(ppmi, num_skills)
    return adjacency_to_edge_index(torch.tensor(adj, dtype=torch.float32))
