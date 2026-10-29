"""
src/baselines/base.py
---------------------
Shared interface, data handling and metrics for the v3 baselines.

Every baseline implements

    fit(train)                                   -> self
    predict(history, next_skill, next_difficulty) -> P(correct) in [0, 1]

where `train` is a split dict exactly as stored in data/processed/windowed_data.pkl
(skill_seq, correct_seq, difficulty_seq, mask, ... as tensors or numpy arrays) and
`history` is a list of past interactions of ONE student, each either a dict
{"skill", "correct", "difficulty"} or a tuple (skill, correct, difficulty), using the
model's internal skill ids (1..num_skills) and difficulty buckets (0 unknown,
1 easy, 2 medium, 3 hard) -- the same ids, buckets and split as the main model.

For speed, evaluation uses predict_sequence(), which returns the predictions for
positions 1..L-1 of one student's sequence (each from the history before it), i.e.
the same targets as src/train.py run_epoch (correct[:, 1:] under mask[:, 1:]).
The default implementation calls predict() on each prefix; most models override it
with an equivalent online update.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

SEQ_KEYS = ("skill_seq", "correct_seq", "difficulty_seq", "problem_seq",
            "response_time_seq", "attempt_seq", "hint_seq")
MIN_PROBLEM_SAMPLES = 5   # same rule as src/dataset.py derive_difficulty


def _np(x):
    """Tensor or array -> numpy array (no torch import needed)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def split_to_sequences(split: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    """Padded split dict -> list of per-student dicts of 1-D arrays (real steps only).
    A list of such per-student dicts is returned unchanged."""
    if isinstance(split, (list, tuple)):
        return list(split)
    mask = _np(split["mask"]).astype(bool)
    arrays = {k: _np(split[k]) for k in SEQ_KEYS if k in split}
    out = []
    for i in range(mask.shape[0]):
        L = int(mask[i].sum())
        seq = {k: v[i, :L] for k, v in arrays.items()}
        seq["skill_seq"] = seq["skill_seq"].astype(np.int64)
        seq["correct_seq"] = seq["correct_seq"].astype(np.int64)
        seq["difficulty_seq"] = seq.get("difficulty_seq", np.zeros(L)).astype(np.int64)
        out.append(seq)
    return out


def normalise_history(history: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """History items (dicts or tuples) -> (skills, corrects, difficulties) arrays."""
    sk, co, di = [], [], []
    for h in history or []:
        if isinstance(h, dict):
            sk.append(int(h.get("skill", h.get("skill_id", 0))))
            co.append(int(h["correct"]))
            di.append(int(h.get("difficulty", h.get("difficulty_bucket", 0)) or 0))
        else:
            sk.append(int(h[0]))
            co.append(int(h[1]))
            di.append(int(h[2]) if len(h) > 2 else 0)
    return (np.asarray(sk, dtype=np.int64), np.asarray(co, dtype=np.int64),
            np.asarray(di, dtype=np.int64))


class BaseKT:
    """Shared interface. Subclasses set `name` and implement fit / predict."""

    name = "base"
    deterministic = True     # True if the fitted model does not depend on the seed

    def __init__(self, num_skills: int = 123, seed: int = 0):
        self.num_skills = int(num_skills)
        self.seed = int(seed)

    def fit(self, train: Dict[str, Any]) -> "BaseKT":
        raise NotImplementedError

    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        raise NotImplementedError

    def predict_sequence(self, skills: np.ndarray, corrects: np.ndarray,
                         diffs: np.ndarray) -> np.ndarray:
        L = len(skills)
        out = np.zeros(max(L - 1, 0), dtype=np.float64)
        for t in range(1, L):
            hist = list(zip(skills[:t], corrects[:t], diffs[:t]))
            out[t - 1] = self.predict(hist, int(skills[t]), int(diffs[t]))
        return out

    def predict_split(self, split: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (pred, target, student_index) over all predicted positions."""
        preds, targets, owners = [], [], []
        for i, s in enumerate(split_to_sequences(split)):
            if len(s["skill_seq"]) < 2:
                continue
            p = self.predict_sequence(s["skill_seq"], s["correct_seq"], s["difficulty_seq"])
            preds.append(p)
            targets.append(s["correct_seq"][1:])
            owners.append(np.full(len(p), i, dtype=np.int64))
        if not preds:
            return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
        return (np.clip(np.concatenate(preds), 0.0, 1.0), np.concatenate(targets).astype(np.int64),
                np.concatenate(owners))


def clip01(p: float) -> float:
    if p != p:  # NaN
        return 0.5
    return float(min(1.0, max(0.0, p)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---- difficulty buckets and re-splits -------------------------------------------

def derive_difficulty_from_sequences(train_seqs: Sequence[Dict[str, np.ndarray]],
                                     min_samples: int = MIN_PROBLEM_SAMPLES) -> Dict[Tuple[int, int], int]:
    """Same rule as src/dataset.py derive_difficulty, computed from (skill, problem,
    correct) of the TRAIN sequences only: per skill, reliable problems (>= min_samples
    responses) are split into rank-based thirds by mean correctness (3 = lowest = hard);
    everything else is 0 (unknown)."""
    sums: Dict[Tuple[int, int], List[float]] = {}
    for s in train_seqs:
        for k, p, c in zip(s["skill_seq"], s["problem_seq"], s["correct_seq"]):
            key = (int(k), int(p))
            v = sums.setdefault(key, [0.0, 0])
            v[0] += float(c)
            v[1] += 1
    by_skill: Dict[int, List[Tuple[float, int]]] = {}
    out: Dict[Tuple[int, int], int] = {}
    for (k, p), (sc, n) in sums.items():
        out[(k, p)] = 0
        if n >= min_samples:
            by_skill.setdefault(k, []).append((sc / n, p))
    for k, items in by_skill.items():
        n = len(items)
        if n < 3:
            continue
        items.sort(key=lambda x: (x[0], x[1]))   # deterministic tie-break by problem id
        for i, (_, p) in enumerate(items):
            out[(k, p)] = 3 if i < n // 3 else (2 if i < (2 * n) // 3 else 1)
    return out


def apply_difficulty(seqs: Sequence[Dict[str, np.ndarray]], diff_map: Dict[Tuple[int, int], int]):
    for s in seqs:
        s["difficulty_seq"] = np.asarray([diff_map.get((int(k), int(p)), 0)
                                          for k, p in zip(s["skill_seq"], s["problem_seq"])], dtype=np.int64)
    return seqs


def sequences_to_split(seqs: Sequence[Dict[str, np.ndarray]], max_len: int | None = None) -> Dict[str, np.ndarray]:
    """Inverse of split_to_sequences (numpy, padded with zeros)."""
    n = len(seqs)
    L = max_len or max((len(s["skill_seq"]) for s in seqs), default=1)
    out = {}
    for k in SEQ_KEYS:
        if seqs and k in seqs[0]:
            dt = np.float32 if k in ("correct_seq", "response_time_seq", "attempt_seq", "hint_seq") else np.int64
            arr = np.zeros((n, L), dtype=dt)
            for i, s in enumerate(seqs):
                m = min(len(s[k]), L)
                arr[i, :m] = s[k][:m]
            out[k] = arr
    mask = np.zeros((n, L), dtype=bool)
    for i, s in enumerate(seqs):
        mask[i, :min(len(s["skill_seq"]), L)] = True
    out["mask"] = mask
    out["student_ids"] = np.arange(n)
    return out


def resplit(all_seqs: Sequence[Dict[str, np.ndarray]], seed: int, train_frac: float = 0.8,
            val_frac: float = 0.1, max_students: int | None = None):
    """Random student-level 80/10/10 re-split for one benchmark seed. Difficulty buckets
    are re-derived from that seed's TRAIN students only (no leakage). Returns
    (train, val, test) lists of sequences (copies)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_seqs))
    if max_students:
        idx = idx[:max_students]
    n = len(idx)
    a, b = int(train_frac * n), int(train_frac * n) + int(val_frac * n)
    pick = lambda ids: [{k: v.copy() for k, v in all_seqs[i].items()} for i in ids]
    train, val, test = pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])
    dm = derive_difficulty_from_sequences(train)
    for part in (train, val, test):
        apply_difficulty(part, dm)
    return train, val, test


# ---- metrics ------------------------------------------------------------------

def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the rank statistic (ties get average ranks); NaN if one class."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def all_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return {
        "auc": auc(y, p),
        "acc": float(((p >= 0.5).astype(np.float64) == y).mean()) if len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        "ece": ece(y, p),
        "n": int(len(y)),
    }


def mean_ci(values: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    """Mean, std (ddof=1), variance and t-based CI of the mean."""
    v = np.asarray([x for x in values if x == x], dtype=np.float64)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "var": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    m = float(v.mean())
    if n == 1:
        return {"mean": m, "std": 0.0, "var": 0.0, "ci_low": m, "ci_high": m, "n": 1}
    from scipy.stats import t
    sd = float(v.std(ddof=1))
    h = float(t.ppf(0.5 + level / 2.0, n - 1) * sd / math.sqrt(n))
    return {"mean": m, "std": sd, "var": sd * sd, "ci_low": m - h, "ci_high": m + h, "n": n}
