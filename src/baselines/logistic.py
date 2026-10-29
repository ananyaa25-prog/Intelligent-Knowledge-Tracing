"""
src/baselines/logistic.py
-------------------------
Logistic regression on counts (scikit-learn).

Features for predicting step t (target skill k, bucket d) from the history before t:
  log1p(#correct on k), log1p(#wrong on k), log1p(#correct overall),
  log1p(#wrong overall), one-hot target skill, one-hot difficulty bucket.
Coefficients on the counts are shared across skills (contrast: PFA, per-skill).
"""

from typing import Any, Dict, List

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from baselines.base import BaseKT, normalise_history, split_to_sequences


class _CountFeatures:
    """Builds the sparse design rows for one student sequence (positions 1..L-1)."""

    def __init__(self, num_skills: int, per_skill: bool):
        self.S = num_skills + 1
        self.per_skill = per_skill
        # column blocks
        self.n_cols = (3 * self.S + 4) if per_skill else (4 + self.S + 4)

    def rows(self, skills, corrects, diffs, start: int = 1):
        """Yields (cols, vals) for each position t >= start, using history < t."""
        S = self.S
        succ: Dict[int, int] = {}
        fail: Dict[int, int] = {}
        tot_s = tot_f = 0
        out_cols: List[np.ndarray] = []
        out_vals: List[np.ndarray] = []
        for t in range(len(skills)):
            k = int(skills[t]) if 0 <= int(skills[t]) < S else 0
            d = min(max(int(diffs[t]), 0), 3)
            if t >= start:
                s_k, f_k = succ.get(k, 0), fail.get(k, 0)
                if self.per_skill:   # PFA: beta_k, gamma_k * succ_k, rho_k * fail_k, diff
                    cols = [k, S + k, 2 * S + k, 3 * S + d]
                    vals = [1.0, float(s_k), float(f_k), 1.0]
                else:                # LR on counts
                    cols = [0, 1, 2, 3, 4 + k, 4 + S + d]
                    vals = [np.log1p(s_k), np.log1p(f_k), np.log1p(tot_s), np.log1p(tot_f), 1.0, 1.0]
                out_cols.append(np.asarray(cols))
                out_vals.append(np.asarray(vals, dtype=np.float64))
            if int(corrects[t]):
                succ[k] = succ.get(k, 0) + 1
                tot_s += 1
            else:
                fail[k] = fail.get(k, 0) + 1
                tot_f += 1
        return out_cols, out_vals

    def matrix(self, seqs):
        indptr, indices, data, ys = [0], [], [], []
        for s in seqs:
            if len(s["skill_seq"]) < 2:
                continue
            cols, vals = self.rows(s["skill_seq"], s["correct_seq"], s["difficulty_seq"])
            for c, v in zip(cols, vals):
                indices.append(c)
                data.append(v)
                indptr.append(indptr[-1] + len(c))
            ys.append(s["correct_seq"][1:])
        X = sparse.csr_matrix((np.concatenate(data) if data else np.zeros(0),
                               np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
                               np.asarray(indptr)), shape=(len(indptr) - 1, self.n_cols))
        y = np.concatenate(ys) if ys else np.zeros(0)
        return X, y


class LogisticCountsBaseline(BaseKT):
    name = "logreg_counts"
    per_skill = False
    C = 1.0

    def fit(self, train: Dict[str, Any]) -> "LogisticCountsBaseline":
        seqs = split_to_sequences(train) if isinstance(train, dict) else train
        self.feat = _CountFeatures(self.num_skills, self.per_skill)
        X, y = self.feat.matrix(seqs)
        self.clf = LogisticRegression(C=self.C, max_iter=500, solver="lbfgs")
        if len(np.unique(y)) < 2:
            self.const = float(y.mean()) if len(y) else 0.5
            self.clf = None
        else:
            self.clf.fit(X, y)
        return self

    def _score(self, cols, vals) -> np.ndarray:
        if not cols:
            return np.zeros(0)
        if self.clf is None:
            return np.full(len(cols), self.const)
        X = sparse.csr_matrix((np.concatenate(vals), np.concatenate(cols),
                               np.cumsum([0] + [len(c) for c in cols])), shape=(len(cols), self.feat.n_cols))
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        sk, co, di = normalise_history(history)
        sk = np.append(sk, int(next_skill))
        co = np.append(co, 0)
        di = np.append(di, int(next_difficulty))
        cols, vals = self.feat.rows(sk, co, di, start=len(sk) - 1)
        return float(self._score(cols, vals)[0])

    def predict_sequence(self, skills, corrects, diffs):
        cols, vals = self.feat.rows(skills, corrects, diffs)
        return self._score(cols, vals)
