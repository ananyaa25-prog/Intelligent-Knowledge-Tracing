"""
src/baselines/elo.py
--------------------
Elo-style student/item rating (in the spirit of Pelanek's Elo for adaptive practice).

    P(correct) = sigmoid(theta + theta_k - b_{k,d})

theta = the student's global ability, theta_k = the student's ability on skill k,
b_{k,d} = difficulty of the item class (skill k, difficulty bucket d).
Updates after each answer use the "uncertainty" step size K(n) = A / (1 + B n),
n = number of earlier updates of that parameter.

fit(): item difficulties b are learned by one Elo pass over the TRAIN students
(student order shuffled with the seed, so the fit depends on the seed). At
prediction time b is frozen and only the student abilities are updated from the
student's own history (no information flows between test students).
"""

from typing import Any, Dict

import numpy as np

from baselines.base import BaseKT, normalise_history, split_to_sequences

A_STEP, B_STEP = 1.0, 0.05


def _k(n: float) -> float:
    return A_STEP / (1.0 + B_STEP * n)


def _sig(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + np.exp(-x))


class EloBaseline(BaseKT):
    name = "elo"
    deterministic = False   # training order depends on the seed

    def fit(self, train: Dict[str, Any]) -> "EloBaseline":
        seqs = split_to_sequences(train) if isinstance(train, dict) else train
        S = self.num_skills + 1
        self.b = np.zeros((S, 4))
        self.nb = np.zeros((S, 4))
        rng = np.random.default_rng(self.seed)
        for i in rng.permutation(len(seqs)):
            s = seqs[i]
            theta, n_theta = 0.0, 0
            th_k: Dict[int, float] = {}
            n_k: Dict[int, int] = {}
            for k, c, d in zip(s["skill_seq"], s["correct_seq"], s["difficulty_seq"]):
                k = int(k) if 0 <= int(k) < S else 0
                d = min(max(int(d), 0), 3)
                p = _sig(theta + th_k.get(k, 0.0) - self.b[k, d])
                err = float(c) - p
                theta += _k(n_theta) * err
                th_k[k] = th_k.get(k, 0.0) + _k(n_k.get(k, 0)) * err
                self.b[k, d] -= _k(self.nb[k, d]) * err
                n_theta += 1
                n_k[k] = n_k.get(k, 0) + 1
                self.nb[k, d] += 1
        return self

    def _run(self, skills, corrects, diffs, n_pred_from: int):
        S = self.b.shape[0]
        theta, n_theta = 0.0, 0
        th_k: Dict[int, float] = {}
        n_k: Dict[int, int] = {}
        out = []
        for t in range(len(skills)):
            k = int(skills[t]) if 0 <= int(skills[t]) < S else 0
            d = min(max(int(diffs[t]), 0), 3)
            p = _sig(theta + th_k.get(k, 0.0) - self.b[k, d])
            if t >= n_pred_from:
                out.append(p)
            if t < len(corrects):
                err = float(corrects[t]) - p
                theta += _k(n_theta) * err
                th_k[k] = th_k.get(k, 0.0) + _k(n_k.get(k, 0)) * err
                n_theta += 1
                n_k[k] = n_k.get(k, 0) + 1
        return np.asarray(out, dtype=np.float64)

    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        sk, co, di = normalise_history(history)
        sk = np.append(sk, int(next_skill))
        di = np.append(di, int(next_difficulty))
        return float(self._run(sk, co, di, len(sk) - 1)[0])

    def predict_sequence(self, skills, corrects, diffs):
        return self._run(skills, corrects, diffs, 1)
