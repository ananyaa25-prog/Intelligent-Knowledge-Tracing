"""
src/baselines/majority.py
-------------------------
Two reference baselines that ignore the student's history:

- MajorityBaseline: always predicts the global train correctness rate
  (a constant, so its AUC is 0.5 by construction; useful for ACC/Brier/ECE).
- SkillMeanBaseline: train correctness rate of the target (skill, difficulty bucket),
  smoothed towards the skill rate and then the global rate (m-estimate, m=5).
"""

from typing import Any, Dict

import numpy as np

from baselines.base import BaseKT, split_to_sequences

SMOOTHING_M = 5.0


class MajorityBaseline(BaseKT):
    name = "majority"

    def fit(self, train: Dict[str, Any]) -> "MajorityBaseline":
        seqs = split_to_sequences(train) if isinstance(train, dict) else train
        ys = np.concatenate([s["correct_seq"][1:] for s in seqs if len(s["correct_seq"]) > 1])
        self.p = float(ys.mean()) if len(ys) else 0.5
        return self

    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        return self.p

    def predict_sequence(self, skills, corrects, diffs):
        return np.full(max(len(skills) - 1, 0), self.p)


class SkillMeanBaseline(BaseKT):
    name = "skill_mean"

    def fit(self, train: Dict[str, Any]) -> "SkillMeanBaseline":
        seqs = split_to_sequences(train) if isinstance(train, dict) else train
        S = self.num_skills + 1
        sk = np.concatenate([s["skill_seq"][1:] for s in seqs if len(s["skill_seq"]) > 1])
        di = np.concatenate([s["difficulty_seq"][1:] for s in seqs if len(s["skill_seq"]) > 1])
        y = np.concatenate([s["correct_seq"][1:] for s in seqs if len(s["skill_seq"]) > 1]).astype(float)
        sk = np.clip(sk, 0, S - 1)
        di = np.clip(di, 0, 3)
        self.global_p = float(y.mean()) if len(y) else 0.5
        n_k = np.bincount(sk, minlength=S).astype(float)
        c_k = np.bincount(sk, weights=y, minlength=S)
        self.skill_p = (c_k + SMOOTHING_M * self.global_p) / (n_k + SMOOTHING_M)
        flat = sk * 4 + di
        n_kd = np.bincount(flat, minlength=S * 4).astype(float).reshape(S, 4)
        c_kd = np.bincount(flat, weights=y, minlength=S * 4).reshape(S, 4)
        self.table = (c_kd + SMOOTHING_M * self.skill_p[:, None]) / (n_kd + SMOOTHING_M)
        return self

    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        k = int(next_skill)
        if k < 0 or k >= self.table.shape[0]:
            return self.global_p
        return float(self.table[k, min(max(int(next_difficulty), 0), 3)])

    def predict_sequence(self, skills, corrects, diffs):
        if len(skills) < 2:
            return np.zeros(0)
        k = np.clip(np.asarray(skills[1:]), 0, self.table.shape[0] - 1)
        d = np.clip(np.asarray(diffs[1:]), 0, 3)
        return self.table[k, d].astype(float)
