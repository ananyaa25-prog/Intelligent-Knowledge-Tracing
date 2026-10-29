"""
src/baselines/gkt_plus.py
-------------------------
GatedKT+ : an improved variant of our model (v3 follow-up). src/model.py and the
existing checkpoints are unchanged; this is a separate model with the same
interface as every baseline.

Changes vs full_gated, each motivated by the v3 benchmark (reports/v3/benchmark.md):
  1. Per-question difficulty (the largest gap to Elo on Eedi / Statics2011):
     question embedding e_q = c_skill + mu_q * d_skill + bucket  (Rasch-style, as in
     AKT, Ghosh et al. 2020) and a learned difficulty offset b_q subtracted from the
     output logit. mu_q and b_q start at 0; questions not seen in the TRAIN split use
     index 0, which stays at 0, so unseen questions fall back to the skill level.
  2. No skill graph (it added only +0.0007 AUC).
  3. Regularisation for longer training: dropout before a 2-layer head, weight decay,
     early stopping on validation AUC (DeepKT.patience).
  Kept from GatedKT: short-term GRU over the last 10 steps, long-term GRU, metadata
  encoder, learned vector gate between short- and long-term state.

GKTPlusElo ("gkt_plus_elo"): GatedKT+ stacked with the Elo baseline: a 2-feature
logistic regression on [logit P_gkt+, logit P_elo] fitted on the VALIDATION split
only (the test split is never used for fitting).

Pretraining (kwargs init_from=<path>): copies every parameter whose name and shape
match from a GatedKT+ checkpoint trained on another dataset (encoders, gate, head);
embeddings differ in size and are trained from scratch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from baselines.base import BaseKT, normalise_history
from baselines.deep_common import DeepKT, to_tensors

HIDDEN = 64
META_DIM = 16


class _GKTPlusNet(nn.Module):
    def __init__(self, num_skills, num_problems, rt_mean, rt_std, hidden=HIDDEN, dropout=0.2, window=10):
        super().__init__()
        from short_term_encoder import ShortTermEncoder
        from long_term_encoder import LongTermEncoder
        from metadata_encoder import MetadataEncoder
        self.S = num_skills
        self.skill_emb = nn.Embedding(num_skills + 1, hidden, padding_idx=0)
        self.skill_var = nn.Embedding(num_skills + 1, hidden)
        self.prob_mu = nn.Embedding(num_problems + 1, 1)
        self.prob_b = nn.Embedding(num_problems + 1, 1)
        nn.init.zeros_(self.prob_mu.weight)
        nn.init.zeros_(self.prob_b.weight)
        self.correct_emb = nn.Embedding(2, hidden)
        self.diff_emb = nn.Embedding(4, hidden)
        self.short_term = ShortTermEncoder(input_dim=hidden, hidden_dim=hidden, window_size=window)
        self.long_term = LongTermEncoder(input_dim=hidden, hidden_dim=hidden)
        self.metadata_encoder = MetadataEncoder(hidden_dim=META_DIM, rt_log_mean=rt_mean, rt_log_std=rt_std)
        self.gate = nn.Linear(2 * hidden + META_DIM, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def question(self, skill, diff, problem):
        return (self.skill_emb(skill) + self.prob_mu(problem) * self.skill_var(skill)
                + self.diff_emb(diff.clamp(0, 3)))

    def forward(self, skill, correct, mask, rt, att, hint, diff, problem=None):
        skill = skill.clamp(0, self.S)
        if problem is None:
            problem = torch.zeros_like(skill)
        e = self.question(skill, diff, problem)
        x = e[:, :-1] + self.correct_emb(correct[:, :-1].clamp(0, 1))
        hm = mask[:, :-1]
        hs = self.short_term(x, hm)
        hl = self.long_term(x * hm.unsqueeze(-1), hm)
        meta = self.metadata_encoder(rt[:, :-1], att[:, :-1], hint[:, :-1])
        g = torch.sigmoid(self.gate(torch.cat([hs, hl, meta], dim=-1)))
        h = self.drop(g * hs + (1 - g) * hl)
        logit = self.head(torch.cat([h, e[:, 1:]], dim=-1)).squeeze(-1) - self.prob_b(problem[:, 1:]).squeeze(-1)
        return torch.sigmoid(logit)


class GKTPlus(DeepKT):
    name = "gkt_plus"
    uses_problem = True
    epochs = 15
    patience = 3
    weight_decay = 1e-5
    dropout = 0.2

    def __init__(self, num_skills: int = 123, seed: int = 0, **kw):
        for k in ("dropout", "weight_decay", "patience", "lr"):
            if k in kw:
                setattr(self, k, kw.pop(k))
        super().__init__(num_skills=num_skills, seed=seed, **kw)

    def build_net(self):
        n_prob = len(getattr(self, "_problem_index", [])) if getattr(self, "_problem_index", None) is not None else 0
        net = _GKTPlusNet(self.num_skills, n_prob, self.rt_stats[0], self.rt_stats[1], dropout=self.dropout)
        src = self.kwargs.get("init_from")
        if src:
            state = torch.load(src, map_location="cpu")
            own = net.state_dict()
            keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape
                    and not k.startswith(("skill_emb", "skill_var", "prob_"))}
            own.update(keep)
            net.load_state_dict(own)
            self.pretrained_params = sorted(keep)
        return net


class GKTPlusElo(BaseKT):
    """GatedKT+ and Elo, combined by a logistic regression fitted on validation data."""

    name = "gkt_plus_elo"
    deterministic = False

    def __init__(self, num_skills: int = 123, seed: int = 0, **kw):
        super().__init__(num_skills=num_skills, seed=seed)
        self.gkt = GKTPlus(num_skills=num_skills, seed=seed, **kw)
        from baselines.elo import EloBaseline
        self.elo = EloBaseline(num_skills=num_skills, seed=seed)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def _features(self, pg, pe):
        return np.column_stack([self._logit(pg), self._logit(pe)])

    def fit(self, train: Dict[str, Any], val: Optional[Dict[str, Any]] = None) -> "GKTPlusElo":
        if val is None:
            raise ValueError("gkt_plus_elo needs a validation split to fit the blend")
        from sklearn.linear_model import LogisticRegression
        self.gkt.fit(train, val)
        self.elo.fit(train)
        pg, y, _ = self.gkt.predict_split(val)
        pe, ye, _ = self.elo.predict_split(val)
        if not np.array_equal(y, ye):
            raise RuntimeError("GatedKT+ and Elo validation targets are not aligned")
        self.blend = LogisticRegression(C=1e3, max_iter=1000).fit(self._features(pg, pe), y)
        self.blend_weights = {"gkt_plus": float(self.blend.coef_[0][0]), "elo": float(self.blend.coef_[0][1]),
                              "bias": float(self.blend.intercept_[0])}
        self.best_val_auc = getattr(self.gkt, "best_val_auc", None)
        return self

    def predict_split(self, split):
        pg, y, owner = self.gkt.predict_split(split)
        pe, ye, _ = self.elo.predict_split(split)
        return self.blend.predict_proba(self._features(pg, pe))[:, 1], y, owner

    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        pg = self.gkt.predict(history, next_skill, next_difficulty)
        pe = self.elo.predict(history, next_skill, next_difficulty)
        return float(self.blend.predict_proba(self._features([pg], [pe]))[0, 1])

    def predict_sequence(self, skills, corrects, diffs):
        pg = self.gkt.predict_sequence(skills, corrects, diffs)
        pe = self.elo.predict_sequence(skills, corrects, diffs)
        if len(pg) == 0:
            return pg
        return self.blend.predict_proba(self._features(pg, pe))[:, 1]

    def save(self, path=None):
        return self.gkt.save(path)
