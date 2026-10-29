"""
src/baselines/gkt_plus.py
-------------------------
GatedKT+ : an improved variant of our model (v3 follow-up). src/model.py and the
existing checkpoints are unchanged; this is a separate model with the same
interface as every baseline.

Changes vs full_gated, each motivated by the v3 benchmark (reports/v3/stage5/benchmark.md):
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

Hint / attempt pathway (kwargs meta_mode=..., v3 follow-up 2). In ASSISTments a hint
request makes the first attempt count as wrong and most multi-attempt rows are wrong
too, so hints and attempts say HOW a student got something wrong. meta_mode:
  "gate" (default, = gkt_plus above): metadata reaches only the gate.
  "full" (gkt_plus_meta): (a) the GRU input uses a 7-way outcome embedding (correct /
     wrong x {1, 2, 3+ attempts} x {hint, no hint}) instead of a binary correct
     embedding, (b) the metadata vector is projected into the GRU input, so both the
     short- and long-term GRU states are updated differently, and (c) the gate also sees
     the recent help-seeking rate (share of hints and of multi-attempt answers over the
     last `window` steps), so it can shift weight between short- and long-term state.
  "gru": (a)+(b) only, gate without metadata.   "none": no metadata anywhere.

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
META_MODES = ("gate", "full", "gru", "none")
N_OUTCOMES = 7


def outcome_class(correct, att, hint):
    """0 correct; 1-3 wrong with 1 / 2 / 3+ attempts, no hint; 4-6 the same with a hint."""
    a = att.clamp(min=1, max=3).long() - 1
    cls = 1 + a + 3 * (hint > 0).long()
    return torch.where(correct.long() > 0, torch.zeros_like(cls), cls)


def help_rates(att, hint, mask, window):
    """Causal share of hint use and of multi-attempt answers over the last `window`
    real steps (inclusive), [B, T, 2]."""
    m = mask.float()
    f = torch.stack([(hint > 0).float() * m, (att > 1).float() * m, m], dim=-1)
    cs = f.cumsum(dim=1)
    if cs.shape[1] > window:
        cs = cs - torch.cat([torch.zeros_like(cs[:, :window]), cs[:, :-window]], dim=1)
    return cs[..., :2] / cs[..., 2:].clamp(min=1.0)


class _GKTPlusNet(nn.Module):
    def __init__(self, num_skills, num_problems, rt_mean, rt_std, hidden=HIDDEN, dropout=0.2, window=10,
                 meta_mode="gate"):
        super().__init__()
        if meta_mode not in META_MODES:
            raise ValueError(f"meta_mode must be one of {META_MODES}")
        self.meta_mode = meta_mode
        self.window = window
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
        gate_in = 2 * hidden + {"gate": META_DIM, "full": META_DIM + 2}.get(meta_mode, 0)
        self.gate = nn.Linear(gate_in, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        if meta_mode in ("full", "gru"):
            self.outcome_emb = nn.Embedding(N_OUTCOMES, hidden)
            self.meta_to_x = nn.Linear(META_DIM, hidden)

    def question(self, skill, diff, problem):
        return (self.skill_emb(skill) + self.prob_mu(problem) * self.skill_var(skill)
                + self.diff_emb(diff.clamp(0, 3)))

    def states(self, skill, correct, mask, rt, att, hint, diff, problem):
        """Fused state h and gate g after each history step (all inputs [B, T])."""
        e = self.question(skill, diff, problem)
        meta = self.metadata_encoder(rt, att, hint) if self.meta_mode != "none" else None
        if self.meta_mode in ("full", "gru"):
            x = e + self.outcome_emb(outcome_class(correct, att, hint)) + self.meta_to_x(meta)
        else:
            x = e + self.correct_emb(correct.clamp(0, 1))
        hs = self.short_term(x, mask)
        hl = self.long_term(x * mask.unsqueeze(-1), mask)
        parts = [hs, hl]
        if self.meta_mode == "gate":
            parts.append(meta)
        elif self.meta_mode == "full":
            parts += [meta, help_rates(att, hint, mask, self.window)]
        g = torch.sigmoid(self.gate(torch.cat(parts, dim=-1)))
        return self.drop(g * hs + (1 - g) * hl), g

    def score(self, h, skill, diff, problem):
        e = self.question(skill, diff, problem)
        return self.head(torch.cat([h, e], dim=-1)).squeeze(-1) - self.prob_b(problem).squeeze(-1)

    def forward(self, skill, correct, mask, rt, att, hint, diff, problem=None, return_gate=False):
        skill = skill.clamp(0, self.S)
        if problem is None:
            problem = torch.zeros_like(skill)
        h, g = self.states(skill[:, :-1], correct[:, :-1], mask[:, :-1], rt[:, :-1], att[:, :-1],
                           hint[:, :-1], diff[:, :-1], problem[:, :-1])
        pred = torch.sigmoid(self.score(h, skill[:, 1:], diff[:, 1:], problem[:, 1:]))
        return (pred, g) if return_gate else pred


class ServingGKTPlus(nn.Module):
    """Wraps a trained _GKTPlusNet in the call interface of src/model.py GatedKTModel
    (forward -> (pred, gate), predict_candidates) so the inference service can serve it
    behind a flag. App questions have no ASSISTments problem id, so the per-question
    difficulty offset is 0 (skill level + difficulty bucket only)."""

    def __init__(self, net: "_GKTPlusNet"):
        super().__init__()
        self.net = net.eval()

    def forward(self, skill_id, correct, mask, response_time, attempt_count, hint_used, difficulty=None,
                edge_index=None, edge_weight=None):
        if difficulty is None:
            difficulty = torch.zeros_like(skill_id)
        return self.net(skill_id.long(), correct.long(), mask.float(), response_time.float(),
                        attempt_count.float(), hint_used.float(), difficulty.long(), return_gate=True)

    def predict_candidates(self, skill_seq, correct_seq, mask, response_time, attempt_count, hint_used,
                           difficulty_seq, candidates, edge_index=None, edge_weight=None):
        if not candidates:
            return torch.zeros(0)
        n = self.net
        skill = skill_seq.long().clamp(0, n.S)
        h, _ = n.states(skill, correct_seq.long(), mask.float(), response_time.float(), attempt_count.float(),
                        hint_used.float(), difficulty_seq.long().clamp(0, 3), torch.zeros_like(skill))
        last = int(mask[0].sum().item()) - 1
        hk = h[0, max(last, 0)].unsqueeze(0).expand(len(candidates), -1)
        cs = torch.tensor([c[0] for c in candidates], dtype=torch.long).clamp(0, n.S)
        cd = torch.tensor([c[1] for c in candidates], dtype=torch.long).clamp(0, 3)
        return torch.sigmoid(n.score(hk, cs, cd, torch.zeros_like(cs)))


class GKTPlus(DeepKT):
    name = "gkt_plus"
    uses_problem = True
    epochs = 15
    patience = 3
    weight_decay = 1e-5
    dropout = 0.2

    meta_mode = "gate"

    def __init__(self, num_skills: int = 123, seed: int = 0, **kw):
        for k in ("dropout", "weight_decay", "patience", "lr", "meta_mode"):
            if k in kw:
                setattr(self, k, kw.pop(k))
        super().__init__(num_skills=num_skills, seed=seed, **kw)

    def build_net(self):
        n_prob = len(getattr(self, "_problem_index", [])) if getattr(self, "_problem_index", None) is not None else 0
        net = _GKTPlusNet(self.num_skills, n_prob, self.rt_stats[0], self.rt_stats[1], dropout=self.dropout,
                          meta_mode=self.meta_mode)
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


class GKTPlusMeta(GKTPlus):
    """GatedKT+ with hints and attempts in the GRU input and the gate (meta_mode "full")."""
    name = "gkt_plus_meta"
    meta_mode = "full"


class GKTPlusMetaGRU(GKTPlus):
    """Ablation: hints / attempts in the GRU input only, gate without metadata."""
    name = "gkt_plus_meta_gru"
    meta_mode = "gru"


class GKTPlusNoMeta(GKTPlus):
    """Ablation: no response time, attempts or hints anywhere."""
    name = "gkt_plus_nometa"
    meta_mode = "none"


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
