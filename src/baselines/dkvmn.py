"""
src/baselines/dkvmn.py
----------------------
DKVMN-lite: a simplified Dynamic Key-Value Memory Network (Zhang et al., 2017).

Simplifications (hence "-lite"): one static key memory of 20 slots (d=64) and a
per-student value memory of 20 x 64; attention w = softmax(q K^T) with the query
q = emb(skill) + emb(bucket); read r = w V; f = tanh(W [r, q]); p = sigmoid(w_out f).
Write after each answer with erase/add vectors from emb(skill, correct, bucket).
No hyper-parameter search; capacity is in the same range as GatedKT.
"""

import torch
import torch.nn as nn

from baselines.deep_common import DeepKT

HIDDEN = 64
SLOTS = 20


class _DKVMNNet(nn.Module):
    def __init__(self, num_skills, hidden=HIDDEN, slots=SLOTS):
        super().__init__()
        self.S = num_skills
        self.q_skill = nn.Embedding(num_skills + 1, hidden, padding_idx=0)
        self.q_diff = nn.Embedding(4, hidden)
        self.v_corr = nn.Embedding(2, hidden)
        self.key = nn.Parameter(torch.randn(slots, hidden) * 0.1)
        self.value0 = nn.Parameter(torch.randn(slots, hidden) * 0.1)
        self.erase = nn.Linear(hidden, hidden)
        self.add = nn.Linear(hidden, hidden)
        self.f = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, skill, correct, mask, rt, att, hint, diff):
        B, L = skill.shape
        skill = skill.clamp(0, self.S)
        q = self.q_skill(skill) + self.q_diff(diff.clamp(0, 3))           # [B, L, H]
        w = torch.softmax(q @ self.key.t(), dim=-1)                       # [B, L, N]
        v = q + self.v_corr(correct.clamp(0, 1))                           # interaction emb
        e = torch.sigmoid(self.erase(v))
        a = torch.tanh(self.add(v))
        M = self.value0.unsqueeze(0).expand(B, -1, -1)
        reads = []
        m = mask.unsqueeze(-1).unsqueeze(-1)
        for t in range(L):
            if t >= 1:
                reads.append((w[:, t].unsqueeze(1) @ M).squeeze(1))       # read before answer t
            if t < L - 1:
                wt = w[:, t].unsqueeze(-1)                                 # [B, N, 1]
                M_new = M * (1 - wt * e[:, t].unsqueeze(1)) + wt * a[:, t].unsqueeze(1)
                M = torch.where(m[:, t] > 0, M_new, M)
        r = torch.stack(reads, dim=1)                                      # [B, L-1, H]
        f = torch.tanh(self.f(torch.cat([r, q[:, 1:]], dim=-1)))
        return torch.sigmoid(self.out(f)).squeeze(-1)


class DKVMNLite(DeepKT):
    name = "dkvmn_lite"

    def build_net(self):
        return _DKVMNNet(self.num_skills)
