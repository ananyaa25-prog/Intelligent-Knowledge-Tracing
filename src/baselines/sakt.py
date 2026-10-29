"""
src/baselines/sakt.py
---------------------
SAKT-lite: a simplified Self-Attentive Knowledge Tracing model (Pandey & Karypis, 2019).

Simplifications (hence "-lite"): one attention block (4 heads, d=64), learned
positional embeddings up to 512 steps, causal mask; the query at step t+1 is
emb(next skill) + emb(next bucket); keys/values are emb(skill, correct, bucket) of
steps <= t plus position; residual + layer norm + feed-forward; sigmoid output.
"""

import torch
import torch.nn as nn

from baselines.deep_common import DeepKT

HIDDEN = 64
HEADS = 4
MAX_POS = 512


class _SAKTNet(nn.Module):
    def __init__(self, num_skills, hidden=HIDDEN, heads=HEADS, dropout=0.1):
        super().__init__()
        self.S = num_skills
        self.skill = nn.Embedding(num_skills + 1, hidden, padding_idx=0)
        self.corr = nn.Embedding(2, hidden)
        self.diff = nn.Embedding(4, hidden)
        self.pos = nn.Embedding(MAX_POS, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.ln2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, skill, correct, mask, rt, att, hint, diff):
        B, L = skill.shape
        skill = skill.clamp(0, self.S)
        diff = diff.clamp(0, 3)
        n = L - 1
        pos = torch.arange(n, device=skill.device).clamp(max=MAX_POS - 1)
        kv = (self.skill(skill[:, :-1]) + self.corr(correct[:, :-1].clamp(0, 1))
              + self.diff(diff[:, :-1]) + self.pos(pos).unsqueeze(0))
        q = self.skill(skill[:, 1:]) + self.diff(diff[:, 1:]) + self.pos(pos).unsqueeze(0)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=skill.device), diagonal=1)
        pad = mask[:, :-1] < 0.5
        # a query whose every key is padding would give NaN; those positions are
        # masked out of the loss anyway, so let them see key 0.
        pad = pad.clone()
        pad[:, 0] = False
        h, _ = self.attn(q, kv, kv, attn_mask=causal, key_padding_mask=pad, need_weights=False)
        h = self.ln1(h + q)
        h = self.ln2(h + self.ffn(h))
        return torch.sigmoid(self.out(h)).squeeze(-1)


class SAKTLite(DeepKT):
    name = "sakt_lite"

    def build_net(self):
        return _SAKTNet(self.num_skills)
