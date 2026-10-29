"""
src/baselines/dkt.py
--------------------
Deep Knowledge Tracing (Piech et al., 2015), GRU or LSTM, hidden size 64 (same as
GatedKT).

- DKTClassic ("dkt_classic"): as published -- input = embedding of (skill, correct),
  output layer with one unit per skill, the next skill's unit is read out. No
  difficulty bucket.
- DKT ("dkt_gru" / "dkt_lstm"): the same recurrent core, but input and target also
  carry the difficulty bucket (input = skill + correct + bucket embeddings; head =
  linear on [h_t, emb(next skill) + emb(next bucket)]), so it sees the same
  features as GatedKT. These are what the benchmark compares against.
"""

import torch
import torch.nn as nn

from baselines.deep_common import DeepKT

HIDDEN = 64


class _DKTNet(nn.Module):
    def __init__(self, num_skills, cell="gru", use_difficulty=True, hidden=HIDDEN):
        super().__init__()
        self.S = num_skills
        self.use_difficulty = use_difficulty
        self.skill_emb = nn.Embedding(num_skills + 1, hidden, padding_idx=0)
        self.correct_emb = nn.Embedding(2, hidden)
        self.diff_emb = nn.Embedding(4, hidden)
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(hidden, hidden, batch_first=True)
        if use_difficulty:
            self.head = nn.Linear(2 * hidden, 1)
        else:
            self.head = nn.Linear(hidden, num_skills + 1)

    def forward(self, skill, correct, mask, rt, att, hint, diff):
        skill = skill.clamp(0, self.S)
        x = self.skill_emb(skill[:, :-1]) + self.correct_emb(correct[:, :-1].clamp(0, 1))
        if self.use_difficulty:
            x = x + self.diff_emb(diff[:, :-1].clamp(0, 3))
        x = x * mask[:, :-1].unsqueeze(-1)
        h, _ = self.rnn(x)
        nxt = skill[:, 1:]
        if self.use_difficulty:
            tgt = self.skill_emb(nxt) + self.diff_emb(diff[:, 1:].clamp(0, 3))
            logit = self.head(torch.cat([h, tgt], dim=-1)).squeeze(-1)
        else:
            logit = self.head(h).gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(logit)


class DKT(DeepKT):
    name = "dkt_gru"
    cell = "gru"
    use_difficulty = True

    def build_net(self):
        return _DKTNet(self.num_skills, self.cell, self.use_difficulty)


class DKTLSTM(DKT):
    name = "dkt_lstm"
    cell = "lstm"


class DKTClassic(DKT):
    name = "dkt_classic"
    use_difficulty = False
