"""
src/baselines/ours.py
---------------------
Thin wrappers that expose our own GatedKT variants (src/model.py, unchanged) through
the shared baseline interface, plus the input ablations used in Stage 5.

Variants (use_graph, use_gate, input neutralisation):
  full_gated      True,  True    -- the served model
  graph_no_gate   True,  False   -- graph signal, naive average instead of the gate
  dkt_only        False, False   -- "DKT+difficulty": our encoders without graph or gate
  no_graph        False, True    -- gate kept, graph context replaced by zeros
  no_metadata     True,  True    -- response time / attempts / hints replaced by constants
  no_difficulty   True,  True    -- difficulty buckets replaced by 0 (unknown)
(no_gate == graph_no_gate.)

`OursVariant.from_checkpoint(cond, seed)` loads the EXISTING v2 checkpoints
data/processed/best_<cond>_seed<seed>.pt read-only (no retraining, nothing
overwritten). `fit()` trains a fresh copy for the multi-seed benchmark; weights are
kept in memory or saved to models_v3/ only.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from baselines.deep_common import DeepKT, PROJECT_ROOT

PROCESSED = PROJECT_ROOT / "data" / "processed"
SKILL_GRAPH_PATH = PROCESSED / "skill_graph.pkl"

VARIANTS: Dict[str, Dict] = {
    "full_gated": {"use_graph": True, "use_gate": True},
    "graph_no_gate": {"use_graph": True, "use_gate": False},
    "dkt_only": {"use_graph": False, "use_gate": False},
    "no_graph": {"use_graph": False, "use_gate": True},
    "no_metadata": {"use_graph": True, "use_gate": True, "neutral_meta": True},
    "no_difficulty": {"use_graph": True, "use_gate": True, "zero_difficulty": True},
}
EXISTING_CHECKPOINT_CONDS = ("full_gated", "graph_no_gate", "dkt_only")
EXISTING_SEEDS = (42, 123, 2024)

_GRAPH_CACHE: Dict[str, torch.Tensor] = {}


def load_graph():
    if not _GRAPH_CACHE:
        from graph_encoder import adjacency_to_edge_index
        with open(SKILL_GRAPH_PATH, "rb") as f:
            g = pickle.load(f)
        ei, ew = adjacency_to_edge_index(torch.tensor(g["adjacency_matrix"], dtype=torch.float32))
        _GRAPH_CACHE["edge_index"], _GRAPH_CACHE["edge_weight"] = ei, ew
    return _GRAPH_CACHE["edge_index"], _GRAPH_CACHE["edge_weight"]


class _GatedNet(nn.Module):
    def __init__(self, num_skills, rt_mean, rt_std, use_graph, use_gate):
        super().__init__()
        from model import GatedKTModel
        self.model = GatedKTModel(num_skills=num_skills, rt_log_mean=rt_mean, rt_log_std=rt_std,
                                  use_graph=use_graph, use_gate=use_gate)
        self.use_graph = use_graph
        if use_graph:
            ei, ew = load_graph()
            self.register_buffer("edge_index", ei, persistent=False)
            self.register_buffer("edge_weight", ew, persistent=False)

    def forward(self, skill, correct, mask, rt, att, hint, diff):
        ei = self.edge_index if self.use_graph else None
        ew = self.edge_weight if self.use_graph else None
        pred, _ = self.model(skill, correct, mask, rt, att, hint, diff, edge_index=ei, edge_weight=ew)
        return pred


class OursVariant(DeepKT):
    name = "full_gated"

    def __init__(self, num_skills: int = 123, seed: int = 0, cond: str = "full_gated", **kw):
        super().__init__(num_skills=num_skills, seed=seed, **kw)
        if cond not in VARIANTS:
            raise ValueError(f"unknown variant {cond}")
        self.cond = cond
        self.name = cond
        self.spec = VARIANTS[cond]

    def build_net(self):
        return _GatedNet(self.num_skills, self.rt_stats[0], self.rt_stats[1],
                         self.spec["use_graph"], self.spec["use_gate"])

    def transform(self, args):
        skill, correct, mask, rt, att, hint, diff = args
        if self.spec.get("neutral_meta"):
            # constant metadata: log1p(rt) equals the train mean, 1 attempt, no hint
            rt = torch.full_like(rt, math.expm1(self.rt_stats[0]))
            att = torch.ones_like(att)
            hint = torch.zeros_like(hint)
        if self.spec.get("zero_difficulty"):
            diff = torch.zeros_like(diff)
        return skill, correct, mask, rt, att, hint, diff

    @classmethod
    def from_checkpoint(cls, cond: str, seed: int, num_skills: int = 123, rt_stats=(8.5, 1.2),
                        path: Optional[Path] = None) -> "OursVariant":
        """Load an existing v2 checkpoint (read-only)."""
        m = cls(num_skills=num_skills, seed=seed, cond=cond, rt_stats=rt_stats)
        m.net = m.build_net()
        p = Path(path or (PROCESSED / f"best_{cond}_seed{seed}.pt"))
        state = torch.load(p, map_location="cpu")
        m.net.model.load_state_dict(state)
        m.net.eval()
        m.checkpoint_path = str(p)
        return m


def rt_stats_from_split(train) -> tuple:
    """Same as src/train.py compute_rt_stats (log1p of real response times)."""
    rt = torch.as_tensor(train["response_time_seq"]) if not isinstance(train["response_time_seq"], torch.Tensor) else train["response_time_seq"]
    mask = torch.as_tensor(train["mask"]).bool()
    lr = torch.log1p(rt[mask].float().clamp(min=0))
    return float(lr.mean().item()), float(lr.std().item())
