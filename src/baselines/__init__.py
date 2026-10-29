"""
src/baselines
-------------
v3 baselines with one shared interface (see base.py):
    fit(train) -> self;  predict(history, next_skill, next_difficulty) -> P in [0, 1]

CLASSICAL holds the cheap models (Stage 2). Deep models are registered in
baselines.deep (Stage 3) so importing this package does not import torch.
"""

from baselines.base import BaseKT  # noqa: F401
from baselines.majority import MajorityBaseline, SkillMeanBaseline
from baselines.logistic import LogisticCountsBaseline
from baselines.pfa import PFABaseline
from baselines.elo import EloBaseline
from baselines.bkt import BKTBaseline, BKTDifficultyBaseline

CLASSICAL = {
    "majority": MajorityBaseline,
    "skill_mean": SkillMeanBaseline,
    "logreg_counts": LogisticCountsBaseline,
    "pfa": PFABaseline,
    "elo": EloBaseline,
    "bkt": BKTBaseline,
    "bkt_diff": BKTDifficultyBaseline,
}


def make_classical(name: str, num_skills: int = 123, seed: int = 0) -> BaseKT:
    return CLASSICAL[name](num_skills=num_skills, seed=seed)
