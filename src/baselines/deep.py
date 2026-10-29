"""
src/baselines/deep.py
---------------------
Registry of the deep baselines (Stage 3) and our GatedKT variants. Importing this
module imports torch (the classical registry in baselines/__init__.py does not).
"""

from baselines.dkt import DKT, DKTLSTM, DKTClassic
from baselines.dkvmn import DKVMNLite
from baselines.sakt import SAKTLite
from baselines.ours import OursVariant, VARIANTS

EXTERNAL_DEEP = {
    "dkt_classic": DKTClassic,
    "dkt_gru": DKT,
    "dkt_lstm": DKTLSTM,
    "dkvmn_lite": DKVMNLite,
    "sakt_lite": SAKTLite,
}
OUR_VARIANTS = list(VARIANTS)   # full_gated, graph_no_gate, dkt_only, no_graph, no_metadata, no_difficulty


def make_deep(name: str, num_skills: int = 123, seed: int = 0, **kw):
    if name in EXTERNAL_DEEP:
        return EXTERNAL_DEEP[name](num_skills=num_skills, seed=seed, **kw)
    if name in VARIANTS:
        return OursVariant(num_skills=num_skills, seed=seed, cond=name, **kw)
    raise KeyError(name)
