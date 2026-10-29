"""
src/baselines/deep_common.py
----------------------------
Shared torch plumbing for the deep baselines (Stage 3) and the wrappers of our own
GatedKT variants, with the same interface as the classical baselines:

    fit(train, val=None) -> self;  predict(history, next_skill, next_difficulty) -> P

Every network implements
    net(skill, correct, mask, rt, attempt, hint, difficulty) -> pred [B, L-1]
predicting positions 1..L-1 from the history before them -- the exact contract of
GatedKTModel.forward and src/train.py run_epoch, so all deep models are trained and
scored on identical targets.

Training: Adam (lr 1e-3), batch 32, masked BCE, gradient clip 5, best-validation-AUC
weights kept in memory (if no val split is given, the last epoch is used). Seeds
fix the initialisation and the batch order. Nothing is written to data/processed/;
optional saving goes to models_v3/.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from baselines.base import BaseKT, auc, normalise_history, _np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_V3 = PROJECT_ROOT / "models_v3"

TENSOR_KEYS = ("skill_seq", "correct_seq", "mask", "response_time_seq", "attempt_seq",
               "hint_seq", "difficulty_seq")
DEFAULT_RT_MS = 5000.0


def set_torch_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def to_tensors(split: Dict[str, Any], max_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
    out = {}
    for k in TENSOR_KEYS:
        v = split[k]
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
        if max_len is not None:
            t = t[:, :max_len]
        out[k] = t
    # drop trailing all-padding columns (cheaper batches)
    real = out["mask"].bool().any(dim=0).nonzero()
    L = int(real.max().item()) + 1 if len(real) else 1
    return {k: v[:, :L] for k, v in out.items()}


def _batch_args(b: Dict[str, torch.Tensor]):
    L = int(b["mask"].sum(dim=1).max().item()) if b["mask"].numel() else 1
    L = max(L, 2)
    return (b["skill_seq"][:, :L].long(), b["correct_seq"][:, :L].long(), b["mask"][:, :L].float(),
            b["response_time_seq"][:, :L].float(), b["attempt_seq"][:, :L].float(),
            b["hint_seq"][:, :L].float(), b["difficulty_seq"][:, :L].long())


def length_bucketed_batches(lens: torch.Tensor, batch_size: int, gen: torch.Generator):
    """Seeded batches of students with similar sequence length (less padding, same
    data and same number of updates as random batches); batch order is shuffled."""
    n = len(lens)
    noise = torch.rand(n, generator=gen)
    order = torch.argsort(lens.float() + noise)          # random order within equal lengths
    batches = [order[s:s + batch_size] for s in range(0, n, batch_size)]
    return [batches[i] for i in torch.randperm(len(batches), generator=gen).tolist()]


def masked_bce(pred, target, mask):
    pred = pred.clamp(1e-6, 1 - 1e-6)
    loss = nn.functional.binary_cross_entropy(pred, target.float(), reduction="none") * mask
    return loss.sum() / mask.sum().clamp(min=1)


class DeepKT(BaseKT):
    """Base class: subclasses implement build_net() -> nn.Module."""

    name = "deep"
    deterministic = False
    epochs = 5
    batch_size = 32
    lr = 1e-3
    grad_clip = 5.0
    max_train_len: Optional[int] = None   # optional truncation of TRAIN sequences (speed)

    def __init__(self, num_skills: int = 123, seed: int = 0, epochs: Optional[int] = None,
                 rt_stats=(8.5, 1.2), verbose: bool = False, **kwargs):
        super().__init__(num_skills=num_skills, seed=seed)
        if epochs is not None:
            self.epochs = int(epochs)
        self.rt_stats = rt_stats
        self.verbose = verbose
        self.kwargs = kwargs
        self.net: Optional[nn.Module] = None
        self.history_log = []

    # -- to override
    def build_net(self) -> nn.Module:
        raise NotImplementedError

    def transform(self, args):
        """Hook to neutralise inputs (used by ablations). Default: identity."""
        return args

    # -- training
    def fit(self, train: Dict[str, Any], val: Optional[Dict[str, Any]] = None) -> "DeepKT":
        set_torch_seed(self.seed)
        self.net = self.build_net()
        tr = to_tensors(train, self.max_train_len)
        va = to_tensors(val) if val is not None else None
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        lens = tr["mask"].sum(dim=1)
        best_auc, best_state = -1.0, None
        for ep in range(1, self.epochs + 1):
            self.net.train()
            for idx in length_bucketed_batches(lens, self.batch_size, gen):
                args = self.transform(_batch_args({k: v[idx] for k, v in tr.items()}))
                pred = self.net(*args)
                loss = masked_bce(pred, args[1][:, 1:], args[2][:, 1:])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                opt.step()
            if va is not None:
                p, y = self._predict_tensors(va)
                v_auc = auc(y, p)
                self.history_log.append({"epoch": ep, "val_auc": v_auc})
                if self.verbose:
                    print(f"[{self.name} seed {self.seed}] epoch {ep} val_auc={v_auc:.4f}")
                if v_auc > best_auc:
                    best_auc = v_auc
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_auc = best_auc if va is not None else None
        self.net.eval()
        return self

    # -- prediction
    @torch.no_grad()
    def _predict_tensors(self, data: Dict[str, torch.Tensor], batch_size: int = 64):
        self.net.eval()
        preds, ys = [], []
        n = data["mask"].shape[0]
        for s in range(0, n, batch_size):
            args = self.transform(_batch_args({k: v[s:s + batch_size] for k, v in data.items()}))
            p = self.net(*args)
            m = args[2][:, 1:].bool()
            preds.append(p[m].double().numpy())
            ys.append(args[1][:, 1:][m].numpy())
        return np.clip(np.concatenate(preds), 0, 1), np.concatenate(ys).astype(np.int64)

    @torch.no_grad()
    def predict_split(self, split: Dict[str, Any]):
        data = to_tensors(split)
        p, y = self._predict_tensors(data)
        lens = (data["mask"].sum(dim=1).long() - 1).clamp(min=0).numpy()
        owner = np.repeat(np.arange(len(lens)), lens)
        return p, y, owner

    @torch.no_grad()
    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        sk, co, di = normalise_history(history)
        meta = [h if isinstance(h, dict) else {} for h in (history or [])]
        L = len(sk) + 1
        skill = torch.zeros(1, max(L, 2), dtype=torch.long)
        corr = torch.zeros_like(skill)
        diff = torch.zeros_like(skill)
        mask = torch.zeros(1, max(L, 2))
        rt = torch.full((1, max(L, 2)), DEFAULT_RT_MS)
        att = torch.ones(1, max(L, 2))
        hint = torch.zeros(1, max(L, 2))
        if len(sk) == 0:
            # no history: a single zero-padding step stands in for "nothing seen yet"
            skill[0, 1] = int(next_skill)
            diff[0, 1] = int(next_difficulty)
            mask[0, :2] = torch.tensor([0.0, 1.0])
            args = self.transform((skill, corr, mask, rt, att, hint, diff))
            return float(self.net(*args)[0, 0].clamp(0, 1).item())
        skill[0, :len(sk)] = torch.as_tensor(sk)
        corr[0, :len(sk)] = torch.as_tensor(co)
        diff[0, :len(sk)] = torch.as_tensor(di)
        for i, h in enumerate(meta):
            rt[0, i] = float(h.get("response_time", h.get("response_time_ms", DEFAULT_RT_MS)))
            att[0, i] = float(h.get("attempt_count", 1))
            hint[0, i] = float(h.get("hint_used", 0))
        skill[0, len(sk)] = min(max(int(next_skill), 0), self.num_skills)
        diff[0, len(sk)] = min(max(int(next_difficulty), 0), 3)
        mask[0, :L] = 1.0
        args = self.transform((skill[:, :L], corr[:, :L], mask[:, :L], rt[:, :L], att[:, :L], hint[:, :L], diff[:, :L]))
        return float(self.net(*args)[0, -1].clamp(0, 1).item())

    def predict_sequence(self, skills, corrects, diffs):
        L = len(skills)
        t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt).view(1, -1)
        args = self.transform((t(skills, torch.long), t(corrects, torch.long), torch.ones(1, L),
                               torch.full((1, L), DEFAULT_RT_MS), torch.ones(1, L), torch.zeros(1, L),
                               t(diffs, torch.long)))
        with torch.no_grad():
            return self.net(*args)[0].double().numpy() if L > 1 else np.zeros(0)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or (MODELS_V3 / f"{self.name}_seed{self.seed}.pt"))
        if path.resolve().parent == (PROJECT_ROOT / "data" / "processed").resolve():
            raise ValueError("v3 never writes checkpoints into data/processed/")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        return path


def clamp_ids(skill, num_skills):
    return skill.clamp(0, num_skills)
