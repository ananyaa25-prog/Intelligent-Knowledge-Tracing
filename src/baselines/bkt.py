"""
src/baselines/bkt.py
--------------------
Bayesian Knowledge Tracing (Corbett & Anderson, 1995) with its own EM fit.

Per skill k: prior L0_k, learn T_k, guess G, slip S; no forgetting. Two variants:

- BKTBaseline ("bkt"): classic, one guess/slip per skill (difficulty bucket ignored).
- BKTDifficultyBaseline ("bkt_diff"): guess/slip per (skill, difficulty bucket), in the
  spirit of KT-IDEM (Pardos & Heffernan, 2011), so it uses the same buckets as the
  main model.

EM = Baum-Welch on each student's per-skill subsequence, vectorised over all
(student, skill) subsequences (rows sorted by length so step t only touches the
rows still active). Parameters start from a seed-dependent random point;
guess and slip are capped at 0.3 (the usual identifiability constraint).
"""

from typing import Any, Dict, List

import numpy as np

from baselines.base import BaseKT, normalise_history, split_to_sequences

MAX_GS = 0.3
EPS = 1e-6


def _build_rows(seqs, S):
    """All (student, skill) subsequences -> rows sorted by length (desc)."""
    rows: List[tuple] = []
    for s in seqs:
        by_k: Dict[int, list] = {}
        for k, c, d in zip(s["skill_seq"], s["correct_seq"], s["difficulty_seq"]):
            k = int(k) if 0 <= int(k) < S else 0
            by_k.setdefault(k, []).append((int(c), min(max(int(d), 0), 3)))
        for k, obs in by_k.items():
            rows.append((k, obs))
    rows.sort(key=lambda r: -len(r[1]))
    R = len(rows)
    Tmax = len(rows[0][1]) if rows else 0
    lens = np.asarray([len(r[1]) for r in rows], dtype=np.int64)
    # step-major ragged storage: at step t the first n_t rows are active
    n_t = np.asarray([int((lens > t).sum()) for t in range(Tmax)], dtype=np.int64)
    obs_t = [np.asarray([rows[r][1][t][0] for r in range(n_t[t])], dtype=np.float64) for t in range(Tmax)]
    dif_t = [np.asarray([rows[r][1][t][1] for r in range(n_t[t])], dtype=np.int64) for t in range(Tmax)]
    skill = np.asarray([r[0] for r in rows], dtype=np.int64)
    return skill, n_t, obs_t, dif_t


class BKTBaseline(BaseKT):
    name = "bkt"
    deterministic = False      # EM starts from a seed-dependent random point
    per_difficulty = False
    n_iter = 25
    tol = 1e-4

    def _init_params(self, S):
        rng = np.random.default_rng(self.seed)
        self.L0 = rng.uniform(0.2, 0.8, S)
        self.T = rng.uniform(0.05, 0.3, S)
        g = rng.uniform(0.05, 0.25, S)
        s = rng.uniform(0.05, 0.25, S)
        self.G = np.repeat(g[:, None], 4, axis=1)
        self.Sl = np.repeat(s[:, None], 4, axis=1)

    def fit(self, train: Dict[str, Any]) -> "BKTBaseline":
        seqs = split_to_sequences(train) if isinstance(train, dict) else train
        S = self.num_skills + 1
        self._init_params(S)
        skill, n_t, obs_t, dif_t = _build_rows(seqs, S)
        self.loglik_trace = []
        if len(n_t) == 0:
            return self
        prev = -np.inf
        for _ in range(self.n_iter):
            ll = self._em_step(skill, n_t, obs_t, dif_t, S)
            self.loglik_trace.append(ll)
            if abs(ll - prev) < self.tol * max(1.0, abs(ll)) * 1e-2:
                break
            prev = ll
        return self

    def _emit(self, k, d, c):
        G = self.G[k, d]
        Sl = self.Sl[k, d]
        e1 = np.where(c > 0.5, 1.0 - Sl, Sl)      # P(obs | known)
        e0 = np.where(c > 0.5, G, 1.0 - G)        # P(obs | not known)
        return e0, e1

    def _em_step(self, skill, n_t, obs_t, dif_t, S):
        Tmax = len(n_t)
        # forward (scaled)
        a0, a1, scale = [], [], []
        for t in range(Tmax):
            n = n_t[t]
            k = skill[:n]
            e0, e1 = self._emit(k, dif_t[t], obs_t[t])
            if t == 0:
                p0 = (1 - self.L0[k]) * e0
                p1 = self.L0[k] * e1
            else:
                Tk = self.T[k]
                p0 = a0[t - 1][:n] * (1 - Tk) * e0
                p1 = (a0[t - 1][:n] * Tk + a1[t - 1][:n]) * e1
            c = p0 + p1 + 1e-300
            a0.append(p0 / c)
            a1.append(p1 / c)
            scale.append(c)
        loglik = float(sum(np.log(c).sum() for c in scale))
        # backward + accumulate sufficient statistics
        R = n_t[0]
        b0 = np.ones(R)
        b1 = np.ones(R)
        sum_L0 = np.zeros(S); cnt_rows = np.zeros(S)
        trans_num = np.zeros(S); trans_den = np.zeros(S)
        g_num = np.zeros((S, 4)); g_den = np.zeros((S, 4))
        s_num = np.zeros((S, 4)); s_den = np.zeros((S, 4))
        for t in range(Tmax - 1, -1, -1):
            n = n_t[t]
            k = skill[:n]
            b0n, b1n = b0[:n], b1[:n]
            g1 = a1[t] * b1n
            g0 = a0[t] * b0n
            z = g0 + g1 + 1e-300
            g0, g1 = g0 / z, g1 / z
            c = obs_t[t]
            d = dif_t[t]
            np.add.at(g_num, (k, d), g0 * c)
            np.add.at(g_den, (k, d), g0)
            np.add.at(s_num, (k, d), g1 * (1 - c))
            np.add.at(s_den, (k, d), g1)
            if t == 0:
                np.add.at(sum_L0, k, g1)
                np.add.at(cnt_rows, k, 1.0)
            if t > 0:
                # xi(unknown at t-1 -> known at t) for rows active at t
                m = n
                kk = skill[:m]
                e0, e1 = self._emit(kk, d, c)
                Tk = self.T[kk]
                xi01 = a0[t - 1][:m] * Tk * e1 * b1n / scale[t]
                np.add.at(trans_num, kk, xi01)
                # gamma(unknown at t-1) for those rows: computed from alpha/beta at t-1
                nb0 = ((1 - Tk) * e0 * b0n + Tk * e1 * b1n) / scale[t]
                nb1 = (e1 * b1n) / scale[t]
                gp0 = a0[t - 1][:m] * nb0
                gp1 = a1[t - 1][:m] * nb1
                np.add.at(trans_den, kk, gp0 / (gp0 + gp1 + 1e-300))
                # rows that END at t-1 (inactive at t) have beta = 1
                b0 = np.ones(n_t[t - 1])
                b1 = np.ones(n_t[t - 1])
                b0[:m] = nb0
                b1[:m] = nb1
        # M-step
        seen = cnt_rows > 0
        self.L0[seen] = np.clip(sum_L0[seen] / cnt_rows[seen], EPS, 1 - EPS)
        tm = trans_den > 0
        self.T[tm] = np.clip(trans_num[tm] / trans_den[tm], EPS, 1 - EPS)
        if self.per_difficulty:
            gm, sm = g_den > 0, s_den > 0
            self.G[gm] = np.clip(g_num[gm] / g_den[gm], EPS, MAX_GS)
            self.Sl[sm] = np.clip(s_num[sm] / s_den[sm], EPS, MAX_GS)
        else:
            gd, sd = g_den.sum(1), s_den.sum(1)
            g = np.where(gd > 0, g_num.sum(1) / np.maximum(gd, 1e-300), self.G[:, 0])
            s = np.where(sd > 0, s_num.sum(1) / np.maximum(sd, 1e-300), self.Sl[:, 0])
            self.G = np.repeat(np.clip(g, EPS, MAX_GS)[:, None], 4, axis=1)
            self.Sl = np.repeat(np.clip(s, EPS, MAX_GS)[:, None], 4, axis=1)
        return loglik

    # ---- online prediction --------------------------------------------------

    def _run(self, skills, corrects, diffs, n_pred_from: int):
        S = len(self.L0)
        pk: Dict[int, float] = {}
        out = []
        for t in range(len(skills)):
            k = int(skills[t]) if 0 <= int(skills[t]) < S else 0
            d = min(max(int(diffs[t]), 0), 3)
            L = pk.get(k, self.L0[k])
            G, Sl = self.G[k, d], self.Sl[k, d]
            p = L * (1 - Sl) + (1 - L) * G
            if t >= n_pred_from:
                out.append(p)
            if t < len(corrects):
                # exact posterior denominators (1 - p computed as a sum, not a
                # difference) and a clamp keep L inside [0, 1] numerically
                if int(corrects[t]):
                    post = L * (1 - Sl) / max(L * (1 - Sl) + (1 - L) * G, 1e-300)
                else:
                    post = L * Sl / max(L * Sl + (1 - L) * (1 - G), 1e-300)
                post = min(max(post, 0.0), 1.0)
                pk[k] = post + (1 - post) * self.T[k]
        return np.asarray(out, dtype=np.float64)

    def predict(self, history, next_skill: int, next_difficulty: int = 0) -> float:
        sk, co, di = normalise_history(history)
        sk = np.append(sk, int(next_skill))
        di = np.append(di, int(next_difficulty))
        return float(self._run(sk, co, di, len(sk) - 1)[0])

    def predict_sequence(self, skills, corrects, diffs):
        return self._run(skills, corrects, diffs, 1)


class BKTDifficultyBaseline(BKTBaseline):
    name = "bkt_diff"
    per_difficulty = True
