"""
scripts/fit_calibration.py
--------------------------
T8b: Temperature scaling (single scalar T on the logit) for the served model.

Data (same priority as T1): validation split of windowed_data.pkl. Validation
students are split 50/50 (seeded): T is fitted on half A by minimising the
negative log-likelihood; Brier, ECE (10 equal-width bins) and AUC are reported
on held-out half B before and after. AUC must be unchanged (asserted).
Outputs: reports/calibration.md, reports/calibration.png, data/processed/calibration.json

Usage: python scripts/fit_calibration.py [--seed 11]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPORTS_DIR, load_service, load_validation_split, seed_everything

import numpy as np
import torch


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if m.sum() == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        conf, acc = p[m].mean(), y[m].mean()
        total += m.sum() / len(p) * abs(conf - acc)
        rows.append((lo, hi, int(m.sum()), float(conf), float(acc)))
    return float(total), rows


def apply_T(p, T):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))


def nll(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    seed_everything(args.seed)
    t0 = time.time()
    svc = load_service()
    val = load_validation_split()
    if val is None:
        REPORTS_DIR.mkdir(exist_ok=True)
        (REPORTS_DIR / "calibration.md").write_text("# Calibration\n\nCould not run: validation split unavailable. "
                                                    "APPLY_CALIBRATION stays false.\n", encoding="utf-8")
        print("calibration could not run")
        return
    model = svc.artifacts.model
    P, Y, S = [], [], []
    n = val["mask"].shape[0]
    with torch.no_grad():
        for i in range(0, n, 32):
            sl = slice(i, i + 32)
            pred, _ = model(val["skill_seq"][sl], val["correct_seq"][sl].long(), val["mask"][sl].float(),
                            val["response_time_seq"][sl], val["attempt_seq"][sl], val["hint_seq"][sl],
                            val["difficulty_seq"][sl], edge_index=svc.artifacts.edge_index,
                            edge_weight=svc.artifacts.edge_weight)
            m = val["mask"][sl][:, 1:].bool()
            P.append(pred[m].numpy())
            Y.append(val["correct_seq"][sl][:, 1:][m].numpy())
            S.append(np.repeat(np.arange(i, min(i + 32, n))[:, None], m.shape[1], axis=1)[m.numpy()])
    p = np.concatenate(P).astype(float)
    y = np.concatenate(Y).astype(float)
    s = np.concatenate(S)
    rng = np.random.default_rng(args.seed)
    stud = rng.permutation(n)
    half_a = set(stud[: n // 2].tolist())
    in_a = np.array([x in half_a for x in s])
    pa, ya, pb, yb = p[in_a], y[in_a], p[~in_a], y[~in_a]

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda T: nll(apply_T(pa, T), ya), bounds=(0.2, 5.0), method="bounded")
    T = float(res.x)
    pb_cal = apply_T(pb, T)

    from sklearn.metrics import roc_auc_score
    auc_before, auc_after = roc_auc_score(yb, pb), roc_auc_score(yb, pb_cal)
    assert abs(auc_before - auc_after) < 1e-9, "temperature scaling changed AUC"
    brier_b, brier_a = float(((pb - yb) ** 2).mean()), float(((pb_cal - yb) ** 2).mean())
    ece_b, rows_b = ece(pb, yb)
    ece_a, rows_a = ece(pb_cal, yb)
    rel_impr = (ece_b - ece_a) / ece_b if ece_b > 0 else 0.0
    gate_c = bool(rel_impr >= 0.10 and abs(auc_before - auc_after) < 1e-9)
    elapsed = time.time() - t0

    out = {"temperature": round(T, 5), "fit_on": "validation half A", "evaluated_on": "validation half B",
           "n_fit": int(len(pa)), "n_eval": int(len(pb)), "brier_before": brier_b, "brier_after": brier_a,
           "ece_before": ece_b, "ece_after": ece_a, "ece_relative_improvement": rel_impr,
           "auc_before": auc_before, "auc_after": auc_after, "gate_c_apply_calibration": gate_c,
           "seed": args.seed, "generated_at": datetime.now(timezone.utc).isoformat()}
    from serving.policy_config import CALIBRATION_PATH
    CALIBRATION_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="perfect calibration")
    for rows, lbl, col in [(rows_b, f"before (ECE {ece_b:.3f})", "#d97706"), (rows_a, f"after T={T:.2f} (ECE {ece_a:.3f})", "#2563eb")]:
        xs = [r[3] for r in rows if r[3] is not None]
        ys = [r[4] for r in rows if r[3] is not None]
        ax.plot(xs, ys, "o-", color=col, label=lbl)
    ax.set_xlabel("mean predicted P(correct) in bin")
    ax.set_ylabel("observed share correct")
    ax.set_title("Reliability diagram (held-out validation half)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    plt.savefig(REPORTS_DIR / "calibration.png", dpi=130)
    plt.close(fig)

    L = ["# Calibration (T8b): temperature scaling", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/fit_calibration.py` (seed {args.seed}, {elapsed:.0f}s).",
         "",
         f"Data: windowed_data.pkl validation split, next-step predictions of the served model (`best_full_gated.pt`). "
         f"Students split 50/50: T fitted on half A ({len(pa)} predictions), metrics on held-out half B ({len(pb)} predictions).",
         "", f"Fitted temperature T = **{T:.4f}** (T > 1 softens, T < 1 sharpens predictions).", "",
         "| Metric (held-out half B) | before | after |", "|---|---|---|",
         f"| Brier score | {brier_b:.4f} | {brier_a:.4f} |",
         f"| ECE (10 equal-width bins) | {ece_b:.4f} | {ece_a:.4f} |",
         f"| AUC | {auc_before:.4f} | {auc_after:.4f} |", "",
         f"ECE relative improvement: {rel_impr:.1%}. AUC unchanged (asserted).", "",
         f"**Gate C (pre-registered: ECE improves >= 10% relative and AUC unchanged): {'PASSES' if gate_c else 'FAILS'}** "
         f"-> demo APPLY_CALIBRATION = {str(gate_c).lower()}.", "",
         "Caveat: calibration is measured on ASSISTments validation data. The quiz uses generated template questions, "
         "so calibration on real quiz answers can only be checked with pilot data (scripts/analyze_logs.py).", "",
         "## Reliability table (half B)", "", "| bin | n | mean predicted before | observed | mean predicted after |", "|---|---|---|---|---|"]
    for rb, ra in zip(rows_b, rows_a):
        L.append(f"| {rb[0]:.1f}-{rb[1]:.1f} | {rb[2]} | {'' if rb[3] is None else f'{rb[3]:.3f}'} | "
                 f"{'' if rb[4] is None else f'{rb[4]:.3f}'} | {'' if ra[3] is None else f'{ra[3]:.3f} (n={ra[2]})'} |")
    L += ["", "![calibration](calibration.png)", ""]
    (REPORTS_DIR / "calibration.md").write_text("\n".join(L), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
