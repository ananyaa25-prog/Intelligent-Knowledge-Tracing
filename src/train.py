# src/train.py

import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from model import GatedKTModel
from graph_encoder import adjacency_to_edge_index

DEVICE = "cpu"
BATCH_SIZE = 32
EPOCHS = 5          # deliberately small first run -- see note below
LR = 1e-3
GRAD_CLIP = 5.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_rt_stats(train_data):
    """Real normalization constants from training data, replacing the
    Stage 6 placeholder. Only computed over REAL (non-padded) positions,
    or the padding zeros would corrupt the mean/std."""
    rt = train_data["response_time_seq"]
    mask = train_data["mask"]
    real_rt = rt[mask]
    log_rt = torch.log1p(real_rt.clamp(min=0))
    return log_rt.mean().item(), log_rt.std().item()


def masked_bce_loss(pred, target, mask):
    """BCE computed ONLY over real (non-padded) positions. Padded
    positions still produce a prediction (the model doesn't know they're
    padding), but they must not contribute to the loss or gradient --
    otherwise you're training on garbage timesteps."""
    loss_fn = nn.BCELoss(reduction="none")
    per_step_loss = loss_fn(pred, target.float())
    per_step_loss = per_step_loss * mask.float()
    return per_step_loss.sum() / mask.float().sum().clamp(min=1)


def masked_auc(pred, target, mask):
    """AUC over only real positions, flattened across the batch."""
    mask_np = mask.bool().numpy().flatten()
    pred_np = pred.detach().numpy().flatten()[mask_np]
    target_np = target.numpy().flatten()[mask_np]
    if len(np.unique(target_np)) < 2:
        return float("nan")  # AUC undefined if batch has only one class
    return roc_auc_score(target_np, pred_np)


def get_batches(data_dict, batch_size):
    n = data_dict["skill_seq"].shape[0]
    indices = torch.randperm(n)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield {k: v[idx] for k, v in data_dict.items() if k != "student_ids"}


def run_epoch(model, data_dict, edge_index, edge_weight, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_batches = 0.0, 0
    all_preds, all_targets, all_masks = [], [], []

    for batch in get_batches(data_dict, BATCH_SIZE):
        skill_id = batch["skill_seq"].long()
        correct = batch["correct_seq"].long()
        mask = batch["mask"].float()
        response_time = batch["response_time_seq"].float()
        attempt_count = batch["attempt_seq"].float()
        hint_used = batch["hint_seq"].float()

        with torch.set_grad_enabled(is_train):
            pred, _ = model(skill_id, correct, mask, response_time,
                             attempt_count, hint_used, edge_index, edge_weight)
            # pred is length T-1 (predicting correct[1:]) -- target and
            # mask must be shifted to match, or shapes silently misalign
            target = correct[:, 1:].float()
            target_mask = mask[:, 1:].float()
            loss = masked_bce_loss(pred, target, target_mask)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        all_preds.append(pred.detach())
        all_targets.append(correct[:, 1:].float())
        all_masks.append(mask[:, 1:])

    avg_loss = total_loss / total_batches
    auc = masked_auc(torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_masks))
    return avg_loss, auc


# train.py — replace the __main__ block

def train_condition(name, use_graph, use_gate, num_skills, train_data, val_data,
                     edge_index, edge_weight, rt_mean, rt_std):
    print(f"\n{'='*50}\nTraining condition: {name}\n{'='*50}")
    model = GatedKTModel(num_skills=num_skills, rt_log_mean=rt_mean, rt_log_std=rt_std,
                          use_graph=use_graph, use_gate=use_gate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_auc = run_epoch(model, train_data, edge_index, edge_weight, optimizer)
        val_loss, val_auc = run_epoch(model, val_data, edge_index, edge_weight, optimizer=None)
        elapsed = time.time() - t0
        print(f"[{name}] Epoch {epoch}: train_auc={train_auc:.4f} val_auc={val_auc:.4f} | {elapsed:.1f}s")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"data/processed/best_{name}.pt")

    print(f"[{name}] BEST val AUC: {best_val_auc:.4f}")
    return best_val_auc


def main():
    print("Loading data...")
    with open("data/processed/windowed_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("data/processed/skill_graph.pkl", "rb") as f:
        graph_data = pickle.load(f)

    num_skills = data["num_skills"]
    train_data = data["splits"]["train"]
    val_data = data["splits"]["val"]

    adj_matrix = torch.tensor(graph_data["adjacency_matrix"], dtype=torch.float32)
    edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
    rt_mean, rt_std = compute_rt_stats(train_data)

    SEEDS = [42, 123, 2024]
    all_results = {"dkt_only": [], "graph_no_gate": [], "full_gated": []}

    for seed in SEEDS:
        print(f"\n{'#'*60}\n### SEED {seed}\n{'#'*60}")
        set_seed(seed)

        auc = train_condition(f"dkt_only_seed{seed}", use_graph=False, use_gate=False,
                               num_skills=num_skills, train_data=train_data, val_data=val_data,
                               edge_index=edge_index, edge_weight=edge_weight,
                               rt_mean=rt_mean, rt_std=rt_std)
        all_results["dkt_only"].append(auc)

        set_seed(seed)  # reset so each condition starts from the same init point
        auc = train_condition(f"graph_no_gate_seed{seed}", use_graph=True, use_gate=False,
                               num_skills=num_skills, train_data=train_data, val_data=val_data,
                               edge_index=edge_index, edge_weight=edge_weight,
                               rt_mean=rt_mean, rt_std=rt_std)
        all_results["graph_no_gate"].append(auc)

        set_seed(seed)
        auc = train_condition(f"full_gated_seed{seed}", use_graph=True, use_gate=True,
                               num_skills=num_skills, train_data=train_data, val_data=val_data,
                               edge_index=edge_index, edge_weight=edge_weight,
                               rt_mean=rt_mean, rt_std=rt_std)
        all_results["full_gated"].append(auc)

    print(f"\n{'='*60}\nFINAL RESULTS ACROSS {len(SEEDS)} SEEDS (val AUC)\n{'='*60}")
    for name, aucs in all_results.items():
        mean = np.mean(aucs)
        std = np.std(aucs)
        print(f"  {name}: {aucs} -> mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    main()