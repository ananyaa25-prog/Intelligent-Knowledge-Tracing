# src/evaluate_test.py

import pickle
import numpy as np
import torch

from model import GatedKTModel
from graph_encoder import adjacency_to_edge_index
from train import run_epoch, compute_rt_stats


def load_and_eval(name, use_graph, use_gate, num_skills, test_data,
                   edge_index, edge_weight, rt_mean, rt_std):
    model = GatedKTModel(num_skills=num_skills, rt_log_mean=rt_mean, rt_log_std=rt_std,
                          use_graph=use_graph, use_gate=use_gate)
    model.load_state_dict(torch.load(f"data/processed/best_{name}.pt"))
    model.eval()
    test_loss, test_auc = run_epoch(model, test_data, edge_index, edge_weight, optimizer=None)
    print(f"[{name}] TEST AUC = {test_auc:.4f}  (test_loss = {test_loss:.4f})")
    return test_auc


def main():
    with open("data/processed/windowed_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("data/processed/skill_graph.pkl", "rb") as f:
        graph_data = pickle.load(f)

    num_skills = data["num_skills"]
    train_data = data["splits"]["train"]
    test_data = data["splits"]["test"]

    adj_matrix = torch.tensor(graph_data["adjacency_matrix"], dtype=torch.float32)
    edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
    rt_mean, rt_std = compute_rt_stats(train_data)

    SEEDS = [42, 123, 2024]
    conditions = {
        "dkt_only": (False, False),
        "graph_no_gate": (True, False),
        "full_gated": (True, True),
    }

    print("\n=== FINAL TEST SET RESULTS (report these, not val AUC) ===")
    for cond_name, (use_graph, use_gate) in conditions.items():
        aucs = []
        for seed in SEEDS:
            checkpoint_name = f"{cond_name}_seed{seed}"
            auc = load_and_eval(checkpoint_name, use_graph, use_gate, num_skills,
                                 test_data, edge_index, edge_weight, rt_mean, rt_std)
            aucs.append(auc)
        print(f"  {cond_name}: {aucs} -> mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}\n")


if __name__ == "__main__":
    main()