# src/evaluate_test.py

import pickle
import torch

from model import GatedKTModel
from graph_encoder import adjacency_to_edge_index
from train import run_epoch  # reuse the same masked-AUC logic, no reimplementation


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

    # Must reuse the SAME rt_mean/rt_std computed from TRAINING data during
    # actual training -- recomputing from test data here would be its own
    # small leak (normalizing test data using test data's own statistics).
    from train import compute_rt_stats
    rt_mean, rt_std = compute_rt_stats(train_data)

    print("\n=== FINAL TEST SET RESULTS (report these, not val AUC) ===")
    load_and_eval("dkt_only", False, False, num_skills, test_data, edge_index, edge_weight, rt_mean, rt_std)
    load_and_eval("graph_no_gate", True, False, num_skills, test_data, edge_index, edge_weight, rt_mean, rt_std)
    load_and_eval("full_gated", True, True, num_skills, test_data, edge_index, edge_weight, rt_mean, rt_std)


if __name__ == "__main__":
    main()