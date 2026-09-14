# src/visualize_gate.py

import pickle
import torch
import matplotlib.pyplot as plt

from model import GatedKTModel
from graph_encoder import adjacency_to_edge_index
from train import compute_rt_stats


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

    model = GatedKTModel(num_skills=num_skills, rt_log_mean=rt_mean, rt_log_std=rt_std,
                          use_graph=True, use_gate=True)
    model.load_state_dict(torch.load("data/processed/best_full_gated.pt"))
    model.eval()

    # Pick one real student with a decent number of real (non-padded)
    # interactions -- a student with only 5 real steps makes a boring plot.
    mask = test_data["mask"]
    real_lengths = mask.sum(dim=1)
    student_idx = torch.argmax(real_lengths).item()  # the student with the longest real history
    n_real = real_lengths[student_idx].item()
    print(f"Selected student index {student_idx} with {n_real} real interactions")

    skill_id = test_data["skill_seq"][student_idx:student_idx+1]
    correct = test_data["correct_seq"][student_idx:student_idx+1]
    m = test_data["mask"][student_idx:student_idx+1]
    response_time = test_data["response_time_seq"][student_idx:student_idx+1]
    attempt_count = test_data["attempt_seq"][student_idx:student_idx+1]
    hint_used = test_data["hint_seq"][student_idx:student_idx+1]

    with torch.no_grad():
        pred, g = model(skill_id, correct, m, response_time, attempt_count,
                         hint_used, edge_index, edge_weight)

    # g has shape [1, seq_len-1, hidden_dim] -- average across the hidden
    # dimension to get one interpretable "overall trust in short-term"
    # scalar per timestep. g close to 1 = trusting recent behavior more,
    # close to 0 = trusting long-term pattern more.
    g_mean = g[0].mean(dim=-1).numpy()
    real_len = int(m[0, 1:].sum().item())  # matches the T-1 shift in forward()
    g_mean_real = g_mean[:real_len]

    g_full = g[0, :real_len].numpy()  # [real_len, hidden_dim] -- no averaging yet

    per_dim_std = g_full.std(axis=0)   # variation ACROSS TIME, per dimension
    per_timestep_std = g_full.std(axis=1)  # variation ACROSS DIMENSIONS, per timestep

    print(f"\nPer-dimension temporal variation (std across time, per dim):")
    print(f"  mean={per_dim_std.mean():.4f}, max={per_dim_std.max():.4f}")
    print(f"Per-timestep cross-dimension spread (std across dims, per step):")
    print(f"  mean={per_timestep_std.mean():.4f}, max={per_timestep_std.max():.4f}")

    print(f"Gate values over this student's real history (first {real_len} steps):")
    print(f"  Mean gate value: {g_mean_real.mean():.4f}")
    print(f"  Min: {g_mean_real.min():.4f}, Max: {g_mean_real.max():.4f}")

    plt.figure(figsize=(12, 4))
    plt.plot(g_mean_real, marker="o", markersize=3)
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Equal trust")
    plt.xlabel("Interaction step")
    plt.ylabel("Gate value (1 = trust short-term, 0 = trust long-term)")
    plt.title(f"Gate behavior over student {student_idx}'s history ({real_len} interactions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/gate_visualization.png", dpi=150)
    print("\nSaved figure to data/processed/gate_visualization.png")


if __name__ == "__main__":
    main()