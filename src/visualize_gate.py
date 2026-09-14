# src/visualize_gate.py
# Produces the heatmap + top-5-dimension figures for ONE student.
# This file does NOT compute the skill-transition statistic --
# that lives entirely in skill_transition_analysis.py, a separate file.

import pickle
import numpy as np
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

    mask = test_data["mask"]
    real_lengths = mask.sum(dim=1)
    student_idx = torch.argmax(real_lengths).item()
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

    real_len = int(m[0, 1:].sum().item())
    g_full = g[0, :real_len].numpy()

    per_dim_std = g_full.std(axis=0)
    per_timestep_std = g_full.std(axis=1)
    g_mean_real = g_full.mean(axis=-1)

    print(f"\nPer-dimension temporal variation (std across time, per dim):")
    print(f"  mean={per_dim_std.mean():.4f}, max={per_dim_std.max():.4f}")
    print(f"Per-timestep cross-dimension spread (std across dims, per step):")
    print(f"  mean={per_timestep_std.mean():.4f}, max={per_timestep_std.max():.4f}")
    print(f"\nGate values over this student's real history (first {real_len} steps):")
    print(f"  Mean gate value: {g_mean_real.mean():.4f}")
    print(f"  Min: {g_mean_real.min():.4f}, Max: {g_mean_real.max():.4f}")

    # --- Heatmap: dimensions x time ---
    dim_order = np.argsort(-per_dim_std)
    g_sorted = g_full[:, dim_order]

    plt.figure(figsize=(12, 6))
    plt.imshow(g_sorted.T, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
               interpolation="nearest")
    plt.colorbar(label="Gate value (1=short-term trust, 0=long-term trust)")
    plt.xlabel("Interaction step")
    plt.ylabel("Hidden dimension (sorted by activity, most active on top)")
    plt.title(f"Per-dimension gate behavior — student {student_idx}, {real_len} interactions")
    plt.tight_layout()
    plt.savefig("data/processed/gate_heatmap.png", dpi=150)
    print("\nSaved heatmap to data/processed/gate_heatmap.png")

    # --- Top-5 most active dimensions, as lines ---
    top5 = dim_order[:5]
    plt.figure(figsize=(12, 4))
    for d in top5:
        plt.plot(g_full[:, d], alpha=0.7, label=f"dim {d}")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    plt.xlabel("Interaction step")
    plt.ylabel("Gate value")
    plt.title("Top 5 most active gate dimensions over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/processed/gate_top5_dims.png", dpi=150)
    print("Saved top-5 dimension plot to data/processed/gate_top5_dims.png")


if __name__ == "__main__":
    main()