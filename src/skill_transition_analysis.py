# src/skill_transition_analysis.py
# Aggregated statistical test across ~100 students: is gate activity
# higher at skill-transition points than elsewhere? This is the ONLY
# file that computes the t-test and Cohen's d. visualize_gate.py does
# NOT contain this code and should not be edited to add it.

import pickle
import numpy as np
import torch
from scipy import stats

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
    model.load_state_dict(torch.load("data/processed/best_full_gated_seed42.pt"))
    model.eval()

    mask = test_data["mask"]
    real_lengths = mask.sum(dim=1)

    candidates = torch.where(real_lengths >= 50)[0]
    print(f"Students with >= 50 real interactions: {len(candidates)}")
    sample_idx = candidates[:100] if len(candidates) > 100 else candidates

    all_activity_at_change = []
    all_activity_no_change = []
    total_changes = 0
    total_steps = 0

    with torch.no_grad():
        for idx in sample_idx:
            idx = idx.item()
            skill_id = test_data["skill_seq"][idx:idx+1]
            correct = test_data["correct_seq"][idx:idx+1]
            m = test_data["mask"][idx:idx+1]
            response_time = test_data["response_time_seq"][idx:idx+1]
            attempt_count = test_data["attempt_seq"][idx:idx+1]
            hint_used = test_data["hint_seq"][idx:idx+1]

            pred, g = model(skill_id, correct, m, response_time, attempt_count,
                             hint_used, edge_index, edge_weight)

            real_len = int(m[0, 1:].sum().item())
            if real_len < 2:
                continue

            g_full = g[0, :real_len].numpy()
            activity = g_full.std(axis=1)

            skill_seq_real = skill_id[0, 1:real_len+1].numpy()
            skill_changed = np.concatenate([[False], skill_seq_real[1:] != skill_seq_real[:-1]])

            total_changes += skill_changed.sum()
            total_steps += real_len

            all_activity_at_change.extend(activity[skill_changed].tolist())
            all_activity_no_change.extend(activity[~skill_changed].tolist())

    all_activity_at_change = np.array(all_activity_at_change)
    all_activity_no_change = np.array(all_activity_no_change)

    print(f"\nAggregated across {len(sample_idx)} students, {total_steps} total steps")
    print(f"Total skill changes observed: {total_changes}")
    print(f"Gate activity AT skill-change steps:  n={len(all_activity_at_change)}, "
          f"mean={all_activity_at_change.mean():.4f}, std={all_activity_at_change.std():.4f}")
    print(f"Gate activity at NON-change steps:     n={len(all_activity_no_change)}, "
          f"mean={all_activity_no_change.mean():.4f}, std={all_activity_no_change.std():.4f}")

    t_stat, p_val = stats.ttest_ind(all_activity_at_change, all_activity_no_change)
    print(f"\nt-test: t={t_stat:.3f}, p={p_val:.6f}")

    pooled_std = np.sqrt((all_activity_at_change.std()**2 + all_activity_no_change.std()**2) / 2)
    cohens_d = (all_activity_at_change.mean() - all_activity_no_change.mean()) / pooled_std
    print(f"Cohen's d (effect size): {cohens_d:.4f}")

    if p_val < 0.05:
        print("Statistically significant difference -- real finding, safe to report.")
    else:
        print("NOT statistically significant -- do not claim a skill-transition "
              "relationship. Report the per-dimension heatmap finding alone.")


if __name__ == "__main__":
    main()