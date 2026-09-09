"""
src/graph_builder.py
----------------------
Stage 2: Build the Confound-Aware Graph (Contribution #1)

Reads the session-windowed data from dataset.py and builds a skill-skill
adjacency matrix using PPMI over curriculum co-occurrence — not raw text,
not accuracy correlation. This is deliberate: see the reasoning notes
inline below for why each rule exists.

Output: a saved adjacency matrix + a sanity-check report you should
actually read before moving to model training.
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations


# ----------------------------
# Config
# ----------------------------
INPUT_PATH = "data/processed/windowed_data.pkl"
OUTPUT_PATH = "data/processed/skill_graph.pkl"

MIN_COOCCURRENCE = 5   # Rule 1 — pairs below this count are dropped entirely


def load_windowed_data(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_counts(splits: dict, num_skills: int):
    """
    Walk every student's sequence (TRAINING SPLIT ONLY — never touch val/test
    here, that would be leakage into your graph structure) and tally:
      - skill_count[s]        = how many windows skill s appears in
      - pair_count[(s1, s2)]  = how many windows both s1 and s2 appear in
      - total_windows         = denominator for probability estimates

    Note: we count PRESENCE per window, not raw frequency. If a skill
    appears 3 times in one window, it counts once for that window. This
    matches "did the curriculum place these skills together," which is
    the actual signal we want — not "how many times did this skill repeat."
    """
    train_data = splits["train"]
    skill_seq = train_data["skill_seq"].numpy()
    window_id_seq = train_data["window_id_seq"].numpy()
    mask = train_data["mask"].numpy()

    skill_count = defaultdict(int)
    pair_count = defaultdict(int)
    total_windows = 0

    num_students = skill_seq.shape[0]
    for i in range(num_students):
        valid_positions = np.where(mask[i])[0]
        if len(valid_positions) == 0:
            continue

        student_skills = skill_seq[i, valid_positions]
        student_windows = window_id_seq[i, valid_positions]

        # group skills by window_id for this student
        window_to_skills = defaultdict(set)
        for skill, w_id in zip(student_skills, student_windows):
            if w_id >= 0:   # -1 means padding, already excluded by mask but double-safe
                window_to_skills[w_id].add(int(skill))

        for w_id, skills_in_window in window_to_skills.items():
            total_windows += 1
            for s in skills_in_window:
                skill_count[s] += 1
            for s1, s2 in combinations(sorted(skills_in_window), 2):
                pair_count[(s1, s2)] += 1

    return skill_count, pair_count, total_windows


def apply_min_count_filter(pair_count: dict, min_count: int = MIN_COOCCURRENCE) -> dict:
    """
    Rule 1: drop any pair that co-occurred fewer than min_count times.

    Why: PMI is unstable on rare events — a pair that co-occurred once or
    twice can produce an inflated score purely from small-sample noise,
    not genuine relatedness. This filter removes that risk before the
    PMI formula ever sees them.
    """
    filtered = {pair: count for pair, count in pair_count.items() if count >= min_count}
    print(f"Pairs before filter: {len(pair_count)}, after filter: {len(filtered)}")
    return filtered


def compute_ppmi(skill_count: dict, pair_count: dict, total_windows: int) -> dict:
    """
    Rule 2 + core formula: compute PMI for each surviving pair, clip
    negative scores to 0 (PPMI). Negative PMI just means "co-occurred
    less than chance would predict" — unreliable and not useful as a
    negative relationship signal, so we discard it rather than trust it.
    """
    ppmi_scores = {}

    for (s1, s2), joint_count in pair_count.items():
        p_s1 = skill_count[s1] / total_windows
        p_s2 = skill_count[s2] / total_windows
        p_joint = joint_count / total_windows

        pmi = np.log2(p_joint / (p_s1 * p_s2))
        ppmi = max(pmi, 0.0)   # Rule 2 — clip negatives to zero

        if ppmi > 0:
            ppmi_scores[(s1, s2)] = ppmi

    return ppmi_scores


def build_adjacency_matrix(ppmi_scores: dict, num_skills: int) -> np.ndarray:
    """
    Arrange PPMI scores into a symmetric [num_skills+1, num_skills+1] matrix
    (index 0 reserved for padding, matching dataset.py's skill re-indexing).
    """
    adj = np.zeros((num_skills + 1, num_skills + 1), dtype=np.float32)

    for (s1, s2), score in ppmi_scores.items():
        adj[s1, s2] = score
        adj[s2, s1] = score   # symmetric — relatedness is not directional here

    return adj


def sanity_check(ppmi_scores: dict, skill_map: dict, top_k: int = 15):
    """
    READ THIS OUTPUT before trusting the graph. Reverse the skill_map to
    show real skill names/IDs, sorted by strength. If the top pairs look
    like random noise instead of curriculum-sensible groupings, something
    upstream is wrong (windowing, or the raw skill_id mapping) — do not
    proceed to model training until this looks reasonable.
    """
    reverse_map = {v: k for k, v in skill_map.items()}
    sorted_pairs = sorted(ppmi_scores.items(), key=lambda x: -x[1])

    print(f"\nTop {top_k} strongest skill relationships (by PPMI):")
    for (s1, s2), score in sorted_pairs[:top_k]:
        raw1, raw2 = reverse_map.get(s1, s1), reverse_map.get(s2, s2)
        print(f"  skill {raw1} <-> skill {raw2}   PPMI = {score:.3f}")

    print(f"\nTotal surviving edges: {len(ppmi_scores)}")
    print("Check the pairs above against known curriculum structure "
          "(e.g. do related topics like Fractions/Ratios show up here?) "
          "before moving to graph_encoder.py.")


def main():
    print("Loading windowed data...")
    data = load_windowed_data(INPUT_PATH)
    splits = data["splits"]
    skill_map = data["skill_map"]
    num_skills = data["num_skills"]

    print("Collecting skill and pair counts from TRAINING split only...")
    skill_count, pair_count, total_windows = collect_counts(splits, num_skills)
    print(f"Total windows counted: {total_windows}")
    print(f"Unique skills observed: {len(skill_count)}")
    print(f"Unique raw pairs (before filtering): {len(pair_count)}")

    filtered_pairs = apply_min_count_filter(pair_count)
    ppmi_scores = compute_ppmi(skill_count, filtered_pairs, total_windows)

    sanity_check(ppmi_scores, skill_map)

    adj_matrix = build_adjacency_matrix(ppmi_scores, num_skills)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "adjacency_matrix": adj_matrix,
            "ppmi_scores": ppmi_scores,
            "skill_map": skill_map,
        }, f)

    print(f"\nSaved adjacency matrix to {OUTPUT_PATH}")
    print(f"Matrix shape: {adj_matrix.shape}")
    print(f"Sparsity: {(adj_matrix > 0).sum()} nonzero entries "
          f"out of {adj_matrix.size} total")


if __name__ == "__main__":
    main()