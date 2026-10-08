"""
src/dataset.py
----------------
Stage 1: Data Input Handling

Reads raw student interaction logs, groups by student, sorts chronologically,
and slices into uniform Session Windows (using real assignment boundaries
where available, falling back to a fixed window size otherwise).

Output: a saved, cleaned, windowed dataset that graph_builder.py and
the model training script both read from — no other file should touch
the raw CSV directly.
"""

import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path


# ----------------------------
# Config — adjust these here, nowhere else
# ----------------------------
RAW_DATA_PATH = "data/raw/skill_builder_data.csv"
OUTPUT_PATH = "data/processed/windowed_data.pkl"

WINDOW_SIZE = 15          # fallback window size if no assignment_id column exists
MIN_INTERACTIONS = 10     # drop students with fewer than this many total interactions

RAW_REQUIRED_COLUMNS = [
    "user_id", "skill_id", "correct",
    "ms_first_response", "attempt_count", "hint_count",
    "order_id", "assignment_id", "problem_id"
]

MIN_PROBLEM_SAMPLES = 5   # problems with fewer TRAIN responses get difficulty 0 (unknown)


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV and keep only the columns we actually need."""
    # This ASSISTments file needs Latin-1 encoding — default UTF-8 fails partway through
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}.")

    df = df[RAW_REQUIRED_COLUMNS].copy()

    # Rename to the internal names the rest of the pipeline expects
    df = df.rename(columns={
        "user_id": "student_id",
        "ms_first_response": "response_time",
    })

    df = df.dropna(subset=["skill_id", "correct"])
         # Some skill_id values are multi-skill composites like "48_63_79" —
    # explode each into its own row so co-occurrence isn't polluted by
    # treating a composite string as a fake standalone skill.
    df["skill_id"] = df["skill_id"].astype(str).str.split("_")
    df = df.explode("skill_id")
    df["skill_id"] = df["skill_id"].astype(float).astype(int)
    df["correct"] = df["correct"].astype(int)
    df["attempt_count"] = df["attempt_count"].fillna(1).astype(int)
    df["response_time"] = df["response_time"].fillna(df["response_time"].median())
    df["problem_id"] = df["problem_id"].fillna(0).astype(int)

    # hint_used doesn't exist as a raw column — derive it from hint_count
    df["hint_count"] = df["hint_count"].fillna(0)
    df["hint_used"] = (df["hint_count"] > 0).astype(int)
    df = df.drop(columns=["hint_count"])

    return df


def filter_short_students(df: pd.DataFrame) -> pd.DataFrame:
    """Drop students with too few interactions to model meaningfully."""
    counts = df.groupby("student_id").size()
    keep_ids = counts[counts >= MIN_INTERACTIONS].index
    return df[df["student_id"].isin(keep_ids)].copy()


def reindex_skills(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Map raw skill_ids to a contiguous 1...num_skills range (0 reserved for padding).
    Save the mapping — you will need it again in graph_builder.py and for
    interpreting results (mapping back to real skill names).
    """
    unique_skills = sorted(df["skill_id"].unique())
    skill_map = {raw_id: new_id + 1 for new_id, raw_id in enumerate(unique_skills)}
    df["skill_id"] = df["skill_id"].map(skill_map)
    return df, skill_map

ASSIGNMENT_GROUP_SIZE = 3  # how many consecutive assignments form one window

def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["student_id", "order_id"]).reset_index(drop=True)

    if "assignment_id" in df.columns:
        # Number each distinct assignment a student moves through, in order
        assignment_change = (
            df.groupby("student_id")["assignment_id"]
            .transform(lambda x: (x != x.shift()).cumsum())
        )
        # Group every ASSIGNMENT_GROUP_SIZE consecutive assignments into one window —
        # this is needed because skill-builder assignments are single-skill by design,
        # so single-assignment windows almost never contain a second skill to co-occur with.
        df["window_id"] = df.groupby("student_id").apply(
            lambda g: ((assignment_change.loc[g.index] - 1) // ASSIGNMENT_GROUP_SIZE)
        ).reset_index(level=0, drop=True)
        print(f"Using assignment_id boundaries, grouped {ASSIGNMENT_GROUP_SIZE} at a time.")
    else:
        df["window_id"] = df.groupby("student_id").cumcount() // WINDOW_SIZE
        print(f"No assignment_id found — falling back to fixed windows of size {WINDOW_SIZE}.")

    return df


def derive_difficulty(train_df: pd.DataFrame, min_samples: int = MIN_PROBLEM_SAMPLES) -> dict:
    """
    Derive per-problem difficulty from the TRAINING split only.

    For each (skill, problem) pair we compute the mean correctness across train
    students, then tertile problems WITHIN each skill by that mean correctness:
      1 = easy (highest train success), 2 = medium, 3 = hard (lowest).
    Problems with too few train samples (unreliable) map to 0 = unknown.

    Returns {(skill_id, problem_id): difficulty_bucket}.
    """
    g = train_df.groupby(["skill_id", "problem_id"])["correct"].agg(["mean", "count"]).reset_index()
    g.columns = ["skill_id", "problem_id", "mean_correct", "count"]
    g["difficulty"] = 0

    for skill, grp in g.groupby("skill_id"):
        # Keep the original index (drop=False) so we can write buckets back to g.
        reliable = grp[grp["count"] >= min_samples].sort_values("mean_correct")
        n = len(reliable)
        if n < 3:
            continue  # too few reliable problems to tertile; leave as 0 (unknown)
        # rank-based thirds: lowest mean correctness -> hard (3), highest -> easy (1)
        buckets = np.empty(n, dtype=np.int64)
        for i in range(n):
            if i < n // 3:
                buckets[i] = 3
            elif i < (2 * n) // 3:
                buckets[i] = 2
            else:
                buckets[i] = 1
        for j, orig_idx in enumerate(reliable.index):
            g.loc[orig_idx, "difficulty"] = buckets[j]

    return {(int(r.skill_id), int(r.problem_id)): int(r.difficulty) for r in g.itertuples()}


def build_sequences(df: pd.DataFrame, max_seq_len: int) -> dict:
    """
    Convert the long table into padded tensors, one row per student,
    ready for the model. Also keeps window_id per interaction so
    graph_builder.py can compute co-occurrence within windows, and the
    per-problem difficulty bucket derived from the training split.
    """
    student_ids = df["student_id"].unique()
    num_students = len(student_ids)

    skill_seq = np.zeros((num_students, max_seq_len), dtype=np.int64)
    correct_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
    response_time_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
    attempt_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
    hint_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
    difficulty_seq = np.zeros((num_students, max_seq_len), dtype=np.int64)
    problem_seq = np.zeros((num_students, max_seq_len), dtype=np.int64)
    window_id_seq = np.full((num_students, max_seq_len), -1, dtype=np.int64)
    mask = np.zeros((num_students, max_seq_len), dtype=bool)

    for i, sid in enumerate(student_ids):
        student_df = df[df["student_id"] == sid].sort_values("order_id")
        length = min(len(student_df), max_seq_len)

        skill_seq[i, :length] = student_df["skill_id"].values[:length]
        correct_seq[i, :length] = student_df["correct"].values[:length]
        response_time_seq[i, :length] = student_df["response_time"].values[:length]
        attempt_seq[i, :length] = student_df["attempt_count"].values[:length]
        hint_seq[i, :length] = student_df["hint_used"].values[:length]
        if "difficulty" in student_df.columns:
            difficulty_seq[i, :length] = student_df["difficulty"].values[:length]
        if "problem_id" in student_df.columns:
            problem_seq[i, :length] = student_df["problem_id"].values[:length]
        window_id_seq[i, :length] = student_df["window_id"].values[:length]
        mask[i, :length] = True

    return {
        "student_ids": student_ids,
        "skill_seq": torch.tensor(skill_seq),
        "correct_seq": torch.tensor(correct_seq),
        "response_time_seq": torch.tensor(response_time_seq),
        "attempt_seq": torch.tensor(attempt_seq),
        "hint_seq": torch.tensor(hint_seq),
        "difficulty_seq": torch.tensor(difficulty_seq),
        "problem_seq": torch.tensor(problem_seq),
        "window_id_seq": torch.tensor(window_id_seq),
        "mask": torch.tensor(mask),
    }


def split_student_ids(df: pd.DataFrame, train_frac=0.8, val_frac=0.1, seed=42) -> dict:
    """Student-level split by student_id — never split mid-sequence, that leaks
    future info. Returns {split_name: set(student_id)} so difficulty can be
    derived from TRAIN students before any tensors are built."""
    student_ids = df["student_id"].unique()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(student_ids))

    train_end = int(train_frac * len(student_ids))
    val_end = train_end + int(val_frac * len(student_ids))

    return {
        "train": set(student_ids[idx[:train_end]]),
        "val": set(student_ids[idx[train_end:val_end]]),
        "test": set(student_ids[idx[val_end:]]),
    }


def split_students(data: dict, split_sets: dict) -> dict:
    """Slice a built data dict into train/val/test using precomputed id sets."""
    result = {}
    for split_name, ids in split_sets.items():
        indices = [i for i, sid in enumerate(data["student_ids"]) if sid in ids]
        result[split_name] = {
            key: (val[indices] if isinstance(val, torch.Tensor) else val[np.array(indices)])
            for key, val in data.items()
        }
    return result


def difficulty_shares(splits: dict) -> dict:
    """Report the fraction of real (non-padded) interactions in each difficulty
    bucket, per split. Used to sanity-check the train-derived bucketing."""
    shares = {}
    for name, d in splits.items():
        mask = d["mask"].numpy().astype(bool)
        diff = d["difficulty_seq"].numpy()[mask]
        total = diff.shape[0]
        shares[name] = {
            "n_interactions": int(total),
            "unknown(0)": float((diff == 0).sum() / total) if total else 0.0,
            "easy(1)": float((diff == 1).sum() / total) if total else 0.0,
            "medium(2)": float((diff == 2).sum() / total) if total else 0.0,
            "hard(3)": float((diff == 3).sum() / total) if total else 0.0,
        }
    return shares


def main():
    print("Loading raw data...")
    df = load_raw(RAW_DATA_PATH)

    print(f"Rows before filtering: {len(df)}")
    df = filter_short_students(df)
    print(f"Rows after dropping short histories: {len(df)}")

    df, skill_map = reindex_skills(df)
    df = assign_windows(df)

    # Student-level split FIRST so difficulty is derived from TRAIN students only.
    split_sets = split_student_ids(df)
    train_df = df[df["student_id"].isin(split_sets["train"])]

    print("Deriving per-problem difficulty from TRAINING split only...")
    diff_map = derive_difficulty(train_df)
    n_reliable = sum(1 for v in diff_map.values() if v > 0)
    print(f"Difficulty buckets derived: {len(diff_map)} (skill, problem) pairs, "
          f"{n_reliable} with a reliable bucket (1/2/3)")
    df["difficulty"] = [diff_map.get((int(s), int(p)), 0)
                        for s, p in zip(df["skill_id"], df["problem_id"])]

    # max_seq_len covers the longest student history (cap it to avoid outlier blowup)
    max_len_actual = df.groupby("student_id").size().max()
    max_seq_len = min(max_len_actual, 500)   # cap — tune if needed
    print(f"Using max_seq_len = {max_seq_len}")

    data = build_sequences(df, max_seq_len)
    splits = split_students(data, split_sets)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "splits": splits,
            "skill_map": skill_map,
            "num_skills": len(skill_map),
            "max_seq_len": max_seq_len,
            "difficulty_map": diff_map,
        }, f)

    print(f"Saved windowed dataset to {OUTPUT_PATH}")
    print(f"Train students: {len(splits['train']['student_ids'])}, "
          f"Val: {len(splits['val']['student_ids'])}, "
          f"Test: {len(splits['test']['student_ids'])}")

    print("\nDifficulty bucket shares per split:")
    shares = difficulty_shares(splits)
    for name, s in shares.items():
        print(f"  {name}: n={s['n_interactions']}  unknown={s['unknown(0)']:.3f} "
              f"easy={s['easy(1)']:.3f} medium={s['medium(2)']:.3f} hard={s['hard(3)']:.3f}")


if __name__ == "__main__":
    main()