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
    "order_id", "assignment_id"
]


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


def build_sequences(df: pd.DataFrame, max_seq_len: int) -> dict:
    """
    Convert the long table into padded tensors, one row per student,
    ready for the model. Also keeps window_id per interaction so
    graph_builder.py can compute co-occurrence within windows.
    """
    student_ids = df["student_id"].unique()
    num_students = len(student_ids)

    skill_seq = np.zeros((num_students, max_seq_len), dtype=np.int64)
    correct_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
    response_time_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
    attempt_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
    hint_seq = np.zeros((num_students, max_seq_len), dtype=np.float32)
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
        window_id_seq[i, :length] = student_df["window_id"].values[:length]
        mask[i, :length] = True

    return {
        "student_ids": student_ids,
        "skill_seq": torch.tensor(skill_seq),
        "correct_seq": torch.tensor(correct_seq),
        "response_time_seq": torch.tensor(response_time_seq),
        "attempt_seq": torch.tensor(attempt_seq),
        "hint_seq": torch.tensor(hint_seq),
        "window_id_seq": torch.tensor(window_id_seq),
        "mask": torch.tensor(mask),
    }


def split_students(data: dict, train_frac=0.8, val_frac=0.1, seed=42) -> dict:
    """Split by student_id — never split mid-sequence, that leaks future info."""
    num_students = len(data["student_ids"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_students)

    train_end = int(train_frac * num_students)
    val_end = train_end + int(val_frac * num_students)

    splits = {"train": idx[:train_end], "val": idx[train_end:val_end], "test": idx[val_end:]}

    result = {}
    for split_name, indices in splits.items():
        result[split_name] = {
            key: (val[indices] if key != "student_ids" else val[indices])
            for key, val in data.items()
        }
    return result


def main():
    print("Loading raw data...")
    df = load_raw(RAW_DATA_PATH)

    print(f"Rows before filtering: {len(df)}")
    df = filter_short_students(df)
    print(f"Rows after dropping short histories: {len(df)}")

    df, skill_map = reindex_skills(df)
    df = assign_windows(df)

    # max_seq_len covers the longest student history (cap it to avoid outlier blowup)
    max_len_actual = df.groupby("student_id").size().max()
    max_seq_len = min(max_len_actual, 500)   # cap — tune if needed
    print(f"Using max_seq_len = {max_seq_len}")

    data = build_sequences(df, max_seq_len)
    splits = split_students(data)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "splits": splits,
            "skill_map": skill_map,
            "num_skills": len(skill_map),
            "max_seq_len": max_seq_len,
        }, f)

    print(f"Saved windowed dataset to {OUTPUT_PATH}")
    print(f"Train students: {len(splits['train']['student_ids'])}, "
          f"Val: {len(splits['val']['student_ids'])}, "
          f"Test: {len(splits['test']['student_ids'])}")


if __name__ == "__main__":
    main()