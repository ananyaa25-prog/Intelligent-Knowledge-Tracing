"""
src/generate_metadata.py
------------------------
Extracts:
1. data/processed/skill_names.json from skill_builder_data.csv & question_bank.json
2. data/processed/sample_test_students.json from test split of windowed_data.pkl
"""

import json
import pickle
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "skill_builder_data.csv"
WINDOWED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "windowed_data.pkl"
QUESTION_BANK_PATH = PROJECT_ROOT / "question_bank.json"
SKILL_NAMES_OUT = PROJECT_ROOT / "data" / "processed" / "skill_names.json"
SAMPLE_STUDENTS_OUT = PROJECT_ROOT / "data" / "processed" / "sample_test_students.json"


def generate_skill_names():
    skill_names = {}

    # Load from question bank first
    if QUESTION_BANK_PATH.exists():
        with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as f:
            qb = json.load(f)
            for item in qb.get("skill_sequence", []):
                skill_names[str(item["raw_skill_id"])] = item["skill_name"]

    # Load from raw CSV
    if RAW_CSV_PATH.exists():
        df = pd.read_csv(
            RAW_CSV_PATH,
            usecols=["skill_id", "skill_name"],
            encoding="ISO-8859-1",
            low_memory=False,
        ).dropna()
        
        # Handle composite skill_ids
        for _, row in df.drop_duplicates(subset=["skill_id"]).iterrows():
            raw_id = str(row["skill_id"])
            name = str(row["skill_name"]).strip()
            if "_" in raw_id:
                parts = raw_id.split("_")
                for p in parts:
                    if p not in skill_names:
                        skill_names[p] = name
            else:
                skill_names[raw_id] = name

    with open(SKILL_NAMES_OUT, "w", encoding="utf-8") as f:
        json.dump(skill_names, f, indent=2)

    print(f"Saved {len(skill_names)} skill names to {SKILL_NAMES_OUT}")


def generate_sample_students():
    with open(WINDOWED_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    test_split = data["splits"]["test"]
    skill_map = data["skill_map"]
    reverse_map = {v: k for k, v in skill_map.items()}
    student_ids = test_split.get("student_ids", [])

    num_students = test_split["skill_seq"].shape[0]
    sample_students = []

    # Pick 15 real test-split students
    count = 0
    for i in range(num_students):
        mask = test_split["mask"][i]
        real_len = int(mask.sum().item())
        if real_len < 10:
            continue

        sid = str(student_ids[i]) if i < len(student_ids) else f"Test-Student-{i+1}"
        history = []
        for t in range(real_len):
            reindexed = int(test_split["skill_seq"][i, t].item())
            raw_skill_id = int(reverse_map[reindexed])
            history.append({
                "skill_id": raw_skill_id,
                "correct": int(test_split["correct_seq"][i, t].item()),
                "response_time_ms": float(test_split["response_time_seq"][i, t].item()),
                "attempt_count": max(1, int(test_split["attempt_seq"][i, t].item())),
                "hint_used": int(test_split["hint_seq"][i, t].item()),
            })

        sample_students.append({
            "student_id": f"Student_{count+1:02d}",
            "original_id": sid,
            "num_interactions": real_len,
            "history": history
        })
        count += 1
        if count >= 15:
            break

    with open(SAMPLE_STUDENTS_OUT, "w", encoding="utf-8") as f:
        json.dump(sample_students, f, indent=2)

    print(f"Saved {len(sample_students)} sample students to {SAMPLE_STUDENTS_OUT}")


if __name__ == "__main__":
    generate_skill_names()
    generate_sample_students()
