"""
src/data/adapters/eedi.py
-------------------------
Adapter for the Eedi / NeurIPS 2020 Education Challenge data (tasks 1-2), placed in
data/external/eedi_neurips2020/ (data.zip or its extracted folders).

Files used (only read, never modified):
  train_data/train_task_1_2.csv            QuestionId, UserId, AnswerId, IsCorrect, ...
  metadata/answer_metadata_task_1_2.csv    AnswerId, DateAnswered, ..., QuizId
  metadata/question_metadata_task_1_2.csv  QuestionId, SubjectId ("[3, 71, 98, 209]")
  metadata/student_metadata_task_1_2.csv   UserId, Gender, DateOfBirth, PremiumPupil
  metadata/subject_metadata.csv            SubjectId, Name, ParentId, Level

Mapping to our schema:
  skill    = the most specific subject of the question (highest Level in its SubjectId list)
  problem  = QuestionId; correct = IsCorrect; order = DateAnswered (ties: AnswerId)
  group    = QuizId (co-occurrence windows for the skill graph)
  metadata = not available -> neutral values + flag (common.py)
Section by the student's AGE at their first answer (DateOfBirth is month-level):
  age < 11.0 -> primary; 11.0 <= age < 16.5 -> secondary; otherwise or no date of
  birth -> excluded (counts reported). UK primary school ends in Year 6 (age 10-11).
A capped, seeded sample of students per section is kept (max_students_per_section).
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from data.adapters.common import ROOT, frame_to_sequences, save_processed

FOLDER = ROOT / "data" / "external" / "eedi_neurips2020"
FILES = {
    "train": "train_data/train_task_1_2.csv",
    "answers": "metadata/answer_metadata_task_1_2.csv",
    "questions": "metadata/question_metadata_task_1_2.csv",
    "students": "metadata/student_metadata_task_1_2.csv",
    "subjects": "metadata/subject_metadata.csv",
}
PRIMARY_MAX_AGE = 11.0
SECONDARY_MAX_AGE = 16.5


def _opener(folder: Path):
    """Returns open(relpath) for either extracted files or data.zip (read-only)."""
    zpath = folder / "data.zip"
    if zpath.exists():
        z = zipfile.ZipFile(zpath)
        names = z.namelist()

        def op(rel):
            cands = [n for n in names if n.endswith(rel)]
            if not cands:
                raise FileNotFoundError(rel)
            return z.open(cands[0])
        return op

    def op(rel):
        hits = list(folder.rglob(Path(rel).name))
        if not hits:
            raise FileNotFoundError(rel)
        return open(hits[0], "rb")
    return op


def available(folder: Path = FOLDER) -> bool:
    try:
        op = _opener(folder)
        for rel in FILES.values():
            op(rel).close()
        return True
    except (FileNotFoundError, zipfile.BadZipFile, OSError):
        return False


def section_for_age(age: Optional[float]) -> Optional[str]:
    if age is None or not np.isfinite(age) or age <= 3:
        return None
    if age < PRIMARY_MAX_AGE:
        return "primary"
    if age < SECONDARY_MAX_AGE:
        return "secondary"
    return None


def load(folder: Path = FOLDER, max_students_per_section: int = 3000, max_len: int = 200,
         seed: int = 42, nrows: Optional[int] = None) -> Dict[str, Any]:
    import pandas as pd
    op = _opener(folder)
    with op(FILES["subjects"]) as f:
        subj = pd.read_csv(f, encoding="utf-8-sig")
    level = dict(zip(subj["SubjectId"], subj["Level"]))
    names = dict(zip(subj["SubjectId"], subj["Name"]))
    with op(FILES["questions"]) as f:
        qm = pd.read_csv(f, encoding="utf-8-sig")

    def deepest(s):
        ids = ast.literal_eval(s) if isinstance(s, str) else []
        ids = [i for i in ids if i in level]
        return max(ids, key=lambda i: (level[i], i)) if ids else -1
    q_skill = {int(q): deepest(s) for q, s in zip(qm["QuestionId"], qm["SubjectId"])}
    with op(FILES["students"]) as f:
        st = pd.read_csv(f, usecols=["UserId", "DateOfBirth"], encoding="utf-8-sig")
    dob = dict(zip(st["UserId"], pd.to_datetime(st["DateOfBirth"], errors="coerce")))
    with op(FILES["train"]) as f:
        tr = pd.read_csv(f, usecols=["QuestionId", "UserId", "AnswerId", "IsCorrect"], nrows=nrows, encoding="utf-8-sig",
                         dtype={"QuestionId": np.int32, "UserId": np.int32, "AnswerId": np.int64, "IsCorrect": np.int8})
    with op(FILES["answers"]) as f:
        am = pd.read_csv(f, usecols=["AnswerId", "DateAnswered", "QuizId"], dtype={"AnswerId": np.float64}, encoding="utf-8-sig")
    am = am.dropna(subset=["AnswerId"])
    am["AnswerId"] = am["AnswerId"].astype(np.int64)
    am["DateAnswered"] = pd.to_datetime(am["DateAnswered"], errors="coerce")
    df = tr.merge(am, on="AnswerId", how="left")
    n_rows_source = len(df)
    df = df.dropna(subset=["DateAnswered"])
    first = df.groupby("UserId")["DateAnswered"].min()
    ages = {u: ((t - dob[u]).days / 365.25 if (u in dob and pd.notna(dob[u])) else None) for u, t in first.items()}
    sec = {u: section_for_age(a) for u, a in ages.items()}
    counts = {"primary": 0, "secondary": 0, "excluded_age_or_dob": 0}
    for v in sec.values():
        counts[v if v else "excluded_age_or_dob"] += 1
    rng = np.random.default_rng(seed)
    keep = {}
    for s in ("primary", "secondary"):
        users = np.array(sorted(u for u, v in sec.items() if v == s))
        # 2x headroom: students with < 10 answers are dropped later (frame_to_sequences)
        pick = rng.permutation(users)[: max_students_per_section * 2]
        for u in pick:
            keep[int(u)] = s
    df = df[df["UserId"].isin(keep.keys())].copy()
    df["skill"] = df["QuestionId"].map(q_skill).astype(np.int64)
    df = df[df["skill"] >= 0]
    out = df.rename(columns={"UserId": "student", "QuestionId": "problem", "IsCorrect": "correct", "QuizId": "group"})
    out["order"] = out["DateAnswered"]
    out = out.sort_values(["student", "order", "AnswerId"], kind="mergesort")
    out["section"] = out["student"].map(keep)
    built = frame_to_sequences(out[["student", "skill", "problem", "correct", "order", "group", "section"]],
                               max_len=max_len)
    # cap per section after the >= 10 answers filter
    seqs = []
    for s in ("primary", "secondary"):
        seqs += [x for x in built["sequences"] if x["section"] == s][:max_students_per_section]
    payload = {
        "name": "eedi", "source": "Eedi / NeurIPS 2020 Education Challenge, tasks 1-2 (public archive)",
        "sequences": seqs, "skill_map": built["skill_map"], "num_skills": built["num_skills"],
        "skill_names": {int(k): names.get(int(k), "") for k in built["skill_map"]},
        "flags": built["flags"], "section_counts_all_students": counts,
        "n_rows_source": int(n_rows_source), "max_len": max_len,
        "max_students_per_section": max_students_per_section, "seed": seed,
        "section_rule": f"age at first answer < {PRIMARY_MAX_AGE} primary; < {SECONDARY_MAX_AGE} secondary; else excluded",
    }
    return payload


def build(**kw) -> Path:
    return save_processed("eedi", load(**kw))
