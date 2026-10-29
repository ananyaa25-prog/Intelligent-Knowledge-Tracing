"""v3 Stage 7: dataset adapters on tiny fixtures (written by the test into a temp
folder in the Eedi file layout; they are test fixtures, not data)."""

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.adapters import eedi  # noqa: E402
from data.adapters.common import NEUTRAL_RT_MS, frame_to_sequences, split_and_bucket  # noqa: E402


def _fixture(folder: Path, n_students=24, per_student=14, seed=0):
    rng = np.random.default_rng(seed)
    (folder / "train_data").mkdir(parents=True)
    (folder / "metadata").mkdir(parents=True)
    pd.DataFrame({"SubjectId": [1, 2, 3, 4], "Name": ["Maths", "Number", "Fractions", "Algebra"],
                  "ParentId": [None, 1, 2, 1], "Level": [0, 1, 2, 1]}).to_csv(folder / "metadata/subject_metadata.csv", index=False)
    pd.DataFrame({"QuestionId": list(range(10)),
                  "SubjectId": ["[1, 2, 3]" if q < 6 else "[1, 4]" for q in range(10)]}).to_csv(
        folder / "metadata/question_metadata_task_1_2.csv", index=False)
    # half the students aged ~9 (primary), half ~13 (secondary), one without a birth date
    dob = ["2009-01-01 00:00:00.000" if u % 2 == 0 else "2005-01-01 00:00:00.000" for u in range(n_students)]
    dob[-1] = None
    pd.DataFrame({"UserId": range(n_students), "Gender": 0, "DateOfBirth": dob, "PremiumPupil": 0}).to_csv(
        folder / "metadata/student_metadata_task_1_2.csv", index=False)
    rows, ans = [], []
    aid = 0
    for u in range(n_students):
        for k in range(per_student):
            q = int(rng.integers(0, 10))
            rows.append({"QuestionId": q, "UserId": u, "AnswerId": aid, "IsCorrect": int(rng.random() < 0.6),
                         "CorrectAnswer": 1, "AnswerValue": 1})
            ans.append({"AnswerId": aid, "DateAnswered": f"2018-0{1 + k // 9}-{1 + k % 9:02d} 10:00:00.000",
                        "Confidence": None, "GroupId": 1, "QuizId": k // 4, "SchemeOfWorkId": 1})
            aid += 1
    pd.DataFrame(rows).to_csv(folder / "train_data/train_task_1_2.csv", index=False)
    pd.DataFrame(ans).to_csv(folder / "metadata/answer_metadata_task_1_2.csv", index=False)
    return len(rows)


def test_eedi_adapter_schema_and_sections(tmp_path):
    n_rows = _fixture(tmp_path)
    assert eedi.available(tmp_path)
    d = eedi.load(folder=tmp_path, max_students_per_section=50, max_len=200)
    seqs = d["sequences"]
    assert d["n_rows_source"] == n_rows
    assert {s["section"] for s in seqs} == {"primary", "secondary"}
    assert d["section_counts_all_students"]["excluded_age_or_dob"] == 1
    assert sum(len(s["skill_seq"]) for s in seqs) <= n_rows               # nothing invented
    for s in seqs:
        L = len(s["skill_seq"])
        assert L >= 10
        assert set(s["skill_seq"]) <= set(range(1, d["num_skills"] + 1))
        assert np.all(s["response_time_seq"] == NEUTRAL_RT_MS) and np.all(s["attempt_seq"] == 1)
        assert np.all(s["hint_seq"] == 0)
    assert "response_time_missing_neutral_value" in d["flags"]
    # skill = most specific subject: questions 0-5 -> Fractions (3), 6-9 -> Algebra (4)
    assert set(d["skill_map"]) == {3, 4}


def test_eedi_not_available_without_files(tmp_path):
    assert not eedi.available(tmp_path / "missing")


def test_section_for_age():
    assert eedi.section_for_age(9.5) == "primary"
    assert eedi.section_for_age(11.0) == "secondary"
    assert eedi.section_for_age(17.0) is None and eedi.section_for_age(None) is None


def test_split_and_bucket_train_only():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"student": np.repeat(np.arange(40), 12), "skill": rng.integers(1, 4, 480),
                       "problem": rng.integers(0, 12, 480), "correct": rng.integers(0, 2, 480),
                       "order": np.tile(np.arange(12), 40)})
    built = frame_to_sequences(df)
    tr, va, te = split_and_bucket(built["sequences"], seed=42)
    assert len(tr) == 32 and len(va) == 4 and len(te) == 4
    from baselines.base import derive_difficulty_from_sequences
    dm = derive_difficulty_from_sequences(tr)
    for s in te:
        assert [dm.get((int(k), int(p)), 0) for k, p in zip(s["skill_seq"], s["problem_seq"])] == list(s["difficulty_seq"])
