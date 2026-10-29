"""
src/data/adapters/statics.py
----------------------------
Adapter for Statics2011 (CMU OLI "Engineering Statics", Fall 2011): real
undergraduate learners. Files in data/external/statics2011/ (fetched by
scripts/fetch_datasets.py from the public mirror in the DKVMN authors' repository,
github.com/jennyzhang0215/DKVMN, data/STATICS; the original source is PSLC DataShop,
whose terms of use apply):

  STATICS_train.csv, STATICS_test.csv   3 lines per student: length / question ids / 0-1 answers
  STATICS_qid_sid_sname                 question id -> skill id, skill name (tab-separated)

Mapping to our schema: problem = question id; skill = skill id from the map;
correct = answer; order = position in the file. Both files are merged and re-split
by student with our own seeded 80/10/10 rule (common.split_and_bucket), so the
protocol matches the other datasets. Response time / attempts / hints are not in the
files -> neutral values + flag. Every student is assigned to section
"undergraduate". Subject caveat: this is engineering statics, not the app's
undergraduate maths topics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from data.adapters.common import ROOT, frame_to_sequences, save_processed

FOLDER = ROOT / "data" / "external" / "statics2011"
FILES = ("STATICS_train.csv", "STATICS_test.csv")
QMAP = "STATICS_qid_sid_sname"


def available(folder: Path = FOLDER) -> bool:
    return all((folder / f).exists() for f in FILES + (QMAP,))


def _read_three_line(path: Path, start_id: int):
    """Yields (student_index, [qids], [answers]) from the 3-line format."""
    lines = [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    sid = start_id
    for i in range(0, len(lines) - 2, 3):
        n = int(lines[i].split(",")[0])
        q = [int(x) for x in lines[i + 1].split(",") if x.strip()][:n]
        a = [int(x) for x in lines[i + 2].split(",") if x.strip()][:n]
        if len(q) != len(a):
            raise ValueError(f"{path.name}: question/answer length mismatch for record {i // 3}")
        yield sid, q, a
        sid += 1


def load(folder: Path = FOLDER, max_len: int = 500) -> Dict[str, Any]:
    import pandas as pd
    qmap = pd.read_csv(folder / QMAP, sep="\t", encoding="utf-8-sig")
    q_skill = dict(zip(qmap["qid"].astype(int), qmap["skillid"].astype(int)))
    names = dict(zip(qmap["skillid"].astype(int), qmap["skillname"].astype(str)))
    rows, nxt = [], 0
    for f in FILES:
        for sid, q, a in _read_three_line(folder / f, nxt):
            for t, (qq, aa) in enumerate(zip(q, a)):
                rows.append({"student": sid, "problem": qq, "skill": q_skill.get(qq, -1), "correct": int(aa),
                             "order": t, "section": "undergraduate"})
            nxt = sid + 1
    df = pd.DataFrame(rows)
    n_rows = len(df)
    unmapped = int((df["skill"] < 0).sum())
    df = df[df["skill"] >= 0]
    built = frame_to_sequences(df, max_len=max_len)
    return {
        "name": "statics2011",
        "source": "Statics2011, CMU OLI Engineering Statics (undergraduate), public mirror of the DKVMN repository "
                  "(original: PSLC DataShop)",
        "sequences": built["sequences"], "skill_map": built["skill_map"], "num_skills": built["num_skills"],
        "skill_names": {int(k): names.get(int(k), "") for k in built["skill_map"]},
        "flags": built["flags"] + (["questions_without_skill_dropped"] if unmapped else []),
        "section_counts_all_students": {"undergraduate": nxt},
        "n_rows_source": int(n_rows), "max_len": max_len, "max_students_per_section": nxt, "seed": 42,
        "section_rule": "all students are undergraduates (university engineering statics course)",
        "rows_dropped_no_skill": unmapped,
    }


def build(max_students_per_section: int = 0, max_len: int = 500, **kw) -> Path:
    # all 333 students are kept (the dataset is small); max_students_per_section is ignored
    return save_processed("statics2011", load(max_len=max_len))
