"""
src/serving/sections.py
-----------------------
Learner-level sections (v3 Stage 6): primary, secondary, undergraduate.

Each section has a grade label, a topic list, difficulty parameters per level and,
per topic, a mapping to ASSISTments 2009 skill ids where a reasonable one exists.
A session started without a section keeps the original 10-topic behaviour.

Validation status (docs/DECISIONS.md D33-D37):
- A section is "validated on real data" only if a real dataset for that level was
  loaded and the model was evaluated on it: `secondary` (ASSISTments 2009 + Eedi
  secondary-age learners) and `primary` (Eedi learners younger than 11, Stage 7).
  `undergraduate` is "transfer mode: not validated on real data" (no dataset).
- Separately, `serving` says whether the app uses the served ASSISTments model for the
  section: only `secondary` ("model"). Primary is validated offline with a model trained
  on Eedi, which is not the served model, so the app stays evidence-only there.
- Topics without an ASSISTments mapping get no model prediction (the predictor sees
  only mapped topics); topic choice then falls back to the evidence blend with a
  neutral prior.

Grade-band split of the ASSISTments skills (by skill NAME, rule in
docs/DECISIONS.md D34): `skill_band(raw_skill_id)` -> "primary" | "secondary" |
"unassigned". No ASSISTments skill is undergraduate level.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

VALIDATED = "validated on real data"
TRANSFER = "transfer mode: not validated on real data"

# Topic ids of section topics are synthetic (they never collide with ASSISTments
# skill ids, which are < 1000): 10xxx primary, 20xxx secondary, 30xxx undergraduate.
SECTIONS: Dict[str, Dict[str, Any]] = {
    "primary": {
        "label": "Primary",
        "grade_label": "Grades 1-5 (ages 6-11)",
        "validation": VALIDATED,
        "serving": "evidence_only",
        "validation_note": "Evaluated offline on real primary-age learners (Eedi / NeurIPS 2020 data, UK pupils "
                           "younger than 11; reports/v3/external_benchmark.md). The served model is trained on "
                           "ASSISTments only, so this section shows no model predictions: topic choice and results "
                           "use the student's own answers.",
        "difficulty": {   # explicit generator parameters per level (see src/questions/generator.py)
            "easy": {"digits": 1, "steps": 1, "negatives": False, "fractions": False},
            "medium": {"digits": 2, "steps": 1, "negatives": False, "fractions": True},
            "hard": {"digits": 3, "steps": 2, "negatives": False, "fractions": True},
        },
        "topics": [
            {"id": 10001, "key": "p_add_sub", "name": "Addition and subtraction", "assist_skills": [58, 74]},
            {"id": 10002, "key": "p_mul_div", "name": "Multiplication and division", "assist_skills": [69]},
            {"id": 10003, "key": "p_fractions", "name": "Fractions", "assist_skills": [48, 50, 280, 64]},
            {"id": 10004, "key": "p_decimals", "name": "Decimals and place value", "assist_skills": [278, 276, 49, 54]},
            {"id": 10005, "key": "p_geometry", "name": "Simple geometry (perimeter and area)", "assist_skills": [42, 296, 298]},
        ],
    },
    "secondary": {
        "label": "Secondary",
        "grade_label": "Grades 6-10 (ages 11-16)",
        "validation": VALIDATED,
        "serving": "model",
        "validation_note": "ASSISTments 2009 skill-builder (middle-school learners; the served model) and Eedi "
                           "secondary-age learners (offline benchmark); reports/v3/by_band.md, "
                           "reports/v3/external_benchmark.md.",
        "difficulty": {
            "easy": {"digits": 1, "steps": 1, "negatives": False, "fractions": False},
            "medium": {"digits": 2, "steps": 2, "negatives": True, "fractions": False},
            "hard": {"digits": 2, "steps": 3, "negatives": True, "fractions": True},
        },
        "topics": [
            {"id": 20001, "key": "s_algebra", "name": "Algebraic expressions", "assist_skills": [165, 371, 340]},
            {"id": 20002, "key": "s_equations", "name": "Linear equations", "assist_skills": [311, 312, 166]},
            {"id": 20003, "key": "s_ratios", "name": "Ratios, rates and percents", "assist_skills": [79, 217, 204, 203]},
            {"id": 20004, "key": "s_geometry", "name": "Geometry (angles, Pythagoras, circles)", "assist_skills": [27, 22, 25, 39]},
            {"id": 20005, "key": "s_statistics", "name": "Statistics (mean, median, mode, range)", "assist_skills": [12, 13, 14, 15]},
        ],
    },
    "undergraduate": {
        "label": "Undergraduate",
        "grade_label": "First-year university",
        "validation": TRANSFER,
        "serving": "evidence_only",
        "validation_note": "No undergraduate dataset was loaded and no ASSISTments skill is at this level; "
                           "predictions are not available, topic choice uses the student's own answers only.",
        "difficulty": {
            "easy": {"digits": 1, "steps": 1, "negatives": False, "fractions": False},
            "medium": {"digits": 1, "steps": 2, "negatives": True, "fractions": False},
            "hard": {"digits": 2, "steps": 3, "negatives": True, "fractions": True},
        },
        "topics": [
            {"id": 30001, "key": "u_calculus", "name": "Calculus basics (derivatives)", "assist_skills": []},
            {"id": 30002, "key": "u_linear_algebra", "name": "Linear algebra (2x2 matrices)", "assist_skills": []},
            {"id": 30003, "key": "u_probability", "name": "Probability", "assist_skills": []},
            {"id": 30004, "key": "u_discrete", "name": "Discrete maths (combinatorics, modular arithmetic)", "assist_skills": []},
        ],
    },
}

VALID_SECTIONS = tuple(SECTIONS)
# "validation" is the label shown to users (was a real dataset for this level loaded and
# evaluated?). "serving" is what the app does: "model" = the served ASSISTments model
# predicts (topics mapped to its skills); "evidence_only" = no model numbers, topic choice
# and profile from the student's own answers (D35, D37).
TOPIC_INDEX: Dict[int, Dict[str, Any]] = {}
for _sec, _cfg in SECTIONS.items():
    for _pos, _t in enumerate(_cfg["topics"], start=1):
        _t.setdefault("section", _sec)
        _t.setdefault("position", _pos)
        TOPIC_INDEX[_t["id"]] = _t


def normalise_section(section: Optional[str]) -> Optional[str]:
    """None / '' / 'default' -> None (original behaviour). Unknown -> ValueError."""
    if section is None:
        return None
    s = str(section).strip().lower()
    if s in ("", "default", "none"):
        return None
    if s not in SECTIONS:
        raise ValueError(f"Unknown section '{section}'. Valid: {', '.join(VALID_SECTIONS)}")
    return s


def section_topics(section: str) -> List[Dict[str, Any]]:
    """Topics in the shape AdaptiveSession expects (position, skill_name, raw_skill_id)."""
    out = []
    for t in SECTIONS[section]["topics"]:
        out.append({"position": t["position"], "skill_name": t["name"], "raw_skill_id": t["id"],
                    "key": t["key"], "assist_skills": list(t["assist_skills"]), "section": section})
    return out


def assist_skill_for_topic(topic_id: int) -> Optional[int]:
    """Representative ASSISTments skill of a section topic (first mapped), or the id
    itself if it is already an ASSISTments skill id, or None if unmapped."""
    t = TOPIC_INDEX.get(int(topic_id))
    if t is None:
        return int(topic_id) if int(topic_id) < 1000 else None
    return int(t["assist_skills"][0]) if t["assist_skills"] else None


def section_info(section: Optional[str]) -> Dict[str, Any]:
    """Public description returned by the API (/sections, /session/start)."""
    if section is None:
        return {"section": None, "label": "Default (ASSISTments topics)", "grade_label": "Grades 6-10",
                "validation": VALIDATED, "validated": True, "serving": "model",
                "validation_note": "Original 10-topic session on ASSISTments 2009 skills."}
    c = SECTIONS[section]
    return {"section": section, "label": c["label"], "grade_label": c["grade_label"],
            "validation": c["validation"], "validated": c["validation"] == VALIDATED,
            "validation_note": c["validation_note"], "serving": c["serving"],
            "topics": [{"id": t["id"], "name": t["name"], "mapped": bool(t["assist_skills"]),
                        "assist_skills": list(t["assist_skills"])} for t in c["topics"]],
            "difficulty": c["difficulty"]}


def all_sections_info() -> List[Dict[str, Any]]:
    return [section_info(s) for s in VALID_SECTIONS]


# ---- grade-band split of ASSISTments skills (rule D34) --------------------------

# Keyword rules on the skill NAME, applied in order; first match wins.
_PRIMARY_PATTERNS = [
    r"whole numbers", r"^equivalent fractions", r"^ordering fractions", r"^ordering positive decimals",
    r"^fraction of", r"addition and subtraction fractions", r"multiplication fractions", r"division fractions",
    r"positive decimals", r"^rounding", r"^estimation", r"^number line", r"reading a ruler", r"divisibility",
    r"^prime number", r"least common multiple", r"greatest common factor", r"perimeter of a polygon",
    r"^area rectangle", r"^area triangle", r"^area parallelogram", r"obtuse, acute, and right", r"^table$",
    r"pattern finding", r"nets of 3d", r"volume rectangular prism",
    r"order of operations \+,-,/,\* \(\) positive reals", r"unit conversion within a system",
]


def band_for_name(name: Optional[str]) -> str:
    """Primary-level content by keyword; any other NAMED ASSISTments skill is
    secondary (the dataset is middle-school maths); unnamed -> unassigned."""
    if not name or not str(name).strip():
        return "unassigned"
    n = str(name).strip().lower()
    for pat in _PRIMARY_PATTERNS:
        if re.search(pat, n):
            return "primary"
    return "secondary"


def load_skill_names() -> Dict[int, str]:
    """ASSISTments skill names: data/processed_v3/skill_names_all.json if present
    (written by scripts/build_by_band.py from the raw CSV, read-only), else the
    10 names in data/processed/skill_names.json."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    for p in (root / "data" / "processed_v3" / "skill_names_all.json", root / "data" / "processed" / "skill_names.json"):
        if p.exists():
            try:
                return {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
            except Exception:
                continue
    return {}


def skill_band(raw_skill_id: int, names: Optional[Dict[int, str]] = None) -> str:
    names = names if names is not None else load_skill_names()
    return band_for_name(names.get(int(raw_skill_id)))


def model_items(history):
    """History as the ASSISTments model may see it: section topic ids are replaced
    by their mapped ASSISTments skill, unmapped section items are dropped. Items
    with any other id are returned unchanged (so the original behaviour, including
    errors for invalid ids, is kept). Works for dicts and pydantic records."""
    out = []
    for it in history or []:
        sid = it.get("skill_id", it.get("topic")) if isinstance(it, dict) else getattr(it, "skill_id", None)
        try:
            tid = int(sid)
        except (TypeError, ValueError):
            out.append(it)
            continue
        if tid not in TOPIC_INDEX:
            out.append(it)
            continue
        m = assist_skill_for_topic(tid)
        if m is None:
            continue
        if isinstance(it, dict):
            out.append(dict(it, skill_id=m, topic=m))
        else:
            out.append(it.copy(update={"skill_id": m}))
    return out
