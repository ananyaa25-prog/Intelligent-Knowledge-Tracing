"""
src/serving/inference_service.py
--------------------------------
FastAPI inference service wrapping the trained Gated Knowledge Tracing model.

Loads the model weights, PPMI graph, and response time normalization stats
ONCE at startup (never per-request).

Supports:
- Per-interaction and batch predictions (/predict, /predict_batch)
- SQLite-backed multi-session tracking (/session/start, /session/:id/answer, /session/:id/predict)
- Cumulative student history mastery tracking (/student/:id/predict, /student/:id/summary)
- Live multi-student dashboard matrix querying SQLite database (/students/matrix)

SCOPE: Mastery tracker only -- takes a student's answer history and outputs
per-skill mastery estimates. Does not generate questions, adapt difficulty,
or select content.
"""

import sys
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import GatedKTModel
from graph_encoder import adjacency_to_edge_index
from train import compute_rt_stats
from serving.session import session_manager
from serving.calibration import maybe_calibrate
from serving.adaptive_session import (
    AdaptiveSession, AdaptiveSessionManager, MAIN_QUESTIONS, TOPIC_BY_ID, TOPIC_IDS,
)
from serving.sections import normalise_section, model_items, all_sections_info

# Base paths
PROJECT_ROOT = SRC_DIR.parent
WINDOWED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "windowed_data.pkl"
SKILL_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "skill_graph.pkl"
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "best_full_gated.pt"
SESSION_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "session_logs.jsonl"
# Optional override (policy_config KT_LOG_PATH), e.g. tests writing to a temp copy.
if os.environ.get("KT_LOG_PATH"):
    SESSION_LOG_PATH = Path(os.environ["KT_LOG_PATH"])

# Fallback checkpoint if best_full_gated.pt is not directly named
FALLBACK_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "best_full_gated_seed123.pt"
SEED42_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "best_full_gated_seed42.pt"


import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Pydantic Request / Response Models
class InteractionRecord(BaseModel):
    skill_id: Any = Field(..., description="Raw skill ID from skill_map (not re-indexed)")
    correct: int = Field(..., ge=0, le=1, description="Binary correctness (0 or 1)")
    response_time_ms: float = Field(..., ge=0.0, description="Response time in milliseconds")
    attempt_count: int = Field(..., ge=1, description="Number of attempts (>= 1)")
    hint_used: int = Field(..., ge=0, le=1, description="Whether hint was used (0 or 1)")
    difficulty_bucket: int = Field(0, ge=0, le=3, description="Difficulty bucket (0=unknown, 1=easy, 2=medium, 3=hard)")


class PredictRequest(BaseModel):
    student_history: List[InteractionRecord] = Field(
        ..., description="Chronological sequence of student interactions (oldest first)"
    )


class PredictResponse(BaseModel):
    skill_mastery: Dict[str, float] = Field(
        ..., description="Map of raw_skill_id to predicted mastery (probability in [0, 1])"
    )
    num_interactions_processed: int = Field(
        ..., description="Number of interactions processed"
    )
    skill_interaction_counts: Optional[Dict[str, int]] = Field(
        default=None, description="Number of interactions recorded per skill"
    )
    low_confidence_skills: Optional[List[str]] = Field(
        default=None, description="Skills with interaction count < 3 (low data confidence)"
    )
    mastery_summary: Optional[str] = Field(
        default=None, description="Auto-generated student-facing synthesis summary"
    )
    message: Optional[str] = Field(
        default=None, description="Optional diagnostic message"
    )


class SessionStartRequest(BaseModel):
    student_id: Optional[str] = Field(None, description="Optional existing student ID (new UUID generated if None)")
    preferred_set: Optional[str] = Field(None, description="Optional set preference ('A' or 'B')")
    source: Optional[str] = Field("live", description="Student source tag ('live' or 'test')")


class AnswerSubmitRequest(BaseModel):
    position: int = Field(..., ge=1, le=10, description="Question position (1-10)")
    submitted_answer: str = Field(..., description="Selected answer option (e.g. 'A', 'B', 'C', 'D' or option text)")
    response_time_ms: float = Field(..., ge=0.0, description="Time taken to answer in ms")
    hint_used: Optional[int] = Field(0, ge=0, le=1, description="Whether hint was viewed (0 or 1)")


class BatchStudentItem(BaseModel):
    student_id: str
    student_history: List[InteractionRecord]


class BatchPredictRequest(BaseModel):
    students: List[BatchStudentItem]


class BatchPredictResponse(BaseModel):
    results: Dict[str, PredictResponse]


class StartSessionRequest(BaseModel):
    student_id: Optional[str] = Field(default=None, description="Student identifier")
    preferred_set: Optional[str] = Field(default=None, description="Optional set preference ('A', 'B', etc. or None for procedural)")
    source: str = Field(..., description="Student source tag ('live' or 'test')")
    section: Optional[str] = Field(default=None, description="v3: primary | secondary | undergraduate (omit for the original topics)")


class StartSessionResponse(BaseModel):
    session_id: str
    student_id: str
    questions: List[Dict[str, Any]]
    prior_completed_sessions: int = 0
    total_prior_interactions: int = 0
    # v3 additive fields (section of the session and its validation label)
    section: Optional[str] = None
    section_label: Optional[str] = None
    validation: Optional[str] = None
    validated: Optional[bool] = None
    model_predictions: Optional[bool] = None


class StudentSummaryResponse(BaseModel):
    student_id: str
    exists: bool
    session_count: int
    completed_session_count: int
    total_interactions: int
    created_at: Optional[str] = None


class AnswerRequest(BaseModel):
    position: Optional[int] = Field(default=None, ge=1, le=20)
    submitted_answer: Any
    response_time_ms: float = Field(..., ge=0.0)
    hint_used: int = Field(default=0, ge=0, le=1)


class AnswerResponse(BaseModel):
    session_id: str
    student_id: str
    position: int
    correct: int
    attempt_count: int
    interactions_count: int
    is_complete: bool


class ModelArtifacts:
    """Singleton-style container holding pre-loaded model, graph, and RT stats."""
    def __init__(self):
        self.model: Optional[GatedKTModel] = None
        self.skill_map: Dict[Any, int] = {}
        self.inv_skill_map: Dict[int, Any] = {}
        self.adj_matrix: Optional[np.ndarray] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self.rt_mean: float = 8.5
        self.rt_std: float = 1.2
        self.num_skills: int = 123
        self.max_seq_len: int = 200
        self.is_loaded: bool = False

    def load(self):
        if self.is_loaded:
            return

        print("[InferenceService] Pre-loading model artifacts...")

        # 1. Load skill graph
        if not SKILL_GRAPH_PATH.exists():
            raise FileNotFoundError(f"Skill graph not found at {SKILL_GRAPH_PATH}")
        with open(SKILL_GRAPH_PATH, "rb") as f:
            graph_data = pickle.load(f)

        self.skill_map = graph_data["skill_map"]
        self.inv_skill_map = {v: k for k, v in self.skill_map.items()}
        self.adj_matrix = graph_data["adjacency_matrix"]
        self.num_skills = graph_data.get("num_skills", len(self.skill_map))

        # Convert adjacency matrix to edge_index and edge_weight
        adj_t = torch.tensor(self.adj_matrix, dtype=torch.float32)
        self.edge_index, self.edge_weight = adjacency_to_edge_index(adj_t)

        # 2. Load windowed data stats if available
        if WINDOWED_DATA_PATH.exists():
            with open(WINDOWED_DATA_PATH, "rb") as f:
                window_data = pickle.load(f)
            train_split = window_data["splits"]["train"]
            self.rt_mean, self.rt_std = compute_rt_stats(train_split)
            self.max_seq_len = train_split["skill_seq"].shape[1]
            print(f"[InferenceService] Computed RT stats: mean={self.rt_mean:.3f}, std={self.rt_std:.3f}")
        else:
            print("[InferenceService] Warning: windowed_data.pkl not found, using default RT stats.")

        # 3. Load model weights
        ckpt_path = None
        for candidate in [MODEL_CHECKPOINT_PATH, FALLBACK_CHECKPOINT_PATH, SEED42_CHECKPOINT_PATH]:
            if candidate.exists():
                ckpt_path = candidate
                break

        if ckpt_path is None:
            raise FileNotFoundError(
                f"No model checkpoint found. Checked: {MODEL_CHECKPOINT_PATH}, {FALLBACK_CHECKPOINT_PATH}, {SEED42_CHECKPOINT_PATH}"
            )

        print(f"[InferenceService] Loading weights from {ckpt_path}...")
        self.model = GatedKTModel(
            num_skills=self.num_skills,
            rt_log_mean=self.rt_mean,
            rt_log_std=self.rt_std,
            use_graph=True,
            use_gate=True,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.is_loaded = True
        print(f"[InferenceService] Model successfully loaded ({self.num_skills} skills, max_seq_len={self.max_seq_len})")


artifacts = ModelArtifacts()


# ---- Adaptive session engine (Phase B) ------------------------------------

def _log_line(obj: Dict[str, Any]) -> None:
    """Append one JSON line to the session log for Phase C evaluation."""
    try:
        with open(SESSION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception as e:  # logging must never break the session
        print(f"[InferenceService] Warning: could not write session log: {e}")


def _store_adaptive(session_id: str, student_id: str, interaction: Dict[str, Any]) -> None:
    session_manager.db.record_adaptive_interaction(session_id, student_id, interaction)


adaptive_manager = AdaptiveSessionManager(store=_store_adaptive, log=_log_line)


def _history_to_tensors(history: List[Dict[str, Any]], max_len: int):
    """Build model input tensors from a list of interaction dicts (raw skill ids)."""
    curr_len = min(len(history), max_len)
    skill_seq = torch.zeros((1, max_len), dtype=torch.long)
    correct_seq = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.zeros((1, max_len), dtype=torch.float32)
    response_time = torch.zeros((1, max_len), dtype=torch.float32)
    attempt_count = torch.zeros((1, max_len), dtype=torch.float32)
    hint_used = torch.zeros((1, max_len), dtype=torch.float32)
    difficulty_seq = torch.zeros((1, max_len), dtype=torch.long)

    for i in range(curr_len):
        item = history[i]
        raw_id = item.get("skill_id", item.get("topic"))
        rid = artifacts.skill_map.get(raw_id)
        if rid is None:
            try:
                rid = artifacts.skill_map.get(int(raw_id))
            except (ValueError, TypeError):
                rid = 1
        skill_seq[0, i] = rid if rid is not None else 1
        correct_seq[0, i] = int(item["correct"])
        mask[0, i] = 1.0
        response_time[0, i] = float(item.get("response_time_ms", 5000.0))
        attempt_count[0, i] = float(item.get("attempt_count", 1))
        hint_used[0, i] = float(item.get("hint_used", 0))
        difficulty_seq[0, i] = int(item.get("difficulty_bucket", 0) or 0)

    return skill_seq, correct_seq, mask, response_time, attempt_count, hint_used, difficulty_seq, curr_len


def _predictor(history: List[Dict[str, Any]], candidates: List[tuple]) -> List[float]:
    """Score candidate (raw_skill_id, difficulty_bucket) pairs as the NEXT question."""
    history = model_items(history)  # v3: section topic ids -> mapped ASSISTments skill (others unchanged)
    # Keep the MOST RECENT interactions if the history is longer than the model window.
    if len(history) > artifacts.max_seq_len:
        history = history[-artifacts.max_seq_len:]
    tensors = _history_to_tensors(history, artifacts.max_seq_len)
    skill_seq, correct_seq, mask, rt, ac, hu, diff_seq, curr_len = tensors
    reindexed_candidates = []
    for raw_id, bucket in candidates:
        rid = artifacts.skill_map.get(raw_id)
        if rid is None:
            try:
                rid = artifacts.skill_map.get(int(raw_id))
            except (ValueError, TypeError):
                rid = None
        reindexed_candidates.append((rid, int(bucket)))

    scores = [0.5] * len(candidates)
    valid_idx = [i for i, (rid, _) in enumerate(reindexed_candidates) if rid is not None]
    if valid_idx:
        with torch.no_grad():
            preds = artifacts.model.predict_candidates(
                skill_seq[:, :curr_len], correct_seq[:, :curr_len], mask[:, :curr_len],
                rt[:, :curr_len], ac[:, :curr_len], hu[:, :curr_len], diff_seq[:, :curr_len],
                [reindexed_candidates[i] for i in valid_idx],
                edge_index=artifacts.edge_index, edge_weight=artifacts.edge_weight,
            )
        for j, i in enumerate(valid_idx):
            # Optional temperature scaling (APPLY_CALIBRATION); identity when disabled.
            scores[i] = max(0.0, min(1.0, maybe_calibrate(float(preds[j].item()))))
    return scores


# ---- v3: checkpoint ensemble for POLICY=uncertainty ---------------------------------
ENSEMBLE_SEEDS = (42, 123, 2024)   # the existing v2 full_gated checkpoints (read-only)
_ensemble_models: List[GatedKTModel] = []


def _load_ensemble() -> List[GatedKTModel]:
    if not _ensemble_models:
        for seed in ENSEMBLE_SEEDS:
            path = PROJECT_ROOT / "data" / "processed" / f"best_full_gated_seed{seed}.pt"
            if not path.exists():
                continue
            m = GatedKTModel(num_skills=artifacts.num_skills, rt_log_mean=artifacts.rt_mean,
                             rt_log_std=artifacts.rt_std, use_graph=True, use_gate=True)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            _ensemble_models.append(m)
    return _ensemble_models


def _ensemble_predictor(history: List[Dict[str, Any]], candidates: List[tuple]) -> List[List[float]]:
    """P(correct) of each candidate from every ensemble member (same inputs and optional
    calibration as _predictor). Used only by POLICY=uncertainty."""
    members = _load_ensemble()
    if len(members) < 2:
        return []
    history = model_items(history)
    if len(history) > artifacts.max_seq_len:
        history = history[-artifacts.max_seq_len:]
    skill_seq, correct_seq, mask, rt, ac, hu, diff_seq, curr_len = _history_to_tensors(history, artifacts.max_seq_len)
    reidx = []
    for raw_id, bucket in candidates:
        rid = artifacts.skill_map.get(raw_id)
        if rid is None:
            try:
                rid = artifacts.skill_map.get(int(raw_id))
            except (ValueError, TypeError):
                rid = None
        reidx.append((rid, int(bucket)))
    valid = [i for i, (rid, _) in enumerate(reidx) if rid is not None]
    out = []
    for m in members:
        scores = [0.5] * len(candidates)
        if valid and curr_len > 0:
            with torch.no_grad():
                preds = m.predict_candidates(
                    skill_seq[:, :curr_len], correct_seq[:, :curr_len], mask[:, :curr_len],
                    rt[:, :curr_len], ac[:, :curr_len], hu[:, :curr_len], diff_seq[:, :curr_len],
                    [reidx[i] for i in valid], edge_index=artifacts.edge_index, edge_weight=artifacts.edge_weight)
            for j, i in enumerate(valid):
                scores[i] = max(0.0, min(1.0, maybe_calibrate(float(preds[j].item()))))
        out.append(scores)
    return out


_predictor.ensemble = _ensemble_predictor   # discovered by AdaptiveSession for POLICY=uncertainty


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts.load()
    yield


app = FastAPI(
    title="Intelligent Knowledge Tracing Inference Service",
    description="FastAPI service serving per-skill mastery predictions from GatedKTModel.",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def find_weak_skill_pairs(skill_mastery: Dict[str, float], cutoff: float = 1.0) -> List[Dict[str, Any]]:
    """
    Task 16: Identifies pairs of skills that are BOTH weak (<50% mastery) and have high curriculum PPMI co-occurrence (>= cutoff).
    Returns list of dicts: {'skill1_id': int, 'skill1_name': str, 'skill2_id': int, 'skill2_name': str, 'ppmi_weight': float}
    """
    if artifacts.adj_matrix is None or not artifacts.skill_map:
        return []

    id_to_name = {str(s["raw_skill_id"]): s["skill_name"] for s in session_manager.bank.skill_sequence}
    weak_skills = [
        raw_id for raw_id, val in skill_mastery.items()
        if val < 0.50
    ]

    weak_pairs = []
    for i, id1_str in enumerate(weak_skills):
        for j, id2_str in enumerate(weak_skills):
            if i >= j:
                continue
            try:
                id1 = int(id1_str)
                id2 = int(id2_str)
            except (ValueError, TypeError):
                continue
            idx1 = artifacts.skill_map.get(id1)
            idx2 = artifacts.skill_map.get(id2)
            if idx1 is not None and idx2 is not None:
                weight = float(artifacts.adj_matrix[idx1, idx2])
                if weight >= cutoff:
                    weak_pairs.append({
                        "skill1_id": id1,
                        "skill1_name": id_to_name.get(str(id1), f"Skill {id1}"),
                        "skill2_id": id2,
                        "skill2_name": id_to_name.get(str(id2), f"Skill {id2}"),
                        "ppmi_weight": round(weight, 4),
                    })
    return weak_pairs


def generate_mastery_summary(
    skill_mastery: Dict[str, float],
    skill_interaction_counts: Optional[Dict[str, int]] = None,
    low_confidence_skills: Optional[List[str]] = None,
    session_count: Optional[int] = None,
) -> str:
    """
    Task 19: Produces a 2-4 sentence student/teacher-facing synthesis summary:
    - Names the single weakest skill (lowest %) and strongest skill (highest %) with actual percentages.
    - If Task 16 finds connected weak curriculum skills, states that connection to practice together.
    - If student data is limited (<3 interactions for majority of skills or session_count <= 1), states an honest disclaimer.
    - Strictly avoids internal model jargon ("gate", "PPMI", "embedding", "GRU", "GAT").
    """
    if not skill_mastery:
        return ""

    id_to_name = {str(s["raw_skill_id"]): s["skill_name"] for s in session_manager.bank.skill_sequence}

    # Focus on monitored sequence skills for clean synthesis
    monitored_mastery = {
        str(k): v for k, v in skill_mastery.items() if str(k) in id_to_name
    }
    if not monitored_mastery:
        monitored_mastery = {str(k): v for k, v in skill_mastery.items()}

    # Find single weakest and single strongest skills
    weakest_id = min(monitored_mastery, key=monitored_mastery.get)
    strongest_id = max(monitored_mastery, key=monitored_mastery.get)

    weakest_name = id_to_name.get(weakest_id, f"Skill {weakest_id}")
    strongest_name = id_to_name.get(strongest_id, f"Skill {strongest_id}")

    weakest_pct = round(monitored_mastery[weakest_id] * 100)
    strongest_pct = round(monitored_mastery[strongest_id] * 100)

    sentences = []

    # Sentence 1: Strongest & Weakest synthesis
    if weakest_id == strongest_id or abs(weakest_pct - strongest_pct) < 1:
        sentences.append(
            f"Your performance is consistent across skills, with overall mastery estimated around {strongest_pct}%."
        )
    else:
        sentences.append(
            f"Your strongest area is {strongest_name} at {strongest_pct}% mastery, while your primary area for improvement is {weakest_name} at {weakest_pct}%."
        )

    # Sentence 2: Relational connection (reuse find_weak_skill_pairs)
    weak_pairs = find_weak_skill_pairs(monitored_mastery, cutoff=1.0)
    if weak_pairs:
        top_pair = max(weak_pairs, key=lambda p: p["ppmi_weight"])
        s1 = top_pair["skill1_name"]
        s2 = top_pair["skill2_name"]
        sentences.append(
            f"Because {s1} and {s2} are closely related topics and both show lower scores, we recommend practicing them together."
        )

    # Sentence 3: Low-confidence / limited data disclaimer
    is_low_data = False
    if session_count is not None and session_count <= 1:
        is_low_data = True
    elif low_confidence_skills is not None and len(monitored_mastery) > 0:
        if len(low_confidence_skills) >= len(monitored_mastery) / 2:
            is_low_data = True
    elif skill_interaction_counts:
        low_count = sum(1 for k in monitored_mastery if skill_interaction_counts.get(str(k), 0) < 3)
        if low_count >= len(monitored_mastery) / 2:
            is_low_data = True

    if is_low_data:
        sentences.append(
            "This estimate is based on limited data so far; complete more sessions for a more reliable picture."
        )

    return " ".join(sentences)


def _process_history(history: List[InteractionRecord]) -> PredictResponse:
    """Internal helper to process a single student history."""
    history = model_items(history)  # v3: section topic ids -> mapped ASSISTments skill (others unchanged)
    n_interactions = len(history)

    # Edge case: History < 2 interactions cannot produce next-step prediction
    if n_interactions < 2:
        return PredictResponse(
            skill_mastery={},
            num_interactions_processed=n_interactions,
            message="Insufficient data: at least 2 interactions are required to estimate mastery."
        )

    # Truncate to max_seq_len if longer (KEEP MOST RECENT interactions)
    max_len = artifacts.max_seq_len
    if len(history) > max_len:
        history = history[-max_len:]

    # Validate skill_ids against skill_map
    reindexed_skills = []
    raw_skill_ids = []
    for idx, item in enumerate(history):
        raw_id = item.skill_id
        matched_key = None
        if raw_id in artifacts.skill_map:
            matched_key = raw_id
        else:
            try:
                int_id = int(raw_id)
                if int_id in artifacts.skill_map:
                    matched_key = int_id
                else:
                    for k in artifacts.skill_map:
                        if int(k) == int_id:
                            matched_key = k
                            break
            except (ValueError, TypeError):
                pass

        if matched_key is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid skill_id '{raw_id}' at index {idx}: not found in skill_map"
            )

        reindexed_skills.append(artifacts.skill_map[matched_key])
        raw_skill_ids.append(matched_key)

    curr_len = len(history)

    # Construct model input tensors [batch=1, max_seq_len]
    skill_seq = torch.zeros((1, max_len), dtype=torch.long)
    correct_seq = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.zeros((1, max_len), dtype=torch.float32)
    response_time = torch.zeros((1, max_len), dtype=torch.float32)
    attempt_count = torch.zeros((1, max_len), dtype=torch.float32)
    hint_used = torch.zeros((1, max_len), dtype=torch.float32)
    difficulty_seq = torch.zeros((1, max_len), dtype=torch.long)

    for i in range(curr_len):
        skill_seq[0, i] = reindexed_skills[i]
        correct_seq[0, i] = history[i].correct
        mask[0, i] = 1.0
        response_time[0, i] = float(history[i].response_time_ms)
        attempt_count[0, i] = float(history[i].attempt_count)
        hint_used[0, i] = float(history[i].hint_used)
        difficulty_seq[0, i] = int(getattr(history[i], "difficulty_bucket", 0) or 0)

    with torch.no_grad():
        pred, _ = artifacts.model(
            skill_seq,
            correct_seq,
            mask,
            response_time,
            attempt_count,
            hint_used,
            difficulty_seq,
            edge_index=artifacts.edge_index,
            edge_weight=artifacts.edge_weight,
        )

    # Step t (0 <= t < curr_len - 1) predicts outcome for interaction t+1 (skill raw_skill_ids[t+1])
    skill_mastery: Dict[str, float] = {}
    for t in range(curr_len - 1):
        target_skill_raw = raw_skill_ids[t + 1]
        predicted_val = float(pred[0, t].item())
        skill_mastery[str(target_skill_raw)] = max(0.0, min(1.0, predicted_val))

    # Ensure all 10 monitored skills or seen skills have an estimate
    target_raw_skills = set(s["raw_skill_id"] for s in session_manager.bank.skill_sequence)
    for raw_id in raw_skill_ids:
        target_raw_skills.add(raw_id)

    missing_skills = [s for s in target_raw_skills if str(s) not in skill_mastery]
    if missing_skills:
        # Use predict_candidates() to score the missing (skill, difficulty) pairs as
        # the NEXT question, replacing the previous manual graph-embedding block.
        pairs = []  # (raw_skill_id, reindexed_id)
        for raw_s in missing_skills:
            reindexed_id = artifacts.skill_map.get(raw_s)
            if reindexed_id is None:
                try:
                    reindexed_id = artifacts.skill_map.get(int(raw_s))
                except (ValueError, TypeError):
                    pass
            if reindexed_id is not None:
                pairs.append((raw_s, reindexed_id))

        if pairs:
            with torch.no_grad():
                scores = artifacts.model.predict_candidates(
                    skill_seq[:, :curr_len], correct_seq[:, :curr_len], mask[:, :curr_len],
                    response_time[:, :curr_len], attempt_count[:, :curr_len],
                    hint_used[:, :curr_len], difficulty_seq[:, :curr_len],
                    [(rid, 0) for _, rid in pairs],
                    edge_index=artifacts.edge_index, edge_weight=artifacts.edge_weight,
                )
            for i, (raw_s, _) in enumerate(pairs):
                val = float(scores[i].item())
                skill_mastery[str(raw_s)] = max(0.0, min(1.0, val))

    # Compute interaction counts and low-confidence flags per skill
    skill_interaction_counts: Dict[str, int] = {}
    for raw_id in raw_skill_ids:
        raw_id_str = str(raw_id)
        skill_interaction_counts[raw_id_str] = skill_interaction_counts.get(raw_id_str, 0) + 1

    low_confidence_skills = [
        s for s, cnt in skill_interaction_counts.items() if cnt < 3
    ]

    summary_text = generate_mastery_summary(
        skill_mastery=skill_mastery,
        skill_interaction_counts=skill_interaction_counts,
        low_confidence_skills=low_confidence_skills
    )

    return PredictResponse(
        skill_mastery=skill_mastery,
        num_interactions_processed=n_interactions,
        skill_interaction_counts=skill_interaction_counts,
        low_confidence_skills=low_confidence_skills,
        mastery_summary=summary_text
    )


@app.get("/health")
def health():
    return {
        "status": "healthy" if artifacts.is_loaded else "initializing",
        "num_skills": artifacts.num_skills,
        "max_seq_len": artifacts.max_seq_len,
        "rt_mean": artifacts.rt_mean,
        "rt_std": artifacts.rt_std,
    }


@app.get("/skills")
def get_skills():
    """Return list of valid raw skill IDs available in skill_map."""
    return {
        "num_skills": artifacts.num_skills,
        "skills": [str(k) for k in artifacts.skill_map.keys()]
    }


@app.get("/sections")
def get_sections():
    """v3: learner-level sections with their topics and validation label."""
    return {"sections": all_sections_info()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return _process_history(request.student_history)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    results = {}
    for student in request.students:
        results[student.student_id] = _process_history(student.student_history)
    return BatchPredictResponse(results=results)


@app.post("/session/start", response_model=StartSessionResponse)
def api_start_session(req: StartSessionRequest):
    st_id = req.student_id
    src = req.source
    # Earlier sessions' answers feed every prediction of the new session (D28).
    try:
        section = normalise_section(req.section)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    prior = session_manager.get_full_history(st_id.strip()) if st_id and st_id.strip() else []
    extra = {"section": section} if section else {}
    sid, st_id = adaptive_manager.start(student_id=st_id, prior_history=prior, **extra)
    session_manager.db.create_session(session_id=sid, student_id=st_id, set_name="adaptive", source=src)
    summary = session_manager.get_student_summary(st_id)
    sec = adaptive_manager.get(sid).section_payload()

    return StartSessionResponse(
        session_id=sid,
        student_id=st_id,
        questions=[],  # adaptive sessions do not generate the quiz up front
        prior_completed_sessions=summary["completed_session_count"],
        total_prior_interactions=summary["total_interactions"],
        **sec,
    )


def _get_adaptive_session(session_id: str) -> AdaptiveSession:
    try:
        return adaptive_manager.get(session_id)
    except KeyError:
        sess_row = session_manager.db.get_session(session_id)
        if not sess_row:
            raise
        interactions = session_manager.db.get_session_interactions(session_id)
        prior = [h for h in session_manager.get_full_history(sess_row["student_id"])
                 if h.get("session_id") != session_id]
        sess = AdaptiveSession.from_history(session_id, sess_row["student_id"], interactions,
                                            prior_history=prior)
        adaptive_manager.sessions[session_id] = sess
        return sess


@app.post("/session/{session_id}/next")
def api_next_question(session_id: str):
    """Returns the next adaptive question with a pre-answer predicted P(correct)."""
    try:
        _get_adaptive_session(session_id)
        q = adaptive_manager.next_question(session_id, predictor=_predictor)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")

    if q is None:
        # Session is over: mark it completed (so it counts as a finished session in
        # summaries and the class matrix) and return graph-based predictions for unasked topics.
        sess = adaptive_manager.get(session_id)
        session_manager.db.complete_session(session_id)
        unasked = sess.unasked_topics()
        predictions = {}
        if len(sess.context()) >= 2 and unasked and sess.uses_model:
            scores = sess._model_predictor(_predictor)(sess.context(), [(t, 2) for t in unasked])
            for t, s in zip(unasked, scores):
                predictions[str(t)] = s
        return {
            "session_id": session_id,
            "is_complete": True,
            "unasked_predictions": predictions,
            "main_questions": MAIN_QUESTIONS,
            "total_answered": len(sess.history),
            "total_questions": len(sess.history),
            "session_mode": sess.session_mode,
            **sess.section_payload(),
        }

    return {
        "session_id": session_id,
        "position": q["position"],
        "topic": q["topic"],
        "skill_name": q["skill_name"],
        "level": q["level"],
        "difficulty_bucket": q["difficulty_bucket"],
        "question_text": q["question_text"],
        "predicted_p": q["predicted_p"],
        "is_probe": q["is_probe"],
        "is_complete": False,
        # adaptive-v2 additive fields
        "total_questions": adaptive_manager.get(session_id).total_questions(),
        "session_mode": adaptive_manager.get(session_id).session_mode,
        "policy": adaptive_manager.get(session_id).policy,
        "decision": q.get("decision"),
        # v3 additive fields
        **adaptive_manager.get(session_id).section_payload(),
        "question_source": q.get("question_source", "bank"),
        "answer_type": q.get("answer_type", "numeric"),
        **({"choices": q["choices"]} if q.get("choices") else {}),
    }


@app.post("/session/{session_id}/answer")
def api_submit_answer(session_id: str, req: AnswerRequest):
    try:
        _get_adaptive_session(session_id)
        res = adaptive_manager.record_answer(
            session_id=session_id,
            submitted_answer=req.submitted_answer,
            response_time_ms=req.response_time_ms,
            hint_used=req.hint_used,
        )
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {
        "session_id": session_id,
        "position": res["position"],
        "correct": res["correct"],
        "difficulty_bucket": res["difficulty_bucket"],
        "level": res["level"],
        "is_probe": bool(res["is_probe"]),
        "is_complete": adaptive_manager.get(session_id).is_complete(),
    }


@app.get("/session/{session_id}/history")
def api_get_session_history(session_id: str):
    try:
        hist = session_manager.get_history(session_id)
        return {"session_id": session_id, "history": hist}
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")


@app.post("/session/{session_id}/predict", response_model=PredictResponse)
def api_predict_session(session_id: str):
    """Computes mastery for the session's student using their cumulative history."""
    try:
        sess = session_manager.get_session(session_id)
        student_id = sess["student_id"]
        hist = session_manager.get_full_history(student_id)
        if not hist:
            hist = session_manager.get_history(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")

    records = [InteractionRecord(**item) for item in hist]
    return _process_history(records)


@app.get("/student/{student_id}/summary", response_model=StudentSummaryResponse)
def api_student_summary(student_id: str):
    summary = session_manager.get_student_summary(student_id)
    return StudentSummaryResponse(**summary)


@app.post("/student/{student_id}/predict", response_model=PredictResponse)
def api_predict_student(student_id: str):
    """Computes mastery for a student from their entire cumulative history across all sessions."""
    hist = session_manager.get_full_history(student_id)
    if not hist:
        return PredictResponse(
            skill_mastery={},
            num_interactions_processed=0,
            message=f"No interaction history found for student '{student_id}'"
        )
    records = [InteractionRecord(**item) for item in hist]
    return _process_history(records)


@app.get("/student/{student_id}/trend")
def api_student_trend(student_id: str):
    """
    Task 9: Evaluates mastery trajectory across historical session completion boundaries.
    Recomputes cumulative mastery prediction using interactions up through each completed session.
    """
    clean_id = student_id.strip()
    boundaries = session_manager.db.get_student_session_boundaries(clean_id)
    if not boundaries:
        # Check if there is any uncompleted history
        hist = session_manager.get_full_history(clean_id)
        if not hist:
            return {
                "student_id": clean_id,
                "total_completed_sessions": 0,
                "total_interactions": 0,
                "trend": [],
                "message": f"No interaction history found for student '{clean_id}'"
            }
        records = [InteractionRecord(**item) for item in hist]
        pred_res = _process_history(records)
        return {
            "student_id": clean_id,
            "total_completed_sessions": 0,
            "total_interactions": len(hist),
            "trend": [{
                "session_index": 1,
                "session_id": "current_session",
                "completed_at": None,
                "cumulative_interactions": len(hist),
                "mastery": pred_res.skill_mastery,
                "skill_interaction_counts": pred_res.skill_interaction_counts,
                "low_confidence_skills": pred_res.low_confidence_skills,
            }],
            "message": "Student has 1 ongoing session. Complete more sessions to reveal a historical trend."
        }

    trend_points = []
    accumulated_interactions = []

    for idx, sess in enumerate(boundaries, 1):
        accumulated_interactions.extend(sess["interactions"])
        records = [InteractionRecord(**item) for item in accumulated_interactions]
        pred_res = _process_history(records)

        trend_points.append({
            "session_index": idx,
            "session_id": sess["session_id"],
            "set_name": sess["set_name"],
            "completed_at": sess["completed_at"],
            "cumulative_interactions": len(accumulated_interactions),
            "mastery": pred_res.skill_mastery,
            "skill_interaction_counts": pred_res.skill_interaction_counts,
            "low_confidence_skills": pred_res.low_confidence_skills,
        })

    return {
        "student_id": clean_id,
        "total_completed_sessions": len(boundaries),
        "total_interactions": len(accumulated_interactions),
        "trend": trend_points,
        "message": "Single session data point -- complete more sessions to reveal trajectory trend." if len(trend_points) == 1 else None
    }


@app.get("/students/matrix")
def api_students_matrix(source: str = "live"):
    """
    Task 7, 10, 11: Returns cumulative mastery predictions and data sufficiency
    counts for completed students in SQLite database, filtered by source.
    """
    completed_students = session_manager.db.get_all_completed_students(source=source)
    student_rows = []

    for st in completed_students:
        st_id = st["student_id"]
        hist = session_manager.get_full_history(st_id)
        counts = session_manager.db.get_student_skill_interaction_counts(st_id)
        records = [InteractionRecord(**item) for item in hist]
        pred_res = _process_history(records)
        student_rows.append({
            "student_id": st_id,
            "source": st.get("source", "live"),
            "completed_sessions": st["completed_sessions"],
            "total_interactions": len(hist),
            "mastery": pred_res.skill_mastery,
            "skill_interaction_counts": counts,
            "low_confidence_skills": [k for k, v in counts.items() if v < 3],
            "last_completed_at": st["last_completed_at"]
        })

    return {
        "student_count": len(student_rows),
        "source": source,
        "students": student_rows
    }


@app.get("/graph/subgraph")
def get_subgraph(student_id: Optional[str] = None, session_id: Optional[str] = None):
    """
    Task 8 & 10: Extracts PPMI subgraph for the 10 quiz skills with exact PPMI weights.
    If student_id or session_id is provided, includes per-node mastery, weak status (<50%),
    and interaction counts for that student.
    """
    target_student_id = student_id
    if session_id and not target_student_id:
        try:
            sess = session_manager.get_session(session_id)
            target_student_id = sess["student_id"]
        except KeyError:
            pass

    skills = session_manager.bank.skill_sequence

    # Calculate student mastery if student_id available
    student_mastery = {}
    skill_counts = {}
    has_completed_session = False
    has_history = False

    if target_student_id:
        summary = session_manager.get_student_summary(target_student_id)
        has_completed_session = summary["completed_session_count"] > 0
        hist = session_manager.get_full_history(target_student_id)
        if hist:
            has_history = True
            records = [InteractionRecord(**item) for item in hist]
            pred_res = _process_history(records)
            student_mastery = pred_res.skill_mastery
            skill_counts = session_manager.db.get_student_skill_interaction_counts(target_student_id)

    nodes = []
    for s in skills:
        raw_id = s["raw_skill_id"]
        raw_id_str = str(raw_id)
        node_data = {
            "id": raw_id,
            "raw_skill_id": raw_id,
            "name": s["skill_name"],
            "position": s["position"],
        }
        if has_history or has_completed_session:
            m_val = student_mastery.get(raw_id_str, 0.5)
            c_val = skill_counts.get(raw_id_str, 0)
            node_data["mastery"] = round(m_val, 4)
            node_data["is_weak"] = bool(m_val < 0.50)
            node_data["interaction_count"] = c_val
            node_data["is_low_data"] = bool(c_val < 3)

        nodes.append(node_data)

    edges = []
    if artifacts.adj_matrix is not None:
        for i, s1 in enumerate(skills):
            id1 = s1["raw_skill_id"]
            idx1 = artifacts.skill_map.get(id1)
            if idx1 is None:
                continue
            for j, s2 in enumerate(skills):
                if i >= j:
                    continue
                id2 = s2["raw_skill_id"]
                idx2 = artifacts.skill_map.get(id2)
                if idx2 is None:
                    continue
                weight = float(artifacts.adj_matrix[idx1, idx2])
                if weight > 0:
                    edges.append({
                        "source": id1,
                        "target": id2,
                        "weight": round(weight, 4)
                    })

    resp = {
        "nodes": nodes,
        "edges": edges,
        "student_id": target_student_id,
        "has_completed_session": has_completed_session,
        "has_history": has_history,
    }

    # Task 16: Connect weak-skill flagging (<50%) to the relation graph with PPMI cutoff
    PPMI_WEAK_RELATION_CUTOFF = 1.0  # Explicit PPMI weight cutoff for surfacing related curriculum weaknesses
    weak_skill_pairs = []
    weak_relational_callouts = []

    if has_history or has_completed_session:
        weak_skill_pairs = find_weak_skill_pairs(student_mastery, cutoff=PPMI_WEAK_RELATION_CUTOFF)
        for pair in weak_skill_pairs:
            weak_relational_callouts.append(
                f"{pair['skill1_name']} and {pair['skill2_name']} are both weak and are related topics (PPMI={pair['ppmi_weight']:.2f}) — consider reviewing them together."
            )
        summary_obj = session_manager.get_student_summary(target_student_id) if target_student_id else {}
        sess_cnt = summary_obj.get("completed_session_count", 1)
        resp["mastery_summary"] = generate_mastery_summary(student_mastery, session_count=sess_cnt)

    resp["weak_skill_pairs"] = weak_skill_pairs
    resp["weak_relational_callouts"] = weak_relational_callouts

    if target_student_id and not has_history and not has_completed_session:
        resp["message"] = "Complete a session to see your skill graph"

    return resp


@app.get("/session/{session_id}/gate")
def get_session_gate(session_id: str):
    """Computes gate activation values across time steps for the given session."""
    return get_session_gate_detail(session_id)


@app.get("/session/{session_id}/gate_detail")
def get_session_gate_detail(session_id: str):
    """
    Task 12: Detailed gate interpretability for a completed session.
    Computes per-dimension variation, top 5 most active dimensions, and temporal gate evolution.
    """
    try:
        sess = session_manager.get_session(session_id)
        student_id = sess["student_id"]
        hist = session_manager.get_history(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")

    if len(hist) < 3:
        return {
            "session_id": session_id,
            "student_id": student_id,
            "available": False,
            "num_interactions": len(hist),
            "message": "Not enough interactions yet for a detailed gate view (at least 3 required)."
        }

    max_len = artifacts.max_seq_len
    curr_len = min(len(hist), max_len)

    skill_seq = torch.zeros((1, max_len), dtype=torch.long)
    correct_seq = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.zeros((1, max_len), dtype=torch.float32)
    response_time = torch.zeros((1, max_len), dtype=torch.float32)
    attempt_count = torch.zeros((1, max_len), dtype=torch.float32)
    hint_used = torch.zeros((1, max_len), dtype=torch.float32)

    for i in range(curr_len):
        raw_id = hist[i]["skill_id"]
        reindexed_id = artifacts.skill_map.get(raw_id, 1)
        skill_seq[0, i] = reindexed_id
        correct_seq[0, i] = hist[i]["correct"]
        mask[0, i] = 1.0
        response_time[0, i] = float(hist[i]["response_time_ms"])
        attempt_count[0, i] = float(hist[i]["attempt_count"])
        hint_used[0, i] = float(hist[i]["hint_used"])

    with torch.no_grad():
        _, g = artifacts.model(
            skill_seq, correct_seq, mask, response_time, attempt_count, hint_used,
            edge_index=artifacts.edge_index, edge_weight=artifacts.edge_weight
        )

    real_len = curr_len - 1
    g_steps = g[0, :real_len].detach().cpu().numpy()

    per_dim_std = g_steps.std(axis=0)
    dim_order = np.argsort(-per_dim_std)
    top5_dims = [int(d) for d in dim_order[:5]]

    top5_series = {}
    for d in top5_dims:
        top5_series[f"dim_{d}"] = [round(float(v), 4) for v in g_steps[:, d]]

    step_means = [round(float(g_steps[t].mean()), 4) for t in range(real_len)]

    return {
        "session_id": session_id,
        "student_id": student_id,
        "available": True,
        "num_interactions": curr_len,
        "num_timesteps": real_len,
        "top5_dimensions": top5_dims,
        "top5_trajectories": top5_series,
        "step_gate_means": step_means,
        "mean_gate": round(float(g_steps.mean()), 4),
        "min_gate": round(float(g_steps.min()), 4),
        "max_gate": round(float(g_steps.max()), 4),
        "interpretation": "Values closer to 1.0 indicate reliance on Short-Term GRU (recent responses); values closer to 0.0 indicate reliance on Long-Term GRU & PPMI Graph context."
    }


@app.get("/session/{session_id}/gate_plot.png")
def get_session_gate_plot(session_id: str):
    """
    Task 12: Generates a 2-panel matplotlib interpretability figure for a completed session.
    """
    detail = get_session_gate_detail(session_id)
    if not detail.get("available", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail.get("message", "Insufficient interactions for gate plot")
        )

    try:
        sess = session_manager.get_session(session_id)
        hist = session_manager.get_history(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    max_len = artifacts.max_seq_len
    curr_len = min(len(hist), max_len)
    real_len = curr_len - 1

    skill_seq = torch.zeros((1, max_len), dtype=torch.long)
    correct_seq = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.zeros((1, max_len), dtype=torch.float32)
    response_time = torch.zeros((1, max_len), dtype=torch.float32)
    attempt_count = torch.zeros((1, max_len), dtype=torch.float32)
    hint_used = torch.zeros((1, max_len), dtype=torch.float32)

    for i in range(curr_len):
        raw_id = hist[i]["skill_id"]
        reindexed_id = artifacts.skill_map.get(raw_id, 1)
        skill_seq[0, i] = reindexed_id
        correct_seq[0, i] = hist[i]["correct"]
        mask[0, i] = 1.0
        response_time[0, i] = float(hist[i]["response_time_ms"])
        attempt_count[0, i] = float(hist[i]["attempt_count"])
        hint_used[0, i] = float(hist[i]["hint_used"])

    with torch.no_grad():
        _, g = artifacts.model(
            skill_seq, correct_seq, mask, response_time, attempt_count, hint_used,
            edge_index=artifacts.edge_index, edge_weight=artifacts.edge_weight
        )

    g_full = g[0, :real_len].detach().cpu().numpy()
    per_dim_std = g_full.std(axis=0)
    dim_order = np.argsort(-per_dim_std)
    g_sorted = g_full[:, dim_order]
    top5 = dim_order[:5]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [1.2, 1]})

    # Heatmap
    im = ax1.imshow(g_sorted.T, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Gate (1=Short-Term, 0=Long-Term & Graph)", fontsize=9)
    ax1.set_xlabel("Interaction Step (Transition t → t+1)", fontsize=9)
    ax1.set_ylabel("Hidden Dims (Most active on top)", fontsize=9)
    ax1.set_title(f"Per-Dimension Gate Behavior — Student '{sess['student_id']}' ({curr_len} interactions)", fontsize=11, fontweight="bold")

    # Line Plot of Top 5
    for d in top5:
        ax2.plot(range(1, real_len + 1), g_full[:, d], marker='o', markersize=4, alpha=0.8, label=f"Dim {d}")
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Equilibrium (0.5)")
    ax2.set_xlabel("Interaction Step (t → t+1)", fontsize=9)
    ax2.set_ylabel("Gate Activation", fontsize=9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Top 5 Most Temporally Active Gate Dimensions", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


# ---- adaptive-v2 additive endpoints (profile, overview, feedback) ----------
from serving.v2_routes import build_router as _build_v2_router
app.include_router(_build_v2_router(sys.modules[__name__]))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
