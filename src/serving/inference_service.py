"""
src/serving/inference_service.py
--------------------------------
FastAPI inference service wrapping the trained Gated Knowledge Tracing model.

Loads the model weights, PPMI graph, and response time normalization stats
ONCE at startup (never per-request).

SCOPE: Mastery tracker only -- takes a student's answer history and outputs
per-skill mastery estimates. Does not generate questions, adapt difficulty,
or select content.
"""

import sys
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status
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

# Base paths
PROJECT_ROOT = SRC_DIR.parent
WINDOWED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "windowed_data.pkl"
SKILL_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "skill_graph.pkl"
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "best_full_gated.pt"

# Fallback checkpoint if best_full_gated.pt is not directly named
FALLBACK_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "best_full_gated_seed123.pt"
SEED42_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "best_full_gated_seed42.pt"


# Pydantic Request / Response Models
class InteractionRecord(BaseModel):
    skill_id: Any = Field(..., description="Raw skill ID from skill_map (not re-indexed)")
    correct: int = Field(..., ge=0, le=1, description="Binary correctness (0 or 1)")
    response_time_ms: float = Field(..., ge=0.0, description="Response time in milliseconds")
    attempt_count: int = Field(..., ge=1, description="Number of attempts (>= 1)")
    hint_used: int = Field(..., ge=0, le=1, description="Whether hint was used (0 or 1)")


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
    message: Optional[str] = Field(
        default=None, description="Informational message for edge cases (e.g. insufficient data)"
    )


class BatchStudentItem(BaseModel):
    student_id: str
    student_history: List[InteractionRecord]


class BatchPredictRequest(BaseModel):
    students: List[BatchStudentItem]


class BatchPredictResponse(BaseModel):
    results: Dict[str, PredictResponse]


class StartSessionResponse(BaseModel):
    session_id: str
    questions: List[Dict[str, Any]]


class AnswerRequest(BaseModel):
    position: int = Field(..., ge=1, le=10)
    submitted_answer: Any
    response_time_ms: float = Field(..., ge=0.0)
    hint_used: int = Field(default=0, ge=0, le=1)


class AnswerResponse(BaseModel):
    session_id: str
    position: int
    correct: int
    attempt_count: int
    interactions_count: int
    is_complete: bool


class ModelArtifacts:
    """Singleton container holding model weights and static inference artifacts."""
    def __init__(self):
        self.model: Optional[GatedKTModel] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self.skill_map: Dict[Any, int] = {}
        self.reverse_skill_map: Dict[int, Any] = {}
        self.num_skills: int = 0
        self.max_seq_len: int = 500
        self.rt_mean: float = 8.5
        self.rt_std: float = 1.2
        self.is_loaded: bool = False

    def load(self):
        print("[InferenceService] Loading model artifacts once at startup...")

        # 1. Load windowed_data.pkl
        if not WINDOWED_DATA_PATH.exists():
            raise FileNotFoundError(f"Missing data file: {WINDOWED_DATA_PATH}")

        with open(WINDOWED_DATA_PATH, "rb") as f:
            data = pickle.load(f)

        self.num_skills = data["num_skills"]
        self.skill_map = data["skill_map"]
        self.max_seq_len = data.get("max_seq_len", 500)
        self.reverse_skill_map = {v: k for k, v in self.skill_map.items()}

        # 2. Compute RT stats strictly from TRAINING split (matches training setup)
        train_data = data["splits"]["train"]
        self.rt_mean, self.rt_std = compute_rt_stats(train_data)
        print(f"[InferenceService] RT Normalization: mean={self.rt_mean:.4f}, std={self.rt_std:.4f}")

        # 3. Load PPMI skill graph
        if not SKILL_GRAPH_PATH.exists():
            raise FileNotFoundError(f"Missing graph file: {SKILL_GRAPH_PATH}")

        with open(SKILL_GRAPH_PATH, "rb") as f:
            graph_data = pickle.load(f)

        adj_matrix = torch.tensor(graph_data["adjacency_matrix"], dtype=torch.float32)
        self.edge_index, self.edge_weight = adjacency_to_edge_index(adj_matrix)
        print(f"[InferenceService] Skill graph loaded: {self.edge_index.shape[1]} directed edges")

        # 4. Find checkpoint
        ckpt_path = MODEL_CHECKPOINT_PATH
        if not ckpt_path.exists():
            if FALLBACK_CHECKPOINT_PATH.exists():
                ckpt_path = FALLBACK_CHECKPOINT_PATH
            elif SEED42_CHECKPOINT_PATH.exists():
                ckpt_path = SEED42_CHECKPOINT_PATH
            else:
                raise FileNotFoundError(f"No checkpoint found at {MODEL_CHECKPOINT_PATH}")

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load artifacts once
    artifacts.load()
    yield
    # Shutdown clean up if needed


app = FastAPI(
    title="Intelligent Knowledge Tracing Inference Service",
    description="FastAPI service serving per-skill mastery predictions from GatedKTModel.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local backend proxy and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process_history(history: List[InteractionRecord]) -> PredictResponse:
    """Internal helper to process a single student history."""
    n_interactions = len(history)

    # Edge case: History < 2 interactions cannot produce next-step prediction
    if n_interactions < 2:
        return PredictResponse(
            skill_mastery={},
            num_interactions_processed=n_interactions,
            message="Insufficient data: at least 2 interactions are required to estimate mastery."
        )

    # Validate skill_ids against skill_map
    reindexed_skills = []
    raw_skill_ids = []
    for idx, item in enumerate(history):
        raw_id = item.skill_id
        # Skill map keys might be int or numpy int; normalize comparison
        matched_key = None
        if raw_id in artifacts.skill_map:
            matched_key = raw_id
        else:
            try:
                # Try integer conversion
                int_id = int(raw_id)
                if int_id in artifacts.skill_map:
                    matched_key = int_id
                else:
                    # Check numpy int keys
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

    # Truncate to max_seq_len if longer (keep latest interactions)
    max_len = artifacts.max_seq_len
    if len(history) > max_len:
        history = history[-max_len:]
        reindexed_skills = reindexed_skills[-max_len:]
        raw_skill_ids = raw_skill_ids[-max_len:]

    curr_len = len(history)

    # Construct model input tensors [batch=1, max_seq_len]
    skill_seq = torch.zeros((1, max_len), dtype=torch.long)
    correct_seq = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.zeros((1, max_len), dtype=torch.float32)
    response_time = torch.zeros((1, max_len), dtype=torch.float32)
    attempt_count = torch.zeros((1, max_len), dtype=torch.float32)
    hint_used = torch.zeros((1, max_len), dtype=torch.float32)

    for i in range(curr_len):
        skill_seq[0, i] = reindexed_skills[i]
        correct_seq[0, i] = history[i].correct
        mask[0, i] = 1.0
        response_time[0, i] = float(history[i].response_time_ms)
        attempt_count[0, i] = float(history[i].attempt_count)
        hint_used[0, i] = float(history[i].hint_used)

    with torch.no_grad():
        pred, _ = artifacts.model(
            skill_seq,
            correct_seq,
            mask,
            response_time,
            attempt_count,
            hint_used,
            artifacts.edge_index,
            artifacts.edge_weight,
        )

    # Map predictions back to (raw_skill_id, predicted_mastery)
    # pred has shape [1, max_len - 1].
    # Step t (0 <= t < curr_len - 1) predicts outcome for interaction t+1 (skill raw_skill_ids[t+1])
    skill_mastery: Dict[str, float] = {}
    for t in range(curr_len - 1):
        target_skill_raw = raw_skill_ids[t + 1]
        predicted_val = float(pred[0, t].item())
        # Clamping to valid [0, 1] range
        predicted_val = max(0.0, min(1.0, predicted_val))
        # Keep latest prediction if skill was seen multiple times
        skill_mastery[str(target_skill_raw)] = predicted_val

    # Ensure any skill present in the history has a mastery estimate.
    # If raw_skill_ids[0] was never queried at t >= 1 (e.g. 10 distinct skills),
    # compute its mastery state directly using the model components after interaction 0.
    all_seen_raw = [r for r in raw_skill_ids]
    for missing_raw in all_seen_raw:
        missing_str = str(missing_raw)
        if missing_str not in skill_mastery:
            all_skill_ids_tensor = torch.arange(artifacts.model.skill_emb.num_embeddings)
            all_skill_emb = artifacts.model.skill_emb(all_skill_ids_tensor)
            if artifacts.model.use_graph:
                graph_emb_table = artifacts.model.graph_encoder(
                    all_skill_emb, artifacts.edge_index, artifacts.edge_weight
                )
            else:
                graph_emb_table = all_skill_emb

            x_0 = artifacts.model.skill_emb(skill_seq[:, 0:1]) + artifacts.model.correct_emb(correct_seq[:, 0:1].long())
            hist_mask_0 = mask[:, 0:1]
            h_short_0 = artifacts.model.short_term(x_0, hist_mask_0)
            h_long_0 = artifacts.model.long_term(x_0, hist_mask_0)
            h_meta_0 = artifacts.model.metadata_encoder(
                response_time[:, 0:1], attempt_count[:, 0:1], hint_used[:, 0:1]
            )

            reindexed_id = artifacts.skill_map[missing_raw]
            h_graph_q = graph_emb_table[reindexed_id].unsqueeze(0).unsqueeze(0)
            if artifacts.model.use_gate:
                pred_q, _ = artifacts.model.gate_fusion(h_short_0, h_long_0, h_graph_q, h_meta_0)
            else:
                h_final_q = (h_short_0 + h_long_0 + h_graph_q) / 3.0
                pred_q = torch.sigmoid(artifacts.model.gate_fusion.output_head(h_final_q)).squeeze(-1)

            val_q = float(pred_q[0, 0].item())
            val_q = max(0.0, min(1.0, val_q))
            skill_mastery[missing_str] = val_q

    return PredictResponse(
        skill_mastery=skill_mastery,
        num_interactions_processed=n_interactions
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
def api_start_session():
    sid, questions = session_manager.start_session()
    return StartSessionResponse(session_id=sid, questions=questions)


@app.post("/session/{session_id}/answer", response_model=AnswerResponse)
def api_submit_answer(session_id: str, req: AnswerRequest):
    try:
        res = session_manager.record_answer(
            session_id=session_id,
            position=req.position,
            submitted_answer=req.submitted_answer,
            response_time_ms=req.response_time_ms,
            hint_used=req.hint_used
        )
        return AnswerResponse(**res)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/session/{session_id}/history")
def api_get_session_history(session_id: str):
    try:
        hist = session_manager.get_history(session_id)
        return {"session_id": session_id, "history": hist}
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")


@app.post("/session/{session_id}/predict", response_model=PredictResponse)
def api_predict_session(session_id: str):
    try:
        hist = session_manager.get_history(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")

    records = [InteractionRecord(**item) for item in hist]
    return _process_history(records)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
