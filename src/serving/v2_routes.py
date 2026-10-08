"""
src/serving/v2_routes.py
------------------------
Additive endpoints for the adaptive-v2 upgrade. Registered on the existing
FastAPI app by inference_service.py (build_router(service_module)), so no
existing route, field or signature changes.

  GET  /session/{id}/profile   mastery / readiness profile (contract B)
  GET  /student/{id}/profile   same, for a student id
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status

from serving.adaptive_session import TOPICS
from serving.mastery import build_profile


class ProfileCache:
    """Per-student profile cache, invalidated whenever the student answers."""

    def __init__(self):
        self._store: Dict[str, Any] = {}

    def get(self, key: str, n_interactions: int):
        hit = self._store.get(key)
        if hit and hit[0] == n_interactions:
            return hit[1]
        return None

    def put(self, key: str, n_interactions: int, value):
        self._store[key] = (n_interactions, value)

    def invalidate_student(self, student_id: str):
        for k in [k for k in self._store if k.split("|", 1)[0] == student_id]:
            self._store.pop(k, None)


profile_cache = ProfileCache()


def _safe_predictor(svc):
    def predictor(history, candidates):
        return svc._predictor(history, candidates)
    return predictor


def student_profile(svc, student_id: str, session_id: Optional[str] = None,
                    use_cache: bool = True) -> Dict[str, Any]:
    """Profile from the student's cumulative history (same source as /session/{id}/predict)."""
    db = svc.session_manager.db
    full = db.get_full_history(student_id)
    if session_id is not None:
        if not full:
            full = db.get_session_interactions(session_id)
        sess_hist = db.get_session_interactions(session_id)
    else:
        last_sid = full[-1].get("session_id") if full else None
        sess_hist = [h for h in full if h.get("session_id") == last_sid] if last_sid else []
    key = f"{student_id}|{session_id or ''}"
    if use_cache:
        cached = profile_cache.get(key, len(full))
        if cached is not None:
            return cached
    prof = build_profile(
        full_history=full,
        topics=[{"raw_skill_id": t["raw_skill_id"], "skill_name": t["skill_name"]} for t in TOPICS],
        predictor=_safe_predictor(svc),
        session_history=sess_hist,
        session_id=session_id,
        student_id=student_id,
    )
    prof["n_interactions"] = len(full)
    if use_cache:
        profile_cache.put(key, len(full), prof)
    return prof


def build_router(svc) -> APIRouter:
    router = APIRouter()

    # Invalidate cached profiles when a new answer is stored.
    try:
        svc.adaptive_manager.on_answer = profile_cache.invalidate_student
    except Exception:
        pass

    @router.get("/session/{session_id}/profile")
    def api_session_profile(session_id: str):
        row = svc.session_manager.db.get_session(session_id)
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found")
        return student_profile(svc, row["student_id"], session_id=session_id)

    @router.get("/student/{student_id}/profile")
    def api_student_profile(student_id: str):
        return student_profile(svc, student_id.strip(), session_id=None)

    return router
