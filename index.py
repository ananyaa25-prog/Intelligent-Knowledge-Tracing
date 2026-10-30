"""
Root entrypoint for Vercel Python runtime service.
Exposes the FastAPI application instance from src.serving.inference_service.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from serving.inference_service import app
