"""
scripts/_services.py
--------------------
Start/stop the real FastAPI service and the Node proxy (cross-platform).
Used by scripts/e2e_http_check.py, scripts/visual_qa.py and scripts/start_demo.py.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, SRC_DIR, find_node


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http_json(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}


def wait_for(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


class Services:
    """Context manager: FastAPI (uvicorn) + Node proxy with the given env."""

    def __init__(self, env_overrides=None, py_port=None, node_port=None, log_dir=None, quiet=True):
        self.env = dict(os.environ)
        self.env.update({k: str(v) for k, v in (env_overrides or {}).items()})
        self.py_port = py_port or free_port()
        self.node_port = node_port or free_port()
        self.log_dir = Path(log_dir) if log_dir else None
        self.quiet = quiet
        self.procs = []

    @property
    def base(self):
        return f"http://127.0.0.1:{self.node_port}"

    def _out(self, name):
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            return open(self.log_dir / f"{name}.log", "w", encoding="utf-8")
        return subprocess.DEVNULL if self.quiet else None

    def start(self):
        py_env = dict(self.env)
        py_env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + py_env.get("PYTHONPATH", "")
        self.procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "serving.inference_service:app", "--host", "127.0.0.1",
             "--port", str(self.py_port)],
            cwd=str(SRC_DIR), env=py_env, stdout=self._out("fastapi"), stderr=subprocess.STDOUT))
        if not wait_for(f"http://127.0.0.1:{self.py_port}/health", 180):
            self.stop()
            raise RuntimeError("FastAPI did not start")
        node = find_node()
        if not node:
            self.stop()
            raise RuntimeError("node executable not found")
        node_env = dict(self.env)
        node_env["PORT"] = str(self.node_port)
        node_env["PYTHON_BASE_URL"] = f"http://127.0.0.1:{self.py_port}"
        self.procs.append(subprocess.Popen(
            [node, "src/index.js"], cwd=str(PROJECT_ROOT / "server"), env=node_env,
            stdout=self._out("node"), stderr=subprocess.STDOUT))
        if not wait_for(f"{self.base}/api/health", 60):
            self.stop()
            raise RuntimeError("Node proxy did not start")
        return self

    def stop(self):
        for p in reversed(self.procs):
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self.procs = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def temp_db_copy(dest_dir) -> Path:
    """SQLite-backup copy of the real DB into dest_dir (empty DB if none exists)."""
    import sqlite3
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dest_dir / "kt_sessions.db"
    real = PROJECT_ROOT / "data" / "processed" / "kt_sessions.db"
    dst = sqlite3.connect(str(dst_path))
    if real.exists():
        src = sqlite3.connect(str(real))
        src.backup(dst)
        src.close()
    dst.close()
    return dst_path
