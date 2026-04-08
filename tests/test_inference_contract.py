"""Contract tests for inference.py submission-checklist compliance."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = REPO_ROOT / "inference.py"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def api_server_url() -> str:
    """Start a local uvicorn server for subprocess-level inference checks."""
    port = _find_free_port()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    health_url = f"http://127.0.0.1:{port}/health"
    started = False
    for _ in range(60):
        if proc.poll() is not None:
            break
        try:
            response = httpx.get(health_url, timeout=1.0)
            if response.status_code == 200:
                started = True
                break
        except Exception:
            pass
        time.sleep(0.25)

    if not started:
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        pytest.fail(f"Failed to start local API server for inference test. stderr:\n{stderr}")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_inference_env_declarations_are_strict() -> None:
    """Ensure env-var declarations match strict submission checklist style."""
    content = INFERENCE_PATH.read_text(encoding="utf-8")

    assert 'API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")' in content
    assert 'MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")' in content
    assert 'HF_TOKEN = os.getenv("HF_TOKEN")' in content
    assert 'LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")' in content


def test_inference_quiet_stdout_is_structured_only(api_server_url: str) -> None:
    """Run inference.py via subprocess and validate strict START/STEP/END stdout."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = "."
    env.pop("HF_TOKEN", None)

    cmd = [
        sys.executable,
        "inference.py",
        "--task",
        "task3",
        "--seed",
        "42",
        "--quiet",
        "--url",
        api_server_url,
    ]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines, "Expected structured stdout lines but got empty output"
    assert lines[0].startswith("[START]")
    assert any(line.startswith("[STEP]") for line in lines)
    assert lines[-1].startswith("[END]")

    allowed_prefixes = ("[START]", "[STEP]", "[END]")
    unexpected = [line for line in lines if not line.startswith(allowed_prefixes)]
    assert not unexpected, f"Found non-structured stdout lines: {unexpected}"
