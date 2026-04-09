"""Contract tests for inference.py submission-checklist compliance."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest

import inference as inference_module


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
            "--no-access-log",
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


@pytest.fixture
def server_entrypoint_url() -> str:
    """Start the canonical server entrypoint used by deployment smoke checks."""
    port = _find_free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["PYTHONPATH"] = "."
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "server.app",
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
        stdout = proc.stdout.read() if proc.stdout is not None else ""
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        pytest.fail(
            "Failed to start server.app entrypoint for inference smoke test.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _make_observation(
    title: str,
    service: str = "api-gateway",
    logs: list[str] | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "incident_id": "INC-TEST-0001",
        "alert_payload": {
            "title": title,
            "service": service,
            "severity": "P1",
            "timestamp": "2026-04-08T10:00:00Z",
        },
        "service_topology": {
            "nodes": [service, "auth-service", "payment-service"],
            "edges": [[service, "auth-service"]],
        },
        "logs": logs or [f"{service} generic diagnostic log"],
        "metrics": metrics or {
            "cpu": 25.0,
            "memory": 35.0,
            "latency_ms": 125.0,
            "error_rate": 0.5,
        },
        "action_history": [],
        "step_number": 0,
        "incident_resolved": False,
    }


def _make_mock_agent(handler) -> inference_module.SREAgent:
    agent = inference_module.SREAgent(api_url="http://testserver", hf_token=None)
    agent.client.close()
    agent.client = httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=inference_module.HTTP_TIMEOUT,
    )
    return agent


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


def test_waits_for_cold_start_before_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"health": 0, "reset": 0}
    observation = _make_observation("[P1] Deployment Failure on auth-service", service="auth-service")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            calls["health"] += 1
            status_code = 503 if calls["health"] < 3 else 200
            payload = {"status": "starting"} if status_code != 200 else {"status": "ok"}
            return httpx.Response(status_code, json=payload)
        if request.url.path == "/reset":
            calls["reset"] += 1
            return httpx.Response(200, json=observation)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    monkeypatch.setattr(inference_module.time, "sleep", lambda *_args, **_kwargs: None)
    agent = _make_mock_agent(handler)
    try:
        result = agent.reset_episode("task1", seed=42)
    finally:
        agent.close()

    assert result["ok"] is True
    assert calls["health"] == 3
    assert calls["reset"] == 1


def test_reset_retries_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"reset": 0}
    observation = _make_observation("[P1] Db Overload on order-service", service="order-service")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/reset":
            calls["reset"] += 1
            if calls["reset"] == 1:
                return httpx.Response(503, json={"error": "warming_up"})
            return httpx.Response(200, json=observation)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    monkeypatch.setattr(inference_module.time, "sleep", lambda *_args, **_kwargs: None)
    agent = _make_mock_agent(handler)
    try:
        result = agent.reset_episode("task1", seed=7)
    finally:
        agent.close()

    assert result["ok"] is True
    assert calls["reset"] == 2


def test_reset_handles_connection_errors_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"reset": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/reset":
            calls["reset"] += 1
            raise httpx.ConnectError("connection refused", request=request)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    monkeypatch.setattr(inference_module.time, "sleep", lambda *_args, **_kwargs: None)
    agent = _make_mock_agent(handler)
    try:
        result = agent.reset_episode("task1", seed=11)
    finally:
        agent.close()

    assert result["ok"] is False
    assert result["error"] == "REQUEST_ERROR"
    assert "ConnectError" in result["message"]
    assert calls["reset"] == inference_module.MAX_RETRIES


def test_reset_failure_emits_start_and_end_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    port = _find_free_port()

    class FailingResetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/reset":
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"reset_failed","message":"boom"}')
                return
            self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), FailingResetHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = "."
    env.pop("HF_TOKEN", None)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "inference.py",
                "--task",
                "task1",
                "--seed",
                "42",
                "--quiet",
                "--url",
                f"http://127.0.0.1:{port}",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert result.returncode == 0
    assert lines[0].startswith("[START]")
    assert lines[-1] == "[END] success=false steps=0 score=0.00 rewards=0.00"


def test_server_entrypoint_smoke_runs_inference_without_crashing(server_entrypoint_url: str) -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = "."
    env.pop("HF_TOKEN", None)

    result = subprocess.run(
        [
            sys.executable,
            "inference.py",
            "--task",
            "task1",
            "--seed",
            "42",
            "--quiet",
            "--url",
            server_entrypoint_url,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert result.returncode == 0
    assert lines[0].startswith("[START]")
    assert lines[-1].startswith("[END]")


def test_title_based_family_inference_ignores_red_herring_logs() -> None:
    observation = _make_observation(
        "[P2] Network Partition on auth-service",
        service="auth-service",
        logs=[
            "Deploy v3.1.0 started successfully",
            "Config reload triggered at 2026-04-08 10:00:00",
            "ERROR auth-service: TCP RST received from api-gateway",
        ],
        metrics={
            "cpu": 22.0,
            "memory": 48.0,
            "latency_ms": 18250.0,
            "error_rate": 51.0,
        },
    )

    assert inference_module.infer_incident_family(observation) == "network_partition"


def test_task3_deterministic_planner_handles_all_hard_seeds(api_server_url: str) -> None:
    agent = inference_module.SREAgent(
        api_url=api_server_url,
        model=inference_module.MODEL_NAME,
        hf_token=None,
    )

    try:
        for seed in range(30):
            result = agent.run_episode(
                task="task3",
                seed=seed,
                verbose=False,
            )
            assert result["steps_used"] <= inference_module.TASK_STEP_LIMITS["task3"]
            assert result["final_score"] >= 0.15
    finally:
        agent.close()
