from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import check_space_release as release_check


def test_wait_for_space_health_retries_until_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"health": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["health"] += 1
        if calls["health"] < 3:
            return httpx.Response(503, json={"status": "starting"})
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=release_check.HTTP_TIMEOUT)
    monkeypatch.setattr(release_check.time, "sleep", lambda *_args, **_kwargs: None)

    try:
        release_check.wait_for_space_health(
            client,
            "https://example-space.hf.space/health",
            timeout_seconds=1.0,
            poll_seconds=0.0,
        )
    finally:
        client.close()

    assert calls["health"] == 3


def test_ensure_raw_inference_is_hardened_rejects_stale_code() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="resp.raise_for_status()")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=release_check.HTTP_TIMEOUT)
    try:
        with pytest.raises(RuntimeError, match="stale raise_for_status"):
            release_check.ensure_raw_inference_is_hardened(
                client,
                "https://huggingface.co/spaces/example/raw/main/inference.py",
            )
    finally:
        client.close()


def test_ensure_raw_manifest_has_three_task_graders_accepts_valid_manifest() -> None:
    manifest = """
name: sre-bench
tasks:
  - id: task1
    grader:
      module: tasks.manifest_graders
      function: grade_task1_manifest
  - id: task2
    grader:
      module: tasks.manifest_graders
      function: grade_task2_manifest
  - id: task3
    grader:
      module: tasks.manifest_graders
      function: grade_task3_manifest
"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=manifest)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=release_check.HTTP_TIMEOUT)
    try:
        release_check.ensure_raw_manifest_has_three_task_graders(
            client,
            "https://huggingface.co/spaces/example/raw/main/openenv.yaml",
        )
    finally:
        client.close()


def test_ensure_raw_manifest_has_three_task_graders_rejects_bad_manifest() -> None:
    manifest = """
name: sre-bench
tasks:
  - id: task1
    grader:
      module: tasks.manifest_graders
      function: grade_task1_manifest
  - id: task2
    grader:
      module: tasks.manifest_graders
"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=manifest)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=release_check.HTTP_TIMEOUT)
    try:
        with pytest.raises(RuntimeError, match="at least 3 task entries"):
            release_check.ensure_raw_manifest_has_three_task_graders(
                client,
                "https://huggingface.co/spaces/example/raw/main/openenv.yaml",
            )
    finally:
        client.close()


def test_ensure_live_tasks_endpoint_has_three_task_graders_accepts_valid_payload() -> None:
    payload = [
        {"id": "task1", "grader": {"module": "tasks.manifest_graders", "function": "grade_task1_manifest"}},
        {"id": "task2", "grader": {"module": "tasks.manifest_graders", "function": "grade_task2_manifest"}},
        {"id": "task3", "grader": {"module": "tasks.manifest_graders", "function": "grade_task3_manifest"}},
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=release_check.HTTP_TIMEOUT)
    try:
        release_check.ensure_live_tasks_endpoint_has_three_task_graders(
            client,
            "https://example-space.hf.space/tasks",
        )
    finally:
        client.close()


def test_ensure_live_tasks_endpoint_has_three_task_graders_rejects_bad_payload() -> None:
    payload = {"tasks": [{"id": "task1", "grader": {"module": "tasks.manifest_graders"}}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=release_check.HTTP_TIMEOUT)
    try:
        with pytest.raises(RuntimeError, match="at least 3 tasks with graders"):
            release_check.ensure_live_tasks_endpoint_has_three_task_graders(
                client,
                "https://example-space.hf.space/tasks",
            )
    finally:
        client.close()


def test_run_remote_smoke_accepts_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python", "inference.py"],
            returncode=0,
            stdout=(
                "[START] task=task1 env=sre-bench model=test-model\n"
                "[STEP] step=1 action=inspect_logs('api-gateway') reward=0.05 done=true error=null\n"
                "[END] success=true steps=1 score=1.00 rewards=0.05\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(release_check.subprocess, "run", fake_run)

    release_check.run_remote_smoke(
        python_executable="python",
        repo_root=Path("."),
        space_url="https://example-space.hf.space",
        task="task1",
        seed=42,
    )


def test_run_remote_smoke_rejects_non_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python", "inference.py"],
            returncode=0,
            stdout="oops\n",
            stderr="",
        )

    monkeypatch.setattr(release_check.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Remote inference smoke failed"):
        release_check.run_remote_smoke(
            python_executable="python",
            repo_root=Path("."),
            space_url="https://example-space.hf.space",
            task="task1",
            seed=42,
        )
