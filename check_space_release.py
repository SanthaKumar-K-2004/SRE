#!/usr/bin/env python3
"""Verify that the deployed Hugging Face Space is ready for submission."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPACE_URL = "https://santhakumar-k-2004-sre-bench.hf.space"
DEFAULT_RAW_URL = "https://huggingface.co/spaces/santhakumar-k-2004/sre-bench/raw/main/inference.py"
DEFAULT_RAW_MANIFEST_URL = "https://huggingface.co/spaces/santhakumar-k-2004/sre-bench/raw/main/openenv.yaml"
DEFAULT_HEALTH_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_SECONDS = 5.0
HTTP_TIMEOUT = httpx.Timeout(20.0, connect=10.0)


def _snippet(response: httpx.Response) -> str:
    """Return a compact response snippet for diagnostics."""
    try:
        body = response.text.strip().replace("\n", " ").replace("\r", " ")
    except Exception as exc:  # pragma: no cover - defensive
        return f"unavailable_response_body({exc.__class__.__name__}: {exc})"
    return body[:200] or response.reason_phrase or "empty_response"


def wait_for_space_health(
    client: httpx.Client,
    health_url: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    """Poll the live Space until /health returns the expected payload."""
    deadline = time.monotonic() + timeout_seconds
    last_error = "health check not started"

    while time.monotonic() < deadline:
        try:
            response = client.get(health_url)
            if response.status_code == 200:
                payload = response.json()
                if payload.get("status") == "ok":
                    print(f"[PASS] Space health is ready: {health_url}")
                    return
                last_error = f"unexpected health payload: {payload}"
            else:
                last_error = f"HTTP {response.status_code}: {_snippet(response)}"
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"

        print(f"[WAIT] Space health not ready yet: {last_error}")
        time.sleep(poll_seconds)

    raise RuntimeError(
        f"Space /health never returned {{'status': 'ok'}} within {timeout_seconds:.0f}s. "
        f"Last error: {last_error}"
    )


def ensure_raw_inference_is_hardened(client: httpx.Client, raw_url: str) -> None:
    """Ensure the deployed Space main branch no longer contains the stale failure path."""
    response = client.get(raw_url)
    response.raise_for_status()
    if "raise_for_status(" in response.text:
        raise RuntimeError("Space raw inference.py still contains stale raise_for_status calls.")
    print(f"[PASS] Space raw inference.py is hardened: {raw_url}")


def _grader_reference_is_valid(value: Any) -> bool:
    if isinstance(value, str):
        return ":" in value
    if isinstance(value, dict):
        module = value.get("module")
        function = value.get("function")
        return isinstance(module, str) and bool(module) and isinstance(function, str) and bool(function)
    return False


def _extract_tasks(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [task for task in payload if isinstance(task, dict)]
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            return [task for task in tasks if isinstance(task, dict)]
    return []


def _validate_tasks_payload(payload: Any, *, source: str) -> int:
    tasks = _extract_tasks(payload)

    if len(tasks) < 3:
        raise RuntimeError(
            f"{source} does not expose at least 3 tasks with graders. Found {len(tasks)}"
        )

    invalid = [task.get("id", "<unknown>") for task in tasks if not _grader_reference_is_valid(task.get("grader"))]
    if invalid:
        raise RuntimeError(
            f"{source} returned tasks with invalid grader metadata: " + ", ".join(invalid)
        )

    return len(tasks)


def ensure_raw_manifest_has_three_task_graders(client: httpx.Client, manifest_url: str) -> None:
    """Ensure the deployed manifest advertises >=3 tasks with grader references."""
    response = client.get(manifest_url)
    response.raise_for_status()
    manifest = yaml.safe_load(response.text) or {}
    count = _validate_tasks_payload(manifest, source=f"Space openenv.yaml ({manifest_url})")
    print(f"[PASS] Space openenv.yaml has {count} task graders: {manifest_url}")


def ensure_live_tasks_endpoint_has_three_task_graders(
    client: httpx.Client,
    tasks_url: str,
    *,
    timeout_seconds: float = 0.0,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> None:
    """Ensure the deployed API exposes >=3 task entries with grader metadata."""
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    last_error = "tasks check not started"

    while True:
        try:
            response = client.get(tasks_url)
            response.raise_for_status()
            count = _validate_tasks_payload(response.json(), source=f"Space /tasks ({tasks_url})")
            print(f"[PASS] Space /tasks exposes {count} task graders: {tasks_url}")
            return
        except Exception as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                last_error = f"HTTP {exc.response.status_code}: {_snippet(exc.response)}"
            else:
                last_error = f"{exc.__class__.__name__}: {exc}"

        if timeout_seconds <= 0 or time.monotonic() >= deadline:
            break

        print(f"[WAIT] Space /tasks not ready yet: {last_error}")
        time.sleep(poll_seconds)

    raise RuntimeError(
        f"Space /tasks never exposed 3 valid task graders within {timeout_seconds:.0f}s. "
        f"Last error: {last_error}"
    )


def run_remote_smoke(
    *,
    python_executable: str,
    repo_root: Path,
    space_url: str,
    task: str,
    seed: int,
) -> None:
    """Run the submission inference script against the live Space."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(repo_root)
    env.pop("HF_TOKEN", None)

    result = subprocess.run(
        [
            python_executable,
            "inference.py",
            "--task",
            task,
            "--seed",
            str(seed),
            "--quiet",
            "--url",
            space_url,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    smoke_ok = (
        result.returncode == 0
        and bool(lines)
        and lines[0].startswith("[START]")
        and lines[-1].startswith("[END]")
    )
    if not smoke_ok:
        raise RuntimeError(
            "Remote inference smoke failed.\n"
            f"returncode={result.returncode}\n"
            f"stdout={result.stdout[-4000:]}\n"
            f"stderr={result.stderr[-4000:]}"
        )

    print(f"[PASS] Remote inference smoke succeeded: {lines[-1]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate live Hugging Face Space submission readiness.")
    parser.add_argument("--space-url", default=DEFAULT_SPACE_URL, help="Base URL for the deployed Space.")
    parser.add_argument("--raw-url", default=DEFAULT_RAW_URL, help="Raw main-branch inference.py URL.")
    parser.add_argument(
        "--manifest-url",
        default=DEFAULT_RAW_MANIFEST_URL,
        help="Raw main-branch openenv.yaml URL.",
    )
    parser.add_argument("--health-timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--task", default="task1", choices=["task1", "task2", "task3"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    health_url = f"{args.space_url.rstrip('/')}/health"
    tasks_url = f"{args.space_url.rstrip('/')}/tasks"

    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        wait_for_space_health(
            client,
            health_url,
            timeout_seconds=args.health_timeout,
            poll_seconds=args.poll_seconds,
        )
        ensure_raw_inference_is_hardened(client, args.raw_url)
        ensure_raw_manifest_has_three_task_graders(client, args.manifest_url)
        ensure_live_tasks_endpoint_has_three_task_graders(
            client,
            tasks_url,
            timeout_seconds=args.health_timeout,
            poll_seconds=args.poll_seconds,
        )

    run_remote_smoke(
        python_executable=sys.executable,
        repo_root=REPO_ROOT,
        space_url=args.space_url,
        task=args.task,
        seed=args.seed,
    )
    print("[PASS] Live Space release readiness checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
