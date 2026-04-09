#!/usr/bin/env python3
"""Verify that the deployed Hugging Face Space is ready for submission."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPACE_URL = "https://santhakumar-k-2004-sre-bench.hf.space"
DEFAULT_RAW_URL = "https://huggingface.co/spaces/santhakumar-k-2004/sre-bench/raw/main/inference.py"
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
    parser.add_argument("--health-timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--task", default="task1", choices=["task1", "task2", "task3"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    health_url = f"{args.space_url.rstrip('/')}/health"

    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        wait_for_space_health(
            client,
            health_url,
            timeout_seconds=args.health_timeout,
            poll_seconds=args.poll_seconds,
        )
        ensure_raw_inference_is_hardened(client, args.raw_url)

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
