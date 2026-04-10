"""
SRE-Bench: FastAPI application.

Endpoints:
    POST /reset - Initialize new episode
    POST /step  - Execute action
    GET  /state - Get current state
    GET  /health - Health check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment import SREBenchEnv, get_task_catalog
from models import (
    HealthResponse,
    ResetRequest,
    SREAction,
    SREObservation,
    SREReward,
    TaskDescriptor,
)


logger = logging.getLogger(__name__)
env: SREBenchEnv | None = None


def _create_env() -> SREBenchEnv:
    """Create the canonical environment instance."""
    return SREBenchEnv()


def initialize_env() -> SREBenchEnv:
    """Create the process-wide environment during startup."""
    global env
    if env is None:
        env = _create_env()
    return env


def get_env() -> SREBenchEnv:
    """Return the initialized environment instance."""
    if env is None:
        raise RuntimeError("Environment not initialized. Startup did not complete.")
    return env


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Bootstrap the environment eagerly and fail fast on startup errors."""
    global env
    environment = None
    try:
        environment = initialize_env()
        logger.info("SRE-Bench environment loaded successfully")
        yield
    except Exception:
        logger.exception("Failed to initialize SRE-Bench environment during startup")
        raise
    finally:
        if environment is not None:
            environment.close()
        env = None


app = FastAPI(
    title="SRE-Bench",
    description=(
        "The first RL environment that trains AI to be the on-call engineer. "
        "OpenEnv-compliant environment with deterministic grading."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint for deployment platforms and quick manual checks.

    Returns a compact API status payload with useful links.
    """
    return {
        "name": "SRE-Bench",
        "status": "ok",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
        "openapi": "/openapi.json",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint required by deployment validators."""
    get_env()
    return HealthResponse(status="ok")


@app.get("/tasks", response_model=list[TaskDescriptor], tags=["Environment"])
async def list_tasks() -> list[TaskDescriptor]:
    """Expose published task metadata for validators that enumerate tasks over HTTP."""
    return [TaskDescriptor.model_validate(task) for task in get_task_catalog()]


@app.post("/reset", response_model=SREObservation, tags=["Environment"])
async def reset_episode(request: Optional[ResetRequest] = None) -> SREObservation:
    """
    Initialize a new episode.

    - task: Task difficulty level (task1=easy, task2=medium, task3=hard)
    - seed: Optional deterministic seed for reproducible incident selection

    Returns the initial observation containing alert payload, logs, metrics,
    and service topology.
    """
    try:
        environment = get_env()
        payload = request or ResetRequest()
        observation = environment.reset(
            task=payload.task.value,
            seed=payload.seed,
        )
        return observation
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATASET_NOT_FOUND",
                "message": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_TASK",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Failed to reset environment: {exc}",
            },
        ) from exc


@app.post("/step", response_model=SREReward, tags=["Environment"])
async def execute_step(action: SREAction) -> SREReward:
    """
    Execute an action in the current episode.

    - action_type: One of 10 valid actions (inspect_logs, check_metrics, etc.)
    - target_service: Optional target service name
    - parameters: Optional additional parameters

    Returns the reward, done flag, and info dict.
    """
    environment = get_env()

    if not environment._initialized:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "NOT_INITIALIZED",
                "message": "Call POST /reset before POST /step",
            },
        )

    try:
        reward = environment.step(action)
        return reward
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "EPISODE_DONE",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Step execution failed: {exc}",
            },
        ) from exc


@app.get("/state", tags=["Environment"])
async def get_state() -> Dict[str, Any]:
    """
    Get the full current environment state.

    Returns a dictionary containing episode state, observation,
    action history, and system health status.
    """
    environment = get_env()

    if not environment._initialized:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "NOT_INITIALIZED",
                "message": "Call POST /reset before GET /state",
            },
        )

    return environment.state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
