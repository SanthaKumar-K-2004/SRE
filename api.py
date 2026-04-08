"""
SRE-Bench: FastAPI Application
HTTP interface for the SRE-Bench reinforcement learning environment.

Endpoints:
    POST /reset  — Initialize new episode
    POST /step   — Execute action
    GET  /state  — Get current state
    GET  /health — Health check
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment import SREBenchEnv
from models import (
    HealthResponse,
    ResetRequest,
    SREAction,
    SREObservation,
    SREReward,
)


# ─── Application Setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="SRE-Bench",
    description=(
        "The first RL environment that trains AI to be the on-call engineer. "
        "OpenEnv-compliant environment with deterministic grading."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Environment Instance ───────────────────────────────────────────────

env: SREBenchEnv | None = None


def get_env() -> SREBenchEnv:
    """Get or create the global environment instance."""
    global env
    if env is None:
        env = SREBenchEnv()
    return env


# ─── Endpoints ──────────────────────────────────────────────────────────────────


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
        "openapi": "/openapi.json",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint — required by HuggingFace Spaces."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=SREObservation, tags=["Environment"])
async def reset_episode(request: Optional[ResetRequest] = None) -> SREObservation:
    """
    Initialize a new episode.

    - **task**: Task difficulty level (task1=easy, task2=medium, task3=hard)
    - **seed**: Optional deterministic seed for reproducible incident selection

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
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATASET_NOT_FOUND",
                "message": str(e),
            },
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_TASK",
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Failed to reset environment: {str(e)}",
            },
        )


@app.post("/step", response_model=SREReward, tags=["Environment"])
async def execute_step(action: SREAction) -> SREReward:
    """
    Execute an action in the current episode.

    - **action_type**: One of 10 valid actions (inspect_logs, check_metrics, etc.)
    - **target_service**: Optional target service name
    - **parameters**: Optional additional parameters

    Returns the reward (per-step and cumulative), done flag, and info dict.
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
    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "EPISODE_DONE",
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"Step execution failed: {str(e)}",
            },
        )


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


# ─── Startup Event ──────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Pre-load the dataset on application startup."""
    try:
        get_env()
        print("✅ SRE-Bench environment loaded successfully")
    except FileNotFoundError:
        print("⚠️  Dataset not found. Run 'python generate_dataset.py' first.")
    except Exception as e:
        print(f"⚠️  Error loading environment: {e}")


# ─── Main Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
