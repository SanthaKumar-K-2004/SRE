---
title: SRE-Bench
sdk: docker
app_port: 7860
pinned: false
---

# SRE-Bench

SRE-Bench is a deterministic reinforcement learning environment for training and evaluating AI agents on on-call incident response workflows.

It simulates the full incident lifecycle:
- receive and acknowledge alerts
- diagnose with logs and metrics
- apply remediation actions
- verify service recovery
- resolve incidents

## Table of Contents
- [What You Get](#what-you-get)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Pre-Submission Checklist (5/5)](#pre-submission-checklist-55)
- [Run Full Project Verification](#run-full-project-verification)
- [GitHub to Hugging Face Sync](#github-to-hugging-face-sync)
- [API Reference](#api-reference)
- [Action Space](#action-space)
- [Tasks and Scoring](#tasks-and-scoring)
- [Project Structure](#project-structure)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What You Get
- OpenEnv-style lifecycle: `reset()`, `step()`, `state()`, `close()`
- 3 task tiers with increasing difficulty:
  - `task1`: alert classification
  - `task2`: root cause analysis
  - `task3`: full remediation
- 90 synthetic incidents (`30` per tier) generated in `data/incidents.json`
- 10 discrete actions with enforced preconditions
- Deterministic grading (no LLM-as-judge variance)
- FastAPI service for local/dev and container deployment

## Requirements
- Python `3.11+` (tested with Python `3.12`)
- `pip`

## Quick Start

## Space URLs

After deployment on Hugging Face Spaces:
- Root: `/`
- Health: `/health`
- Docs: `/docs`

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd SRE-BENCH-main
```

### 2. Create and activate a virtual environment

PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Bash:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the synthetic dataset
```bash
python generate_dataset.py
```

This creates `data/incidents.json`.

### 5. Run tests

PowerShell:
```powershell
$env:PYTHONPATH='.'
pytest --cov=. tests/
```

Bash:
```bash
PYTHONPATH=. pytest --cov=. tests/
```

### 6. Start the API
```bash
uvicorn api:app --reload --port 7860
```

Open:
- Swagger docs: `http://localhost:7860/docs`
- Health check: `http://localhost:7860/health`

### 7. Run the baseline agent
```bash
python inference.py --task task1 --seed 42
python inference.py --task task2 --seed 42
python inference.py --task task3 --seed 42
python inference.py --all-tasks --seed 42
```

Optional/required environment variables for `inference.py`:
- `ENV_BASE_URL` (default: `http://localhost:7860`) for environment endpoints (`/health`, `/reset`, `/step`)
- `API_BASE_URL` (no default) for evaluator LiteLLM/OpenAI proxy base URL
- `API_KEY` (no default) for evaluator LiteLLM/OpenAI proxy API key
- `MODEL_NAME` (default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `HF_TOKEN` (optional compatibility fallback only when `API_KEY` is absent)
- `LOCAL_IMAGE_NAME` (optional; only needed for `from_docker_image()` workflows)

If `API_BASE_URL` or `API_KEY` is missing, `inference.py` falls back to a deterministic rule-based agent.

## Pre-Submission Checklist (5/5)

Use this mapping to satisfy the submission portal checks:

1. Sample `inference.py` flow followed
   - `inference.py` uses env-configured API + model routing and a deterministic agent loop.
2. Environment variables present in `inference.py`
   - `ENV_BASE_URL`, `API_BASE_URL`, `API_KEY`, `MODEL_NAME`, `HF_TOKEN`, optional `LOCAL_IMAGE_NAME`.
3. Defaults set only for `ENV_BASE_URL` and `MODEL_NAME`
   - `API_BASE_URL`, `API_KEY`, and `HF_TOKEN` intentionally have no default values.
4. LLM calls use OpenAI client configured via variables
   - `from openai import OpenAI` in `inference.py` and `OpenAI(base_url=API_BASE_URL, api_key=API_KEY)`.
5. Stdout structured logs exact
   - `python inference.py --task task3 --seed 42 --quiet` prints only `[START]`, `[STEP]`, `[END]` lines.

## Run Full Project Verification

Use `verify_sre_bench.py` for end-to-end checks across models, dataset logic, graders, API behavior, and deployment readiness.

PowerShell:
```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
python verify_sre_bench.py
```

Bash:
```bash
PYTHONUTF8=1 PYTHONPATH=. python verify_sre_bench.py
```

Targeted gate checks:

```bash
PYTHONUTF8=1 PYTHONPATH=. python verify_sre_bench.py --gate phase1
PYTHONUTF8=1 PYTHONPATH=. python verify_sre_bench.py --gate phase2
```

Final remote submission guard:

```bash
bash validate-submission.sh
```

PowerShell equivalent:

```powershell
.\validate-submission.ps1
```

This final guard expects GitHub `main` and Hugging Face Space `main` to already be in sync.

## GitHub to Hugging Face Sync

To prevent GitHub/Space drift, this repo uses `.github/workflows/sync-space.yml`:

1. Push to `main`.
2. Workflow runs `pytest -q` and `python verify_sre_bench.py`.
3. If both pass, workflow mirrors `main` to `https://huggingface.co/spaces/santhakumar-k-2004/sre-bench` using `--force-with-lease`.

Required repository secret (one of the following):

- `HF_SPACE_WRITE_TOKEN` (preferred): Hugging Face token with write access to `santhakumar-k-2004/sre-bench`.
- `HF_TOKEN` (fallback): same write access; used only when `HF_SPACE_WRITE_TOKEN` is not configured.

If neither secret is configured, the workflow still runs validation but skips Space sync and emits a warning.

Optional strict mode:

- Set repository variable `REQUIRE_SPACE_SYNC=true` to fail the workflow when neither token secret is configured.
- Leave it unset (or set to `false`) to keep default non-blocking validation mode.

Submission flow (recommended):

1. Run local gates: `pytest -q`, `python verify_sre_bench.py --gate phase1`, `python verify_sre_bench.py --gate phase2`, `python verify_sre_bench.py`.
2. `git push origin main`
3. Wait for GitHub Actions job `sync-space` to succeed.
4. Run `bash validate-submission.sh` (Linux/macOS) or `.\validate-submission.ps1` (PowerShell)
5. Confirm the live Space is healthy at `https://santhakumar-k-2004-sre-bench.hf.space/health`.
6. Confirm the raw Space code is updated at `https://huggingface.co/spaces/santhakumar-k-2004/sre-bench/raw/main/inference.py`.
7. Let the team lead click `Update submission`.

Evaluator LLM criteria note:

- The evaluator expects at least one LLM call through injected `API_BASE_URL` + `API_KEY`.
- Do not bypass proxy routing with hardcoded provider URLs or alternate credentials.

What `validate-submission.sh` enforces:

1. Runs the current local gates and the full verifier.
2. Confirms `origin/main` and Space `main` point to the same commit.
3. Waits for Space `/health` to return `200 {"status":"ok"}`.
4. Confirms remote `inference.py` no longer contains stale `raise_for_status(`.
5. Runs `python inference.py --task task1 --seed 42 --quiet --url https://santhakumar-k-2004-sre-bench.hf.space`.

Troubleshooting note:

- GitHub failure emails may reference older commits (for example `695bb05`) even after a newer commit run is green. Always check the latest workflow run on `main` by timestamp and commit SHA.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Get current environment state |
| `GET` | `/docs` | Swagger UI |

### `POST /reset`
Request:
```json
{
  "task": "task1",
  "seed": 42
}
```

### `POST /step`
Request:
```json
{
  "action_type": "inspect_logs",
  "target_service": "api-gateway"
}
```

## Action Space

| Action | Purpose | Key Preconditions |
| --- | --- | --- |
| `inspect_logs` | Inspect service logs | None |
| `check_metrics` | Check CPU/memory/latency/error rate | None |
| `check_service` | Check service health status | None |
| `restart_service` | Restart a service | Must call `inspect_logs` first |
| `scale_up` | Increase replicas | Must call `check_metrics` first; only valid for capacity-style incidents |
| `rollback_deploy` | Roll back deployment | Must call `inspect_logs` first; only valid for deploy/config incidents |
| `verify_endpoint` | Verify remediation success | Remediation must be applied first |
| `resolve` | Close the incident | Must call `verify_endpoint` first |
| `check_topology` | Inspect dependency graph | None |
| `acknowledge_alert` | Acknowledge alert receipt | None |

## Tasks and Scoring

| Task | Difficulty | Max Steps | Grading Focus |
| --- | --- | --- | --- |
| `task1` | Easy | `1` | Classification accuracy (`incident_type`, `severity`, `primary_fault_service`) |
| `task2` | Medium | `8` | Root cause quality, affected chain quality, step efficiency |
| `task3` | Hard | `15` | LCS action-sequence match, behavior penalties/bonuses, verification completeness |

All graders return scores in the range `[0.0, 1.0]`.

## Project Structure

```text
.
+-- api.py
+-- environment.py
+-- generate_dataset.py
+-- inference.py
+-- models.py
+-- rewards.py
+-- state_machine.py
+-- verify_sre_bench.py
+-- validate-submission.sh
+-- Dockerfile
+-- requirements.txt
+-- README.md
+-- data/
|   +-- incidents.json
+-- tasks/
|   +-- __init__.py
|   +-- task1.py
|   +-- task2.py
|   +-- task3.py
+-- tests/
    +-- test_api.py
    +-- test_environment.py
    +-- test_graders.py
    +-- test_models.py
    +-- test_state_machine.py
```

## Docker

Build:
```bash
docker build -t sre-bench .
```

Run:
```bash
docker run -p 7860:7860 sre-bench
```

Test:
```bash
curl http://localhost:7860/health
```

## Troubleshooting

- `Dataset not found ... run generate_dataset.py first`
  - Run `python generate_dataset.py`.
- `ModuleNotFoundError: No module named 'models'` when running pytest
  - Run tests with `PYTHONPATH=.` (or `$env:PYTHONPATH='.'` on PowerShell).
- Unicode print errors on Windows (for scripts with symbols)
  - Run with `PYTHONUTF8=1` (or `$env:PYTHONUTF8='1'` on PowerShell).
- Port conflict on `7860`
  - Stop the existing process or use another port: `uvicorn api:app --port <new_port>`.

## License

No `LICENSE` file is currently present in this repository. Add one before publishing or distributing.

