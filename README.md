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
- [Run Full Project Verification](#run-full-project-verification)
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

Optional environment variables:
- `HF_TOKEN` (for Hugging Face Router model calls)
- `HF_MODEL` (override default model)

If `HF_TOKEN` is not set, `inference.py` falls back to a rule-based agent.

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
