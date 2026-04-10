"""Integration tests for FastAPI endpoints using TestClient."""

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def test_dataset():
    """Create a minimal dataset for API testing."""
    incidents = []
    for diff in ("easy", "medium", "hard"):
        for i in range(3):
            inc_type = ["cpu_spike", "memory_leak", "deployment_failure"][i]
            incidents.append({
                "incident_id": f"INC-{diff.upper()}-API{i:02d}",
                "difficulty": diff,
                "incident_type": inc_type,
                "alert_payload": {
                    "title": f"[P2] {inc_type} on api-gateway",
                    "service": "api-gateway",
                    "severity": "P2",
                    "timestamp": "2026-04-06T12:00:00Z",
                    "runbook_url": "",
                },
                "service_topology": {
                    "nodes": ["api-gateway", "auth-service", "payment-service"],
                    "edges": [["api-gateway", "auth-service"]],
                },
                "logs": [f"log line {j}" for j in range(50)],
                "metrics": {"cpu": 92.0, "memory": 50.0, "latency_ms": 200.0, "error_rate": 5.0},
                "gold_root_cause": f"Root cause for {inc_type}",
                "gold_triggered_by": "api-gateway",
                "gold_affected_chain": ["api-gateway", "auth-service"],
                "gold_action_sequence": [
                    "acknowledge_alert", "inspect_logs", "check_metrics",
                    "restart_service", "verify_endpoint", "resolve",
                ],
                "red_herrings": ["red herring"] if diff == "hard" else [],
            })

    tmp_dir = tempfile.mkdtemp()
    data_path = os.path.join(tmp_dir, "incidents.json")
    with open(data_path, "w") as f:
        json.dump(incidents, f)

    yield data_path

    os.remove(data_path)
    os.rmdir(tmp_dir)


@pytest.fixture(scope="module")
def client(test_dataset):
    """Create a TestClient with test dataset."""
    # Override the global env before importing app
    import api as api_module
    from environment import SREBenchEnv

    api_module.env = SREBenchEnv(data_path=test_dataset)

    with TestClient(api_module.app) as c:
        yield c




class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_status_and_docs(self, client):
        response = client.get("/")
        data = response.json()
        assert data["status"] == "ok"
        assert data["docs"] == "/docs"
        assert data["tasks"] == "/tasks"

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"


class TestResetEndpoint:
    def test_reset_without_body_returns_200(self, client):
        response = client.post("/reset")
        assert response.status_code == 200

    def test_reset_null_body_returns_200(self, client):
        response = client.post(
            "/reset",
            data="null",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200

    def test_reset_returns_200(self, client):
        response = client.post("/reset", json={"task": "task1", "seed": 0})
        assert response.status_code == 200

    def test_reset_accepts_task_id_alias(self, client):
        response = client.post("/reset", json={"task_id": "task1", "seed": 0})
        assert response.status_code == 200

    def test_reset_returns_observation(self, client):
        response = client.post("/reset", json={"task": "task1", "seed": 0})
        data = response.json()
        assert "incident_id" in data
        assert "alert_payload" in data
        assert "logs" in data
        assert "metrics" in data
        assert "service_topology" in data
        assert data["step_number"] == 0
        assert data["incident_resolved"] is False

    def test_reset_task1(self, client):
        response = client.post("/reset", json={"task": "task1", "seed": 0})
        data = response.json()
        assert "EASY" in data["incident_id"]

    def test_reset_task2(self, client):
        response = client.post("/reset", json={"task": "task2", "seed": 0})
        data = response.json()
        assert "MEDIUM" in data["incident_id"]

    def test_reset_task3(self, client):
        response = client.post("/reset", json={"task": "task3", "seed": 0})
        data = response.json()
        assert "HARD" in data["incident_id"]

    def test_reset_deterministic(self, client):
        r1 = client.post("/reset", json={"task": "task1", "seed": 42})
        r2 = client.post("/reset", json={"task": "task1", "seed": 42})
        assert r1.json()["incident_id"] == r2.json()["incident_id"]

    def test_reset_invalid_task(self, client):
        response = client.post("/reset", json={"task": "task99"})
        assert response.status_code == 422


class TestStepEndpoint:
    def test_step_returns_200(self, client):
        client.post("/reset", json={"task": "task2", "seed": 0})
        response = client.post("/step", json={
            "action_type": "inspect_logs",
            "target_service": "api-gateway",
        })
        assert response.status_code == 200

    def test_step_returns_reward(self, client):
        client.post("/reset", json={"task": "task2", "seed": 0})
        response = client.post("/step", json={
            "action_type": "inspect_logs",
            "target_service": "api-gateway",
        })
        data = response.json()
        assert "value" in data
        assert "cumulative" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_in_range(self, client):
        client.post("/reset", json={"task": "task2", "seed": 0})
        response = client.post("/step", json={
            "action_type": "check_metrics",
            "target_service": "api-gateway",
        })
        data = response.json()
        assert -1.0 <= data["value"] <= 1.0

    def test_step_invalid_action(self, client):
        client.post("/reset", json={"task": "task2", "seed": 0})
        response = client.post("/step", json={
            "action_type": "nonexistent_action",
        })
        assert response.status_code == 422

    def test_step_before_reset(self, client):
        # Create a fresh client scenario by resetting state
        import api as api_module
        api_module.env._initialized = False

        response = client.post("/step", json={
            "action_type": "inspect_logs",
        })
        assert response.status_code == 400
        assert "NOT_INITIALIZED" in response.json()["detail"]["error"]

        # Restore
        client.post("/reset", json={"task": "task1", "seed": 0})

    def test_step_precondition_failure(self, client):
        client.post("/reset", json={"task": "task2", "seed": 0})
        response = client.post("/step", json={
            "action_type": "restart_service",
            "target_service": "api-gateway",
        })
        data = response.json()
        assert data["info"]["precondition_failed"] is True

    def test_complete_episode_via_api(self, client):
        client.post("/reset", json={"task": "task3", "seed": 0})

        actions = [
            {"action_type": "acknowledge_alert"},
            {"action_type": "inspect_logs", "target_service": "api-gateway"},
            {"action_type": "check_metrics", "target_service": "api-gateway"},
            {"action_type": "check_service", "target_service": "api-gateway"},
            {"action_type": "restart_service", "target_service": "api-gateway"},
            {"action_type": "verify_endpoint", "target_service": "api-gateway"},
            {"action_type": "resolve"},
        ]

        last_response = None
        for action in actions:
            response = client.post("/step", json=action)
            assert response.status_code == 200
            last_response = response.json()
            if last_response["done"]:
                break

        assert last_response["done"] is True
        assert "final_score" in last_response["info"]
        assert "grader_score" in last_response["info"]


class TestStateEndpoint:
    def test_state_returns_200(self, client):
        client.post("/reset", json={"task": "task1", "seed": 0})
        response = client.get("/state")
        assert response.status_code == 200

    def test_state_returns_dict(self, client):
        client.post("/reset", json={"task": "task1", "seed": 0})
        response = client.get("/state")
        data = response.json()
        assert "incident_id" in data
        assert "step_count" in data
        assert "observation" in data

    def test_state_before_reset(self, client):
        import api as api_module
        api_module.env._initialized = False

        response = client.get("/state")
        assert response.status_code == 400

        # Restore
        client.post("/reset", json={"task": "task1", "seed": 0})


class TestTasksEndpoint:
    def test_tasks_returns_200(self, client):
        response = client.get("/tasks")
        assert response.status_code == 200

    def test_tasks_returns_three_tasks_with_graders(self, client):
        response = client.get("/tasks")
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3
        assert {"task1", "task2", "task3"}.issubset({task["id"] for task in data})
        assert data[0]["grader"]["module"] == "tasks.manifest_graders"
        assert "function" in data[0]["grader"]


class TestOpenAPIDocs:
    def test_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "SRE-Bench"
        assert "/reset" in schema["paths"]
        assert "/step" in schema["paths"]
        assert "/state" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/tasks" in schema["paths"]
