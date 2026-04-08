"""Tests for SREBenchEnv core environment lifecycle."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from models import ActionType, SREAction, TaskType
from environment import SREBenchEnv


@pytest.fixture
def dataset_path():
    """Create a minimal dataset for testing."""
    incidents = []

    # Generate 3 incidents per difficulty (minimum for testing)
    for diff in ("easy", "medium", "hard"):
        for i in range(3):
            inc_type = ["cpu_spike", "memory_leak", "deployment_failure"][i]
            incidents.append({
                "incident_id": f"INC-{diff.upper()}-TEST{i:02d}",
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
                "red_herrings": ["red herring 1"] if diff == "hard" else [],
            })

    # Write to temp file
    tmp_dir = tempfile.mkdtemp()
    data_path = os.path.join(tmp_dir, "incidents.json")
    with open(data_path, "w") as f:
        json.dump(incidents, f)

    yield data_path

    # Cleanup
    os.remove(data_path)
    os.rmdir(tmp_dir)


@pytest.fixture
def env(dataset_path):
    """Create an environment with test dataset."""
    environment = SREBenchEnv(data_path=dataset_path)
    yield environment
    environment.close()


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task="task1", seed=0)
        assert obs.incident_id is not None
        assert len(obs.incident_id) > 0
        assert obs.step_number == 0
        assert obs.incident_resolved is False
        assert obs.action_history == []

    def test_reset_task1_selects_easy(self, env):
        obs = env.reset(task="task1", seed=0)
        assert "EASY" in obs.incident_id

    def test_reset_task2_selects_medium(self, env):
        obs = env.reset(task="task2", seed=0)
        assert "MEDIUM" in obs.incident_id

    def test_reset_task3_selects_hard(self, env):
        obs = env.reset(task="task3", seed=0)
        assert "HARD" in obs.incident_id

    def test_reset_deterministic_with_seed(self, env):
        obs1 = env.reset(task="task1", seed=42)
        id1 = obs1.incident_id

        obs2 = env.reset(task="task1", seed=42)
        id2 = obs2.incident_id

        assert id1 == id2

    def test_reset_different_seeds_different_incidents(self, env):
        obs1 = env.reset(task="task1", seed=0)
        obs2 = env.reset(task="task1", seed=1)
        # With only 3 incidents, seeds 0 and 1 should give different incidents
        assert obs1.incident_id != obs2.incident_id

    def test_reset_has_logs(self, env):
        obs = env.reset(task="task1", seed=0)
        assert len(obs.logs) == 50

    def test_reset_has_metrics(self, env):
        obs = env.reset(task="task1", seed=0)
        assert obs.metrics.cpu > 0
        assert obs.metrics.memory > 0

    def test_reset_has_topology(self, env):
        obs = env.reset(task="task1", seed=0)
        assert len(obs.service_topology.nodes) >= 2

    def test_reset_clears_previous_episode(self, env):
        env.reset(task="task1", seed=0)
        env.step(SREAction(action_type=ActionType.INSPECT_LOGS))
        
        obs = env.reset(task="task1", seed=0)
        assert obs.step_number == 0
        assert obs.action_history == []


class TestStep:
    def test_step_returns_reward(self, env):
        env.reset(task="task2", seed=0)
        reward = env.step(SREAction(action_type=ActionType.INSPECT_LOGS))
        assert reward.value is not None
        assert isinstance(reward.done, bool)

    def test_step_increments_step_number(self, env):
        env.reset(task="task2", seed=0)
        env.step(SREAction(action_type=ActionType.INSPECT_LOGS))
        state = env.state()
        assert state["step_count"] == 1

    def test_step_tracks_action_history(self, env):
        env.reset(task="task2", seed=0)
        env.step(SREAction(action_type=ActionType.INSPECT_LOGS, target_service="api-gateway"))
        state = env.state()
        assert len(state["action_history"]) == 1

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(SREAction(action_type=ActionType.INSPECT_LOGS))

    def test_step_after_done_returns_done(self, env):
        env.reset(task="task1", seed=0)
        # Task 1 has max 1 step
        env.step(SREAction(action_type=ActionType.INSPECT_LOGS))
        reward = env.step(SREAction(action_type=ActionType.CHECK_METRICS))
        assert reward.done is True

    def test_precondition_failure_handled(self, env):
        env.reset(task="task2", seed=0)
        # Try restart without inspect_logs
        reward = env.step(SREAction(action_type=ActionType.RESTART_SERVICE, target_service="api-gateway"))
        assert reward.info.get("precondition_failed") is True

    def test_full_episode_workflow(self, env):
        env.reset(task="task3", seed=0)

        actions = [
            SREAction(action_type=ActionType.ACKNOWLEDGE_ALERT),
            SREAction(action_type=ActionType.INSPECT_LOGS, target_service="api-gateway"),
            SREAction(action_type=ActionType.CHECK_METRICS, target_service="api-gateway"),
            SREAction(action_type=ActionType.CHECK_SERVICE, target_service="api-gateway"),
            SREAction(action_type=ActionType.RESTART_SERVICE, target_service="api-gateway"),
            SREAction(action_type=ActionType.VERIFY_ENDPOINT, target_service="api-gateway"),
            SREAction(action_type=ActionType.RESOLVE),
        ]

        for action in actions:
            reward = env.step(action)
            if reward.done:
                break

        assert reward.done is True
        assert "final_score" in reward.info

    def test_reward_values_in_range(self, env):
        env.reset(task="task2", seed=0)
        for action_type in [ActionType.INSPECT_LOGS, ActionType.CHECK_METRICS, ActionType.CHECK_SERVICE]:
            reward = env.step(SREAction(action_type=action_type, target_service="api-gateway"))
            assert -1.0 <= reward.value <= 1.0
            if reward.done:
                break


class TestState:
    def test_state_returns_dict(self, env):
        env.reset(task="task1", seed=0)
        state = env.state()
        assert isinstance(state, dict)
        assert "incident_id" in state
        assert "step_count" in state

    def test_state_without_reset(self, env):
        state = env.state()
        assert "error" in state

    def test_state_includes_observation(self, env):
        env.reset(task="task1", seed=0)
        state = env.state()
        assert "observation" in state

    def test_state_reflects_actions(self, env):
        env.reset(task="task2", seed=0)
        env.step(SREAction(action_type=ActionType.INSPECT_LOGS, target_service="api-gateway"))
        state = env.state()
        assert state["diagnosis_done"] is True
        assert state["step_count"] == 1


class TestClose:
    def test_close_resets_state(self, env):
        env.reset(task="task1", seed=0)
        env.close()
        assert not env._initialized

    def test_close_allows_new_reset(self, env):
        env.reset(task="task1", seed=0)
        env.close()
        obs = env.reset(task="task1", seed=0)
        assert obs.incident_id is not None


class TestMaxSteps:
    def test_task1_max_1_step(self, env):
        env.reset(task="task1", seed=0)
        r1 = env.step(SREAction(action_type=ActionType.INSPECT_LOGS))
        assert r1.done is True

    def test_task2_max_8_steps(self, env):
        env.reset(task="task2", seed=0)
        for i in range(8):
            action_types = [
                ActionType.ACKNOWLEDGE_ALERT,
                ActionType.INSPECT_LOGS,
                ActionType.CHECK_METRICS,
                ActionType.CHECK_SERVICE,
                ActionType.CHECK_TOPOLOGY,
                ActionType.RESTART_SERVICE,
                ActionType.VERIFY_ENDPOINT,
                ActionType.RESOLVE,
            ]
            reward = env.step(SREAction(
                action_type=action_types[i],
                target_service="api-gateway",
            ))
            if reward.done:
                break
        assert reward.done is True


class TestDatasetLoading:
    def test_missing_dataset_raises(self):
        with pytest.raises(FileNotFoundError):
            SREBenchEnv(data_path="/nonexistent/path/incidents.json")
