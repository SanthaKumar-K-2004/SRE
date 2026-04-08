"""Tests for the IncidentStateMachine: preconditions, transitions, loop detection."""

import pytest

from models import ActionType, IncidentData, IncidentType, Severity
from state_machine import IncidentStateMachine, PreconditionError


def _make_incident(**overrides) -> IncidentData:
    """Create a test incident with sensible defaults."""
    defaults = {
        "incident_id": "INC-TEST-000001",
        "difficulty": "easy",
        "incident_type": IncidentType.CPU_SPIKE,
        "alert_payload": {
            "title": "[P2] CPU Spike on api-gateway",
            "service": "api-gateway",
            "severity": "P2",
            "timestamp": "2026-04-06T12:00:00Z",
            "runbook_url": "https://runbooks.internal/cpu-spike",
        },
        "service_topology": {
            "nodes": ["api-gateway", "auth-service", "payment-service", "database-primary"],
            "edges": [
                ["api-gateway", "auth-service"],
                ["api-gateway", "payment-service"],
                ["payment-service", "database-primary"],
            ],
        },
        "logs": [f"[2026-04-06 12:00:{i:02d}] ERROR api-gateway: CPU at 95%" for i in range(50)],
        "metrics": {"cpu": 95.0, "memory": 50.0, "latency_ms": 200.0, "error_rate": 5.0},
        "gold_root_cause": "Runaway process on api-gateway consuming excessive CPU",
        "gold_triggered_by": "api-gateway",
        "gold_affected_chain": ["api-gateway", "auth-service"],
        "gold_action_sequence": [
            "acknowledge_alert", "inspect_logs", "check_metrics",
            "restart_service", "verify_endpoint", "resolve",
        ],
        "red_herrings": [],
    }
    defaults.update(overrides)
    return IncidentData(**defaults)


class TestPreconditions:
    def test_inspect_logs_no_precondition(self):
        sm = IncidentStateMachine(_make_incident())
        result = sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        assert result["status"] == "success"

    def test_check_metrics_no_precondition(self):
        sm = IncidentStateMachine(_make_incident())
        result = sm.execute_action(ActionType.CHECK_METRICS, "api-gateway")
        assert result["status"] == "success"

    def test_restart_requires_inspect_logs(self):
        sm = IncidentStateMachine(_make_incident())
        with pytest.raises(PreconditionError) as exc_info:
            sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        assert "inspect_logs" in str(exc_info.value)

    def test_restart_succeeds_after_inspect(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        result = sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        assert result["status"] in ("success", "partial")

    def test_verify_requires_remediation(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        with pytest.raises(PreconditionError) as exc_info:
            sm.execute_action(ActionType.VERIFY_ENDPOINT, "api-gateway")
        assert "remediation" in str(exc_info.value).lower()

    def test_verify_succeeds_after_remediation(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        result = sm.execute_action(ActionType.VERIFY_ENDPOINT, "api-gateway")
        assert result["status"] in ("success", "failure")

    def test_resolve_requires_verification(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        with pytest.raises(PreconditionError) as exc_info:
            sm.execute_action(ActionType.RESOLVE)
        assert "verify" in str(exc_info.value).lower()

    def test_resolve_succeeds_after_full_workflow(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        sm.execute_action(ActionType.VERIFY_ENDPOINT, "api-gateway")
        result = sm.execute_action(ActionType.RESOLVE)
        assert result["status"] == "success"
        assert sm.is_done

    def test_scale_up_requires_metrics(self):
        sm = IncidentStateMachine(_make_incident())
        with pytest.raises(PreconditionError):
            sm.execute_action(ActionType.SCALE_UP, "api-gateway")

    def test_scale_up_blocked_for_wrong_type(self):
        incident = _make_incident(incident_type=IncidentType.CONFIG_ERROR)
        sm = IncidentStateMachine(incident)
        sm.execute_action(ActionType.CHECK_METRICS, "api-gateway")
        with pytest.raises(PreconditionError):
            sm.execute_action(ActionType.SCALE_UP, "api-gateway")

    def test_rollback_requires_logs(self):
        incident = _make_incident(incident_type=IncidentType.DEPLOYMENT_FAILURE)
        sm = IncidentStateMachine(incident)
        with pytest.raises(PreconditionError):
            sm.execute_action(ActionType.ROLLBACK_DEPLOY, "api-gateway")

    def test_rollback_succeeds_for_deploy_failure(self):
        incident = _make_incident(incident_type=IncidentType.DEPLOYMENT_FAILURE)
        sm = IncidentStateMachine(incident)
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        result = sm.execute_action(ActionType.ROLLBACK_DEPLOY, "api-gateway")
        assert result["status"] == "success"


class TestStateTransitions:
    def test_initial_state(self):
        sm = IncidentStateMachine(_make_incident())
        assert not sm.is_done
        assert sm.state.step_count == 0
        assert sm.state.action_history == []
        assert sm.state.system_state["api-gateway"] == "degraded"

    def test_step_count_increments(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        assert sm.state.step_count == 1
        sm.execute_action(ActionType.CHECK_METRICS, "api-gateway")
        assert sm.state.step_count == 2

    def test_action_history_tracked(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.ACKNOWLEDGE_ALERT)
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        assert len(sm.state.action_history) == 2
        assert "acknowledge_alert" in sm.state.action_history[0]

    def test_diagnosis_done_after_inspect(self):
        sm = IncidentStateMachine(_make_incident())
        assert not sm.state.diagnosis_done
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        assert sm.state.diagnosis_done

    def test_remediation_applied_after_restart(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        assert not sm.state.remediation_applied
        sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        assert sm.state.remediation_applied

    def test_service_goes_healthy_after_correct_remediation(self):
        sm = IncidentStateMachine(_make_incident())
        assert sm.state.system_state["api-gateway"] == "degraded"
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        assert sm.state.system_state["api-gateway"] == "healthy"

    def test_cannot_act_after_resolve(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        sm.execute_action(ActionType.RESTART_SERVICE, "api-gateway")
        sm.execute_action(ActionType.VERIFY_ENDPOINT, "api-gateway")
        sm.execute_action(ActionType.RESOLVE)

        assert sm.is_done
        is_valid, reason = sm.check_precondition(ActionType.INSPECT_LOGS)
        assert not is_valid


class TestLoopDetection:
    def test_no_loop_on_first_action(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        assert not sm.detect_action_loop()

    def test_loop_detected_on_repeat(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.CHECK_METRICS, "api-gateway")
        sm.execute_action(ActionType.CHECK_METRICS, "api-gateway")
        assert sm.detect_action_loop()

    def test_no_loop_on_different_actions(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.INSPECT_LOGS, "api-gateway")
        sm.execute_action(ActionType.CHECK_METRICS, "api-gateway")
        assert not sm.detect_action_loop()


class TestCascadePropagation:
    def test_cascade_path_from_primary(self):
        sm = IncidentStateMachine(_make_incident())
        path = sm.get_cascade_path("api-gateway")
        assert "api-gateway" in path
        assert len(path) >= 1

    def test_cascade_path_unknown_service(self):
        sm = IncidentStateMachine(_make_incident())
        path = sm.get_cascade_path("nonexistent-service")
        assert path == ["nonexistent-service"]


class TestGetStateDict:
    def test_state_dict_keys(self):
        sm = IncidentStateMachine(_make_incident())
        state = sm.get_state_dict()
        expected_keys = {
            "incident_id", "step_count", "action_history",
            "system_state", "diagnosis_done", "remediation_applied",
            "verification_done", "episode_done", "cumulative_reward",
            "alert_acknowledged",
        }
        assert expected_keys.issubset(set(state.keys()))

    def test_state_dict_reflects_actions(self):
        sm = IncidentStateMachine(_make_incident())
        sm.execute_action(ActionType.ACKNOWLEDGE_ALERT)
        state = sm.get_state_dict()
        assert state["alert_acknowledged"] is True
        assert state["step_count"] == 1
