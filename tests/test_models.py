"""Tests for Pydantic models and enum validation."""

import pytest
from models import (
    ActionType,
    AlertPayload,
    EpisodeState,
    HealthResponse,
    IncidentData,
    IncidentType,
    ResetRequest,
    ServiceMetrics,
    ServiceTopology,
    Severity,
    SREAction,
    SREObservation,
    SREReward,
    TaskType,
)


# ─── Enum Tests ─────────────────────────────────────────────────────────────────


class TestEnums:
    def test_action_type_values(self):
        assert ActionType.INSPECT_LOGS.value == "inspect_logs"
        assert ActionType.RESTART_SERVICE.value == "restart_service"
        assert ActionType.RESOLVE.value == "resolve"
        assert len(ActionType) == 10

    def test_severity_values(self):
        assert Severity.P1.value == "P1"
        assert Severity.P4.value == "P4"
        assert len(Severity) == 4

    def test_incident_type_values(self):
        assert IncidentType.CPU_SPIKE.value == "cpu_spike"
        assert IncidentType.CASCADE_FAILURE.value == "cascade_failure"
        assert len(IncidentType) == 8

    def test_task_type_values(self):
        assert TaskType.TASK1.value == "task1"
        assert TaskType.TASK3.value == "task3"
        assert len(TaskType) == 3

    def test_invalid_action_type(self):
        with pytest.raises(ValueError):
            ActionType("nonexistent_action")

    def test_invalid_severity(self):
        with pytest.raises(ValueError):
            Severity("P5")


# ─── Model Validation Tests ────────────────────────────────────────────────────


class TestAlertPayload:
    def test_valid_alert(self):
        alert = AlertPayload(
            title="[P1] CPU Spike on api-gateway",
            service="api-gateway",
            severity=Severity.P1,
            timestamp="2026-04-06T12:00:00Z",
            runbook_url="https://runbooks.internal/cpu-spike",
        )
        assert alert.severity == Severity.P1
        assert alert.service == "api-gateway"

    def test_default_runbook_url(self):
        alert = AlertPayload(
            title="Test Alert",
            service="test-service",
            severity=Severity.P3,
            timestamp="2026-04-06T12:00:00Z",
        )
        assert alert.runbook_url == ""

    def test_invalid_severity_in_alert(self):
        with pytest.raises(ValueError):
            AlertPayload(
                title="Test",
                service="test",
                severity="CRITICAL",
                timestamp="2026-04-06T12:00:00Z",
            )


class TestServiceMetrics:
    def test_valid_metrics(self):
        metrics = ServiceMetrics(
            cpu=85.5, memory=60.0, latency_ms=150.0, error_rate=2.5
        )
        assert metrics.cpu == 85.5

    def test_cpu_range_validation(self):
        with pytest.raises(ValueError):
            ServiceMetrics(cpu=150.0, memory=50.0, latency_ms=100.0, error_rate=1.0)

    def test_negative_latency(self):
        with pytest.raises(ValueError):
            ServiceMetrics(cpu=50.0, memory=50.0, latency_ms=-10.0, error_rate=1.0)


class TestSREAction:
    def test_valid_action(self):
        action = SREAction(
            action_type=ActionType.INSPECT_LOGS,
            target_service="payment-service",
        )
        assert action.action_type == ActionType.INSPECT_LOGS
        assert action.target_service == "payment-service"

    def test_action_without_target(self):
        action = SREAction(action_type=ActionType.ACKNOWLEDGE_ALERT)
        assert action.target_service is None

    def test_action_with_parameters(self):
        action = SREAction(
            action_type=ActionType.SCALE_UP,
            target_service="api-gateway",
            parameters={"replicas": 6},
        )
        assert action.parameters["replicas"] == 6

    def test_invalid_action_type_string(self):
        with pytest.raises(ValueError):
            SREAction(action_type="invalid_action")


class TestSREReward:
    def test_valid_reward(self):
        reward = SREReward(value=0.05, cumulative=0.15, done=False, info={})
        assert reward.value == 0.05
        assert not reward.done

    def test_negative_reward(self):
        reward = SREReward(value=-0.15, cumulative=-0.10, done=False)
        assert reward.value == -0.15

    def test_default_values(self):
        reward = SREReward(value=0.0)
        assert reward.cumulative == 0.0
        assert reward.done is False
        assert reward.info == {}


class TestSREObservation:
    def test_valid_observation(self):
        obs = SREObservation(
            incident_id="INC-EASY-abc123",
            alert_payload=AlertPayload(
                title="Test",
                service="test-svc",
                severity=Severity.P2,
                timestamp="2026-04-06T12:00:00Z",
            ),
            service_topology=ServiceTopology(
                nodes=["svc-a", "svc-b"],
                edges=[["svc-a", "svc-b"]],
            ),
            metrics=ServiceMetrics(
                cpu=50.0, memory=50.0, latency_ms=100.0, error_rate=1.0
            ),
        )
        assert obs.incident_id == "INC-EASY-abc123"
        assert obs.step_number == 0
        assert obs.incident_resolved is False
        assert obs.action_history == []

    def test_observation_serialization(self):
        obs = SREObservation(
            incident_id="INC-EASY-abc123",
            alert_payload=AlertPayload(
                title="Test",
                service="test-svc",
                severity=Severity.P2,
                timestamp="2026-04-06T12:00:00Z",
            ),
            service_topology=ServiceTopology(
                nodes=["svc-a"],
                edges=[],
            ),
            metrics=ServiceMetrics(
                cpu=50.0, memory=50.0, latency_ms=100.0, error_rate=1.0
            ),
        )
        data = obs.model_dump()
        assert data["incident_id"] == "INC-EASY-abc123"
        assert data["alert_payload"]["severity"] == "P2"

        # Round-trip
        obs2 = SREObservation(**data)
        assert obs2.incident_id == obs.incident_id


class TestEpisodeState:
    def test_default_state(self):
        state = EpisodeState()
        assert state.step_count == 0
        assert state.action_history == []
        assert state.diagnosis_done is False
        assert state.remediation_applied is False
        assert state.episode_done is False
        assert state.cumulative_reward == 0.0

    def test_state_mutation(self):
        state = EpisodeState()
        state.step_count = 5
        state.diagnosis_done = True
        state.action_history.append("inspect_logs")
        assert state.step_count == 5
        assert len(state.action_history) == 1


class TestResetRequest:
    def test_defaults(self):
        req = ResetRequest()
        assert req.task == TaskType.TASK1
        assert req.seed is None

    def test_with_seed(self):
        req = ResetRequest(task=TaskType.TASK3, seed=42)
        assert req.task == TaskType.TASK3
        assert req.seed == 42

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError):
            ResetRequest(seed=-1)


class TestIncidentData:
    def test_valid_incident(self):
        incident = IncidentData(
            incident_id="INC-EASY-abc123",
            difficulty="easy",
            incident_type=IncidentType.CPU_SPIKE,
            alert_payload=AlertPayload(
                title="CPU Spike",
                service="api-gateway",
                severity=Severity.P2,
                timestamp="2026-04-06T12:00:00Z",
            ),
            service_topology=ServiceTopology(
                nodes=["api-gateway", "auth-service"],
                edges=[["api-gateway", "auth-service"]],
            ),
            metrics=ServiceMetrics(
                cpu=92.5, memory=50.0, latency_ms=100.0, error_rate=1.0
            ),
            gold_root_cause="CPU spike due to runaway process",
            gold_triggered_by="api-gateway",
            gold_affected_chain=["api-gateway"],
            gold_action_sequence=["inspect_logs", "restart_service", "verify_endpoint", "resolve"],
        )
        assert incident.difficulty == "easy"
        assert incident.incident_type == IncidentType.CPU_SPIKE
        assert len(incident.gold_action_sequence) == 4
