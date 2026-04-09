"""Tests for task graders including hypothesis property-based tests."""

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from models import IncidentData, IncidentType, Severity
from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import grade_task3


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
            "runbook_url": "",
        },
        "service_topology": {
            "nodes": ["api-gateway", "auth-service", "payment-service"],
            "edges": [["api-gateway", "auth-service"]],
        },
        "logs": [f"log line {i}" for i in range(50)],
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


# ─── Task 1 Grader Tests ───────────────────────────────────────────────────────


class TestTask1Grader:
    def test_perfect_score(self):
        incident = _make_incident()
        submission = {
            "incident_type": "cpu_spike",
            "severity": "P2",
            "primary_fault_service": "api-gateway",
        }
        score = grade_task1(submission, incident)
        assert score == 0.99

    def test_all_wrong(self):
        incident = _make_incident()
        submission = {
            "incident_type": "cascade_failure",
            "severity": "P4",
            "primary_fault_service": "nonexistent-service",
        }
        score = grade_task1(submission, incident)
        assert score < 0.5

    def test_correct_type_wrong_severity(self):
        incident = _make_incident()
        submission = {
            "incident_type": "cpu_spike",
            "severity": "P4",
            "primary_fault_service": "api-gateway",
        }
        score = grade_task1(submission, incident)
        assert 0.6 <= score <= 0.8  # type + service correct

    def test_partial_type_match(self):
        incident = _make_incident()
        submission = {
            "incident_type": "memory_leak",  # Same resource family
            "severity": "P2",
            "primary_fault_service": "api-gateway",
        }
        score = grade_task1(submission, incident)
        assert score > 0.5  # Partial credit for related type

    def test_adjacent_severity(self):
        incident = _make_incident()
        submission = {
            "incident_type": "cpu_spike",
            "severity": "P1",  # One level off from P2
            "primary_fault_service": "api-gateway",
        }
        score = grade_task1(submission, incident)
        assert score > 0.7  # Partial credit for adjacent severity

    def test_affected_chain_service_credit(self):
        incident = _make_incident()
        submission = {
            "incident_type": "cpu_spike",
            "severity": "P2",
            "primary_fault_service": "auth-service",  # In affected chain
        }
        score = grade_task1(submission, incident)
        assert score > 0.5

    def test_empty_submission(self):
        incident = _make_incident()
        score = grade_task1({}, incident)
        assert score == 0.01

    def test_score_always_in_range(self):
        incident = _make_incident()
        for itype in ["cpu_spike", "memory_leak", "cascade_failure", "unknown"]:
            for sev in ["P1", "P2", "P3", "P4", "CRIT"]:
                for svc in ["api-gateway", "unknown", ""]:
                    score = grade_task1(
                        {"incident_type": itype, "severity": sev, "primary_fault_service": svc},
                        incident,
                    )
                    assert 0.0 < score < 1.0


# ─── Task 2 Grader Tests ───────────────────────────────────────────────────────


class TestTask2Grader:
    def test_perfect_submission(self):
        incident = _make_incident()
        submission = {
            "root_cause": "Runaway process on api-gateway consuming excessive CPU",
            "triggered_by": "api-gateway",
            "affected_chain": ["api-gateway", "auth-service"],
            "incident_type": "cpu_spike",
        }
        score = grade_task2(submission, incident, steps_used=3, action_history=[
            "inspect_logs", "check_metrics", "check_service",
        ])
        assert score > 0.5

    def test_wrong_root_cause(self):
        incident = _make_incident()
        submission = {
            "root_cause": "Network cable unplugged",
            "triggered_by": "nonexistent",
            "affected_chain": [],
        }
        score = grade_task2(submission, incident, steps_used=8, action_history=[
            "inspect_logs",
        ])
        assert score < 0.5

    def test_efficiency_bonus_fewer_steps(self):
        incident = _make_incident()
        submission = {
            "root_cause": "CPU consuming excessive",
            "triggered_by": "api-gateway",
            "affected_chain": ["api-gateway"],
        }
        score_fast = grade_task2(submission, incident, steps_used=2, action_history=[
            "inspect_logs", "check_metrics",
        ])
        score_slow = grade_task2(submission, incident, steps_used=7, action_history=[
            "inspect_logs", "check_metrics", "check_service",
            "check_topology", "inspect_logs:auth-service",
            "check_metrics:auth-service", "check_service:auth-service",
        ])
        assert score_fast >= score_slow

    def test_empty_chain(self):
        incident = _make_incident()
        submission = {
            "root_cause": "CPU spike",
            "triggered_by": "api-gateway",
            "affected_chain": [],
        }
        score = grade_task2(submission, incident, steps_used=3, action_history=[
            "inspect_logs", "check_metrics", "check_service",
        ])
        assert 0.0 < score < 1.0

    def test_score_range(self):
        incident = _make_incident()
        submission = {"root_cause": "", "triggered_by": "", "affected_chain": []}
        score = grade_task2(submission, incident, steps_used=1, action_history=["inspect_logs"])
        assert 0.0 < score < 1.0


# ─── Task 3 Grader Tests ───────────────────────────────────────────────────────


class TestTask3Grader:
    def test_perfect_sequence(self):
        incident = _make_incident()
        gold_seq = incident.gold_action_sequence
        episode_state = {
            "verification_done": True,
            "episode_done": True,
            "alert_acknowledged": True,
        }
        score = grade_task3(gold_seq, incident, episode_state)
        assert score > 0.7

    def test_empty_sequence(self):
        incident = _make_incident()
        score = grade_task3([], incident, {
            "verification_done": False,
            "episode_done": False,
            "alert_acknowledged": False,
        })
        assert score == 0.01

    def test_partial_sequence(self):
        incident = _make_incident()
        partial = ["acknowledge_alert", "inspect_logs", "check_metrics"]
        score = grade_task3(partial, incident, {
            "verification_done": False,
            "episode_done": False,
            "alert_acknowledged": True,
        })
        assert 0.0 < score < 0.7

    def test_wrong_order_penalized(self):
        incident = _make_incident()
        wrong_order = ["restart_service", "inspect_logs", "verify_endpoint", "resolve"]
        score_wrong = grade_task3(wrong_order, incident, {
            "verification_done": True,
            "episode_done": True,
            "alert_acknowledged": False,
        })

        correct_order = ["inspect_logs", "restart_service", "verify_endpoint", "resolve"]
        score_correct = grade_task3(correct_order, incident, {
            "verification_done": True,
            "episode_done": True,
            "alert_acknowledged": False,
        })
        assert score_correct >= score_wrong

    def test_loop_penalized(self):
        incident = _make_incident()
        with_loop = [
            "inspect_logs", "inspect_logs", "inspect_logs",
            "restart_service", "verify_endpoint", "resolve",
        ]
        score = grade_task3(with_loop, incident, {
            "verification_done": True,
            "episode_done": True,
            "alert_acknowledged": False,
        })
        assert score < 0.9  # Looping should reduce score


# ─── Hypothesis Property-Based Tests ───────────────────────────────────────────


class TestGraderRangeSafety:
    @given(
        itype=st.sampled_from(["cpu_spike", "memory_leak", "network_partition",
                                "deployment_failure", "db_overload", "cascade_failure",
                                "config_error", "dependency_timeout", "unknown"]),
        severity=st.sampled_from(["P1", "P2", "P3", "P4", "CRITICAL", ""]),
        service=st.text(min_size=0, max_size=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_task1_grader_always_in_range(self, itype, severity, service):
        incident = _make_incident()
        submission = {
            "incident_type": itype,
            "severity": severity,
            "primary_fault_service": service,
        }
        score = grade_task1(submission, incident)
        assert 0.0 < score < 1.0

    @given(
        steps=st.integers(min_value=0, max_value=20),
        num_actions=st.integers(min_value=0, max_value=15),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_task2_grader_always_in_range(self, steps, num_actions):
        incident = _make_incident()
        actions = ["inspect_logs", "check_metrics", "check_service"][:num_actions]
        submission = {
            "root_cause": "some cause",
            "triggered_by": "api-gateway",
            "affected_chain": ["api-gateway"],
        }
        score = grade_task2(submission, incident, steps_used=steps, action_history=actions)
        assert 0.0 < score < 1.0

    @given(
        num_actions=st.integers(min_value=0, max_value=15),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_task3_grader_always_in_range(self, num_actions):
        incident = _make_incident()
        actions_pool = [
            "acknowledge_alert", "inspect_logs", "check_metrics",
            "check_service", "restart_service", "verify_endpoint", "resolve",
        ]
        import random
        random.seed(num_actions)
        actions = [random.choice(actions_pool) for _ in range(num_actions)]
        
        state = {
            "verification_done": "verify_endpoint" in actions,
            "episode_done": "resolve" in actions,
            "alert_acknowledged": "acknowledge_alert" in actions,
        }
        score = grade_task3(actions, incident, state)
        assert 0.0 < score < 1.0
