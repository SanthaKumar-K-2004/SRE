"""
SRE-Bench Task 1: Alert Classification Grader
Difficulty: Easy | Max Steps: 1 | Expected Score: 0.75 – 0.90

Grades agent's ability to correctly classify:
  - incident_type (0.40 weight)
  - severity (0.30 weight)
  - primary_fault_service (0.30 weight)

All matching is deterministic enum comparison — no LLM-as-judge.
"""

from __future__ import annotations

from typing import Any, Dict

from models import IncidentData
from tasks.score_utils import open_interval_score


def grade_task1(
    submission: Dict[str, Any],
    incident: IncidentData,
) -> float:
    """
    Grade a Task 1 (Alert Classification) submission.

    Args:
        submission: Agent's classification output containing:
            - incident_type: str
            - severity: str
            - primary_fault_service: str
        incident: The ground-truth incident data.

    Returns:
        Score in (0, 1), normalized to 0.01..0.99.
    """
    score = 0.0

    # ── incident_type matching (0.40) ──
    predicted_type = _normalize(submission.get("incident_type", ""))
    gold_type = _normalize(incident.incident_type.value)

    if predicted_type == gold_type:
        score += 0.40
    elif _partial_type_match(predicted_type, gold_type):
        score += 0.15  # Partial credit for related types

    # ── severity matching (0.30) ──
    predicted_severity = _normalize(submission.get("severity", ""))
    gold_severity = _normalize(incident.alert_payload.severity.value)

    if predicted_severity == gold_severity:
        score += 0.30
    elif _adjacent_severity(predicted_severity, gold_severity):
        score += 0.15  # Partial credit for one-level-off severity

    # ── primary_fault_service matching (0.30) ──
    predicted_service = _normalize(submission.get("primary_fault_service", ""))
    gold_service = _normalize(incident.alert_payload.service)

    if predicted_service == gold_service:
        score += 0.30
    elif predicted_service in [_normalize(s) for s in incident.gold_affected_chain]:
        score += 0.10  # Partial credit for identifying an affected service

    return open_interval_score(score)


def _normalize(value: str) -> str:
    """Normalize a string for comparison: lowercase, strip, replace hyphens."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _partial_type_match(predicted: str, gold: str) -> bool:
    """
    Check if the predicted type is in the same family as the gold type.
    e.g., 'cpu_spike' and 'db_overload' are both resource issues.
    """
    resource_types = {"cpu_spike", "memory_leak", "db_overload"}
    network_types = {"network_partition", "dependency_timeout"}
    deploy_types = {"deployment_failure", "config_error"}

    for family in (resource_types, network_types, deploy_types):
        if predicted in family and gold in family:
            return True
    return False


def _adjacent_severity(predicted: str, gold: str) -> bool:
    """Check if severities are one level apart (e.g., P1 vs P2)."""
    severity_order = {"p1": 0, "p2": 1, "p3": 2, "p4": 3}
    p_idx = severity_order.get(predicted, -1)
    g_idx = severity_order.get(gold, -1)

    if p_idx < 0 or g_idx < 0:
        return False
    return abs(p_idx - g_idx) == 1
