"""
SRE-Bench Task 2: Root Cause Analysis Grader
Difficulty: Medium | Max Steps: 8 | Expected Score: 0.40 – 0.60

Grades agent's ability to:
  - Identify root cause (0.40 weight)
  - Trace affected service chain (0.30 weight)
  - Achieve diagnostic efficiency (0.30 weight — fewer steps = higher bonus)

All matching is deterministic — no LLM-as-judge.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from models import IncidentData
from tasks.score_utils import open_interval_score


MAX_STEPS_TASK2 = 8


def grade_task2(
    submission: Dict[str, Any],
    incident: IncidentData,
    steps_used: int,
    action_history: List[str],
) -> float:
    """
    Grade a Task 2 (Root Cause Analysis) submission.

    Args:
        submission: Agent's analysis output containing:
            - root_cause: str (free text or enum)
            - triggered_by: str (service name)
            - affected_chain: List[str] (service names)
        incident: The ground-truth incident data.
        steps_used: Number of steps the agent took.
        action_history: List of actions taken during the episode.

    Returns:
        Score in (0, 1), normalized to 0.01..0.99.
    """
    score = 0.0

    # ── Root cause identification (0.40) ──
    score += _grade_root_cause(submission, incident)

    # ── Affected chain analysis (0.30) ──
    score += _grade_affected_chain(submission, incident)

    # ── Diagnostic efficiency (0.30) ──
    score += _grade_efficiency(steps_used, action_history)

    return open_interval_score(score)


def _grade_root_cause(
    submission: Dict[str, Any],
    incident: IncidentData,
) -> float:
    """Grade root cause identification — up to 0.40 points."""
    sub_score = 0.0

    # Check triggered_by service
    predicted_trigger = _normalize(submission.get("triggered_by", ""))
    gold_trigger = _normalize(incident.gold_triggered_by)

    if predicted_trigger == gold_trigger:
        sub_score += 0.20
    elif predicted_trigger in {_normalize(s) for s in incident.gold_affected_chain}:
        sub_score += 0.08  # Partial: identified an affected service

    # Check root cause description — keyword matching
    predicted_cause = _normalize(submission.get("root_cause", ""))
    gold_cause = _normalize(incident.gold_root_cause)

    # Extract key terms from gold root cause
    gold_keywords = _extract_keywords(gold_cause)
    predicted_keywords = _extract_keywords(predicted_cause)

    if gold_keywords and predicted_keywords:
        overlap = gold_keywords & predicted_keywords
        keyword_ratio = len(overlap) / max(1, len(gold_keywords))
        sub_score += 0.20 * keyword_ratio

    # Check incident type match (bonus if they identified the type correctly)
    predicted_type = _normalize(submission.get("incident_type", ""))
    gold_type = _normalize(incident.incident_type.value)
    if predicted_type == gold_type:
        sub_score = min(0.40, sub_score + 0.05)

    return min(0.40, sub_score)


def _grade_affected_chain(
    submission: Dict[str, Any],
    incident: IncidentData,
) -> float:
    """Grade affected chain analysis — up to 0.30 points."""
    predicted_chain = submission.get("affected_chain", [])
    gold_chain = incident.gold_affected_chain

    if not predicted_chain or not gold_chain:
        return 0.0

    predicted_set: Set[str] = {_normalize(s) for s in predicted_chain}
    gold_set: Set[str] = {_normalize(s) for s in gold_chain}

    if not gold_set:
        return 0.0

    # Compute set overlap (Jaccard-like)
    intersection = predicted_set & gold_set
    union = predicted_set | gold_set

    if not union:
        return 0.0

    # Precision and recall
    precision = len(intersection) / max(1, len(predicted_set))
    recall = len(intersection) / max(1, len(gold_set))

    # F1-like score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # Bonus for correct ordering (if chain order matters)
    order_bonus = 0.0
    if len(intersection) >= 2:
        # Check if the first element matches (root of cascade)
        if predicted_chain and gold_chain:
            if _normalize(predicted_chain[0]) == _normalize(gold_chain[0]):
                order_bonus = 0.05

    return min(0.30, round(f1 * 0.25 + order_bonus, 4))


def _grade_efficiency(
    steps_used: int,
    action_history: List[str],
) -> float:
    """
    Grade diagnostic efficiency — up to 0.30 points.
    Fewer steps = higher bonus. Penalize loops and redundant actions.
    """
    if steps_used == 0:
        return 0.0

    # Base efficiency: ratio of max steps unused
    step_ratio = max(0.0, (MAX_STEPS_TASK2 - steps_used) / MAX_STEPS_TASK2)
    efficiency_score = step_ratio * 0.20

    # Bonus for using diagnostic actions (inspect_logs, check_metrics, check_topology)
    diagnostic_actions = {"inspect_logs", "check_metrics", "check_topology", "check_service"}
    actions_used = {a.split(":")[0] for a in action_history}
    diagnostic_used = actions_used & diagnostic_actions

    if len(diagnostic_used) >= 2:
        efficiency_score += 0.05
    if len(diagnostic_used) >= 3:
        efficiency_score += 0.05

    # Penalize repeated actions (loops)
    unique_actions = len(set(action_history))
    total_actions = len(action_history)
    if total_actions > 0:
        loop_ratio = 1.0 - (unique_actions / total_actions)
        efficiency_score -= loop_ratio * 0.10

    return min(0.30, max(0.0, round(efficiency_score, 4)))


def _normalize(value: str) -> str:
    """Normalize string for comparison."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from a root cause description."""
    # Stop words to filter out
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "on", "in", "at",
        "to", "for", "of", "with", "by", "from", "and", "or", "not",
        "due", "after", "before", "during", "caused", "causing",
    }

    words = text.replace("_", " ").replace("-", " ").split()
    return {w for w in words if len(w) > 2 and w not in stop_words}
