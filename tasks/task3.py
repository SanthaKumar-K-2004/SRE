"""
SRE-Bench Task 3: Full Remediation Grader
Difficulty: Hard | Max Steps: 15 | Expected Score: 0.15 – 0.30

Grades agent's complete incident remediation using:
  - LCS (Longest Common Subsequence) of action sequence vs gold (0.50 weight)
  - Behavior penalties for destructive/random actions (0.30 weight)
  - Verification bonus (0.20 weight)

All matching is deterministic — no LLM-as-judge.
"""

from __future__ import annotations

from typing import Any, Dict, List

from models import IncidentData
from tasks.score_utils import open_interval_score


MAX_STEPS_TASK3 = 15


def grade_task3(
    action_history: List[str],
    incident: IncidentData,
    episode_state: Dict[str, Any],
) -> float:
    """
    Grade a Task 3 (Full Remediation) submission.

    Args:
        action_history: Complete list of actions taken by the agent.
        incident: The ground-truth incident data.
        episode_state: Final episode state dictionary.

    Returns:
        Score in (0, 1), normalized to 0.01..0.99.
    """
    score = 0.0

    gold_sequence = incident.gold_action_sequence

    # ── LCS sequence matching (0.50) ──
    score += _grade_sequence_lcs(action_history, gold_sequence)

    # ── Behavior penalties (0.30) ──
    score += _grade_behavior(action_history, episode_state)

    # ── Verification bonus (0.20) ──
    score += _grade_verification(episode_state)

    return open_interval_score(score)


def _grade_sequence_lcs(
    agent_actions: List[str],
    gold_actions: List[str],
) -> float:
    """
    Score based on Longest Common Subsequence between agent and gold sequences.
    Returns up to 0.50 points.
    """
    if not gold_actions:
        return 0.0

    # Normalize action names (strip target service if present)
    agent_normalized = [a.split(":")[0] for a in agent_actions]
    gold_normalized = [a.split(":")[0] for a in gold_actions]

    lcs_length = _compute_lcs_length(agent_normalized, gold_normalized)

    # LCS ratio relative to gold sequence length
    lcs_ratio = lcs_length / len(gold_normalized)

    # Scale to 0.50 max
    return round(min(0.50, lcs_ratio * 0.50), 4)


def _compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute the length of the Longest Common Subsequence (LCS).
    Uses dynamic programming — O(n*m) time and space.
    """
    n, m = len(seq1), len(seq2)

    if n == 0 or m == 0:
        return 0

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def _grade_behavior(
    action_history: List[str],
    episode_state: Dict[str, Any],
) -> float:
    """
    Grade behavior quality — starts at 0.30, penalties subtracted.
    Penalizes destructive, looping, and premature actions.
    """
    # No actions taken = no behavior to grade
    if not action_history:
        return 0.0

    behavior_score = 0.30

    actions_normalized = [a.split(":")[0] for a in action_history]

    # ── Penalty: Destructive action before diagnosis ──
    destructive_actions = {"restart_service", "scale_up", "rollback_deploy"}
    diagnostic_actions = {"inspect_logs", "check_metrics", "check_service"}

    diagnosis_step = None
    for i, action in enumerate(actions_normalized):
        if action in diagnostic_actions:
            diagnosis_step = i
            break

    for i, action in enumerate(actions_normalized):
        if action in destructive_actions:
            if diagnosis_step is None or i < diagnosis_step:
                behavior_score -= 0.10
                break  # Only penalize once

    # ── Penalty: Loop detection (repeated consecutive actions) ──
    loop_count = 0
    for i in range(1, len(action_history)):
        if action_history[i] == action_history[i - 1]:
            loop_count += 1

    behavior_score -= min(0.10, loop_count * 0.03)

    # ── Penalty: Premature resolution ──
    if "resolve" in actions_normalized:
        resolve_idx = actions_normalized.index("resolve")
        if "verify_endpoint" not in actions_normalized[:resolve_idx]:
            behavior_score -= 0.07

    # ── Penalty: Too many actions (inefficiency) ──
    if len(action_history) > MAX_STEPS_TASK3 * 0.8:
        behavior_score -= 0.05

    # ── Bonus: Correct ordering (acknowledge → diagnose → remediate → verify → resolve) ──
    phase_order = _check_phase_order(actions_normalized)
    if phase_order:
        behavior_score += 0.05

    return max(0.0, min(0.30, round(behavior_score, 4)))


def _check_phase_order(actions: List[str]) -> bool:
    """
    Check if actions follow the correct phase order:
    1. Acknowledge/Diagnose  2. Remediate  3. Verify  4. Resolve
    """
    phase_map = {
        "acknowledge_alert": 0,
        "inspect_logs": 1,
        "check_metrics": 1,
        "check_service": 1,
        "check_topology": 1,
        "restart_service": 2,
        "scale_up": 2,
        "rollback_deploy": 2,
        "verify_endpoint": 3,
        "resolve": 4,
    }

    last_phase = -1
    for action in actions:
        phase = phase_map.get(action, -1)
        if phase < 0:
            continue
        if phase < last_phase:
            return False  # Out of order
        last_phase = phase

    return True


def _grade_verification(episode_state: Dict[str, Any]) -> float:
    """
    Grade whether the agent properly verified the fix.
    Returns up to 0.20 points.
    """
    verification_score = 0.0

    # Did the agent verify the endpoint?
    if episode_state.get("verification_done", False):
        verification_score += 0.12

    # Did the agent resolve the incident?
    if episode_state.get("episode_done", False):
        verification_score += 0.05

    # Did the agent acknowledge the alert first?
    if episode_state.get("alert_acknowledged", False):
        verification_score += 0.03

    return min(0.20, round(verification_score, 4))


def get_lcs_sequence(seq1: List[str], seq2: List[str]) -> List[str]:
    """
    Return the actual LCS (not just length) for debugging/analysis.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the actual subsequence
    result = []
    i, j = n, m
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            result.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(result))
