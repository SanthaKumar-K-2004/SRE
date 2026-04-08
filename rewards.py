"""
SRE-Bench: Reward Calculation Engine
Per-step reward signals, loop detection, and cumulative tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models import ActionType, IncidentData, EpisodeState


class RewardEngine:
    """
    Computes per-step rewards based on action relevance, causal reasoning,
    and penalizes destructive/random behaviors.
    """

    # ─── Reward Constants ───────────────────────────────────────────────────

    REWARD_INSPECT_RELEVANT_LOGS = 0.05
    REWARD_CHECK_RELEVANT_METRICS = 0.05
    REWARD_CORRECT_ROOT_CAUSE = 0.40
    REWARD_CORRECT_REMEDIATION = 0.15
    REWARD_VERIFICATION = 0.20
    REWARD_ACKNOWLEDGE = 0.02

    PENALTY_DESTRUCTIVE_BEFORE_DIAGNOSIS = -0.15
    PENALTY_LOOP_DETECTION = -0.10
    PENALTY_PREMATURE_RESOLUTION = -0.10
    PENALTY_INVALID_ACTION = -0.05
    PENALTY_WRONG_TARGET = -0.03

    def __init__(self, incident: IncidentData):
        self.incident = incident
        self._primary_service = incident.alert_payload.service
        self._affected_services = set(incident.gold_affected_chain)
        self._gold_sequence = incident.gold_action_sequence
        self._cumulative = 0.0
        self._step_rewards: List[float] = []

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative

    @property
    def step_history(self) -> List[float]:
        return self._step_rewards.copy()

    def compute_reward(
        self,
        action: ActionType,
        target_service: Optional[str],
        episode_state: EpisodeState,
        action_result: Dict[str, Any],
        precondition_failed: bool = False,
    ) -> float:
        """
        Compute the reward for a single step.
        Returns a float reward value.
        """
        if precondition_failed:
            reward = self.PENALTY_INVALID_ACTION
            self._apply_reward(reward)
            return reward

        reward = 0.0
        target = target_service or self._primary_service

        # ── Check for loop (repeated identical action) ──
        if self._is_loop(episode_state.action_history):
            reward += self.PENALTY_LOOP_DETECTION

        # ── Action-specific rewards ──
        if action == ActionType.ACKNOWLEDGE_ALERT:
            reward += self.REWARD_ACKNOWLEDGE

        elif action == ActionType.INSPECT_LOGS:
            if target in self._affected_services or target == self._primary_service:
                reward += self.REWARD_INSPECT_RELEVANT_LOGS
            else:
                reward += 0.01  # Small reward for any inspection

        elif action == ActionType.CHECK_METRICS:
            if target == self._primary_service or target == self.incident.gold_triggered_by:
                reward += self.REWARD_CHECK_RELEVANT_METRICS
            else:
                reward += 0.01

        elif action == ActionType.CHECK_SERVICE:
            if target in self._affected_services:
                reward += 0.03
            else:
                reward += 0.01

        elif action == ActionType.CHECK_TOPOLOGY:
            # Useful for cascade and network partition incidents
            if self.incident.incident_type.value in ("cascade_failure", "network_partition", "dependency_timeout"):
                reward += 0.05
            else:
                reward += 0.02

        elif action in (
            ActionType.RESTART_SERVICE,
            ActionType.SCALE_UP,
            ActionType.ROLLBACK_DEPLOY,
        ):
            # Remediation actions
            if not episode_state.diagnosis_done:
                # Destructive action before diagnosis
                reward += self.PENALTY_DESTRUCTIVE_BEFORE_DIAGNOSIS
            else:
                # Check if this is the correct remediation
                if action.value in self._gold_sequence:
                    reward += self.REWARD_CORRECT_REMEDIATION
                    if target == self._primary_service:
                        reward += 0.05  # Bonus for correct target
                else:
                    reward += 0.02  # Partial credit for attempting remediation

        elif action == ActionType.VERIFY_ENDPOINT:
            if episode_state.remediation_applied:
                verified = action_result.get("verified", False)
                if verified:
                    reward += self.REWARD_VERIFICATION
                else:
                    reward += 0.05  # Partial credit for attempting verification
            else:
                reward += self.PENALTY_INVALID_ACTION

        elif action == ActionType.RESOLVE:
            if not episode_state.verification_done:
                reward += self.PENALTY_PREMATURE_RESOLUTION
            else:
                # Final resolution — score based on efficiency
                steps_used = episode_state.step_count
                gold_steps = len(self._gold_sequence)
                if steps_used <= gold_steps:
                    reward += 0.10  # Efficiency bonus
                elif steps_used <= gold_steps * 1.5:
                    reward += 0.05
                # else no bonus

        # ── Clamp per-step reward ──
        reward = max(-1.0, min(1.0, reward))

        self._apply_reward(reward)
        return reward

    def _apply_reward(self, reward: float) -> None:
        """Track reward and update cumulative total."""
        self._step_rewards.append(reward)
        self._cumulative += reward

    def _is_loop(self, action_history: List[str]) -> bool:
        """Detect if the last two actions are identical."""
        if len(action_history) < 2:
            return False
        return action_history[-1] == action_history[-2]

    def compute_efficiency_bonus(self, steps_used: int, max_steps: int) -> float:
        """Compute bonus based on how efficiently the agent solved the incident."""
        if steps_used == 0:
            return 0.0
        ratio = 1.0 - (steps_used / max_steps)
        return max(0.0, round(ratio * 0.3, 4))

    def get_reward_breakdown(self) -> Dict[str, Any]:
        """Return a detailed breakdown of rewards for debugging."""
        return {
            "cumulative_reward": round(self._cumulative, 4),
            "total_steps": len(self._step_rewards),
            "per_step_rewards": [round(r, 4) for r in self._step_rewards],
            "avg_reward": round(
                sum(self._step_rewards) / max(1, len(self._step_rewards)), 4
            ),
        }
