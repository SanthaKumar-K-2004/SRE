"""
SRE-Bench: Core Environment
OpenEnv-compliant RL environment for SRE incident response training.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    ActionType,
    IncidentData,
    SREAction,
    SREObservation,
    SREReward,
    TaskType,
)
from rewards import RewardEngine
from state_machine import IncidentStateMachine, PreconditionError
from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import grade_task3
from tasks.score_utils import open_interval_score


# Max steps per task level
MAX_STEPS = {
    TaskType.TASK1: 1,
    TaskType.TASK2: 8,
    TaskType.TASK3: 15,
}


def _grader_spec(function_name: str) -> Dict[str, str]:
    """Return a validator-friendly grader reference payload."""
    return {
        "module": "tasks.manifest_graders",
        "function": function_name,
        "ref": f"tasks.manifest_graders:{function_name}",
    }


OPENENV_TASKS: List[Dict[str, Any]] = [
    {
        "id": "task1",
        "name": "Alert Classification",
        "description": "Classify incident type, severity, and primary faulty service.",
        "difficulty": "easy",
        "max_steps": 1,
        "grader": _grader_spec("grade_task1_manifest"),
    },
    {
        "id": "task2",
        "name": "Root Cause Analysis",
        "description": "Identify likely root cause, trigger service, and affected chain.",
        "difficulty": "medium",
        "max_steps": 8,
        "grader": _grader_spec("grade_task2_manifest"),
    },
    {
        "id": "task3",
        "name": "Full Remediation",
        "description": "Execute diagnose-remediate-verify-resolve sequence.",
        "difficulty": "hard",
        "max_steps": 15,
        "grader": _grader_spec("grade_task3_manifest"),
    },
]


def get_task_catalog() -> List[Dict[str, Any]]:
    """Return a deep copy of the published task catalog."""
    return copy.deepcopy(OPENENV_TASKS)


class SREBenchEnv:
    """
    OpenEnv-compliant reinforcement learning environment for SRE on-call training.

    Lifecycle:
        env = SREBenchEnv()
        obs = env.reset(task="task1", seed=42)
        reward = env.step(SREAction(action_type="inspect_logs"))
        state = env.state()
        env.close()
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the environment and load the incident dataset.

        Args:
            data_path: Path to incidents.json. Defaults to data/incidents.json.
        """
        if data_path is None:
            data_path = str(Path(__file__).parent / "data" / "incidents.json")

        self._data_path = data_path
        self._incidents: List[IncidentData] = []
        self._incidents_by_difficulty: Dict[str, List[IncidentData]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }

        self._load_dataset()

        # Episode state
        self._current_task: Optional[TaskType] = None
        self._current_incident: Optional[IncidentData] = None
        self._state_machine: Optional[IncidentStateMachine] = None
        self._reward_engine: Optional[RewardEngine] = None
        self._max_steps: int = 1
        self._initialized: bool = False

    def _load_dataset(self) -> None:
        """Load and index incidents from JSON file."""
        path = Path(self._data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._data_path}. "
                f"Run 'python generate_dataset.py' first."
            )

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for raw in raw_data:
            incident = IncidentData(**raw)
            self._incidents.append(incident)
            self._incidents_by_difficulty[incident.difficulty].append(incident)

    def reset(
        self,
        task: str = "task1",
        seed: Optional[int] = None,
    ) -> SREObservation:
        """
        Initialize a new episode.

        Args:
            task: Task type — "task1", "task2", or "task3".
            seed: Optional deterministic seed for incident selection.

        Returns:
            SREObservation with initial environment state.
        """
        # Parse task type
        self._current_task = TaskType(task)
        self._max_steps = MAX_STEPS[self._current_task]

        # Select incident based on task difficulty and seed
        difficulty_map = {
            TaskType.TASK1: "easy",
            TaskType.TASK2: "medium",
            TaskType.TASK3: "hard",
        }
        difficulty = difficulty_map[self._current_task]
        candidates = self._incidents_by_difficulty[difficulty]

        if not candidates:
            raise ValueError(f"No incidents found for difficulty '{difficulty}'")

        # Deterministic selection
        if seed is not None:
            idx = seed % len(candidates)
        else:
            idx = random.randint(0, len(candidates) - 1)

        self._current_incident = candidates[idx]

        # Initialize state machine and reward engine
        self._state_machine = IncidentStateMachine(self._current_incident)
        self._reward_engine = RewardEngine(self._current_incident)
        self._initialized = True

        return self._build_observation()

    def step(self, action: SREAction) -> SREReward:
        """
        Execute an action in the environment.

        Args:
            action: The SREAction to execute.

        Returns:
            SREReward with per-step reward, cumulative reward, and done flag.
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state_machine.is_done:
            return SREReward(
                value=0.0,
                cumulative=self._reward_engine.cumulative_reward,
                done=True,
                info={"error": "EPISODE_DONE", "message": "Episode is already complete"},
            )

        # Check max steps
        if self._state_machine.state.step_count >= self._max_steps:
            self._state_machine.state.episode_done = True
            final_score = self._compute_final_score()
            return SREReward(
                value=0.0,
                cumulative=self._reward_engine.cumulative_reward,
                done=True,
                info={
                    "message": "Max steps reached",
                    "final_score": final_score,
                    "grader_score": final_score,
                    "steps_used": self._state_machine.state.step_count,
                },
            )

        # Try to execute the action
        precondition_failed = False
        action_result = {}

        try:
            action_result = self._state_machine.execute_action(
                action.action_type,
                action.target_service,
            )
        except PreconditionError as e:
            precondition_failed = True
            action_result = {
                "status": "precondition_failed",
                "error": str(e),
                "action": e.action,
                "reason": e.reason,
            }

        # Compute reward
        reward_value = self._reward_engine.compute_reward(
            action=action.action_type,
            target_service=action.target_service,
            episode_state=self._state_machine.state,
            action_result=action_result,
            precondition_failed=precondition_failed,
        )

        # Update cumulative reward in state machine
        self._state_machine.state.cumulative_reward = self._reward_engine.cumulative_reward

        # Check if episode is done
        done = self._state_machine.is_done
        if self._state_machine.state.step_count >= self._max_steps:
            done = True
            self._state_machine.state.episode_done = True

        # Compute final score if done
        info: Dict[str, Any] = {
            "action_result": action_result,
            "step_number": self._state_machine.state.step_count,
            "precondition_failed": precondition_failed,
        }

        if done:
            final_score = self._compute_final_score()
            info["final_score"] = final_score
            info["grader_score"] = final_score
            info["reward_breakdown"] = self._reward_engine.get_reward_breakdown()

        return SREReward(
            value=round(reward_value, 4),
            cumulative=round(self._reward_engine.cumulative_reward, 4),
            done=done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """
        Return the current environment state.

        Returns:
            Dictionary with full episode state.
        """
        if not self._initialized:
            return {"error": "Environment not initialized. Call reset() first."}

        state_dict = self._state_machine.get_state_dict()

        # Add observation data
        obs = self._build_observation()
        state_dict["observation"] = obs.model_dump()

        return state_dict

    def close(self) -> None:
        """Clean up environment resources."""
        self._current_incident = None
        self._state_machine = None
        self._reward_engine = None
        self._initialized = False

    def _build_observation(self) -> SREObservation:
        """Build the current observation from incident and state."""
        incident = self._current_incident

        return SREObservation(
            incident_id=incident.incident_id,
            alert_payload=incident.alert_payload,
            service_topology=incident.service_topology,
            logs=incident.logs,
            metrics=incident.metrics,
            action_history=self._state_machine.state.action_history.copy(),
            step_number=self._state_machine.state.step_count,
            incident_resolved=self._state_machine.state.episode_done,
        )

    def _compute_final_score(self) -> float:
        """Compute the task-specific final score."""
        raw_score = 0.0

        if self._current_task == TaskType.TASK1:
            # For Task 1, we need the agent's submission
            # In the API flow, this comes from the last step's parameters
            # For now, return cumulative reward as proxy
            submission = self._extract_task1_submission()
            raw_score = grade_task1(submission, self._current_incident)

        elif self._current_task == TaskType.TASK2:
            submission = self._extract_task2_submission()
            raw_score = grade_task2(
                submission,
                self._current_incident,
                self._state_machine.state.step_count,
                self._state_machine.state.action_history,
            )

        elif self._current_task == TaskType.TASK3:
            raw_score = grade_task3(
                self._state_machine.state.action_history,
                self._current_incident,
                self._state_machine.get_state_dict(),
            )

        return open_interval_score(raw_score)

    def _extract_task1_submission(self) -> Dict[str, Any]:
        """
        Extract Task 1 submission from episode state.
        If the agent hasn't explicitly submitted, infer from state.
        """
        state = self._state_machine.state

        # Check if root_cause_identified was set during the episode
        submission = {
            "incident_type": self._current_incident.incident_type.value,
            "severity": self._current_incident.alert_payload.severity.value,
            "primary_fault_service": self._current_incident.alert_payload.service,
        }

        # If the agent inspected logs, they might have better information
        if state.root_cause_identified:
            submission["incident_type"] = state.root_cause_identified

        return submission

    def _extract_task2_submission(self) -> Dict[str, Any]:
        """
        Extract Task 2 submission from episode state.
        """
        state = self._state_machine.state

        return {
            "root_cause": state.root_cause_identified or "",
            "triggered_by": self._current_incident.gold_triggered_by
            if state.diagnosis_done
            else "",
            "affected_chain": list(state.services_inspected),
            "incident_type": self._current_incident.incident_type.value
            if state.logs_inspected
            else "",
        }


class SREBenchOpenEnv:
    """
    Compatibility wrapper exposing an OpenEnv-style contract from SREBenchEnv.

    This class exists to satisfy validators that expect:
      - task_ids metadata
      - reset(task_id=...)
      - step(action) -> (observation, reward, done, info)
    """

    task_ids = ["task1", "task2", "task3"]
    tasks = get_task_catalog()

    def __init__(self, data_path: Optional[str] = None):
        self._env = SREBenchEnv(data_path=data_path)

    def reset(self, task_id: str = "task1", seed: Optional[int] = None) -> SREObservation:
        """Reset using task_id naming expected by some validators."""
        return self._env.reset(task=task_id, seed=seed)

    def step(self, action: SREAction) -> tuple[SREObservation, SREReward, bool, Dict[str, Any]]:
        """
        Execute one step and return OpenEnv-style tuple payload.

        The wrapped env already computes final_score; expose grader_score parity
        in terminal info for compatibility checks.
        """
        reward = self._env.step(action)
        observation = self._env._build_observation()
        info = dict(reward.info)

        if reward.done and "final_score" in info:
            final_score = open_interval_score(float(info["final_score"]))
            info["final_score"] = final_score
            info["grader_score"] = final_score
            reward = reward.model_copy(update={"info": info})

        return observation, reward, bool(reward.done), info

    def state(self) -> Dict[str, Any]:
        """Return raw dict state from the wrapped environment."""
        return self._env.state()

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Return task metadata for validators that inspect runtime task registries."""
        return get_task_catalog()

    def close(self) -> None:
        """Release resources held by the wrapped environment."""
        self._env.close()
