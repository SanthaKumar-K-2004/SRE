"""
Compatibility grader wrappers referenced by openenv.yaml.

These wrappers are intentionally permissive with input signatures so they can
be imported and called by a variety of validator harnesses.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from models import IncidentData
from tasks.score_utils import open_interval_score
from tasks.task1 import grade_task1
from tasks.task2 import grade_task2
from tasks.task3 import grade_task3


def _as_incident(value: Any) -> Optional[IncidentData]:
    if isinstance(value, IncidentData):
        return value
    if isinstance(value, dict):
        try:
            return IncidentData(**value)
        except Exception:
            return None
    return None


def _extract_incident(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[IncidentData]:
    for key in ("incident", "gold_incident", "task_incident"):
        incident = _as_incident(kwargs.get(key))
        if incident is not None:
            return incident
    for value in args:
        incident = _as_incident(value)
        if incident is not None:
            return incident
    return None


def _extract_mapping(
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    keys: Iterable[str],
    explicit_names: Iterable[str],
) -> Dict[str, Any]:
    for name in explicit_names:
        value = kwargs.get(name)
        if isinstance(value, dict):
            return value
    target_keys = set(keys)
    for value in args:
        if isinstance(value, dict) and (target_keys & set(value.keys())):
            return value
    for value in kwargs.values():
        if isinstance(value, dict) and (target_keys & set(value.keys())):
            return value
    return {}


def _extract_float_hint(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[float]:
    candidates: List[Any] = list(args) + list(kwargs.values())
    while candidates:
        value = candidates.pop(0)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("grader_score", "final_score", "score", "value"):
                if key in value and isinstance(value[key], (int, float)):
                    return float(value[key])
            candidates.extend(value.values())
    return None


def grade_task1_manifest(*args: Any, **kwargs: Any) -> float:
    """Wrapper for Task 1 grader with broad signature compatibility."""
    incident = _extract_incident(args, kwargs)
    submission = _extract_mapping(
        args,
        kwargs,
        keys=("incident_type", "severity", "primary_fault_service"),
        explicit_names=("submission", "action_params", "payload"),
    )
    if incident is not None:
        return open_interval_score(float(grade_task1(submission, incident)))

    hint = _extract_float_hint(args, kwargs)
    if hint is not None:
        return open_interval_score(hint)
    return open_interval_score(0.0)


def grade_task2_manifest(*args: Any, **kwargs: Any) -> float:
    """Wrapper for Task 2 grader with broad signature compatibility."""
    incident = _extract_incident(args, kwargs)
    submission = _extract_mapping(
        args,
        kwargs,
        keys=("root_cause", "triggered_by", "affected_chain", "incident_type"),
        explicit_names=("submission", "action_params", "payload"),
    )

    steps_used = int(kwargs.get("steps_used", kwargs.get("steps_taken", 0)) or 0)
    action_history = kwargs.get("action_history") or kwargs.get("actions") or []
    if not isinstance(action_history, list):
        action_history = []

    if incident is not None:
        return open_interval_score(
            float(grade_task2(submission, incident, steps_used=steps_used, action_history=action_history))
        )

    hint = _extract_float_hint(args, kwargs)
    if hint is not None:
        return open_interval_score(hint)
    return open_interval_score(0.0)


def grade_task3_manifest(*args: Any, **kwargs: Any) -> float:
    """Wrapper for Task 3 grader with broad signature compatibility."""
    incident = _extract_incident(args, kwargs)
    action_history = kwargs.get("action_history") or kwargs.get("actions") or []
    if not isinstance(action_history, list):
        action_history = []

    episode_state = kwargs.get("episode_state") or kwargs.get("state") or {}
    if not isinstance(episode_state, dict):
        episode_state = {}

    for value in args:
        if isinstance(value, list) and not action_history:
            action_history = value
        if isinstance(value, dict) and not episode_state:
            episode_state = value

    if incident is not None:
        return open_interval_score(float(grade_task3(action_history, incident, episode_state)))

    hint = _extract_float_hint(args, kwargs)
    if hint is not None:
        return open_interval_score(hint)
    return open_interval_score(0.0)


__all__ = [
    "grade_task1_manifest",
    "grade_task2_manifest",
    "grade_task3_manifest",
]
