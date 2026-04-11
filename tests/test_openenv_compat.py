"""OpenEnv manifest entrypoint compatibility tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from models import ActionType, SREAction, SREObservation, SREReward


REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_valid_grader_ref(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith("tasks.manifest_graders:")
    if isinstance(value, dict):
        return value.get("module") == "tasks.manifest_graders" and bool(value.get("function"))
    return False


def _load_manifest_entrypoint_class():
    manifest = yaml.safe_load((REPO_ROOT / "openenv.yaml").read_text(encoding="utf-8"))
    entrypoint = manifest["entrypoint"]
    module_path, class_name = entrypoint.split(":")
    module_name = module_path.replace("/", ".").replace(".py", "")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def test_manifest_entrypoint_reset_and_step_contract():
    env_cls = _load_manifest_entrypoint_class()
    env = env_cls()

    try:
        observation = env.reset("task1", seed=0)
        assert isinstance(observation, SREObservation)

        next_observation, reward, done, info = env.step(
            SREAction(action_type=ActionType.ACKNOWLEDGE_ALERT)
        )
        assert isinstance(next_observation, SREObservation)
        assert isinstance(reward, SREReward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_manifest_entrypoint_exposes_three_tasks_with_graders():
    env_cls = _load_manifest_entrypoint_class()
    env = env_cls()

    try:
        tasks = env.list_tasks() if hasattr(env, "list_tasks") else getattr(env, "tasks", [])
        assert isinstance(tasks, list)
        assert len(tasks) >= 3
        assert {"task1", "task2", "task3"}.issubset({task.get("id") for task in tasks})
        assert all("grader" in task for task in tasks[:3])
        assert all(_is_valid_grader_ref(task["grader"]) for task in tasks[:3])
    finally:
        env.close()


def test_terminal_info_includes_grader_score_with_final_score_parity():
    env_cls = _load_manifest_entrypoint_class()
    env = env_cls()

    try:
        env.reset("task1", seed=0)
        _, _, done, info = env.step(SREAction(action_type=ActionType.ACKNOWLEDGE_ALERT))
        assert done is True
        assert "final_score" in info
        assert "grader_score" in info
        assert 0.0 < float(info["grader_score"]) < 1.0
        assert float(info["grader_score"]) == float(info["final_score"])
    finally:
        env.close()
