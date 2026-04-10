"""Packaging contract checks for OpenEnv validation compatibility."""

from __future__ import annotations

import importlib
from pathlib import Path
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_server_app_exists() -> None:
    assert (REPO_ROOT / "server" / "app.py").exists()


def test_uv_lock_exists() -> None:
    assert (REPO_ROOT / "uv.lock").exists()


def test_pyproject_contains_openenv_dependency_and_server_script() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    deps = data["project"]["dependencies"]
    scripts = data["project"]["scripts"]

    assert any(dep.startswith("openenv") and not dep.startswith("openenv-core") for dep in deps)
    assert scripts.get("server") == "server.app:main"


def test_openenv_manifest_contains_entrypoint_models_and_graders() -> None:
    manifest_path = REPO_ROOT / "openenv.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest.get("entrypoint") == "environment.py:SREBenchOpenEnv"

    models = manifest.get("models", {})
    assert models.get("observation") == "models.SREObservation"
    assert models.get("action") == "models.SREAction"
    assert models.get("reward") == "models.SREReward"

    task_map = {task.get("id"): task for task in manifest.get("tasks", [])}
    assert task_map["task1"]["grader"]["module"] == "tasks.manifest_graders"
    assert task_map["task1"]["grader"]["function"] == "grade_task1_manifest"
    assert task_map["task2"]["grader"]["module"] == "tasks.manifest_graders"
    assert task_map["task2"]["grader"]["function"] == "grade_task2_manifest"
    assert task_map["task3"]["grader"]["module"] == "tasks.manifest_graders"
    assert task_map["task3"]["grader"]["function"] == "grade_task3_manifest"
    assert manifest.get("api", {}).get("tasks", {}).get("path") == "/tasks"


def test_manifest_has_three_importable_task_graders() -> None:
    manifest_path = REPO_ROOT / "openenv.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])

    assert len(tasks) >= 3

    importable = 0
    for task in tasks:
        grader_ref = task.get("grader")
        assert isinstance(grader_ref, dict)
        module_name = grader_ref.get("module")
        function_name = grader_ref.get("function")
        assert isinstance(module_name, str)
        assert isinstance(function_name, str)
        module = importlib.import_module(module_name)
        assert hasattr(module, function_name)
        assert callable(getattr(module, function_name))
        importable += 1

    assert importable >= 3
