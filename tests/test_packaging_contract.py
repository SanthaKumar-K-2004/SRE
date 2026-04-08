"""Packaging contract checks for OpenEnv validation compatibility."""

from __future__ import annotations

from pathlib import Path
import tomllib


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
