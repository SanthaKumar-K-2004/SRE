"""Workflow contract checks for GitHub-to-Space sync hardening."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sync_space_workflow_defaults_to_strict_sync() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "sync-space.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    steps = workflow["jobs"]["validate-and-sync"]["steps"]
    resolve_step = next(step for step in steps if step["name"] == "Resolve Hugging Face write token")
    script = resolve_step["run"]

    assert 'strict_mode="true"' in script
    assert '[ "${strict_mode_raw}" = "false" ]' in script
    assert "explicitly set REQUIRE_SPACE_SYNC=false to opt out" in script
