"""Startup regressions for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_startup_fails_fast_when_environment_initialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api as api_module

    def boom() -> object:
        raise FileNotFoundError("missing incidents.json")

    monkeypatch.setattr(api_module, "env", None)
    monkeypatch.setattr(api_module, "_create_env", boom)

    with pytest.raises(FileNotFoundError, match="missing incidents.json"):
        with TestClient(api_module.app):
            pass
