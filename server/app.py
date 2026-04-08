"""Canonical OpenEnv server entrypoint.

This module exposes `app` and `main()` so validators can find a server script.
"""

from __future__ import annotations

import os

import uvicorn

from api import app


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
