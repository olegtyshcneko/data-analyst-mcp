"""Session singleton — dataset registry + a single DuckDB connection.

The session is module-level state. Tests must call :func:`reset` at the
start of every case via the autouse fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from typing import Any


def get_datasets() -> dict[str, Any]:
    """Stub — returns a sentinel so the empty-dict test fails."""
    return {"_sentinel": True}


def reset() -> None:
    """Stub."""
    return None
