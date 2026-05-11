"""Session singleton — dataset registry + a single DuckDB connection.

The session is module-level state. Tests must call :func:`reset` at the
start of every case via the autouse fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from typing import Any

_datasets: dict[str, Any] = {}


def get_datasets() -> dict[str, Any]:
    """Return the live datasets registry (mutating this mutates the session)."""
    return _datasets


def reset() -> None:
    """Clear the datasets registry. The DuckDB connection persists."""
    _datasets.clear()
