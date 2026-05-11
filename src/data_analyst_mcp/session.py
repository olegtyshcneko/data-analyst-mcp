"""Session singleton — dataset registry + a single DuckDB connection.

The session is module-level state. Tests must call :func:`reset` at the
start of every case via the autouse fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class DatasetEntry:
    """One row in the session's dataset registry."""

    path: str
    read_options: dict[str, Any]
    format: str
    rows: int
    columns: list[dict[str, str]]
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))


_datasets: dict[str, DatasetEntry] = {}


def register(
    *,
    name: str,
    path: str,
    read_options: dict[str, Any],
    format: str,
    rows: int,
    columns: list[dict[str, str]],
) -> None:
    """Insert (or replace) a dataset entry under ``name``."""
    _datasets[name] = DatasetEntry(
        path=path,
        read_options=dict(read_options),
        format=format,
        rows=rows,
        columns=list(columns),
    )


def get_datasets() -> dict[str, DatasetEntry]:
    """Return the live datasets registry (mutating this mutates the session)."""
    return _datasets


def reset() -> None:
    """Clear the datasets registry. The DuckDB connection persists."""
    _datasets.clear()
