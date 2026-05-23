"""Session singleton — dataset registry + model registry + a single DuckDB connection.

The session is module-level state. Tests must call :func:`reset` at the
start of every case via the autouse fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb


@dataclass(frozen=True)
class DatasetEntry:
    """One row in the session's dataset registry."""

    path: str
    read_options: dict[str, Any]
    format: str
    rows: int
    columns: list[dict[str, str]]
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class ModelEntry:
    """One row in the session's fitted-model registry.

    ``_result`` carries the live statsmodels Results object for downstream
    ``predict`` / ``evaluate_model`` calls. It is **never** serialized into
    a JSON tool response — only the metadata fields are. The recorder
    rehydrates models at notebook-replay time by re-fitting on the training
    dataset (see ``training_dataset_hash`` for the drift guard).
    """

    name: str
    kind: str  # "ols" | "logistic" | "poisson" | "negbin"
    formula: str
    fitted_on_dataset: str
    n_obs: int
    fitted_at: datetime
    training_dataset_hash: str
    _result: Any  # statsmodels Results object, in-process only


_datasets: dict[str, DatasetEntry] = {}
_models: dict[str, ModelEntry] = {}
_connection: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the per-process DuckDB connection, creating it on first use."""
    global _connection
    if _connection is None:
        import duckdb as _duckdb

        _connection = _duckdb.connect()
    return _connection


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


def register_model(
    *,
    name: str,
    kind: str,
    formula: str,
    fitted_on_dataset: str,
    n_obs: int,
    training_dataset_hash: str,
    result: Any,
) -> None:
    """Insert a fitted model under ``name``. Raises ``KeyError`` on collision.

    Callers that need the structured-error path should check membership
    against :func:`get_models` first; the raw ``KeyError`` is a footgun
    guard for any code path that bypasses validation.
    """
    if name in _models:
        raise KeyError(f"Model name {name!r} is already registered.")
    _models[name] = ModelEntry(
        name=name,
        kind=kind,
        formula=formula,
        fitted_on_dataset=fitted_on_dataset,
        n_obs=n_obs,
        fitted_at=datetime.now(UTC),
        training_dataset_hash=training_dataset_hash,
        _result=result,
    )


def get_models() -> dict[str, ModelEntry]:
    """Return the live model registry (mutating this mutates the session)."""
    return _models


def get_model(name: str) -> ModelEntry | None:
    """Look up a registered model by name. Returns ``None`` when absent."""
    return _models.get(name)


def reset() -> None:
    """Drop registered tables and clear both registries. Connection persists.

    The DuckDB connection is intentionally kept open so that ``con`` handles
    held by callers (e.g. test fixtures, the live tool layer) remain valid
    across resets. Only the registry-known tables are dropped so unrelated
    in-memory state on the connection is left alone. Model entries are
    dropped alongside datasets — Python GC handles cleanup of the live
    statsmodels Results objects.
    """
    if _connection is not None:
        for name in list(_datasets.keys()):
            _connection.execute(f'DROP TABLE IF EXISTS "{name}"')
    _datasets.clear()
    _models.clear()
