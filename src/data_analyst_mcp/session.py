"""Session singleton — dataset registry + model registry + a single DuckDB connection.

The session is module-level state. Tests must call :func:`reset` at the
start of every case via the autouse fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from data_analyst_mcp.provenance import compute_source_hash

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
    # When a derived dataset (materialize_query) overwrites a file-backed
    # entry, base_loader retains the original file loader as
    # ``{"path", "format", "read_options"}`` so the recorder can re-create the
    # base table before the derived SQL — which often self-references the same
    # name (transform-in-place) — runs at notebook-replay time.
    base_loader: dict[str, Any] | None = None
    # Content hash of the source file at registration time (the recorder's
    # drift-guard anchor). ``sentinel:``-prefixed when there is no
    # verifiable file. Default covers direct constructions in tests.
    source_hash: str = "sentinel:unset"
    # Monotonic per-session registration revision stamped by register().
    # Identity of the *registration*, not the content: replacement through
    # any tool (materialize_query, load_dataset, split_dataset) gets a fresh
    # value even when source_hash stays constant (per-format sentinels,
    # byte-identical reloads). Default covers direct constructions in tests;
    # register() always stamps >= 0.
    revision: int = -1
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
    # Fit-time registration revision of the training dataset — the model
    # guard's identity anchor: the recorder trusts the current entry only
    # when this matches entry.revision (or base_loader["revision"] for a
    # fit on the pre-overwrite file-backed state). -1 for direct
    # constructions in tests.
    training_dataset_revision: int = -1
    # Fit-time loader identity {"path", "format", "read_options"} — proves a
    # later same-name reload has the same loading semantics, which a content
    # hash alone cannot (identical bytes re-parse differently under changed
    # read_options).
    training_loader: dict[str, Any] | None = None


_datasets: dict[str, DatasetEntry] = {}
_models: dict[str, ModelEntry] = {}
_connection: duckdb.DuckDBPyConnection | None = None
_revision_counter = 0


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return the per-process DuckDB connection, creating it on first use.

    The connection is created with filesystem access disabled
    (``enable_external_access=false``). This is the security boundary for the
    ``query`` / ``materialize_query`` tools, which execute agent-supplied SQL:
    it blocks *every* DuckDB file-read vector at once — ``read_csv('/etc/passwd')``
    and friends, the ``SELECT * FROM '/etc/passwd.csv'`` replacement scan,
    ``glob('/etc/*')``, ``COPY``, ``ATTACH``, extension installs — rather than
    chasing a per-function denylist (which replacement scans defeat outright).

    The setting is a one-way latch in DuckDB: once off it cannot be re-enabled
    while the database is running, so no agent-supplied SQL can turn it back on
    (verified: ``SET``/``PRAGMA enable_external_access=true`` both raise). File
    loading, which legitimately needs disk access, runs on a separate
    short-lived connection instead — see :func:`read_file_as_df`.
    """
    global _connection
    if _connection is None:
        import duckdb as _duckdb

        _connection = _duckdb.connect()
        # Must be set before any query runs; it can never be re-enabled after.
        _connection.execute("SET enable_external_access=false")
    return _connection


def read_file_as_df(read_call: str) -> Any:
    """Read a file via a throwaway connection *with* filesystem access and
    return the rows as a pandas DataFrame.

    File loading is deliberately isolated from :func:`get_connection`: the main
    connection has ``enable_external_access=false`` so untrusted query SQL can
    never touch the host filesystem. This short-lived connection is the only
    place disk (or ``s3://`` / ``http`` — ``httpfs`` auto-loads here) access
    happens. The caller registers the returned DataFrame into the main
    connection in memory, so the loaded data ends up queryable there without
    ever exposing the filesystem to the query path.

    ``read_call`` is a fully-rendered DuckDB ``read_*`` table-function call
    built by the trusted loader (``tools.datasets._build_read_call``) — never
    agent-supplied SQL.
    """
    import duckdb as _duckdb

    loader = _duckdb.connect()
    try:
        return loader.execute(f"SELECT * FROM {read_call}").df()
    finally:
        loader.close()


def register(
    *,
    name: str,
    path: str,
    read_options: dict[str, Any],
    format: str,
    rows: int,
    columns: list[dict[str, str]],
    base_loader: dict[str, Any] | None = None,
) -> None:
    """Insert (or replace) a dataset entry under ``name``."""
    global _revision_counter
    _datasets[name] = DatasetEntry(
        path=path,
        read_options=dict(read_options),
        format=format,
        rows=rows,
        columns=list(columns),
        base_loader=dict(base_loader) if base_loader is not None else None,
        source_hash=compute_source_hash(path),
        revision=_revision_counter,
    )
    _revision_counter += 1


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
    training_dataset_revision: int = -1,
    training_loader: dict[str, Any] | None = None,
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
        training_dataset_revision=training_dataset_revision,
        training_loader=dict(training_loader) if training_loader is not None else None,
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

    Derived datasets (``format == "derived"``) share this same cleanup path:
    the DROP TABLE loop iterates every registered name regardless of format,
    so a derived table created by ``materialize_query`` is dropped and its
    registry entry cleared just like any file-backed dataset.
    """
    global _revision_counter
    if _connection is not None:
        for name in list(_datasets.keys()):
            # Escape double quotes in the table name by doubling them for SQL.
            escaped_name = name.replace('"', '""')
            _connection.execute(f'DROP TABLE IF EXISTS "{escaped_name}"')
    _datasets.clear()
    _models.clear()
    _revision_counter = 0
