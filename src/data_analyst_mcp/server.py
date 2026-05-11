"""FastMCP server entry point.

Stdio transport only. Stdout is reserved for the MCP protocol — every log
record goes to stderr via the module-level handler configured below.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from data_analyst_mcp.errors import build_error
from data_analyst_mcp.tools import datasets as _datasets
from data_analyst_mcp.tools import query as _query
from data_analyst_mcp.tools import stats as _stats

_handler = logging.StreamHandler(stream=sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("data_analyst_mcp")

mcp: FastMCP = FastMCP("data-analyst-mcp")


@mcp.tool()
def query(sql: str, limit: int = 50) -> dict[str, Any]:
    """Run a read-only SQL query against the registered datasets.

    Accepts SELECT / WITH / DESCRIBE / SHOW / EXPLAIN / PRAGMA show_tables;
    rejects writes (INSERT / UPDATE / DELETE / DROP / CREATE / SET). A
    LIMIT is auto-applied if not present so result rows stay bounded.
    Returns rows + column names + total_rows (via a separate COUNT(*) over
    the same query) + execution_time_ms.
    """
    try:
        payload = _query.QueryInput(sql=sql, limit=limit)
        return _query.query(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("query failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def describe_column(name: str, column: str, bins: int = 20) -> dict[str, Any]:
    """Deep-dive a single column of a registered dataset.

    Numeric columns: returns the full quantile vector (1/5/10/25/50/75/
    90/95/99), skewness, kurtosis, IQR, histogram counts honoring ``bins``,
    and IQR + z>3 outliers with 5 example rows. Categorical columns: full
    value counts (capped at 50 with "other" bucket) and entropy. Temporal
    columns: counts by year/month/weekday/hour.
    """
    try:
        payload = _datasets.DescribeColumnInput(name=name, column=column, bins=bins)
        return _datasets.describe_column(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("describe_column failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def profile_dataset(name: str, sample_rows: int = 5) -> dict[str, Any]:
    """Produce a full EDA profile for a registered dataset.

    Reports total rows + columns, per-column dtype/null counts/distinct
    counts, numeric stats (min/max/mean/median/std/p25/p75/p99/zeros/
    negatives), string-length stats, temporal stats, top-5 most-frequent
    values per column, heuristic flags (``looks_like_id``,
    ``looks_like_categorical``, ``looks_like_timestamp``,
    ``high_cardinality``, ``mostly_null``, ``constant``,
    ``mixed_dtype_suspect``), a head sample of ``sample_rows`` rows, and a
    short list of suggested next actions. This is the headline EDA tool.
    """
    try:
        payload = _datasets.ProfileDatasetInput(name=name, sample_rows=sample_rows)
        return _datasets.profile_dataset(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("profile_dataset failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def list_datasets() -> dict[str, Any]:
    """List every dataset currently registered in this session.

    Each entry reports name, row count, column count, and the registration
    timestamp so the agent can pick a target for downstream tools without
    re-loading.
    """
    try:
        return _datasets.list_datasets()
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("list_datasets failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def load_dataset(
    path: str,
    name: str | None = None,
    read_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a file as a named dataset queryable by all other tools.

    The path extension picks the DuckDB reader (``.csv``/``.tsv``/``.parquet``/
    ``.json``/``.jsonl``/``.xlsx``). ``read_options`` is forwarded into the
    reader for cases where auto-detection fails (e.g. semicolon-delimited
    files). Reports row count, column names + dtypes, file size, and any
    parser warnings.
    """
    try:
        payload = _datasets.LoadDatasetInput(path=path, name=name, read_options=read_options)
        return _datasets.load_dataset(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("load_dataset failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def correlate(
    name: str,
    columns: list[str] | None = None,
    method: str = "pearson",
    plot: bool = True,
) -> dict[str, Any]:
    """Correlation matrix across numeric columns of a registered dataset.

    Method picks Pearson (default), Spearman, or Kendall. When ``plot`` is
    true, a base64-encoded PNG heatmap is also returned. When ``columns``
    is omitted, every numeric column in the dataset is used.
    """
    try:
        payload = _stats.CorrelateInput(
            name=name, columns=columns, method=method, plot=plot
        )
        return _stats.correlate(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("correlate failed")
        return build_error(type="internal", message=str(exc))


def main() -> None:  # pragma: no cover - exercised by the console-script smoke test
    """Run the MCP server on stdio. Console-script entry point."""
    logger.info("starting data-analyst-mcp on stdio")
    mcp.run()
