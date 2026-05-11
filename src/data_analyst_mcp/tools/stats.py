"""Statistical tools — correlate, compare_groups, test_hypothesis."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error

_NUMERIC_DTYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
}


def _is_numeric_dtype(dtype: str) -> bool:
    """True if a DuckDB dtype represents a numeric column."""
    base = dtype.split("(")[0].strip().upper()
    return base in _NUMERIC_DTYPES


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'

logger = logging.getLogger(__name__)


class CorrelateInput(BaseModel):
    """Inputs for ``correlate``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of column names to correlate. When omitted, every "
            "numeric column in the dataset is used."
        ),
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description="Correlation method: pearson, spearman, or kendall.",
    )
    plot: bool = Field(
        default=True,
        description="When true, include a base64-encoded PNG heatmap in the response.",
    )


def correlate(payload: CorrelateInput) -> dict[str, Any]:
    """Compute a correlation matrix across numeric columns."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    available = {c["name"] for c in entry.columns}
    if payload.columns is not None:
        missing = [c for c in payload.columns if c not in available]
        if missing:
            return build_error(
                type="column_not_found",
                message=f"Columns not in dataset {payload.name!r}: {missing}.",
                hint=f"Available: {sorted(available)}",
            )
        chosen = list(payload.columns)
    else:
        chosen = [c["name"] for c in entry.columns if _is_numeric_dtype(c["dtype"])]
        if not chosen:
            return build_error(
                type="no_numeric_columns",
                message=f"Dataset {payload.name!r} has no numeric columns.",
                hint="Pass an explicit `columns` list, or cast columns to numeric first.",
            )

    matrix = _build_corr_matrix(payload.name, chosen, payload.method)

    return {
        "ok": True,
        "method": payload.method,
        "labels": list(chosen),
        "matrix": matrix,
    }


def _build_corr_matrix(
    dataset_name: str, columns: list[str], method: str
) -> list[list[float]]:
    """Materialize columns then compute the correlation matrix."""
    from scipy import stats as _sps

    con = session.get_connection()
    table = _quote(dataset_name)
    select_cols = ", ".join(_quote(c) for c in columns)
    df = con.execute(f"SELECT {select_cols} FROM {table}").df()
    n = len(columns)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            a = df[columns[i]].to_numpy()
            b = df[columns[j]].to_numpy()
            if method == "pearson":
                r = float(_sps.pearsonr(a, b).statistic)
            elif method == "spearman":
                r = float(_sps.spearmanr(a, b).statistic)
            else:
                r = float(_sps.kendalltau(a, b).statistic)
            matrix[i][j] = r
            matrix[j][i] = r
    return matrix
