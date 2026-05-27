"""Multi-column outlier detection — IQR / z-score / Mahalanobis / Isolation Forest."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error

logger = logging.getLogger(__name__)


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


class FindOutliersInput(BaseModel):
    """Inputs for ``find_outliers``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    columns: list[str] = Field(min_length=1)
    method: Literal["iqr", "zscore", "mahalanobis", "isolation_forest"]
    threshold: float | None = None
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)
    limit: int = Field(default=50, ge=1, le=10_000)


def find_outliers(payload: FindOutliersInput) -> dict[str, Any]:
    """Detect outliers via one of four methods over a numeric column subset."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    available = {c["name"] for c in entry.columns}
    missing = [c for c in payload.columns if c not in available]
    if missing:
        return build_error(
            type="column_not_found",
            message=f"Columns not in dataset {payload.name!r}: {missing}.",
            hint=f"Available: {sorted(available)}",
        )
    dtype_by_name = {c["name"]: c["dtype"] for c in entry.columns}
    for col in payload.columns:
        if not _is_numeric_dtype(dtype_by_name[col]):
            return build_error(
                type="non_numeric_column",
                message=(
                    f"Column {col!r} has non-numeric dtype "
                    f"{dtype_by_name[col]!r}; find_outliers requires numeric columns."
                ),
                hint="Cast the column to a numeric type or pick a different column.",
            )

    if payload.method == "iqr":
        return _iqr_method(payload)
    if payload.method == "zscore":
        return _zscore_method(payload)
    return build_error(type="internal", message="method dispatch not yet implemented")


def _materialize_columns_df(name: str, columns: list[str]) -> Any:
    """Pull the column subset of ``name`` as a pandas DataFrame via DuckDB."""
    con = session.get_connection()
    quoted_cols = ", ".join('"' + c.replace('"', '""') + '"' for c in columns)
    quoted_table = '"' + name.replace('"', '""') + '"'
    return con.execute(f"SELECT {quoted_cols} FROM {quoted_table}").df()


def _iqr_method(payload: FindOutliersInput) -> dict[str, Any]:
    """IQR per-column flag + union row score = max normalized excess."""
    from data_analyst_mcp.tools._outlier_helpers import iqr_column_mask

    threshold = payload.threshold if payload.threshold is not None else 1.5
    return _per_column_union(
        payload,
        method_label="iqr",
        threshold=threshold,
        per_column_fn=lambda values: iqr_column_mask(values, threshold=threshold),
    )


def _zscore_method(payload: FindOutliersInput) -> dict[str, Any]:
    """Z-score per-column flag + union row score = max |z|."""
    from data_analyst_mcp.tools._outlier_helpers import zscore_column_mask

    # Custom threshold lands in the zscore-custom-threshold cycle.
    threshold = 3.0
    return _per_column_union(
        payload,
        method_label="zscore",
        threshold=threshold,
        per_column_fn=lambda values: zscore_column_mask(values, threshold=threshold),
    )


def _per_column_union(
    payload: FindOutliersInput,
    *,
    method_label: str,
    threshold: float,
    per_column_fn: Any,
) -> dict[str, Any]:
    """Apply ``per_column_fn`` to every selected column and union the flags.

    Each column contributes a boolean mask and a per-row score; the row
    is flagged when any column flags it, and its row-level score is the
    max across columns. ``per_column_flags`` records the indices flagged
    by each column independently.
    """
    import numpy as np

    df = _materialize_columns_df(payload.name, list(payload.columns))
    n_rows = len(df)
    row_mask = np.zeros(n_rows, dtype=bool)
    row_score = np.zeros(n_rows, dtype=float)
    per_column_flags: dict[str, list[int]] = {}
    for col in payload.columns:
        mask, score = per_column_fn(df[col].to_numpy())
        row_mask |= mask
        row_score = np.maximum(row_score, score)
        per_column_flags[col] = [int(i) for i in np.nonzero(mask)[0].tolist()]

    flagged_indices = np.nonzero(row_mask)[0]
    n_outliers = int(flagged_indices.size)
    order = sorted(flagged_indices.tolist(), key=lambda i: row_score[i], reverse=True)
    truncated = n_outliers > payload.limit
    chosen = order[: payload.limit]
    outliers = [
        {
            "row_index": int(i),
            "score": float(row_score[i]),
            "values": {col: _json_value(df[col].iloc[i]) for col in payload.columns},
        }
        for i in chosen
    ]

    return {
        "ok": True,
        "method": method_label,
        "n_outliers": n_outliers,
        "n_rows_scored": n_rows,
        "outliers": outliers,
        "truncated": truncated,
        "threshold_used": float(threshold),
        "warnings": [],
        "per_column_flags": per_column_flags,
    }


def _json_value(value: Any) -> Any:
    """Coerce a pandas scalar to a JSON-serializable Python primitive."""
    import math

    import numpy as np

    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
