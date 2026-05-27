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
    # method dispatch will be added in later TDD cycles
    return build_error(type="internal", message="method dispatch not yet implemented")
