"""Dataset registration + EDA tools."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp.errors import build_error

logger = logging.getLogger(__name__)


class LoadDatasetInput(BaseModel):
    """Inputs for ``load_dataset``."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "Local filesystem path or s3:// URL to a CSV, TSV, Parquet, "
            "Excel, JSON, or JSONL file. The extension determines the "
            "DuckDB reader used."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Name to register the dataset under so other tools can refer to "
            "it. When omitted a slugified version of the filename stem is used."
        ),
    )
    read_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional DuckDB reader options forwarded into the read_* call, "
            "e.g. {\"header\": false, \"delim\": \";\"} for CSV. Use this "
            "only when auto-detection produces wrong results."
        ),
    )


def load_dataset(payload: LoadDatasetInput) -> dict[str, Any]:
    """Stub — always returns ok=True so the unsupported-format test fails."""
    return {"ok": True, "name": payload.name or "", "rows": 0, "columns": []}
