"""Statistical tools — correlate, compare_groups, test_hypothesis."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error

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
    return {"ok": True}
