"""Modeling tool — fit_model (OLS / logistic / Poisson)."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error

logger = logging.getLogger(__name__)


class FitModelInput(BaseModel):
    """Inputs for ``fit_model``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    formula: str = Field(
        ...,
        description="Wilkinson-style formula, e.g. 'price ~ sqft + C(neighborhood)'.",
    )
    kind: Literal["ols", "logistic", "poisson"] = Field(
        default="ols",
        description="Model family: ols (linear), logistic (binary), or poisson (counts).",
    )
    robust: bool = Field(
        default=False,
        description="When true and kind=ols, use HC3 heteroskedasticity-robust standard errors.",
    )


def fit_model(payload: FitModelInput) -> dict[str, Any]:
    """Fit an OLS / logistic / Poisson regression and return coefficients + diagnostics."""
    if payload.name not in session.get_datasets():
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    return build_error(
        type="not_implemented",
        message="fit_model is not yet implemented.",
    )
