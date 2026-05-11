"""Modeling tool — fit_model (OLS / logistic / Poisson)."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error

logger = logging.getLogger(__name__)


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _smf() -> Any:
    """Return ``statsmodels.formula.api`` as an untyped module."""
    import statsmodels.formula.api as _smf  # type: ignore[reportMissingTypeStubs]

    return _smf


def _materialize_dataframe(name: str) -> Any:
    """Materialize a registered dataset as a pandas DataFrame via DuckDB."""
    con = session.get_connection()
    return con.execute(f"SELECT * FROM {_quote(name)}").df()


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
    try:
        df = _materialize_dataframe(payload.name)
        return _fit_dispatch(payload, df)
    except _FormulaError as fe:
        return build_error(
            type="formula_error",
            message=str(fe),
            hint=(
                "Verify column names exist and the formula parses, e.g. "
                "'y ~ x + C(group)'."
            ),
        )


class _FormulaError(Exception):
    """Internal marker for formula / patsy / column-binding failures."""


def _fit_dispatch(payload: FitModelInput, df: Any) -> dict[str, Any]:
    """Pick the right statsmodels entry point and translate failures."""
    smf = _smf()
    try:
        if payload.kind == "ols":
            cov_type = "HC3" if payload.robust else "nonrobust"
            m = smf.ols(payload.formula, data=df).fit(cov_type=cov_type)
        elif payload.kind == "logistic":
            m = smf.logit(payload.formula, data=df).fit(disp=0)
        else:  # poisson
            m = smf.poisson(payload.formula, data=df).fit(disp=0)
    except Exception as exc:
        # Patsy / NameError / column-binding failures all bubble up here.
        raise _FormulaError(str(exc)) from exc

    return {
        "ok": True,
        "coefficients": _coefficients(m),
    }


def _coefficients(m: Any) -> list[dict[str, Any]]:
    """Build the per-coefficient envelope from a fitted statsmodels result."""
    params: Any = m.params
    bse: Any = m.bse
    tvals: Any = m.tvalues
    pvals: Any = m.pvalues
    ci: Any = m.conf_int()
    out: list[dict[str, Any]] = []
    for name in params.index:
        out.append(
            {
                "name": str(name),
                "estimate": float(params[name]),
                "std_err": float(bse[name]),
                "t": float(tvals[name]),
                "p_value": float(pvals[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
            }
        )
    return out
