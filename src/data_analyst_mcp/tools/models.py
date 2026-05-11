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
        "fit": _fit_block(m, payload.kind),
        "diagnostics": _diagnostics(m, payload.kind),
    }


def _diagnostics(m: Any, kind: str) -> dict[str, Any]:
    """Build the diagnostics envelope.

    OLS gets the full battery (Breusch-Pagan, Durbin-Watson, Jarque-Bera,
    condition number). Logistic / Poisson only get ``condition_number`` —
    BP/DW/JB are OLS-specific residual diagnostics and are reported as
    ``None`` for non-OLS kinds to keep a stable shape.
    """
    import numpy as np

    out: dict[str, Any] = {"condition_number": _condition_number(m)}
    if kind == "ols":
        from statsmodels.stats.diagnostic import het_breuschpagan  # type: ignore[reportMissingTypeStubs]
        from statsmodels.stats.stattools import (  # type: ignore[reportMissingTypeStubs]
            durbin_watson,
            jarque_bera,
        )

        bp = het_breuschpagan(np.asarray(m.resid), np.asarray(m.model.exog))
        jb = jarque_bera(np.asarray(m.resid))
        out["breusch_pagan_p"] = float(bp[1])
        out["durbin_watson"] = float(durbin_watson(np.asarray(m.resid)))
        out["jarque_bera_p"] = float(jb[1])
        out["vif"] = _vif_per_coefficient(m)
    else:
        out["breusch_pagan_p"] = None
        out["durbin_watson"] = None
        out["jarque_bera_p"] = None
        out["vif"] = None
    return out


def _vif_per_coefficient(m: Any) -> dict[str, float]:
    """Compute the variance-inflation factor for each non-intercept regressor."""
    import numpy as np
    from statsmodels.stats.outliers_influence import (  # type: ignore[reportMissingTypeStubs]
        variance_inflation_factor,
    )

    exog: Any = np.asarray(m.model.exog)
    names: list[str] = list(m.model.exog_names)
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        if name == "Intercept":
            continue
        out[name] = float(variance_inflation_factor(exog, i))
    return out


def _condition_number(m: Any) -> float:
    """Return the condition number of the design matrix as a float.

    OLS results expose ``condition_number`` directly; for GLM-family results
    we fall back to ``numpy.linalg.cond`` on the exog matrix so the field is
    always populated.
    """
    cn = getattr(m, "condition_number", None)
    if cn is not None:
        return float(cn)
    import numpy as np

    return float(np.linalg.cond(np.asarray(m.model.exog)))


def _fit_block(m: Any, kind: str) -> dict[str, Any]:
    """Build the goodness-of-fit envelope.

    OLS reports ``r_squared`` / ``adj_r_squared``; logistic and Poisson swap
    those for ``pseudo_r_squared`` (McFadden's R^2 via ``m.prsquared``).
    """
    out: dict[str, Any] = {
        "aic": float(m.aic),
        "bic": float(m.bic),
        "n_obs": int(m.nobs),
        "df_resid": int(m.df_resid),
    }
    if kind == "ols":
        out["r_squared"] = float(m.rsquared)
        out["adj_r_squared"] = float(m.rsquared_adj)
    else:
        out["pseudo_r_squared"] = float(m.prsquared)
    return out


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
