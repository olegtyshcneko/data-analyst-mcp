"""Modeling tool — fit_model (OLS / logistic / Poisson)."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

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
        result = _fit_dispatch(payload, df)
    except _FormulaError as fe:
        return build_error(
            type="formula_error",
            message=str(fe),
            hint=(
                "Verify column names exist and the formula parses, e.g. "
                "'y ~ x + C(group)'."
            ),
        )
    _record_fit_model(payload, result)
    return result


def _record_fit_model(payload: FitModelInput, result: dict[str, Any]) -> None:
    """Append a markdown + code cell pair describing the fit_model call."""
    if not result.get("ok"):
        return
    md_lines = [
        f"### Fitted {payload.kind.upper()} model on `{payload.name}`",
        f"- Formula: `{payload.formula}`",
        f"- {result['interpretation']}",
    ]
    fit = result["fit"]
    if "r_squared" in fit:
        md_lines.append(f"- R² = {fit['r_squared']:.4f} (adj {fit['adj_r_squared']:.4f})")
    elif "pseudo_r_squared" in fit:
        md_lines.append(f"- pseudo-R² = {fit['pseudo_r_squared']:.4f}")
    md_lines.append(f"- AIC = {fit['aic']:.2f}, BIC = {fit['bic']:.2f}")
    if result["warnings"]:
        md_lines.append(f"- Warnings: {', '.join(result['warnings'])}")
    md = "\n".join(md_lines)
    code = _code_for_fit(payload)
    get_recorder().record(markdown=md, code=code, tool_name="fit_model")


def _code_for_fit(payload: FitModelInput) -> str:
    """Render a reproducible statsmodels snippet for the fit_model call."""
    fn = {"ols": "ols", "logistic": "logit", "poisson": "poisson"}[payload.kind]
    fit_args = ""
    if payload.kind == "ols" and payload.robust:
        fit_args = 'cov_type="HC3"'
    elif payload.kind in {"logistic", "poisson"}:
        fit_args = "disp=0"
    return (
        f"import statsmodels.formula.api as smf\n"
        f'df = con.sql("SELECT * FROM {payload.name}").df()\n'
        f'model = smf.{fn}("{payload.formula}", data=df).fit({fit_args})\n'
        f"model.summary()"
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
        else:
            raise _FormulaError(f"kind={payload.kind!r} not yet supported")
    except Exception as exc:
        # Patsy / NameError / column-binding failures all bubble up here.
        raise _FormulaError(str(exc)) from exc

    diagnostics = _diagnostics(m, payload.kind)
    coefficients = _coefficients(m)
    return {
        "ok": True,
        "coefficients": coefficients,
        "fit": _fit_block(m, payload.kind),
        "diagnostics": diagnostics,
        "warnings": _warnings(m, diagnostics, payload.kind),
        "interpretation": _interpretation(coefficients, payload.kind),
    }


def _interpretation(coefficients: list[dict[str, Any]], kind: str) -> str:
    """Return a 2-3 sentence plain-English summary of the fitted model.

    Picks the strongest non-intercept signal (smallest p-value) and reports
    its direction and magnitude in kind-appropriate terms:
      - OLS: a unit change in the predictor moves the outcome by ``estimate``;
      - logistic: the predictor's odds-ratio (``exp(estimate) - 1``);
      - Poisson: the multiplicative effect on the expected count (``exp``).
    """
    import math

    non_intercept = [c for c in coefficients if c["name"] != "Intercept"]
    if not non_intercept:
        return "Model fit succeeded; only an intercept term was estimated."
    strongest = min(non_intercept, key=lambda c: c["p_value"])
    direction = "positive" if strongest["estimate"] >= 0 else "negative"
    name = strongest["name"]
    p = strongest["p_value"]
    est = strongest["estimate"]
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    if kind == "logistic":
        odds = math.exp(est) - 1
        pct = odds * 100.0
        return (
            f"Strongest predictor: `{name}` ({direction} effect, {sig} at α=0.05, p={p:.4g}). "
            f"A one-unit increase changes the odds by ~{pct:.1f}% (odds ratio = {math.exp(est):.3f})."
        )
    if kind == "poisson":
        mult = math.exp(est)
        pct = (mult - 1.0) * 100.0
        return (
            f"Strongest predictor: `{name}` ({direction} effect, {sig} at α=0.05, p={p:.4g}). "
            f"A one-unit increase multiplies the expected count by {mult:.3f} (~{pct:.1f}%)."
        )
    return (
        f"Strongest predictor: `{name}` ({direction} effect, {sig} at α=0.05, p={p:.4g}). "
        f"A one-unit increase moves the response by {est:.4g} units."
    )


def _warnings(m: Any, diagnostics: dict[str, Any], kind: str) -> list[str]:
    """Translate diagnostic numbers into a list of human-readable warning tags.

    ``high_multicollinearity`` fires for any kind that has a design matrix
    with VIF > 10 — we compute VIFs internally even for logistic/poisson so
    the warning fires consistently, though only OLS surfaces them in the
    ``diagnostics.vif`` block per spec.
    """
    out: list[str] = []
    vif = diagnostics.get("vif")
    if not isinstance(vif, dict):
        vif = _vif_per_coefficient(m)
    if any(v > 10 for v in vif.values()):
        out.append("high_multicollinearity")
    bp = diagnostics.get("breusch_pagan_p")
    if kind == "ols" and isinstance(bp, float) and bp < 0.05:
        out.append("heteroskedasticity")
    jb = diagnostics.get("jarque_bera_p")
    if kind == "ols" and isinstance(jb, float) and jb < 0.05:
        out.append("non_normal_residuals")
    return out


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
