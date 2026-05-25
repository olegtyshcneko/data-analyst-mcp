"""Modeling tool — fit_model (OLS / logistic / Poisson / negative binomial).

Also hosts ``list_models`` — a read-only inspection helper that mirrors
``list_datasets`` (no recorder cell, no state mutation).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)

# Above this file size (bytes) we skip content-hashing in favour of a
# cheap ``(path, mtime, size)`` tuple — content-hash on a 5 GB CSV is
# slow enough that the recorder pause is user-visible. Documented as a
# weaker drift guarantee in §Open question 2 of the model_registry
# proposal.
_HASH_CONTENT_CEILING_BYTES = 100 * 1024 * 1024


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _smf() -> Any:
    """Return ``statsmodels.formula.api`` as an untyped module."""
    import statsmodels.formula.api as _smf  # type: ignore[reportMissingTypeStubs]

    return _smf


def _sm_diagnostic() -> Any:
    """Return ``statsmodels.stats.diagnostic`` as an untyped module."""
    import statsmodels.stats.diagnostic as _d  # type: ignore[reportMissingTypeStubs]

    return _d


def _sm_stattools() -> Any:
    """Return ``statsmodels.stats.stattools`` as an untyped module."""
    import statsmodels.stats.stattools as _s  # type: ignore[reportMissingTypeStubs]

    return _s


def _sm_outliers() -> Any:
    """Return ``statsmodels.stats.outliers_influence`` as an untyped module."""
    import statsmodels.stats.outliers_influence as _o  # type: ignore[reportMissingTypeStubs]

    return _o


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
    kind: Literal["ols", "logistic", "poisson", "negbin"] = Field(
        default="ols",
        description=(
            "Model family: ols (linear), logistic (binary), poisson (counts), or "
            "negbin (NB2 negative binomial for overdispersed counts)."
        ),
    )
    robust: bool = Field(
        default=False,
        description="When true and kind=ols, use HC3 heteroskedasticity-robust standard errors.",
    )
    model_name: str | None = Field(
        default=None,
        description=(
            "Optional name to register the fitted result under in the session "
            "model registry. When provided, subsequent predict / evaluate_model "
            "/ list_models calls can reference it. Must be a non-empty string "
            "with no whitespace; duplicates are rejected (model_name_collision)."
        ),
    )


def _is_valid_model_name(name: str) -> bool:
    """Validate registry handle: non-empty, no whitespace."""
    if not name:
        return False
    return not any(ch.isspace() for ch in name)


def compute_training_dataset_hash(path: str) -> str:
    """Hash a training dataset for the recorder's drift guard.

    Files up to ``_HASH_CONTENT_CEILING_BYTES`` are content-hashed
    (SHA-256 of bytes). Larger files fall back to a cheap
    ``(path, mtime, size)`` tuple — documented as a weaker guarantee in
    the proposal. In-memory datasets (``path == "(dataframe)"``) and any
    other non-file path are tagged with a sentinel so the recorder's
    rehydration cell can detect them and skip the hash assert without
    silently mismatching.
    """
    if not os.path.isfile(path):
        # In-memory datasets, s3:// URLs, anything we cannot stat get a
        # stable sentinel so collisions across runs are deterministic.
        return f"sentinel:no-file:{path}"
    try:
        size = os.path.getsize(path)
    except OSError:
        return f"sentinel:stat-failed:{path}"
    if size <= _HASH_CONTENT_CEILING_BYTES:
        h = hashlib.sha256()
        # Stream in 1 MB chunks; SHA-256 of a 100 MB file at ~500 MB/s is
        # under a quarter second on commodity hardware.
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    # Above the ceiling: fall back to (path, mtime, size). Weaker guarantee
    # — a careful edit that preserves mtime + size will not trigger the
    # drift assert — but content-hashing 5 GB is too slow for an
    # interactive session.
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    fallback_hash = hashlib.sha256(f"{path}|{mtime}|{size}".encode()).hexdigest()
    return f"fallback:{fallback_hash}"


def fit_model(payload: FitModelInput) -> dict[str, Any]:
    """Fit an OLS / logistic / Poisson / negbin regression and return coefficients + diagnostics."""
    if payload.name not in session.get_datasets():
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    # Pre-fit guard: robust SE is not supported on negative binomial.
    if payload.kind == "negbin" and payload.robust:
        return build_error(
            type="robust_not_supported",
            message="robust=True is not supported for kind='negbin'.",
            hint="NB2 robust SE is not implemented in this server. Set `robust=False`.",
        )
    # Model-registry validation runs *before* the fit so we never burn
    # compute on a fit whose result we can't store.
    if payload.model_name is not None:
        if not _is_valid_model_name(payload.model_name):
            return build_error(
                type="model_name_invalid",
                message=(
                    f"Model name {payload.model_name!r} is invalid: must be a "
                    "non-empty string with no whitespace."
                ),
                hint="Match the dataset-name rules: identifier-style strings only.",
            )
        if payload.model_name in session.get_models():
            return build_error(
                type="model_name_collision",
                message=(f"A model named {payload.model_name!r} is already registered."),
                hint=(
                    "Use a different name or call session.reset() to clear the "
                    "registry (no delete_model tool by design)."
                ),
            )
    try:
        df = _materialize_dataframe(payload.name)
        # Pre-fit guard: NB requires non-negative integer endog.
        extra_warnings: list[str] = []
        if payload.kind == "negbin":
            df, validation_error, coerce_warning = _validate_negbin_endog(df, payload.formula)
            if validation_error is not None:
                return validation_error
            if coerce_warning is not None:
                extra_warnings.append(coerce_warning)
        result = _fit_dispatch(payload, df)
    except _FormulaError as fe:
        return build_error(
            type="formula_error",
            message=str(fe),
            hint=("Verify column names exist and the formula parses, e.g. 'y ~ x + C(group)'."),
        )
    if result.get("ok") and extra_warnings:
        # Prepend pre-fit warnings so they appear before model-derived ones.
        existing: list[str] = list(result.get("warnings") or [])
        result["warnings"] = extra_warnings + existing
    # Extract the live statsmodels Results before recording / returning so
    # JSON serialization never sees it. ``_result`` is keyed underscore-
    # prefixed precisely to make leaks visually obvious if a future caller
    # forgets to strip it.
    live_result = result.pop("_result", None)
    if result.get("ok") and payload.model_name is not None and live_result is not None:
        ds_path = session.get_datasets()[payload.name].path
        n_obs_val = int(result["fit"]["n_obs"])
        session.register_model(
            name=payload.model_name,
            kind=payload.kind,
            formula=payload.formula,
            fitted_on_dataset=payload.name,
            n_obs=n_obs_val,
            training_dataset_hash=compute_training_dataset_hash(ds_path),
            result=live_result,
        )
        result["model_name"] = payload.model_name
    _record_fit_model(payload, result)
    return result


def _resolve_endog_name(formula: str) -> str:
    """Extract the left-hand-side column name from a Wilkinson formula."""
    lhs = formula.split("~", 1)[0].strip()
    # Strip surrounding function wrappers like `log(y)` → `y`. Conservative
    # parse: if the LHS is an identifier (no parens), return as-is; otherwise
    # bail out and let patsy raise a normal formula_error downstream.
    if lhs.isidentifier():
        return lhs
    return lhs


def _validate_negbin_endog(df: Any, formula: str) -> tuple[Any, dict[str, Any] | None, str | None]:
    """Validate that the negbin endog column is non-negative integer-valued.

    Returns ``(df, error_envelope, coerce_warning)``:
      - ``df`` may be a copy with the endog cast to integer dtype when the
        original was float-but-integer-valued (e.g. ``[1.0, 2.0]``).
      - ``error_envelope`` is the structured ``build_error`` dict when the
        endog has negatives, NaN, or non-integer floats; ``None`` otherwise.
      - ``coerce_warning`` is ``"coerced_float_to_int"`` when the column was
        float dtype but every value was integer-valued, otherwise ``None``.
    """
    import numpy as np
    import pandas as pd  # type: ignore[reportMissingTypeStubs]

    endog_name = _resolve_endog_name(formula)
    if endog_name not in df.columns:
        # Let patsy raise the column-missing error during dispatch.
        return df, None, None
    col: Any = df[endog_name]
    # NaN check first — affects both int and float-typed columns (extension dtypes can be int+NA).
    n_nan: int = int(col.isna().sum())
    if n_nan > 0:
        return (
            df,
            build_error(
                type="negbin_requires_nonneg_int",
                message=(
                    f"Negative binomial requires non-negative integer counts; "
                    f"column '{endog_name}' has {n_nan} NaN values."
                ),
                hint=(
                    "Drop or impute missing rows before fitting, e.g. "
                    f"df = df.dropna(subset=['{endog_name}'])."
                ),
            ),
            None,
        )
    values: Any = np.asarray(col)
    if pd.api.types.is_float_dtype(col.dtype):  # type: ignore[reportUnknownMemberType]
        non_integer = np.sum(values != np.floor(values))
        if int(non_integer) > 0:
            return (
                df,
                build_error(
                    type="negbin_requires_nonneg_int",
                    message=(
                        f"Negative binomial requires non-negative integer counts; "
                        f"column '{endog_name}' has {int(non_integer)} non-integer values."
                    ),
                    hint=(
                        "NB models discrete counts; cast or aggregate the column "
                        "to integers before fitting."
                    ),
                ),
                None,
            )
        if np.any(values < 0):
            n_neg = int(np.sum(values < 0))
            return (
                df,
                build_error(
                    type="negbin_requires_nonneg_int",
                    message=(
                        f"Negative binomial requires non-negative integer counts; "
                        f"column '{endog_name}' has {n_neg} negative values."
                    ),
                    hint=(
                        "Negative outcomes are incompatible with a count model; "
                        "check whether the column is signed by mistake."
                    ),
                ),
                None,
            )
        # All values are integer-valued floats — coerce to int and warn.
        df = df.copy()
        df[endog_name] = values.astype(np.int64)
        return df, None, "coerced_float_to_int"
    # Integer dtype path — just check non-negativity.
    if np.any(values < 0):
        n_neg = int(np.sum(values < 0))
        return (
            df,
            build_error(
                type="negbin_requires_nonneg_int",
                message=(
                    f"Negative binomial requires non-negative integer counts; "
                    f"column '{endog_name}' has {n_neg} negative values."
                ),
                hint=(
                    "Negative outcomes are incompatible with a count model; "
                    "check whether the column is signed by mistake."
                ),
            ),
            None,
        )
    return df, None, None


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
    fn = {
        "ols": "ols",
        "logistic": "logit",
        "poisson": "poisson",
        "negbin": "negativebinomial",
    }[payload.kind]
    fit_args = ""
    if payload.kind == "ols" and payload.robust:
        fit_args = 'cov_type="HC3"'
    elif payload.kind in {"logistic", "poisson", "negbin"}:
        fit_args = "disp=0"
    return (
        f"import statsmodels.formula.api as smf\n"
        f'df = con.sql("SELECT * FROM {payload.name}").df()\n'
        f'model = smf.{fn}("{payload.formula}", data=df).fit({fit_args})\n'
        f"model.summary()"
    )


class _FormulaError(Exception):
    """Internal marker for formula / patsy / column-binding failures."""


def _coerce_bool_columns(df: Any) -> Any:
    """Cast every boolean column to a numeric dtype patsy can consume.

    statsmodels.logit refuses a single-column boolean endog (it treats bool
    as a 2-level categorical and emits a 2-column dummy matrix). CSVs that
    encode 0/1 outcomes as ``true``/``false`` therefore fail without manual
    casting. Two flavours of boolean show up in practice:

    * numpy ``bool`` — what DuckDB returns for a non-null BOOLEAN column;
    * pandas extension ``BooleanDtype`` — what DuckDB returns when the
      column has any NULL values.

    Both are coerced to a float ``Float64`` (preserving NA as ``NaN`` so
    statsmodels drops those rows itself) before the fit dispatch.
    """
    import pandas as pd  # type: ignore[reportMissingTypeStubs]

    bool_cols = [
        c
        for c in df.columns
        if pd.api.types.is_bool_dtype(df[c].dtype)  # type: ignore[reportUnknownMemberType]
    ]
    if not bool_cols:
        return df
    df = df.copy()
    for c in bool_cols:
        # Float (with NaN for NA) lets statsmodels' missing-handling drop
        # rows uniformly across endog and exog instead of barfing on a
        # pandas extension dtype that numpy doesn't understand.
        df[c] = df[c].astype("Float64").astype(float)
    return df


_NEGBIN_CONVERGENCE_HINT = (
    "MLE did not converge. Common causes: (1) very small alpha — "
    "try Poisson; (2) collinear predictors — check VIF in the OLS "
    "fit; (3) numerical scale — center/scale numerics."
)


def _fit_dispatch(payload: FitModelInput, df: Any) -> dict[str, Any]:
    """Pick the right statsmodels entry point and translate failures."""
    import patsy  # type: ignore[reportMissingTypeStubs]

    smf = _smf()
    df = _coerce_bool_columns(df)
    if payload.kind == "negbin":
        # NB MLE can fail numerically (singular hessian, log-of-zero in the
        # likelihood) on degenerate fixtures like perfect separation. Those
        # raise generic Exceptions from inside scipy/statsmodels that look
        # nothing like a patsy/formula error, so they must be reported as
        # ``convergence_failed`` rather than ``formula_error``.
        try:
            # TODO: patsy's `offset(log(x))` term is rejected by smf.negativebinomial
            # ("name 'offset' is not defined") on the current statsmodels version.
            # Re-evaluate when adding offset support (proposal §Open question 1);
            # the fallback will be an explicit `offset_column: str | None = None`
            # input on FitModelInput.
            m = smf.negativebinomial(payload.formula, data=df).fit(disp=False)
        except (patsy.PatsyError, NameError) as exc:
            raise _FormulaError(str(exc)) from exc
        except Exception as exc:
            return build_error(
                type="convergence_failed",
                message=f"Negative binomial MLE did not converge: {exc}",
                hint=_NEGBIN_CONVERGENCE_HINT,
            )
    else:
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

    if payload.kind == "negbin":
        converged = bool(m.mle_retvals.get("converged", False))
        if not converged:
            return build_error(
                type="convergence_failed",
                message="Negative binomial MLE did not converge.",
                hint=_NEGBIN_CONVERGENCE_HINT,
            )

    diagnostics = _diagnostics(m, payload.kind)
    coefficients = _coefficients(m, kind=payload.kind)
    fit = _fit_block(m, payload.kind)
    return {
        "ok": True,
        "coefficients": coefficients,
        "fit": fit,
        "diagnostics": diagnostics,
        "warnings": _warnings(m, diagnostics, payload.kind, fit),
        "interpretation": _interpretation(coefficients, payload.kind),
        # Live statsmodels Results object — stripped before the response is
        # returned by ``fit_model``. The model registry takes a reference
        # when ``model_name`` is supplied; otherwise it is GC'd at return.
        "_result": m,
    }


def _interpretation(coefficients: list[dict[str, Any]], kind: str) -> str:
    """Return a 2-3 sentence plain-English summary of the fitted model.

    Picks the strongest non-intercept signal (smallest p-value) and reports
    its direction and magnitude in kind-appropriate terms:
      - OLS: a unit change in the predictor moves the outcome by ``estimate``;
      - logistic: the predictor's odds-ratio (``exp(estimate) - 1``);
      - Poisson: the multiplicative effect on the expected count (``exp``);
      - negbin: same multiplicative count interpretation as Poisson, with
        the IRR (incidence rate ratio) phrase called out by name.
    """
    import math

    # The NB fit appends a synthetic ``alpha`` parameter alongside the
    # regression coefficients; exclude it from the "strongest predictor"
    # scan so the interpretation always talks about a real covariate.
    non_intercept = [c for c in coefficients if c["name"] not in ("Intercept", "alpha")]
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
    if kind == "negbin":
        irr = math.exp(est)
        return (
            f"Strongest predictor: `{name}` ({direction} effect, {sig} at α=0.05, p={p:.4g}). "
            f"A one-unit increase multiplies the expected count by "
            f"exp(β)={irr:.4g} (incidence rate ratio, IRR={irr:.4g})."
        )
    return (
        f"Strongest predictor: `{name}` ({direction} effect, {sig} at α=0.05, p={p:.4g}). "
        f"A one-unit increase moves the response by {est:.4g} units."
    )


def _warnings(m: Any, diagnostics: dict[str, Any], kind: str, fit: dict[str, Any]) -> list[str]:
    """Translate diagnostic numbers into a list of human-readable warning tags.

    ``high_multicollinearity`` fires for any kind that has a design matrix
    with VIF > 10 — we compute VIFs internally even for logistic/poisson so
    the warning fires consistently, though only OLS surfaces them in the
    ``diagnostics.vif`` block per spec.
    """
    out: list[str] = []
    raw_vif: Any = diagnostics.get("vif")
    vif: dict[str, float]
    if isinstance(raw_vif, dict):
        items: list[tuple[Any, Any]] = list(raw_vif.items())  # type: ignore[reportUnknownVariableType,reportUnknownArgumentType]
        vif = {str(k): float(v) for k, v in items}
    else:
        vif = _vif_per_coefficient(m)
    if any(v > 10 for v in vif.values()):
        out.append("high_multicollinearity")
    bp = diagnostics.get("breusch_pagan_p")
    if kind == "ols" and isinstance(bp, float) and bp < 0.05:
        out.append("heteroskedasticity")
    jb = diagnostics.get("jarque_bera_p")
    if kind == "ols" and isinstance(jb, float) and jb < 0.05:
        out.append("non_normal_residuals")
    if kind == "poisson" and _poisson_dispersion(m) > 1.5:
        out.append("overdispersion")
    if kind == "negbin":
        alpha = fit.get("dispersion_alpha")
        alpha_se = fit.get("dispersion_alpha_se")
        # ``underdispersion_vs_negbin`` means NB collapsed back to Poisson:
        # alpha is effectively zero AND statistically indistinguishable from
        # it. A pearson_chi2/df near 1 is the *desired* outcome for any
        # well-fit NB2 (genuine NB or true-Poisson alike), so it cannot
        # discriminate on its own — gate on alpha instead. When alpha
        # collapses to ~0 the hessian inversion can fail and ``alpha_se``
        # comes back as ``None`` — that *is* the degeneracy signal, so
        # treat missing SE as a positive collapse indicator alongside the
        # alpha/SE < 2 ratio check.
        if (
            isinstance(alpha, float)
            and alpha < 0.05
            and (
                alpha_se is None
                or (isinstance(alpha_se, float) and alpha_se > 0.0 and alpha / alpha_se < 2.0)
            )
        ):
            out.append("underdispersion_vs_negbin")
        if (
            isinstance(alpha, float)
            and isinstance(alpha_se, float)
            and alpha > 0.0
            and alpha_se / alpha > 0.5
        ):
            out.append("unstable_dispersion")
    return out


def _poisson_dispersion(m: Any) -> float:
    """Pearson chi-squared / df_resid as the Poisson dispersion estimate."""
    import numpy as np

    resid: Any = np.asarray(m.resid_pearson)
    df: int = int(m.df_resid)
    if df <= 0:
        return 0.0
    return float(np.sum(resid * resid) / df)


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
        diag = _sm_diagnostic()
        st = _sm_stattools()
        bp = diag.het_breuschpagan(np.asarray(m.resid), np.asarray(m.model.exog))
        jb = st.jarque_bera(np.asarray(m.resid))
        out["breusch_pagan_p"] = float(bp[1])
        out["durbin_watson"] = float(st.durbin_watson(np.asarray(m.resid)))
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

    outliers = _sm_outliers()
    exog: Any = np.asarray(m.model.exog)
    names: list[str] = list(m.model.exog_names)
    n_exog_cols = exog.shape[1]
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        if name == "Intercept":
            continue
        # The NB fit appends an ``alpha`` row to ``exog_names`` but the design
        # matrix has only regression columns — skip any name without a column.
        if i >= n_exog_cols:
            continue
        out[name] = float(outliers.variance_inflation_factor(exog, i))
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
    if kind == "negbin":
        import math

        import numpy as np

        alpha = float(m.params["alpha"])
        # Hessian inversion can fail when alpha collapses to ~0 (the NB
        # degenerates to Poisson), leaving ``m.bse["alpha"]`` as NaN. Expose
        # that as ``None`` rather than NaN so downstream callers don't divide
        # by it and pydantic's JSON serializer doesn't silently coerce NaN.
        alpha_se_raw = float(m.bse["alpha"])
        alpha_se: float | None = None if math.isnan(alpha_se_raw) else alpha_se_raw
        df_resid = int(m.df_resid)
        resid_p: Any = np.asarray(m.resid_pearson)
        pearson_over_df = float(np.sum(resid_p * resid_p) / df_resid) if df_resid > 0 else 0.0
        out["dispersion_alpha"] = alpha
        out["dispersion_alpha_se"] = alpha_se
        out["pearson_chi2_over_df"] = pearson_over_df
    return out


def _coefficients(m: Any, kind: str = "ols") -> list[dict[str, Any]]:
    """Build the per-coefficient envelope from a fitted statsmodels result.

    For ``kind="negbin"`` the dispersion parameter ``alpha`` is reported in
    ``fit.dispersion_alpha`` rather than as a row in ``coefficients`` — it
    is a scale parameter, not a regression coefficient.
    """
    params: Any = m.params
    bse: Any = m.bse
    tvals: Any = m.tvalues
    pvals: Any = m.pvalues
    ci: Any = m.conf_int()
    out: list[dict[str, Any]] = []
    for name in params.index:
        if kind == "negbin" and str(name) == "alpha":
            continue
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


def list_models() -> dict[str, Any]:
    """Return every registered model with kind, formula, training dataset, n_obs, fitted_at.

    Read-only inspection — does not emit a recorder cell (mirrors
    ``list_datasets``). Iteration order is registration order (Python 3.7+
    dict guarantee).
    """
    models = [
        {
            "name": entry.name,
            "kind": entry.kind,
            "formula": entry.formula,
            "fitted_on_dataset": entry.fitted_on_dataset,
            "n_obs": entry.n_obs,
            "fitted_at": entry.fitted_at.isoformat(),
        }
        for entry in session.get_models().values()
    ]
    return {"ok": True, "models": models}
