"""Model evaluation tool — held-out (or in-sample) metric battery.

Dispatches by model kind:

- Logistic: ROC-AUC, PR-AUC, Brier, log-loss (all sklearn), confusion
  matrix at ``threshold``, plus a quantile calibration table.
- OLS: RMSE, MAE, R², adjusted R².
- Poisson / NegBin: RMSE on counts, MAE, Pearson χ², deviance.

See ``docs/proposals/model_registry.md`` §``evaluate_model``.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


def _pd() -> Any:
    """Return ``pandas`` as untyped to keep strict pyright clean."""
    import pandas as _pd_mod  # type: ignore[reportMissingTypeStubs]

    return _pd_mod


def _sklearn_metrics() -> Any:
    """Return ``sklearn.metrics`` as untyped — strict pyright + sklearn stubs are noisy."""
    import sklearn.metrics as _m  # type: ignore[reportMissingTypeStubs]

    return _m


def _materialize_dataframe(name: str) -> Any:
    """Materialize a registered dataset as a pandas DataFrame via DuckDB."""
    con = session.get_connection()
    return con.execute(f'SELECT * FROM "{name}"').df()


def _formula_outcome(formula: str) -> str:
    """Extract the outcome column name (LHS of ``~``).

    Conservative: if the LHS is a bare identifier we return it; if it's
    wrapped in ``Q("col with spaces")`` we unwrap to the inner column
    name; any other wrapping (e.g. ``log(y)``) is returned as-is and
    downstream column-presence checks will fail with a sensible error.
    """
    import re

    if "~" not in formula:
        return formula.strip()
    lhs = formula.split("~", 1)[0].strip()
    q_match = re.fullmatch(r'Q\(\s*(["\'])(.+)\1\s*\)', lhs)
    if q_match:
        return q_match.group(2)
    return lhs


class EvaluateModelInput(BaseModel):
    """Inputs for ``evaluate_model`` — see proposal §Input."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        ...,
        description="Registry handle from a previous fit_model call.",
    )
    dataset: str = Field(
        ...,
        description=(
            "Dataset to evaluate on. Must contain the outcome column and "
            "all predictors referenced by the model's formula."
        ),
    )
    threshold: float = Field(
        default=0.5,
        description=(
            "Classification threshold for confusion-matrix-derived metrics "
            "(logistic only). Used to compute accuracy / precision / recall / "
            "F1 / tn-fp-fn-tp at the operating point."
        ),
    )
    n_calibration_bins: int = Field(
        default=10,
        description=(
            "Number of quantile bins in the logistic calibration table. "
            "Range [2, 50]. Auto-reduced to min(10, n_obs // 20) on small "
            "datasets; reduces to calibration: null with a note when n is "
            "too small to support even 2 bins."
        ),
    )


def _record_evaluate(payload: EvaluateModelInput, result: dict[str, Any]) -> None:
    """Append a markdown + code cell pair for the evaluate_model call.

    Markdown summarizes the headline metrics; the code cell rehydrates
    them in-notebook via sklearn (logistic), numpy (OLS), or numpy
    (Poisson / negbin), against the model object that the setup cell
    re-fit. Per Open question 3: include the calibration DataFrame in
    the code-cell output; the markdown shows summary stats only.
    """
    if not result.get("ok"):
        return
    entry = session.get_model(payload.model_name)
    if entry is None:  # pragma: no cover - guarded earlier in entry point
        return
    metrics = result.get("metrics", {})
    outcome = _formula_outcome(entry.formula)
    md_lines = [
        f"### Evaluated `{payload.model_name}` on `{payload.dataset}`",
    ]
    if entry.kind == "logistic":
        md_lines.append(
            f"- ROC-AUC = {metrics['roc_auc']:.3f}, "
            f"PR-AUC = {metrics['pr_auc']:.3f}, "
            f"Brier = {metrics['brier']:.3f}"
        )
        md_lines.append(
            f"- Accuracy = {metrics['accuracy']:.3f}, "
            f"F1 = {metrics['f1']:.3f} at threshold {payload.threshold}"
        )
        cal = result.get("calibration")
        if cal:
            gaps = [abs(row["mean_observed"] - row["mean_predicted"]) for row in cal]
            md_lines.append(f"- Calibration: {len(cal)} bins, max decile gap = {max(gaps):.3f}")
        else:
            md_lines.append(f"- Calibration: null ({result.get('calibration_note', '')})")
        code = (
            f"from sklearn.metrics import roc_auc_score, average_precision_score, "
            f"brier_score_loss, log_loss\n"
            f'y_true = {payload.dataset}_df["{outcome}"]\n'
            f"y_pred = {payload.model_name}.predict({payload.dataset}_df)\n"
            f'print(f"ROC-AUC = {{roc_auc_score(y_true, y_pred):.3f}}")\n'
            f'print(f"PR-AUC  = {{average_precision_score(y_true, y_pred):.3f}}")\n'
            f'print(f"Brier   = {{brier_score_loss(y_true, y_pred):.3f}}")\n'
            f"# Calibration table (quantile bins):\n"
            f"_q = pd.qcut(y_pred, q={payload.n_calibration_bins}, "
            f"duplicates='drop', labels=False)\n"
            f"calibration = pd.DataFrame({{'bin': _q + 1, 'y_true': y_true, "
            f"'y_pred': y_pred}}).groupby('bin').agg(\n"
            f"    mean_predicted=('y_pred', 'mean'), "
            f"mean_observed=('y_true', 'mean'), n=('y_true', 'size')\n"
            f").reset_index()\n"
            f"calibration"
        )
    elif entry.kind == "ols":
        md_lines.append(
            f"- RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}, "
            f"R² = {metrics['r_squared']:.4f} (adj {metrics['adj_r_squared']:.4f})"
        )
        code = (
            f"import numpy as np\n"
            f'y_true = {payload.dataset}_df["{outcome}"].to_numpy()\n'
            f"y_pred = {payload.model_name}.predict({payload.dataset}_df).to_numpy()\n"
            f'print(f"RMSE = {{np.sqrt(np.mean((y_true - y_pred)**2)):.4f}}")\n'
            f'print(f"MAE  = {{np.mean(np.abs(y_true - y_pred)):.4f}}")'
        )
    else:  # poisson / negbin
        md_lines.append(
            f"- RMSE = {metrics['rmse']:.4f}, "
            f"Pearson χ² = {metrics['pearson_chi2']:.2f}, "
            f"deviance = {metrics['deviance']:.2f}"
        )
        code = (
            f"import numpy as np\n"
            f'y_true = {payload.dataset}_df["{outcome}"].to_numpy()\n'
            f"mu = {payload.model_name}.predict({payload.dataset}_df).to_numpy()\n"
            f'print(f"Pearson χ² = {{np.sum((y_true - mu)**2 / mu):.2f}}")'
        )
    md = "\n".join(md_lines)
    get_recorder().record(markdown=md, code=code, tool_name="evaluate_model")


def evaluate_model(payload: EvaluateModelInput) -> dict[str, Any]:
    """Compute held-out / in-sample metrics for a registered model."""
    # n_calibration_bins range validation up-front — error type is
    # independent of dataset state.
    if not (2 <= payload.n_calibration_bins <= 50):
        return build_error(
            type="n_calibration_bins_out_of_range",
            message=(f"n_calibration_bins must be in [2, 50]; got {payload.n_calibration_bins}."),
            hint="Pick a bin count in [2, 50].",
        )

    entry = session.get_model(payload.model_name)
    if entry is None:
        known = sorted(session.get_models().keys())
        return build_error(
            type="model_not_found",
            message=f"No model named {payload.model_name!r} registered.",
            hint=f"Known model names: {known}." if known else "Registry is empty.",
        )
    if payload.dataset not in session.get_datasets():
        return build_error(
            type="dataset_not_found",
            message=f"No dataset named {payload.dataset!r} registered.",
            hint="Call list_datasets to see what is available.",
        )

    df: Any = _materialize_dataframe(payload.dataset)
    outcome = _formula_outcome(entry.formula)
    if outcome not in df.columns:
        return build_error(
            type="outcome_column_missing",
            message=(
                f"Outcome column {outcome!r} (from formula {entry.formula!r}) "
                f"is not in dataset {payload.dataset!r}."
            ),
            hint="Use a dataset that contains the outcome column for evaluation.",
        )

    # Drop NaN rows in outcome OR predictors (we reuse patsy via the model's
    # predict by relying on it dropping internally; for outcome NaN we drop
    # rows explicitly here so the metric arrays line up).
    y_raw: Any = df[outcome]
    warning_flags: list[str] = []

    # Dtype validation per proposal table.
    dtype_error = _validate_outcome_dtype(entry.kind, y_raw, warning_flags)
    if dtype_error is not None:
        return dtype_error

    m: Any = entry._result  # type: ignore[reportPrivateUsage]
    pd_mod: Any = _pd()
    # statsmodels predict on the raw df honors the model's missing-handling;
    # we filter to non-NaN outcome rows ourselves so y_true and y_pred align.
    not_null: Any = y_raw.notna()
    df_eval: Any = df.loc[not_null]
    try:
        y_pred_raw: Any = m.predict(df_eval)
    except Exception as exc:  # pragma: no cover - patsy-bound, hard to hit in test
        return build_error(
            type="prediction_failed",
            message=f"Model.predict failed: {exc}",
            hint="Check that the dataset's columns match the training formula.",
        )
    # statsmodels predict may drop additional NaN-predictor rows; align by
    # the index it returns.
    y_pred_series: Any = pd_mod.Series(y_pred_raw)
    kept: Any = y_pred_series.dropna().index
    y_pred: Any = np.asarray(y_pred_series.loc[kept])
    # Align y_true to the same surviving index.
    y_true: Any = np.asarray(pd_mod.Series(y_raw.loc[not_null]).loc[kept])
    n_obs = len(y_true)

    if entry.kind == "logistic":
        metrics, calibration, calibration_note = _logistic_metrics(
            y_true=y_true,
            y_pred=y_pred,
            threshold=payload.threshold,
            n_bins=payload.n_calibration_bins,
        )
        confusion = _confusion_matrix(y_true, y_pred, payload.threshold)
        out = {
            "ok": True,
            "model_name": payload.model_name,
            "dataset": payload.dataset,
            "metrics": metrics,
            "confusion_matrix": confusion,
            "calibration": calibration,
            "n_obs": n_obs,
            "warnings": warning_flags,
        }
        if calibration_note:
            out["calibration_note"] = calibration_note
    elif entry.kind == "ols":
        metrics = _ols_metrics(y_true=y_true, y_pred=y_pred, m=m)
        out = {
            "ok": True,
            "model_name": payload.model_name,
            "dataset": payload.dataset,
            "metrics": metrics,
            "n_obs": n_obs,
            "warnings": warning_flags,
        }
    else:  # poisson / negbin
        metrics = _count_metrics(y_true=y_true, mu=y_pred, kind=entry.kind)
        out = {
            "ok": True,
            "model_name": payload.model_name,
            "dataset": payload.dataset,
            "metrics": metrics,
            "n_obs": n_obs,
            "warnings": warning_flags,
        }

    _record_evaluate(payload, out)
    return out


def _validate_outcome_dtype(
    kind: str, y_raw: Any, warning_flags: list[str]
) -> dict[str, Any] | None:
    """Outcome dtype validation per proposal §Outcome dtype validation table."""
    pd_mod: Any = _pd()
    api_types: Any = pd_mod.api.types
    if kind == "logistic":
        # binary 0/1 or boolean. NaN allowed (filtered later) but only
        # finite values are checked.
        finite: Any = y_raw.dropna()
        if bool(api_types.is_bool_dtype(finite.dtype)):
            return None
        unique = set(np.unique(np.asarray(finite)).tolist())
        if not unique.issubset({0, 1}):
            return build_error(
                type="outcome_dtype_mismatch",
                message=(
                    f"Logistic evaluate_model requires a binary outcome (0/1 or "
                    f"boolean); got distinct values {sorted(unique)}."
                ),
                hint="Coerce the outcome to {0, 1} before evaluating.",
            )
        return None
    if kind == "ols":
        if bool(api_types.is_bool_dtype(y_raw.dtype)):
            # legitimate LPM use — warn but proceed.
            warning_flags.append("boolean_outcome_lpm")
            return None
        if bool(api_types.is_string_dtype(y_raw.dtype)) or bool(
            api_types.is_object_dtype(y_raw.dtype)
        ):
            return build_error(
                type="outcome_dtype_mismatch",
                message="OLS evaluate_model requires a numeric outcome.",
                hint="Cast the outcome column to a numeric dtype.",
            )
        return None
    # poisson / negbin: non-negative integer-valued.
    finite_counts: Any = y_raw.dropna()
    arr: Any = np.asarray(finite_counts)
    if bool(api_types.is_float_dtype(finite_counts.dtype)):
        non_int = float(np.sum(arr != np.floor(arr)))
        if non_int > 0 or float(np.min(arr, initial=0.0)) < 0.0:
            return build_error(
                type="outcome_dtype_mismatch",
                message=(
                    f"{kind} evaluate_model requires non-negative integer counts; "
                    f"got float values with {int(non_int)} non-integer entries."
                ),
                hint="Cast to int and drop negatives before evaluating.",
            )
        return None
    if bool(api_types.is_integer_dtype(finite_counts.dtype)):
        if int(np.min(arr, initial=0)) < 0:
            return build_error(
                type="outcome_dtype_mismatch",
                message=f"{kind} evaluate_model requires non-negative counts.",
                hint="Drop or absolute-value the negative entries.",
            )
        return None
    return build_error(
        type="outcome_dtype_mismatch",
        message=f"{kind} evaluate_model requires numeric counts; got {finite_counts.dtype}.",
        hint="Cast the outcome column to int.",
    )


def _logistic_metrics(
    *,
    y_true: Any,
    y_pred: Any,
    threshold: float,
    n_bins: int,
) -> tuple[dict[str, float], list[dict[str, Any]] | None, str | None]:
    """ROC-AUC / PR-AUC / Brier / log-loss / accuracy / F1 + calibration."""
    skm: Any = _sklearn_metrics()

    y_true_i: Any = y_true.astype(int)
    y_pred_class: Any = (y_pred >= threshold).astype(int)
    metrics: dict[str, float] = {
        "roc_auc": float(skm.roc_auc_score(y_true_i, y_pred)),
        "pr_auc": float(skm.average_precision_score(y_true_i, y_pred)),
        "brier": float(skm.brier_score_loss(y_true_i, y_pred)),
        "log_loss": float(skm.log_loss(y_true_i, np.clip(y_pred, 1e-15, 1 - 1e-15))),
        "accuracy": float(np.mean(np.asarray(y_pred_class == y_true_i, dtype=float))),
        "precision": float(skm.precision_score(y_true_i, y_pred_class, zero_division=0)),
        "recall": float(skm.recall_score(y_true_i, y_pred_class, zero_division=0)),
        "f1": float(skm.f1_score(y_true_i, y_pred_class, zero_division=0)),
    }

    # Calibration auto-reduction. The proposal: ``min(10, n_obs // 20)``
    # when the dataset is small. If that reduces below 2, return null.
    n = len(y_true_i)
    effective_bins = min(n_bins, max(2, n // 20))
    if n // 20 < 2:
        return metrics, None, f"dataset too small for calibration (n_obs={n})"
    if effective_bins < 2:
        return metrics, None, f"dataset too small for calibration (n_obs={n})"

    # Quantile binning on predicted probabilities — pandas qcut, drop dup
    # edges (heavy ties → may yield fewer bins than requested).
    pd_mod: Any = _pd()
    try:
        bin_idx: Any = pd_mod.qcut(y_pred, q=effective_bins, duplicates="drop", labels=False)
    except ValueError:
        return metrics, None, f"insufficient variance for {effective_bins} bins"
    cal_df: Any = pd_mod.DataFrame(
        {"bin": np.asarray(bin_idx) + 1, "y_true": y_true_i, "y_pred": y_pred}
    )
    grouped: Any = (
        cal_df.groupby("bin")
        .agg(
            mean_predicted=("y_pred", "mean"),
            mean_observed=("y_true", "mean"),
            n=("y_true", "size"),
        )
        .reset_index()
    )
    calibration: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        calibration.append(
            {
                "bin": int(row["bin"]),
                "mean_predicted": float(row["mean_predicted"]),
                "mean_observed": float(row["mean_observed"]),
                "n": int(row["n"]),
            }
        )
    return metrics, calibration, None


def _confusion_matrix(y_true: Any, y_pred: Any, threshold: float) -> dict[str, int]:
    """TN / FP / FN / TP at ``threshold``."""
    y_true_i: Any = y_true.astype(int)
    pred_cls: Any = (y_pred >= threshold).astype(int)
    tp = int(np.sum((pred_cls == 1) & (y_true_i == 1)))
    tn = int(np.sum((pred_cls == 0) & (y_true_i == 0)))
    fp = int(np.sum((pred_cls == 1) & (y_true_i == 0)))
    fn = int(np.sum((pred_cls == 0) & (y_true_i == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _ols_metrics(*, y_true: Any, y_pred: Any, m: Any) -> dict[str, float]:
    """RMSE / MAE / R² / adjusted R² on a regression dataset.

    Adjusted R² uses the *training* model's parameter count, applied to
    the *evaluation* sample size.
    """
    err: Any = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    n = len(y_true)
    ss_res = float(np.sum(err * err))
    y_arr: Any = np.asarray(y_true, dtype=float)
    ss_tot = float(np.sum((y_arr - float(cast(Any, np.mean(y_arr)))) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    # Number of regressors (exclude intercept by convention).
    p = len(m.params) - 1
    adj = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float("nan")
    err_sq: Any = np.asarray(err * err, dtype=float)
    mse = float(cast(Any, np.mean(err_sq)))
    abs_err: Any = np.asarray(np.abs(err), dtype=float)
    mae = float(cast(Any, np.mean(abs_err)))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj),
    }


def _count_metrics(*, y_true: Any, mu: Any, kind: str) -> dict[str, float]:
    """Poisson / NegBin: RMSE / MAE / Pearson χ² / deviance."""
    err: Any = np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float)
    err_sq: Any = np.asarray(err * err, dtype=float)
    mse = float(cast(Any, np.mean(err_sq)))
    rmse = float(np.sqrt(mse))
    abs_err: Any = np.asarray(np.abs(err), dtype=float)
    mae = float(cast(Any, np.mean(abs_err)))
    # Avoid div-by-zero — Pearson χ² is undefined where μ == 0.
    safe_mu: Any = np.where(mu > 0, mu, np.nan)
    pearson = float(np.nansum((y_true - mu) ** 2 / safe_mu))
    # Poisson deviance: 2 * Σ y*log(y/μ) - (y - μ). NegBin deviance is
    # the same up to the α-dependent term — we report Poisson deviance for
    # both as a Pearson-comparable goodness-of-fit signal (NB-α dispersion
    # is already in the model's fit block).
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_y: Any = np.where(y_true > 0, y_true, np.nan)
        log_term: Any = np.where(y_true > 0, safe_y * np.log(safe_y / safe_mu), 0.0)
        deviance = float(2.0 * np.nansum(log_term - (y_true - mu)))
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_chi2": pearson,
        "deviance": deviance,
    }


# Public aliases shared with cross_validate (tools/crossval.py).
formula_outcome = _formula_outcome
validate_outcome_dtype = _validate_outcome_dtype
count_metrics = _count_metrics
