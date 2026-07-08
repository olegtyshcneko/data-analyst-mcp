"""k-fold cross-validation — re-fitting complement to ``evaluate_model``.

Fits are ephemeral: there is no ``model_name`` parameter and the model
registry is never touched. A full-data preflight fit goes through
``models.fit_prepared`` (fit_model's exact validation path), surfacing
fit_model's whole error taxonomy before any fold work; the preflight's
patsy design matrices are then sliced per fold, so categorical levels
are encoded globally and a fold-local level can never crash scoring
(spec §5.11d).
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder
from data_analyst_mcp.tools import evaluate as _evaluate
from data_analyst_mcp.tools import models as _models

logger = logging.getLogger(__name__)


def _sm() -> Any:
    """Return ``statsmodels.api`` as an untyped module."""
    import statsmodels.api as _sm_mod  # type: ignore[reportMissingTypeStubs]

    return _sm_mod


def _sklearn_metrics() -> Any:
    """Return ``sklearn.metrics`` as untyped — strict pyright + sklearn stubs are noisy."""
    import sklearn.metrics as _m  # type: ignore[reportMissingTypeStubs]

    return _m


def _materialize_dataframe(name: str) -> Any:
    """Materialize a registered dataset as a pandas DataFrame via DuckDB."""
    con = session.get_connection()
    quoted = '"' + name.replace('"', '""') + '"'
    return con.execute(f"SELECT * FROM {quoted}").df()


class CrossValidateInput(BaseModel):
    """Inputs for ``cross_validate``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    formula: str
    kind: Literal["ols", "logistic", "poisson", "negbin"] = "ols"
    robust: bool = False
    k: int = 5
    seed: int = 42
    threshold: float = 0.5


def _fold_ids(y: Any, k: int, seed: int, stratified: bool) -> Any:
    """Fold id per row via a RandomState permutation.

    Stratified mode (logistic) permutes within each outcome class so every
    fold keeps the class balance; callers must have verified each class
    has >= k members first.
    """
    n = len(y)
    fold = np.empty(n, dtype=int)
    rng = np.random.RandomState(seed)
    if stratified:
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            perm = idx[rng.permutation(len(idx))]
            fold[perm] = np.arange(len(perm)) % k
    else:
        perm = rng.permutation(n)
        fold[perm] = np.arange(n) % k
    return fold


def _fit_fold(kind: str, robust: bool, y_tr: Any, X_tr: Any) -> Any:
    """Array-interface statsmodels fit for one training slice."""
    sm = _sm()
    if kind == "ols":
        return sm.OLS(y_tr, X_tr).fit(cov_type="HC3" if robust else "nonrobust")
    if kind == "logistic":
        return sm.Logit(y_tr, X_tr).fit(disp=0)
    if kind == "poisson":
        return sm.Poisson(y_tr, X_tr).fit(disp=0)
    return sm.NegativeBinomial(y_tr, X_tr).fit(disp=0)


def _classify_fold_failure(kind: str, exc: Exception | None, result: Any | None) -> str:
    """Map a fold-local fit failure to fit_model's error-type strings.

    Mirrors ``_fit_logistic_or_error``'s taxonomy: for logistic,
    ``PerfectSeparationError`` / ``LinAlgError`` signal a degenerate
    logit and map to ``perfect_separation`` (the design matrix is built
    globally and full-rank — the preflight fit succeeded — so the
    rank-deficiency → formula_error branch cannot apply fold-locally).
    """
    if exc is not None:
        from numpy.linalg import LinAlgError
        from statsmodels.tools.sm_exceptions import (  # type: ignore[reportMissingTypeStubs]
            PerfectSeparationError,
        )

        if kind == "logistic" and isinstance(exc, (PerfectSeparationError, LinAlgError)):
            return "perfect_separation"
        return "convergence_failed"
    # Returned-but-degenerate fit (logistic/negbin non-convergence).
    if kind == "logistic" and result is not None:
        degenerate = _models._detect_logistic_separation(result)  # type: ignore[reportPrivateUsage]
        if degenerate is not None:
            return str(degenerate["error"]["type"])
    return "convergence_failed"


def _fold_converged(kind: str, result: Any) -> bool:
    """MLE-family fits must report convergence; OLS always converges."""
    if kind == "ols":
        return True
    return bool(result.mle_retvals.get("converged", False))


def _fold_metrics(kind: str, y_true: Any, y_pred: Any, threshold: float) -> dict[str, float]:
    """Held-out fold metrics — same families as evaluate_model."""
    if kind == "logistic":
        skm: Any = _sklearn_metrics()
        y_i: Any = y_true.astype(int)
        y_cls: Any = (y_pred >= threshold).astype(int)
        clipped: Any = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return {
            "roc_auc": float(skm.roc_auc_score(y_i, y_pred)),
            "pr_auc": float(skm.average_precision_score(y_i, y_pred)),
            "brier": float(skm.brier_score_loss(y_i, y_pred)),
            "log_loss": float(skm.log_loss(y_i, clipped, labels=[0, 1])),
            "accuracy": float(np.mean(np.asarray(y_cls == y_i, dtype=float))),
            "precision": float(skm.precision_score(y_i, y_cls, zero_division=0)),
            "recall": float(skm.recall_score(y_i, y_cls, zero_division=0)),
            "f1": float(skm.f1_score(y_i, y_cls, zero_division=0)),
        }
    if kind == "ols":
        err: Any = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        err_sq: Any = np.asarray(err * err, dtype=float)
        ss_res = float(np.sum(err_sq))
        y_arr: Any = np.asarray(y_true, dtype=float)
        ss_tot = float(np.sum((y_arr - float(cast(Any, np.mean(y_arr)))) ** 2))
        mse = float(cast(Any, np.mean(err_sq)))
        abs_err: Any = np.asarray(np.abs(err), dtype=float)
        mae = float(cast(Any, np.mean(abs_err)))
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": mae,
            "r_squared": 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        }
    return _evaluate.count_metrics(y_true=y_true, mu=y_pred, kind=kind)


def cross_validate(payload: CrossValidateInput) -> dict[str, Any]:
    """k-fold cross-validated metrics for a formula on a dataset."""
    if payload.name not in session.get_datasets():
        return build_error(
            type="dataset_not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    if payload.kind == "negbin" and payload.robust:
        return build_error(
            type="robust_not_supported",
            message="robust=True is not supported for kind='negbin'.",
            hint="NB2 robust SE is not implemented in this server. Set `robust=False`.",
        )
    if not (2 <= payload.k <= 20):
        return build_error(
            type="k_out_of_range",
            message=f"k must be in [2, 20]; got {payload.k}.",
            hint="Pick a fold count in [2, 20].",
        )
    if not (0.0 < payload.threshold < 1.0):
        return build_error(
            type="threshold_out_of_range",
            message=f"threshold must be in the open interval (0, 1); got {payload.threshold}.",
            hint="Pick a threshold strictly between 0 and 1.",
        )

    df: Any = _materialize_dataframe(payload.name)
    outcome = _evaluate.formula_outcome(payload.formula)
    warning_flags: list[str] = []
    if outcome in df.columns:
        dtype_error = _evaluate.validate_outcome_dtype(payload.kind, df[outcome], warning_flags)
        if dtype_error is not None:
            return dtype_error

    fm_payload = _models.FitModelInput(
        name=payload.name,
        formula=payload.formula,
        kind=payload.kind,
        robust=payload.robust,
        model_name=None,
    )
    try:
        full = _models.fit_prepared(fm_payload, df)
    except _models.FormulaError as fe:
        return build_error(
            type="formula_error",
            message=str(fe),
            hint=("Verify column names exist and the formula parses, e.g. 'y ~ x + C(group)'."),
        )
    if not full.get("ok"):
        return full  # perfect_separation / convergence_failed / negbin dtype
    m: Any = full.pop("_result")
    warning_flags.extend(list(full.get("warnings") or []))

    y: Any = np.asarray(m.model.endog, dtype=float)
    X: Any = np.asarray(m.model.exog, dtype=float)
    n = len(y)
    dropped_rows = int(len(df) - n)
    if n < payload.k:
        return build_error(
            type="k_out_of_range",
            message=(
                f"k={payload.k} exceeds the {n} usable rows after NaN drops "
                f"({dropped_rows} dropped)."
            ),
            hint="Lower k or clean the missing predictor/outcome rows.",
        )
    stratified = payload.kind == "logistic"
    if stratified:
        for cls in (0, 1):
            n_cls = int(np.sum(y == cls))
            if n_cls < payload.k:
                return build_error(
                    type="outcome_class_too_small",
                    message=(
                        f"Stratified {payload.k}-fold CV needs at least {payload.k} rows "
                        f"of each outcome class; class {cls} has {n_cls}."
                    ),
                    hint="Lower k or gather more minority-class rows.",
                )
    fold: Any = _fold_ids(y, payload.k, payload.seed, stratified)

    n_params = int(X.shape[1])
    for i in range(payload.k):
        train_size = int(np.sum(fold != i))
        if train_size <= n_params:
            return build_error(
                type="fold_too_small",
                message=(
                    f"Fold {i} would train on {train_size} rows but the design "
                    f"matrix has {n_params} parameters."
                ),
                hint="Lower k or simplify the formula.",
            )

    per_fold: list[dict[str, float] | None] = []
    fold_sizes: list[int] = []
    fold_failures: list[dict[str, Any]] = []
    for i in range(payload.k):
        tr: Any = fold != i
        te: Any = ~tr
        fold_sizes.append(int(np.sum(te)))
        try:
            res: Any = _fit_fold(payload.kind, payload.robust, y[tr], X[tr])
        except Exception as exc:
            fold_failures.append(
                {"fold": i, "error_type": _classify_fold_failure(payload.kind, exc, None)}
            )
            per_fold.append(None)
            continue
        if not _fold_converged(payload.kind, res):
            fold_failures.append(
                {"fold": i, "error_type": _classify_fold_failure(payload.kind, None, res)}
            )
            per_fold.append(None)
            continue
        y_pred: Any = np.asarray(res.predict(X[te]))
        per_fold.append(_fold_metrics(payload.kind, y[te], y_pred, payload.threshold))

    successes = [p for p in per_fold if p is not None]
    if not successes:
        return build_error(
            type="cv_fit_failed",
            message=f"All {payload.k} folds failed to fit.",
            hint="See fit_model on the full dataset for a diagnosable single-fit error.",
        )
    if fold_failures:
        warning_flags.append("fold_failures")

    metric_keys = list(successes[0].keys())
    metrics: dict[str, Any] = {}
    for key in metric_keys:
        vals = [p[key] for p in successes]
        metrics[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "per_fold": [None if p is None else p[key] for p in per_fold],
        }

    result: dict[str, Any] = {
        "ok": True,
        "name": payload.name,
        "formula": payload.formula,
        "kind": payload.kind,
        "k": payload.k,
        "seed": payload.seed,
        "stratified": stratified,
        "metrics": metrics,
        "fold_sizes": fold_sizes,
        "n_obs": n,
        "dropped_rows": dropped_rows,
        "fold_failures": fold_failures,
        "warnings": warning_flags,
        "interpretation": _interpretation(payload, metrics, fold_failures),
    }
    _record_cross_validate(payload, result)
    return result


_PRIMARY_METRIC = {"ols": "rmse", "logistic": "roc_auc", "poisson": "rmse", "negbin": "rmse"}


def _interpretation(
    payload: CrossValidateInput, metrics: dict[str, Any], fold_failures: list[dict[str, Any]]
) -> str:
    """2-3 sentence summary anchored on the kind's primary metric."""
    key = _PRIMARY_METRIC[payload.kind]
    m = metrics[key]
    text = (
        f"{payload.k}-fold cross-validation of {payload.kind} model "
        f"'{payload.formula}' on '{payload.name}': {key} = "
        f"{m['mean']:.4g} ± {m['std']:.4g} across folds."
    )
    if fold_failures:
        text += f" {len(fold_failures)} fold(s) failed to fit and were excluded."
    return text


_SMF_FN = {"ols": "ols", "logistic": "logit", "poisson": "poisson", "negbin": "negativebinomial"}
_SM_CLASS = {"ols": "OLS", "logistic": "Logit", "poisson": "Poisson", "negbin": "NegativeBinomial"}


def _cv_cell_source(payload: CrossValidateInput) -> str:
    """Self-contained notebook cell reproducing the CV table.

    Rebuilds the design matrices via the same smf formula fit the live
    preflight ran, assigns folds with the identical RandomState calls,
    then loops array-interface fits + metrics. ``con`` / ``np`` / ``pd``
    / ``sm`` / ``smf`` come from the setup cell. Known limitation
    (matches fit_model's ``_code_for_fit`` cells): the cell calls smf
    directly without the live path's boolean-column coercion, so a
    boolean logistic outcome that succeeded live needs a manual cast at
    replay. Fold-local fit failures are skipped with try/except, the
    same exclusion the live aggregates apply.
    """
    smf_fn = _SMF_FN[payload.kind]
    sm_cls = _SM_CLASS[payload.kind]
    if payload.kind == "ols":
        full_fit_args = 'cov_type="HC3"' if payload.robust else ""
        fold_fit_args = 'cov_type="HC3"' if payload.robust else ""
    else:
        full_fit_args = "disp=0"
        fold_fit_args = "disp=0"
    lines = [
        f"_cv_df = con.sql('SELECT * FROM \"{payload.name}\"').df()",
        f"_cv_full = smf.{smf_fn}({payload.formula!r}, data=_cv_df).fit({full_fit_args})",
        "_cv_y = np.asarray(_cv_full.model.endog, dtype=float)",
        "_cv_X = np.asarray(_cv_full.model.exog, dtype=float)",
        f"_cv_rng = np.random.RandomState({payload.seed})",
        "_cv_fold = np.empty(len(_cv_y), dtype=int)",
    ]
    if payload.kind == "logistic":
        lines += [
            "for _cls in (0, 1):",
            "    _idx = np.where(_cv_y == _cls)[0]",
            "    _p = _idx[_cv_rng.permutation(len(_idx))]",
            f"    _cv_fold[_p] = np.arange(len(_p)) % {payload.k}",
        ]
    else:
        lines += [
            "_perm = _cv_rng.permutation(len(_cv_y))",
            f"_cv_fold[_perm] = np.arange(len(_cv_y)) % {payload.k}",
        ]
    lines += [
        "_cv_rows = []",
        f"for _i in range({payload.k}):",
        "    _tr = _cv_fold != _i",
        "    try:",
        f"        _res = sm.{sm_cls}(_cv_y[_tr], _cv_X[_tr]).fit({fold_fit_args})",
        "    except Exception:",
        "        continue  # fold-local fit failure — excluded from live aggregates too",
        "    _te_y = _cv_y[~_tr]",
        "    _pred = np.asarray(_res.predict(_cv_X[~_tr]))",
    ]
    if payload.kind == "logistic":
        lines = [
            "from sklearn import metrics as _skm",
            *lines,
            "    _ti = _te_y.astype(int)",
            f"    _cls_pred = (_pred >= {payload.threshold}).astype(int)",
            "    _cv_rows.append({",
            "        'fold': _i,",
            "        'roc_auc': _skm.roc_auc_score(_ti, _pred),",
            "        'pr_auc': _skm.average_precision_score(_ti, _pred),",
            "        'brier': _skm.brier_score_loss(_ti, _pred),",
            "        'log_loss': _skm.log_loss(_ti, np.clip(_pred, 1e-15, 1 - 1e-15), labels=[0, 1]),",
            "        'accuracy': float(np.mean(_cls_pred == _ti)),",
            "        'precision': _skm.precision_score(_ti, _cls_pred, zero_division=0),",
            "        'recall': _skm.recall_score(_ti, _cls_pred, zero_division=0),",
            "        'f1': _skm.f1_score(_ti, _cls_pred, zero_division=0),",
            "    })",
        ]
    elif payload.kind == "ols":
        lines += [
            "    _err = _te_y - _pred",
            "    _ss_tot = float(np.sum((_te_y - np.mean(_te_y)) ** 2))",
            "    _cv_rows.append({",
            "        'fold': _i,",
            "        'rmse': float(np.sqrt(np.mean(_err ** 2))),",
            "        'mae': float(np.mean(np.abs(_err))),",
            "        'r_squared': 1.0 - float(np.sum(_err ** 2)) / _ss_tot if _ss_tot > 0 else 0.0,",
            "    })",
        ]
    else:  # poisson / negbin
        lines += [
            "    _err = _te_y - _pred",
            "    _safe_mu = np.where(_pred > 0, _pred, np.nan)",
            "    _safe_y = np.where(_te_y > 0, _te_y, np.nan)",
            "    _log_term = np.where(_te_y > 0, _safe_y * np.log(_safe_y / _safe_mu), 0.0)",
            "    _cv_rows.append({",
            "        'fold': _i,",
            "        'rmse': float(np.sqrt(np.mean(_err ** 2))),",
            "        'mae': float(np.mean(np.abs(_err))),",
            "        'pearson_chi2': float(np.nansum(_err ** 2 / _safe_mu)),",
            "        'deviance': float(2.0 * np.nansum(_log_term - (_te_y - _pred))),",
            "    })",
        ]
    lines += ["pd.DataFrame(_cv_rows).set_index('fold').agg(['mean', 'std'])"]
    return "\n".join(lines)


def _record_cross_validate(payload: CrossValidateInput, result: dict[str, Any]) -> None:
    """Markdown + code cell for the CV table."""
    key = _PRIMARY_METRIC[payload.kind]
    m = result["metrics"][key]
    md = (
        f"### {payload.k}-fold CV of {payload.kind} on `{payload.name}`\n\n"
        f"- Formula: `{payload.formula}`\n"
        f"- {key} = {m['mean']:.4g} ± {m['std']:.4g}\n"
        f"- {result['interpretation']}"
    )
    code = _cv_cell_source(payload)
    get_recorder().record(markdown=md, code=code, tool_name="cross_validate")
