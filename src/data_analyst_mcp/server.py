"""FastMCP server entry point.

Stdio transport only. Stdout is reserved for the MCP protocol — every log
record goes to stderr via the module-level handler configured below.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from data_analyst_mcp.errors import build_error
from data_analyst_mcp.tools import datasets as _datasets
from data_analyst_mcp.tools import evaluate as _evaluate
from data_analyst_mcp.tools import missingness as _missingness
from data_analyst_mcp.tools import models as _models
from data_analyst_mcp.tools import multitest as _multitest
from data_analyst_mcp.tools import notebook as _notebook
from data_analyst_mcp.tools import plots as _plots
from data_analyst_mcp.tools import predict as _predict
from data_analyst_mcp.tools import query as _query
from data_analyst_mcp.tools import stats as _stats

_handler = logging.StreamHandler(stream=sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("data_analyst_mcp")

mcp: FastMCP = FastMCP("data-analyst-mcp")


@mcp.tool()
def query(sql: str, limit: int = 50) -> dict[str, Any]:
    """Run a read-only SQL query against the registered datasets.

    Accepts SELECT / WITH / DESCRIBE / SHOW / EXPLAIN / PRAGMA show_tables;
    rejects writes (INSERT / UPDATE / DELETE / DROP / CREATE / SET). A
    LIMIT is auto-applied if not present so result rows stay bounded.
    Returns rows + column names + total_rows (via a separate COUNT(*) over
    the same query) + execution_time_ms.
    """
    try:
        payload = _query.QueryInput(sql=sql, limit=limit)
        return _query.query(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("query failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def describe_column(name: str, column: str, bins: int = 20) -> dict[str, Any]:
    """Deep-dive a single column of a registered dataset.

    Numeric columns: returns the full quantile vector (1/5/10/25/50/75/
    90/95/99), skewness, kurtosis, IQR, histogram counts honoring ``bins``,
    and IQR + z>3 outliers with 5 example rows. Categorical columns: full
    value counts (capped at 50 with "other" bucket) and entropy. Temporal
    columns: counts by year/month/weekday/hour.
    """
    try:
        payload = _datasets.DescribeColumnInput(name=name, column=column, bins=bins)
        return _datasets.describe_column(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("describe_column failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def profile_dataset(name: str, sample_rows: int = 5) -> dict[str, Any]:
    """Produce a full EDA profile for a registered dataset.

    Reports total rows + columns, per-column dtype/null counts/distinct
    counts, numeric stats (min/max/mean/median/std/p25/p75/p99/zeros/
    negatives), string-length stats, temporal stats, top-5 most-frequent
    values per column, heuristic flags (``looks_like_id``,
    ``looks_like_categorical``, ``looks_like_timestamp``,
    ``high_cardinality``, ``mostly_null``, ``constant``,
    ``mixed_dtype_suspect``), a head sample of ``sample_rows`` rows, and a
    short list of suggested next actions. This is the headline EDA tool.
    """
    try:
        payload = _datasets.ProfileDatasetInput(name=name, sample_rows=sample_rows)
        return _datasets.profile_dataset(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("profile_dataset failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def analyze_missingness(
    name: str,
    columns: list[str] | None = None,
    pattern_top_k: int = 10,
    pairwise_corr_threshold: float = 0.1,
    run_mcar_test: bool = True,
) -> dict[str, Any]:
    """Diagnose missingness structure on a registered dataset (v1, descriptive).

    Per-column null stats, top-K (col → is_null) patterns, pairwise
    φ-correlation between null indicators, and severity-sorted
    suggestions (drop/binarize for >50%-null columns; structural alerts
    when nulls partition cleanly across a categorical; co-missing
    callouts at |φ| > 0.5). ``columns`` defaults to every column;
    ``pattern_top_k`` is in [1, 100]; ``pairwise_corr_threshold`` in
    [0.0, 1.0]. ``run_mcar_test`` defaults to True for forward-compat
    with v1.1 (Little's MCAR) — in v1 it returns
    ``mcar_not_yet_implemented``; pass ``run_mcar_test=False`` to get
    the descriptive output. ``mcar_test`` is always ``null`` in v1.
    """
    try:
        payload = _missingness.AnalyzeMissingnessInput.model_validate(
            {
                "name": name,
                "columns": columns,
                "pattern_top_k": pattern_top_k,
                "pairwise_corr_threshold": pairwise_corr_threshold,
                "run_mcar_test": run_mcar_test,
            }
        )
        return _missingness.analyze_missingness(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("analyze_missingness failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def list_datasets() -> dict[str, Any]:
    """List every dataset currently registered in this session.

    Each entry reports name, row count, column count, and the registration
    timestamp so the agent can pick a target for downstream tools without
    re-loading.
    """
    try:
        return _datasets.list_datasets()
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("list_datasets failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def load_dataset(
    path: str,
    name: str | None = None,
    read_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a file as a named dataset queryable by all other tools.

    The path extension picks the DuckDB reader (``.csv``/``.tsv``/``.parquet``/
    ``.json``/``.jsonl``/``.xlsx``). ``read_options`` is forwarded into the
    reader for cases where auto-detection fails (e.g. semicolon-delimited
    files). Reports row count, column names + dtypes, file size, and any
    parser warnings.
    """
    try:
        payload = _datasets.LoadDatasetInput(path=path, name=name, read_options=read_options)
        return _datasets.load_dataset(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("load_dataset failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def correlate(
    name: str,
    columns: list[str] | None = None,
    method: str = "pearson",
    plot: bool = True,
) -> dict[str, Any]:
    """Correlation matrix across numeric columns of a registered dataset.

    Method picks Pearson (default), Spearman, or Kendall. When ``plot`` is
    true, a base64-encoded PNG heatmap is also returned. When ``columns``
    is omitted, every numeric column in the dataset is used.
    """
    try:
        payload = _stats.CorrelateInput.model_validate(
            {"name": name, "columns": columns, "method": method, "plot": plot}
        )
        return _stats.correlate(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("correlate failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def compare_groups(
    name: str,
    group_column: str,
    metric_column: str,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """Compare a metric across groups and pick an appropriate test.

    Reads dtype of ``metric_column``: numeric → t-test / Welch / Mann-Whitney
    (2 groups) or ANOVA / Kruskal-Wallis (>2 groups), based on Shapiro-Wilk
    normality + Levene equal-variance checks. Categorical → chi-square, or
    Fisher's exact when 2×2 with a low expected cell count. Returns the
    selected test, statistic, p-value, effect size, sample sizes,
    assumption-check results, and a plain-English interpretation.
    """
    try:
        payload = _stats.CompareGroupsInput(
            name=name,
            group_column=group_column,
            metric_column=metric_column,
            groups=groups,
        )
        return _stats.compare_groups(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("compare_groups failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def test_hypothesis(
    kind: str,
    name: str | None = None,
    group_column: str | None = None,
    metric_column: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
    table: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Run a named statistical hypothesis test.

    ``kind`` is one of ``t_test``, ``welch``, ``mann_whitney``, ``chi_square``,
    ``fisher``, ``anova``, ``kruskal``, ``ks``. Two-sample tests need ``name``,
    ``group_column``, ``metric_column``, ``group_a``, ``group_b``; ANOVA and
    Kruskal-Wallis need ``name``, ``group_column``, ``metric_column``;
    chi-square and Fisher need a ``table`` contingency matrix. Returns a
    uniform shape with ``test``, ``statistic``, ``p_value``, ``effect_size``,
    ``df``, ``n_a``, ``n_b``, ``interpretation``.
    """
    try:
        from pydantic import TypeAdapter

        if kind not in _stats.ALLOWED_KINDS:
            return build_error(
                type="invalid_kind",
                message=f"Unknown kind {kind!r}.",
                hint=f"Allowed kinds: {sorted(_stats.ALLOWED_KINDS)}.",
            )
        adapter: TypeAdapter[Any] = TypeAdapter(_stats.TestHypothesisInput)
        raw: dict[str, Any] = {"kind": kind}
        if name is not None:
            raw["name"] = name
        if group_column is not None:
            raw["group_column"] = group_column
        if metric_column is not None:
            raw["metric_column"] = metric_column
        if group_a is not None:
            raw["group_a"] = group_a
        if group_b is not None:
            raw["group_b"] = group_b
        if table is not None:
            raw["table"] = table
        payload = adapter.validate_python(raw)
        return _stats.test_hypothesis(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("test_hypothesis failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def adjust_pvalues(
    p_values: list[float],
    method: str = "bh",
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Correct a family of raw p-values for multiple testing.

    ``method`` is one of ``bonferroni`` / ``sidak`` / ``holm`` (FWER) or
    ``bh`` / ``by`` (FDR); defaults to ``bh`` (Benjamini–Hochberg). ``alpha``
    is the significance threshold and affects only the ``rejected`` column.
    ``labels`` is an optional list of row labels echoed unchanged into the
    output (must be the same length as ``p_values`` when provided; duplicates
    are allowed). Output rows are returned in input order; each row carries
    ``label``, ``p_raw``, ``p_adj``, and ``rejected``. Empty ``p_values`` is
    valid and returns an empty result with ``n_tests=0``.
    """
    try:
        from pydantic import ValidationError

        try:
            payload = _multitest.AdjustPvaluesInput.model_validate(
                {
                    "p_values": p_values,
                    "method": method,
                    "alpha": alpha,
                    "labels": labels,
                }
            )
        except ValidationError as ve:
            for err in ve.errors():
                if err.get("loc") == ("method",):
                    return build_error(
                        type="unknown_method",
                        message=f"Unknown method {method!r}.",
                        hint=("Allowed methods: ['bh', 'bonferroni', 'by', 'holm', 'sidak']."),
                    )
            raise
        return _multitest.adjust_pvalues(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("adjust_pvalues failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def fit_model(
    name: str,
    formula: str,
    kind: str = "ols",
    robust: bool = False,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Fit an OLS / logistic / Poisson / negative binomial regression with diagnostics.

    ``formula`` is Wilkinson-style (patsy), e.g. ``price ~ sqft + C(area)``.
    ``kind`` picks the model family: ``ols`` (default, linear), ``logistic``
    (binary outcome via logit), ``poisson`` (non-negative counts), or
    ``negbin`` (NB2 negative binomial — the remedy when a Poisson fit emits
    the ``overdispersion`` warning). ``robust=True`` switches OLS to HC3
    heteroskedasticity-robust standard errors (coefficients unchanged, std
    errors recomputed); it is rejected for ``negbin``. ``model_name`` is
    an optional non-empty, whitespace-free handle: when provided, the
    fitted result is stored in the session model registry for downstream
    ``predict`` / ``evaluate_model`` / ``list_models`` calls; duplicates
    are rejected with ``model_name_collision``. Returns coefficients
    (with std errors, t / z stats, p-values, 95% CIs), fit statistics
    (R^2 / adj-R^2 / pseudo-R^2 / AIC / BIC / n_obs / df_resid, plus
    ``dispersion_alpha`` / ``dispersion_alpha_se`` / ``pearson_chi2_over_df``
    for ``negbin``), diagnostics (Breusch-Pagan, Durbin-Watson, Jarque-Bera,
    condition number, VIF for OLS), a ``warnings`` list
    (``high_multicollinearity``, ``heteroskedasticity``,
    ``non_normal_residuals``, ``overdispersion``,
    ``underdispersion_vs_negbin``, ``unstable_dispersion``), a 2-3
    sentence plain-English interpretation, and (when stored) a
    ``model_name`` field echoing the registry handle.
    """
    try:
        from pydantic import ValidationError

        try:
            payload = _models.FitModelInput.model_validate(
                {
                    "name": name,
                    "formula": formula,
                    "kind": kind,
                    "robust": robust,
                    "model_name": model_name,
                }
            )
        except ValidationError as ve:
            for err in ve.errors():
                if err.get("loc") == ("kind",):
                    return build_error(
                        type="invalid_kind",
                        message=f"Unknown kind {kind!r}.",
                        hint="Allowed kinds: ['logistic', 'negbin', 'ols', 'poisson'].",
                    )
            raise
        return _models.fit_model(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("fit_model failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def list_models() -> dict[str, Any]:
    """List every model currently registered in this session.

    Each entry reports ``name``, ``kind``, ``formula``, ``fitted_on_dataset``,
    ``n_obs``, and ``fitted_at`` so the agent can pick a registry handle for
    downstream ``predict`` / ``evaluate_model`` calls without re-fitting.
    Entries are returned in registration order. Read-only — does not emit
    a notebook cell (same convention as ``list_datasets``).
    """
    try:
        return _models.list_models()
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("list_models failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def predict(
    model_name: str,
    dataset: str,
    output: str = "response",
    threshold: float = 0.5,
    limit: int = 50,
    cursor: str | None = None,
    include_se: bool = False,
) -> dict[str, Any]:
    """Score a registered model on a registered dataset.

    ``model_name`` is the registry handle from a previous ``fit_model``
    call. ``dataset`` is the dataset to score — it must contain every
    predictor referenced by the model's formula; the outcome column is
    optional. ``output`` is ``response`` (default — probability for
    logistic, expected count for Poisson / negbin, identity for OLS),
    ``link`` (linear predictor η), or ``class`` (logistic-only thresholded
    label). ``threshold`` is the decision threshold for ``output='class'``,
    in the open interval (0, 1). ``include_se`` is OLS-only and adds
    per-row ``se_mean`` / ``mean_ci_lower`` / ``mean_ci_upper``. Returns
    ``predictions`` (one row per non-dropped input row, ``y_pred`` plus
    ``y_class`` when classifying, plus SE block when requested),
    ``dropped_rows`` (rows patsy dropped due to NaN predictors),
    ``total_rows``, ``truncated`` / ``cursor`` pagination. ``row_index``
    is 0-indexed in the source dataset (non-contiguous after drops) so
    callers can SQL-join predictions back to the source.
    """
    try:
        from pydantic import ValidationError

        try:
            payload = _predict.PredictInput.model_validate(
                {
                    "model_name": model_name,
                    "dataset": dataset,
                    "output": output,
                    "threshold": threshold,
                    "limit": limit,
                    "cursor": cursor,
                    "include_se": include_se,
                }
            )
        except ValidationError as ve:
            for err in ve.errors():
                if err.get("loc") == ("output",):
                    return build_error(
                        type="invalid_output",
                        message=f"Unknown output mode {output!r}.",
                        hint="Allowed: ['link', 'response', 'class'].",
                    )
            raise
        return _predict.predict(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("predict failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def evaluate_model(
    model_name: str,
    dataset: str,
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate a registered model on a dataset and return held-out metrics.

    ``model_name`` is the registry handle; ``dataset`` must contain the
    model's outcome column plus every predictor. ``threshold`` is the
    classification cutoff used for accuracy / precision / recall / F1 /
    confusion matrix (logistic only). ``n_calibration_bins`` (range
    [2, 50], auto-reduced on small datasets) controls the quantile
    calibration table; if reduced below 2 the response carries
    ``calibration: null`` with a note. Dispatch by model kind:
    logistic → ROC-AUC, PR-AUC, Brier, log-loss, accuracy/precision/
    recall/F1, confusion matrix, calibration table; OLS → RMSE, MAE,
    R², adjusted R²; poisson / negbin → RMSE, MAE, Pearson χ², deviance.
    """
    try:
        payload = _evaluate.EvaluateModelInput.model_validate(
            {
                "model_name": model_name,
                "dataset": dataset,
                "threshold": threshold,
                "n_calibration_bins": n_calibration_bins,
            }
        )
        return _evaluate.evaluate_model(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("evaluate_model failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def plot(
    name: str,
    kind: str,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    title: str | None = None,
    bins: int | None = None,
) -> dict[str, Any]:
    """Render a chart of a registered dataset to a base64-encoded PNG.

    ``kind`` is one of ``hist``, ``bar``, ``line``, ``scatter``, ``box``,
    ``violin``, ``heatmap``. Each kind requires its own columns: ``hist``
    needs ``x``; ``bar`` needs ``x`` (and may aggregate with ``y``); ``line``
    and ``scatter`` need both ``x`` and ``y``; ``box`` and ``violin`` need
    ``y`` (and may group by ``x``); ``heatmap`` visualizes the correlation
    matrix of every numeric column. ``hue`` is an optional color-group
    column. ``title`` adds a chart title. ``bins`` overrides the histogram
    bin count. Returns ``{ok, png_base64, width, height}``.
    """
    try:
        from pydantic import ValidationError

        try:
            payload = _plots.PlotInput.model_validate(
                {
                    "name": name,
                    "kind": kind,
                    "x": x,
                    "y": y,
                    "hue": hue,
                    "title": title,
                    "bins": bins,
                }
            )
        except ValidationError as ve:
            for err in ve.errors():
                if err.get("loc") == ("kind",):
                    return build_error(
                        type="invalid_kind",
                        message=f"Unknown kind {kind!r}.",
                        hint=(
                            "Allowed kinds: ['bar', 'box', 'heatmap', 'hist', "
                            "'line', 'scatter', 'violin']."
                        ),
                    )
            raise
        return _plots.plot(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("plot failed")
        return build_error(type="internal", message=str(exc))


@mcp.tool()
def emit_notebook(
    path: str | None = None,
    include_outputs: bool = False,
) -> dict[str, Any]:
    """Serialize the recorded session to a runnable Jupyter notebook.

    Writes a ``.ipynb`` containing a setup cell (imports + DuckDB connection
    + ``CREATE OR REPLACE TABLE`` reloads for every registered dataset)
    followed by one markdown + one code cell per successfully-executed tool
    call. When ``path`` is omitted the file lands at
    ``./session_<YYYYmmdd_HHMMSS>.ipynb`` in the current working directory.
    ``include_outputs`` is an API-stability placeholder: outputs are never
    captured during MCP calls, so re-execute with ``jupyter nbconvert --execute``
    to materialize them.
    """
    try:
        payload = _notebook.EmitNotebookInput(path=path, include_outputs=include_outputs)
        return _notebook.emit_notebook(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("emit_notebook failed")
        return build_error(type="internal", message=str(exc))


def main() -> None:  # pragma: no cover - exercised by the console-script smoke test
    """Run the MCP server on stdio. Console-script entry point."""
    logger.info("starting data-analyst-mcp on stdio")
    mcp.run()
