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
from data_analyst_mcp.tools import models as _models
from data_analyst_mcp.tools import notebook as _notebook
from data_analyst_mcp.tools import plots as _plots
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
def fit_model(
    name: str,
    formula: str,
    kind: str = "ols",
    robust: bool = False,
) -> dict[str, Any]:
    """Fit an OLS / logistic / Poisson regression with diagnostics.

    ``formula`` is Wilkinson-style (patsy), e.g. ``price ~ sqft + C(area)``.
    ``kind`` picks the model family: ``ols`` (default, linear), ``logistic``
    (binary outcome via logit), or ``poisson`` (non-negative counts).
    ``robust=True`` switches OLS to HC3 heteroskedasticity-robust standard
    errors (coefficients unchanged, std errors recomputed). Returns
    coefficients (with std errors, t / z stats, p-values, 95% CIs), fit
    statistics (R^2 / adj-R^2 / pseudo-R^2 / AIC / BIC / n_obs / df_resid),
    diagnostics (Breusch-Pagan, Durbin-Watson, Jarque-Bera, condition number,
    VIF for OLS), a ``warnings`` list (``high_multicollinearity``,
    ``heteroskedasticity``, ``non_normal_residuals``, ``overdispersion``),
    and a 2-3 sentence plain-English interpretation.
    """
    try:
        from pydantic import ValidationError

        try:
            payload = _models.FitModelInput.model_validate(
                {"name": name, "formula": formula, "kind": kind, "robust": robust}
            )
        except ValidationError as ve:
            for err in ve.errors():
                if err.get("loc") == ("kind",):
                    return build_error(
                        type="invalid_kind",
                        message=f"Unknown kind {kind!r}.",
                        hint="Allowed kinds: ['logistic', 'ols', 'poisson'].",
                    )
            raise
        return _models.fit_model(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("fit_model failed")
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
