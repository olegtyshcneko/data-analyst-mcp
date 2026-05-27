"""Tests for the plot tool — characterization style.

Per spec §3 "Characterization tests, not pure TDD": pixel-perfect TDD on
matplotlib is wasted effort. Each plot kind asserts:
  1. ``result["ok"] is True``
  2. PNG magic-byte prefix on ``base64.b64decode(result["png_base64"])``
  3. ``len(decoded_bytes) >= 5000`` (non-trivial image)
  4. ``result["width"]`` and ``result["height"]`` are positive ints

Error paths get strict red/green TDD — they are behavior, not visual.
"""

from __future__ import annotations

import base64

import pandas as pd

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(result: dict) -> bytes:
    """Shared characterization assertion: ok + PNG header + size + dims."""
    assert result["ok"] is True, result
    raw = base64.b64decode(result["png_base64"])
    assert raw[:8] == _PNG_MAGIC
    assert len(raw) >= 5000
    assert isinstance(result["width"], int) and result["width"] > 0
    assert isinstance(result["height"], int) and result["height"] > 0
    return raw


def test_plot_unknown_dataset_returns_not_found(call_tool):
    result = call_tool("plot", {"name": "nope", "kind": "hist", "x": "x"})
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_plot_invalid_kind_returns_invalid_kind(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"x": [1, 2, 3]}))
    result = call_tool("plot", {"name": "tiny", "kind": "pie", "x": "x"})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_kind"


def test_plot_missing_column_returns_column_not_found(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"x": [1, 2, 3]}))
    result = call_tool("plot", {"name": "tiny", "kind": "hist", "x": "nope"})
    assert result["ok"] is False
    assert result["error"]["type"] == "column_not_found"


def test_plot_scatter_without_y_returns_missing_required_param(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = call_tool("plot", {"name": "tiny", "kind": "scatter", "x": "a"})
    assert result["ok"] is False
    assert result["error"]["type"] == "missing_required_param"


# === per-kind characterization ===


_NUMERIC_DF = pd.DataFrame(
    {
        "x": list(range(100)),
        "y": [i * 0.5 + 1.0 for i in range(100)],
        "g": ["A", "B"] * 50,
    }
)


def test_plot_hist_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "hist", "x": "x"})
    _assert_valid_png(result)


def test_plot_bar_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "bar", "x": "g"})
    _assert_valid_png(result)


def test_plot_line_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "line", "x": "x", "y": "y"})
    _assert_valid_png(result)


def test_plot_scatter_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "scatter", "x": "x", "y": "y", "hue": "g"})
    _assert_valid_png(result)


def test_plot_box_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "box", "x": "g", "y": "y"})
    _assert_valid_png(result)


def test_plot_violin_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "violin", "x": "g", "y": "y"})
    _assert_valid_png(result)


def test_plot_heatmap_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "heatmap"})
    _assert_valid_png(result)


def test_plot_title_changes_rendered_bytes(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    plain = call_tool("plot", {"name": "d", "kind": "hist", "x": "x"})
    titled = call_tool("plot", {"name": "d", "kind": "hist", "x": "x", "title": "Hello"})
    _assert_valid_png(plain)
    _assert_valid_png(titled)
    assert plain["png_base64"] != titled["png_base64"]


def test_plot_success_records_markdown_and_code_cells(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("d", _NUMERIC_DF)
    call_tool("plot", {"name": "d", "kind": "hist", "x": "x"})
    cells = get_recorder().cells
    assert len(cells) == 2
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert cells[0]["metadata"]["tool_name"] == "plot"
    assert cells[1]["metadata"]["tool_name"] == "plot"


def test_plot_error_records_no_cells(call_tool):
    from data_analyst_mcp.recorder import get_recorder

    result = call_tool("plot", {"name": "nope", "kind": "hist", "x": "x"})
    assert result["ok"] is False
    assert get_recorder().cells == []


def test_plot_tool_is_registered_via_fastmcp(server):
    import asyncio

    async def _list() -> list[str]:
        tools = await server.list_tools()
        return [t.name for t in tools]

    names = asyncio.run(_list())
    assert "plot" in names


def test_plot_png_roundtrips_through_pil(call_tool, load_df_into_session):
    from io import BytesIO

    from PIL import Image

    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "scatter", "x": "x", "y": "y"})
    raw = _assert_valid_png(result)
    with Image.open(BytesIO(raw)) as im:
        assert im.format == "PNG"
        w, h = im.size
    assert w == result["width"]
    assert h == result["height"]


# === regression_line + residual_diagnostic ===


def _ols_fixture_df(seed: int = 0, n: int = 200) -> pd.DataFrame:
    """Synthetic OLS dataset for diagnostic-plot tests."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    y = 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_regression_line_unknown_model_returns_model_not_found(call_tool):
    result = call_tool("regression_line", {"model_name": "nope", "predictor": "x1"})
    assert result["ok"] is False
    assert result["error"]["type"] == "model_not_found"


def _logistic_fixture_df(seed: int = 0, n: int = 500) -> pd.DataFrame:
    """Synthetic logistic dataset for OLS-only guard tests."""
    import numpy as np
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    eta = 0.5 * x1 - 0.3 * x2
    y = rng.binomial(1, expit(eta))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_regression_line_on_logistic_model_returns_ols_only(call_tool, load_df_into_session):
    df = _logistic_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "logistic", "model_name": "m_log"},
    )
    assert r["ok"], r
    result = call_tool("regression_line", {"model_name": "m_log", "predictor": "x1"})
    assert result["ok"] is False
    assert result["error"]["type"] == "regression_diagnostics_ols_only"


def test_regression_line_unknown_predictor_returns_column_not_found(
    call_tool, load_df_into_session
):
    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    result = call_tool("regression_line", {"model_name": "m", "predictor": "x_nope"})
    assert result["ok"] is False
    assert result["error"]["type"] == "column_not_found"


def test_regression_line_categorical_predictor_returns_non_numeric_predictor(
    call_tool, load_df_into_session
):
    """Patsy expands `y ~ group` to exog name `group[T.B]`. Requesting that
    predictor name surfaces as non_numeric_predictor because the underlying
    `group` column is string-typed in the training DataFrame."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, size=n)
    group = np.where(rng.uniform(size=n) > 0.5, "A", "B")
    y = 2.0 * x + (group == "A").astype(float) + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x, "group": group})
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x + group", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    # patsy expanded "group" → "group[T.B]" — that name IS in exog_names,
    # but it's a categorical dummy, not a numeric column.
    result = call_tool("regression_line", {"model_name": "m", "predictor": "group[T.B]"})
    assert result["ok"] is False
    assert result["error"]["type"] == "non_numeric_predictor"


def test_regression_line_returns_valid_png(call_tool, load_df_into_session):
    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    result = call_tool("regression_line", {"model_name": "m", "predictor": "x1"})
    _assert_valid_png(result)
    assert result["model_name"] == "m"
    assert result["plot_kind"] == "regression_line"


def test_regression_line_scatter_point_count_matches_endog(call_tool, load_df_into_session):
    """Slice 7: the scatter has one point per training-set row (len(endog))."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        RegressionLineInput,
        _build_regression_line_figure,  # type: ignore[reportPrivateUsage]
    )

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    entry = _session.get_model("m")
    assert entry is not None
    payload = RegressionLineInput(model_name="m", predictor="x1")
    fig = _build_regression_line_figure(entry, df, payload)
    ax = fig.axes[0]
    scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
    assert scatter_collections, "expected at least one PathCollection (scatter)"
    n_points = len(scatter_collections[0].get_offsets())
    assert n_points == len(entry._result.model.endog) == len(df)


def test_regression_line_fit_line_slope_matches_params(call_tool, load_df_into_session):
    """Slice 7: the rise/run of the rendered fit line equals
    ``entry._result.params[predictor]`` (we sweep the predictor while
    holding every other column at its mean, so β_other × mean is just a
    constant intercept shift — the line's slope is exactly β_predictor)."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import _compute_fit_line  # type: ignore[reportPrivateUsage]

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    entry = _session.get_model("m")
    assert entry is not None
    x_grid, y_pred, _, _ = _compute_fit_line(entry._result, df, "x1")
    slope = float((y_pred[-1] - y_pred[0]) / (x_grid[-1] - x_grid[0]))
    expected = float(entry._result.params["x1"])
    assert abs(slope - expected) < 1e-8


def test_regression_line_renders_95_ci_band(call_tool, load_df_into_session):
    """Slice 8: a 95 % mean-CI band patch is rendered on the axes
    (non-empty PolyCollection from ``fill_between``)."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        RegressionLineInput,
        _build_regression_line_figure,  # type: ignore[reportPrivateUsage]
        _compute_fit_line,  # type: ignore[reportPrivateUsage]
    )

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    entry = _session.get_model("m")
    assert entry is not None
    # The helper returns real CI bounds (lower < upper at every point).
    _, _, ci_lower, ci_upper = _compute_fit_line(entry._result, df, "x1")
    assert (ci_lower < ci_upper).all()
    # And the rendered figure has a PolyCollection patch from fill_between.
    fig = _build_regression_line_figure(
        entry, df, RegressionLineInput(model_name="m", predictor="x1")
    )
    ax = fig.axes[0]
    fill_collections = [
        c for c in ax.collections if "PolyCollection" in c.__class__.__name__
    ]
    assert fill_collections, "expected at least one fill_between PolyCollection on the axes"
