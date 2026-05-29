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


def test_regression_line_small_dataset_fewer_than_grid_points(call_tool, load_df_into_session):
    """A training set smaller than the dense fit grid (n_points=100) must
    still render. The prediction grid has n_points rows regardless of
    len(df); slicing it from the (shorter) training DataFrame raises a
    length-mismatch ValueError that surfaces as a generic `internal` error."""
    df = _ols_fixture_df(n=50)
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    result = call_tool("regression_line", {"model_name": "m", "predictor": "x1"})
    _assert_valid_png(result)
    assert result["plot_kind"] == "regression_line"


def test_regression_line_with_nan_rows_in_training_data(call_tool, load_df_into_session):
    """statsmodels drops rows with a NaN in any model variable during fit, so
    model.endog is shorter than df[predictor]. The scatter must align its
    x-values to the surviving rows; plotting the full predictor column against
    the shorter endog is a length mismatch that crashes matplotlib (surfaced
    as a generic `internal` error)."""
    import numpy as np

    df = _ols_fixture_df(n=200)
    df.loc[:9, "x2"] = np.nan  # 10 rows dropped during fit (NaN in a predictor)
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    result = call_tool("regression_line", {"model_name": "m", "predictor": "x1"})
    _assert_valid_png(result)


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
    fill_collections = [c for c in ax.collections if "PolyCollection" in c.__class__.__name__]
    assert fill_collections, "expected at least one fill_between PolyCollection on the axes"


# === residual_diagnostic ===


def test_residual_diagnostic_unknown_model_returns_model_not_found(call_tool):
    result = call_tool("residual_diagnostic", {"model_name": "nope"})
    assert result["ok"] is False
    assert result["error"]["type"] == "model_not_found"


def test_residual_diagnostic_on_logistic_returns_ols_only(call_tool, load_df_into_session):
    df = _logistic_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "logistic", "model_name": "m_log"},
    )
    assert r["ok"], r
    result = call_tool("residual_diagnostic", {"model_name": "m_log"})
    assert result["ok"] is False
    assert result["error"]["type"] == "regression_diagnostics_ols_only"


def test_residual_diagnostic_resid_vs_fitted_single_axes(call_tool, load_df_into_session):
    """Slice 11: kind='resid_vs_fitted' produces a single-axes figure."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        ResidualDiagnosticInput,
        _build_residual_diagnostic_figure,  # type: ignore[reportPrivateUsage]
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
    payload = ResidualDiagnosticInput(model_name="m", kind="resid_vs_fitted")
    fig = _build_residual_diagnostic_figure(entry, payload)
    assert len(fig.axes) == 1
    # Also verify the round-trip through the tool produces a valid PNG.
    result = call_tool("residual_diagnostic", {"model_name": "m", "kind": "resid_vs_fitted"})
    _assert_valid_png(result)
    assert result["plot_kind"] == "resid_vs_fitted"
    assert result["model_name"] == "m"


def test_residual_diagnostic_qq_single_axes(call_tool, load_df_into_session):
    """Slice 12: kind='qq' produces a single-axes figure."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        ResidualDiagnosticInput,
        _build_residual_diagnostic_figure,  # type: ignore[reportPrivateUsage]
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
    fig = _build_residual_diagnostic_figure(
        entry, ResidualDiagnosticInput(model_name="m", kind="qq")
    )
    assert len(fig.axes) == 1
    result = call_tool("residual_diagnostic", {"model_name": "m", "kind": "qq"})
    _assert_valid_png(result)
    assert result["plot_kind"] == "qq"


def test_residual_diagnostic_scale_location_single_axes(call_tool, load_df_into_session):
    """Slice 13: kind='scale_location' produces a single-axes figure."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        ResidualDiagnosticInput,
        _build_residual_diagnostic_figure,  # type: ignore[reportPrivateUsage]
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
    fig = _build_residual_diagnostic_figure(
        entry, ResidualDiagnosticInput(model_name="m", kind="scale_location")
    )
    assert len(fig.axes) == 1
    result = call_tool("residual_diagnostic", {"model_name": "m", "kind": "scale_location"})
    _assert_valid_png(result)
    assert result["plot_kind"] == "scale_location"


def test_residual_diagnostic_all_produces_2x2_grid(call_tool, load_df_into_session):
    """Slice 14: kind='all' (the default) produces a 2×2 grid (4 axes)."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        ResidualDiagnosticInput,
        _build_residual_diagnostic_figure,  # type: ignore[reportPrivateUsage]
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
    fig = _build_residual_diagnostic_figure(
        entry, ResidualDiagnosticInput(model_name="m", kind="all")
    )
    assert len(fig.axes) == 4
    result = call_tool("residual_diagnostic", {"model_name": "m"})  # kind defaults to "all"
    _assert_valid_png(result)
    assert result["plot_kind"] == "all"


def test_residual_diagnostic_resid_vs_fitted_has_lowess_overlay(call_tool, load_df_into_session):
    """Slice 15: resid_vs_fitted axes carry >= 2 line2d objects — one for
    the y=0 reference, one for the LOWESS smoothed overlay."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        ResidualDiagnosticInput,
        _build_residual_diagnostic_figure,  # type: ignore[reportPrivateUsage]
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
    fig = _build_residual_diagnostic_figure(
        entry, ResidualDiagnosticInput(model_name="m", kind="resid_vs_fitted")
    )
    ax = fig.axes[0]
    # axhline (y=0) + LOWESS plot() → at least 2 Line2D objects.
    assert len(ax.lines) >= 2


def test_residual_diagnostic_all_4th_panel_uses_cooks_distance(call_tool, load_df_into_session):
    """Slice 16: the 4th panel of kind='all' renders residuals vs leverage
    with Cook's distance reference contours — the leverage and cooks_d
    arrays must come from ``entry._result.get_influence().cooks_distance``
    (verified via a non-empty array and >=4 lines on the panel from the
    reference contours)."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.plots import (
        ResidualDiagnosticInput,
        _build_residual_diagnostic_figure,  # type: ignore[reportPrivateUsage]
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
    # First confirm get_influence returns a non-empty cooks_distance.
    cooks_d, _p_vals = entry._result.get_influence().cooks_distance
    assert len(cooks_d) == len(df)
    # The "all" figure's 4th panel must carry the Cook's reference contours.
    fig = _build_residual_diagnostic_figure(
        entry, ResidualDiagnosticInput(model_name="m", kind="all")
    )
    panel_4 = fig.axes[3]
    # Cook's distance reference contours: two D values (0.5 and 1.0), each
    # rendered as ± boundary → at least 4 reference Line2D objects.
    assert len(panel_4.lines) >= 4


def test_residual_diagnostic_resid_vs_fitted_recorder_cell_plots(call_tool, load_df_into_session):
    """``residual_diagnostic`` with ``kind='resid_vs_fitted'`` must emit a
    cell that scatter-plots residuals against fitted values and overlays
    the LOWESS smoother."""
    from data_analyst_mcp.recorder import get_recorder

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool("residual_diagnostic", {"model_name": "m", "kind": "resid_vs_fitted"})
    assert r["ok"], r

    code_cells = [c for c in get_recorder().cells if c["cell_type"] == "code"]
    source = code_cells[-1]["source"]
    assert "scatter(_fitted, _resid" in source
    assert "axhline(0" in source
    assert "lowess" in source
    assert "plt.show" in source or "fig.savefig" in source


def test_residual_diagnostic_scale_location_recorder_cell_plots(call_tool, load_df_into_session):
    """``residual_diagnostic`` with ``kind='scale_location'`` must emit a
    cell that plots ``sqrt(|standardized residuals|)`` against fitted
    values with the LOWESS overlay."""
    from data_analyst_mcp.recorder import get_recorder

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool("residual_diagnostic", {"model_name": "m", "kind": "scale_location"})
    assert r["ok"], r

    code_cells = [c for c in get_recorder().cells if c["cell_type"] == "code"]
    source = code_cells[-1]["source"]
    assert "sqrt" in source
    assert "_std_resid" in source
    assert "lowess" in source
    assert "plt.show" in source or "fig.savefig" in source


def test_residual_diagnostic_all_recorder_cell_plots_2x2_grid(call_tool, load_df_into_session):
    """``residual_diagnostic`` with ``kind='all'`` must emit a 2×2 grid:
    resid-vs-fitted, Q-Q, scale-location, residuals-vs-leverage with
    Cook's distance reference contours."""
    from data_analyst_mcp.recorder import get_recorder

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool("residual_diagnostic", {"model_name": "m", "kind": "all"})
    assert r["ok"], r

    code_cells = [c for c in get_recorder().cells if c["cell_type"] == "code"]
    source = code_cells[-1]["source"]
    assert "subplots(2, 2" in source
    assert "probplot" in source
    assert "lowess" in source
    assert "hat_matrix_diag" in source
    assert "plt.show" in source or "fig.savefig" in source


def test_residual_diagnostic_qq_recorder_cell_plots(call_tool, load_df_into_session):
    """``residual_diagnostic`` with ``kind='qq'`` must emit a cell that
    actually renders a Q-Q plot — the legacy cell computed residuals and
    stopped at a comment, producing nothing on replay.
    """
    from data_analyst_mcp.recorder import get_recorder

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool("residual_diagnostic", {"model_name": "m", "kind": "qq"})
    assert r["ok"], r

    code_cells = [c for c in get_recorder().cells if c["cell_type"] == "code"]
    source = code_cells[-1]["source"]
    assert "probplot" in source, "qq cell must call scipy.stats.probplot"
    assert "plt.show" in source or "fig.savefig" in source


def test_regression_line_recorder_cell_is_faithful_reproducer(call_tool, load_df_into_session):
    """The recorded code cell for ``regression_line`` must be a faithful
    reproducer of the live render: a linspace grid over the predictor,
    other predictors held at mean / first-value, ``get_prediction(...)
    .summary_frame()`` for the band, ``fill_between`` for the CI ribbon.

    Earlier code emitted ``_grid = _df.copy()`` and plotted the unsorted
    raw predictor → a tangled line at replay. We assert on the source
    string because the alternative (exec the cell and introspect axes) is
    heavy and equally brittle.
    """
    from data_analyst_mcp.recorder import get_recorder

    df = _ols_fixture_df()
    load_df_into_session("d", df)
    r = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x1 + x2", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    r = call_tool("regression_line", {"model_name": "m", "predictor": "x1"})
    assert r["ok"], r

    cells = get_recorder().cells
    # The most recent code cell came from regression_line.
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert code_cells, "no code cell was recorded"
    source = code_cells[-1]["source"]
    # Faithful-reproducer markers — every one of these is in the live path
    # _compute_fit_line + _build_regression_line_figure transliterates to.
    assert "linspace" in source, "fit-line grid must be linspace over the predictor range"
    assert "get_prediction" in source
    assert "summary_frame" in source
    assert "fill_between" in source
    # The legacy bug: copying _df unchanged and reusing it as both x and grid.
    assert "_df.copy()" not in source


def test_cooks_distance_contour_matches_canonical_formula() -> None:
    """The Cook's distance boundary helper returns ``|r| = sqrt(D · p · (1-h) / h)``.

    Earlier code used ``sqrt(D · h / (1-h)^2)`` (off by orders of magnitude
    at typical leverage values). At ``h=0.05, D=1.0, p=3`` the correct
    boundary is ``|r| ≈ 7.5498``; this test pins that value to <1e-6
    tolerance and also re-derives it across a grid.
    """
    import math

    import numpy as np

    from data_analyst_mcp.tools.plots import (
        _cooks_distance_contour,  # type: ignore[reportPrivateUsage]
    )

    h = 0.05
    d_ref = 1.0
    p = 3
    expected_at_h = math.sqrt(d_ref * p * (1.0 - h) / h)
    assert abs(expected_at_h - 7.549834435270749) < 1e-9

    lev_grid = np.array([h])
    lev_out, pos, neg = _cooks_distance_contour(d_ref, lev_grid, p)
    # Helper echoes the grid back unchanged.
    assert lev_out is lev_grid or np.array_equal(lev_out, lev_grid)
    assert abs(float(pos[0]) - expected_at_h) < 1e-6
    assert abs(float(neg[0]) + expected_at_h) < 1e-6
    # And the relation holds across a finer grid too.
    lev_grid = np.linspace(0.01, 0.5, 50)
    _, pos, neg = _cooks_distance_contour(0.5, lev_grid, 4)
    expected = np.sqrt(0.5 * 4 * (1.0 - lev_grid) / lev_grid)
    assert np.allclose(pos, expected, atol=1e-9)
    assert np.allclose(neg, -expected, atol=1e-9)
