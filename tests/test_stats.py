"""Tests for the statistical tools (correlate, compare_groups, test_hypothesis).

Every assertion against a numeric statistic / p-value / effect-size is
hard-pinned to a value computed independently in scipy/statsmodels with a
fixed seed, at ≤1e-4 tolerance. See the comment above each assertion for
the source of the expected number.
"""

from __future__ import annotations

import pandas as pd
import pytest


# === correlate ===


def test_correlate_unknown_dataset_returns_not_found(call_tool):
    result = call_tool("correlate", {"name": "nope"})
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_correlate_unknown_column_returns_column_not_found(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    result = call_tool("correlate", {"name": "tiny", "columns": ["x", "zzz"]})
    assert result["ok"] is False
    assert result["error"]["type"] == "column_not_found"


def test_correlate_no_numeric_columns_returns_error(call_tool, load_df_into_session):
    load_df_into_session(
        "strs", pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    )
    result = call_tool("correlate", {"name": "strs"})
    assert result["ok"] is False
    assert result["error"]["type"] == "no_numeric_columns"


_XYZ_DF = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5], "z": [5, 7, 6, 9, 8]})


def test_correlate_pearson_known_answer(call_tool, load_df_into_session):
    load_df_into_session("xyz", _XYZ_DF)
    result = call_tool(
        "correlate",
        {"name": "xyz", "columns": ["x", "y", "z"], "method": "pearson", "plot": False},
    )
    assert result["ok"] is True
    # scipy.stats.pearsonr on the _XYZ_DF columns:
    #   x,y → 0.7745966692414835
    #   x,z → 0.7999999999999999
    #   y,z → 0.5163977794943223
    m = result["matrix"]
    labels = result["labels"]
    idx = {lab: i for i, lab in enumerate(labels)}
    assert m[idx["x"]][idx["y"]] == pytest.approx(0.7745966692, abs=1e-4)
    assert m[idx["x"]][idx["z"]] == pytest.approx(0.8000000000, abs=1e-4)
    assert m[idx["y"]][idx["z"]] == pytest.approx(0.5163977795, abs=1e-4)


def test_correlate_spearman_known_answer(call_tool, load_df_into_session):
    load_df_into_session("xyz", _XYZ_DF)
    result = call_tool(
        "correlate",
        {"name": "xyz", "columns": ["x", "y", "z"], "method": "spearman", "plot": False},
    )
    # scipy.stats.spearmanr on _XYZ_DF:
    #   x,y → 0.7378647873726218
    #   x,z → 0.7999999999999999
    #   y,z → 0.31622776601683794
    m = result["matrix"]
    labels = result["labels"]
    idx = {lab: i for i, lab in enumerate(labels)}
    assert m[idx["x"]][idx["y"]] == pytest.approx(0.7378647874, abs=1e-4)
    assert m[idx["x"]][idx["z"]] == pytest.approx(0.8000000000, abs=1e-4)
    assert m[idx["y"]][idx["z"]] == pytest.approx(0.3162277660, abs=1e-4)


def test_correlate_kendall_known_answer(call_tool, load_df_into_session):
    load_df_into_session("xyz", _XYZ_DF)
    result = call_tool(
        "correlate",
        {"name": "xyz", "columns": ["x", "y", "z"], "method": "kendall", "plot": False},
    )
    # scipy.stats.kendalltau on _XYZ_DF:
    #   x,y → 0.6708203932499368
    #   x,z → 0.6
    #   y,z → 0.22360679774997894
    m = result["matrix"]
    labels = result["labels"]
    idx = {lab: i for i, lab in enumerate(labels)}
    assert m[idx["x"]][idx["y"]] == pytest.approx(0.6708203932, abs=1e-4)
    assert m[idx["x"]][idx["z"]] == pytest.approx(0.6000000000, abs=1e-4)
    assert m[idx["y"]][idx["z"]] == pytest.approx(0.2236067977, abs=1e-4)


def test_correlate_plot_true_returns_heatmap_png_base64(call_tool, load_df_into_session):
    import base64

    load_df_into_session("xyz", _XYZ_DF)
    result = call_tool("correlate", {"name": "xyz", "plot": True})
    assert "heatmap_png_base64" in result
    raw = base64.b64decode(result["heatmap_png_base64"])
    # PNG magic bytes
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_correlate_records_markdown_and_code_cells(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("xyz", _XYZ_DF)
    call_tool("correlate", {"name": "xyz", "plot": False})
    cells = get_recorder().cells
    assert len(cells) == 2
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert cells[0]["metadata"]["tool_name"] == "correlate"
