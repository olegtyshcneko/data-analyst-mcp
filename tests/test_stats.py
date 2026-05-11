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


# === test_hypothesis ===


def _make_two_group_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["A"] * 5 + ["B"] * 5,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )


def test_test_hypothesis_t_test_known_answer(call_tool, load_df_into_session):
    load_df_into_session("pairs", _make_two_group_df())
    result = call_tool(
        "test_hypothesis",
        {
            "kind": "t_test",
            "name": "pairs",
            "group_column": "g",
            "metric_column": "v",
            "group_a": "A",
            "group_b": "B",
        },
    )
    # scipy.stats.ttest_ind([1,2,3,4,5], [3,4,5,6,7], equal_var=True)
    #   statistic=-2.0, p=0.08051623795726262, df=8.0
    # Cohen's d (pooled, ddof=1) = -1.2649110640673518
    assert result["ok"] is True
    assert result["test"] == "t_test"
    assert result["statistic"] == pytest.approx(-2.0, abs=1e-4)
    assert result["p_value"] == pytest.approx(0.0805162380, abs=1e-4)
    assert result["df"] == pytest.approx(8.0, abs=1e-9)
    assert result["effect_size"]["name"] == "cohens_d"
    assert result["effect_size"]["value"] == pytest.approx(-1.2649110641, abs=1e-4)
    assert result["n_a"] == 5
    assert result["n_b"] == 5


def _make_welch_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["A"] * 5 + ["B"] * 8,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 12.0, 15.0],
        }
    )


def test_test_hypothesis_welch_known_answer(call_tool, load_df_into_session):
    load_df_into_session("wpairs", _make_welch_df())
    result = call_tool(
        "test_hypothesis",
        {
            "kind": "welch",
            "name": "wpairs",
            "group_column": "g",
            "metric_column": "v",
            "group_a": "A",
            "group_b": "B",
        },
    )
    # scipy.stats.ttest_ind([1,2,3,4,5], [3,4,5,6,7,10,12,15], equal_var=False)
    #   statistic=-2.88789438750785, p=0.01671412329574786, df=9.664541257500337
    assert result["ok"] is True
    assert result["test"] == "welch"
    assert result["statistic"] == pytest.approx(-2.8878943875, abs=1e-4)
    assert result["p_value"] == pytest.approx(0.0167141233, abs=1e-4)
    assert result["df"] == pytest.approx(9.6645412575, abs=1e-4)
    assert result["effect_size"]["name"] == "cohens_d"
    assert result["n_a"] == 5
    assert result["n_b"] == 8


def test_test_hypothesis_mann_whitney_known_answer(call_tool, load_df_into_session):
    load_df_into_session("pairs", _make_two_group_df())
    result = call_tool(
        "test_hypothesis",
        {
            "kind": "mann_whitney",
            "name": "pairs",
            "group_column": "g",
            "metric_column": "v",
            "group_a": "A",
            "group_b": "B",
        },
    )
    # scipy.stats.mannwhitneyu([1,2,3,4,5], [3,4,5,6,7], alternative='two-sided')
    #   U=4.5, p=0.11384629800665805
    # rank_biserial = 1 - 2*U/(n1*n2) = 1 - 2*4.5/25 = 0.64
    assert result["ok"] is True
    assert result["test"] == "mann_whitney"
    assert result["statistic"] == pytest.approx(4.5, abs=1e-4)
    assert result["p_value"] == pytest.approx(0.1138462980, abs=1e-4)
    assert result["effect_size"]["name"] == "rank_biserial"
    assert result["effect_size"]["value"] == pytest.approx(0.64, abs=1e-4)
    assert result["n_a"] == 5
    assert result["n_b"] == 5
