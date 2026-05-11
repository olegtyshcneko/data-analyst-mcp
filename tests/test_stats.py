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
