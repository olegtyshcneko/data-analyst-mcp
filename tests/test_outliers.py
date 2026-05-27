"""Tests for the ``find_outliers`` tool.

Every numeric assertion uses hand-computed or scipy/numpy/sklearn-derived
expected values at ≤1e-4 tolerance. Synthetic fixtures use a fixed seed so
runs are reproducible byte-for-byte.
"""

from __future__ import annotations

from typing import Any

import pytest


# === shared ===


def test_find_outliers_unknown_dataset_returns_not_found(call_tool: Any) -> None:
    result = call_tool(
        "find_outliers",
        {"name": "nope", "columns": ["x"], "method": "iqr"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_find_outliers_unknown_column_returns_column_not_found(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session("tiny", pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    result = call_tool(
        "find_outliers",
        {"name": "tiny", "columns": ["x", "zzz"], "method": "iqr"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "column_not_found"


def test_find_outliers_non_numeric_column_returns_typed_error(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session(
        "mixed",
        pd.DataFrame({"x": [1, 2, 3], "g": ["a", "b", "c"]}),
    )
    result = call_tool(
        "find_outliers",
        {"name": "mixed", "columns": ["x", "g"], "method": "iqr"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "non_numeric_column"


# === iqr ===


def test_find_outliers_iqr_default_threshold_flags_extreme_value(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Default threshold k=1.5 flags the planted extreme value.

    Data: 100 nominal values in [0, 1) plus one extreme value at 1000.
    With k=1.5, the extreme must be flagged; the nominal values must not.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed=0)
    values = rng.uniform(0.0, 1.0, size=100).tolist()
    values.append(1000.0)
    load_df_into_session("d", pd.DataFrame({"v": values}))

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr"},
    )

    assert result["ok"] is True
    assert result["method"] == "iqr"
    assert result["threshold_used"] == 1.5
    # The extreme value (row 100, score=1000) must be flagged.
    indices = {o["row_index"] for o in result["outliers"]}
    assert 100 in indices
    assert result["n_outliers"] >= 1


def test_find_outliers_iqr_known_answer_score(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Hand-computed IQR + score for [1..10, 100].

    Q1 = np.quantile(..., 0.25) = 3.5
    Q3 = np.quantile(..., 0.75) = 8.5
    IQR = 5.0; upper fence = 8.5 + 1.5 * 5 = 16.0
    Value 100 → flagged, score = (100 - 16) / 5 = 16.8
    """
    import pandas as pd

    load_df_into_session(
        "d",
        pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]}),
    )
    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr"},
    )
    assert result["ok"] is True
    assert result["n_outliers"] == 1
    flagged = result["outliers"][0]
    assert flagged["row_index"] == 10
    assert flagged["score"] == pytest.approx(16.8, abs=1e-4)
    assert flagged["values"] == {"v": 100.0}


def test_find_outliers_iqr_per_column_aggregation(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Rows flagged by any column must be in the union; per_column_flags
    must report the right index for each column independently."""
    import pandas as pd

    # Column a: row 5 is extreme (100). Column b: row 9 is extreme (200).
    # All other rows are 1..10 in both columns. Both rows must be flagged
    # at the row level; per_column_flags["a"] = [5], ["b"] = [9].
    a = [1, 2, 3, 4, 5, 100, 7, 8, 9, 10]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 200]
    load_df_into_session("d", pd.DataFrame({"a": a, "b": b}))

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["a", "b"], "method": "iqr"},
    )
    assert result["ok"] is True
    flagged = {o["row_index"] for o in result["outliers"]}
    assert 5 in flagged
    assert 9 in flagged
    assert result["per_column_flags"]["a"] == [5]
    assert result["per_column_flags"]["b"] == [9]


def test_find_outliers_iqr_custom_threshold_widens_band(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """A larger threshold k must flag *fewer* values for the same data.

    Construct a column whose only mild-outlier rows sit between
    Q3 + 1.5·IQR and Q3 + 3.0·IQR. With k=1.5 they are flagged; with
    k=3.0 they are not.
    """
    import pandas as pd

    # Values 1..10 plus a mild outlier at 18. With Q1=3.25, Q3=7.75,
    # IQR=4.5 → upper fence at 1.5 is 7.75 + 6.75 = 14.5 (18 > 14.5 →
    # flagged at k=1.5). Upper fence at 3.0 is 7.75 + 13.5 = 21.25
    # (18 < 21.25 → not flagged at k=3.0).
    data = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18]})
    load_df_into_session("d", data)

    r15 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr", "threshold": 1.5},
    )
    r30 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr", "threshold": 3.0},
    )

    assert r15["threshold_used"] == 1.5
    assert r30["threshold_used"] == 3.0
    assert r15["n_outliers"] == 1
    assert r30["n_outliers"] == 0
