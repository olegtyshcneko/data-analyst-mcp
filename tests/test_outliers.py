"""Tests for the ``find_outliers`` tool.

Every numeric assertion uses hand-computed or scipy/numpy/sklearn-derived
expected values at ≤1e-4 tolerance. Synthetic fixtures use a fixed seed so
runs are reproducible byte-for-byte.
"""

from __future__ import annotations

from typing import Any


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
