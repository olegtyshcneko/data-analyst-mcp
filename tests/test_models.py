"""Tests for the modeling tool (fit_model).

Every assertion against a numeric coefficient / std-error / fit-statistic is
hard-pinned to a value computed independently in statsmodels with a fixed
random seed (or against a published statsmodels demo dataset), at the
tolerance noted next to the assertion. See the comment above each
assertion for the source of the expected number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_fit_model_unknown_dataset_returns_not_found(call_tool):
    result = call_tool(
        "fit_model", {"name": "nope", "formula": "y ~ x", "kind": "ols"}
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_fit_model_unknown_kind_returns_invalid_kind(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}))
    result = call_tool(
        "fit_model", {"name": "tiny", "formula": "y ~ x", "kind": "probit"}
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_kind"
