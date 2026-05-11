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
