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
