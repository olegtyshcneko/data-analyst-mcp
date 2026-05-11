"""Tests for the result-formatting helpers."""

from __future__ import annotations

import base64

import pytest


def test_truncate_rows_under_limit_returns_full_rows() -> None:
    from data_analyst_mcp.formatting import truncate_rows

    rows = [{"a": 1}, {"a": 2}, {"a": 3}]
    out = truncate_rows(rows, limit=5)

    assert out["rows"] == rows
    assert out["total_rows"] == 3
    assert out["truncated"] is False
    assert out["cursor"] is None
