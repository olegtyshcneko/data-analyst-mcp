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


def test_truncate_rows_over_limit_truncates_with_cursor() -> None:
    from data_analyst_mcp.formatting import truncate_rows

    rows = [{"i": i} for i in range(10)]
    out = truncate_rows(rows, limit=3)

    assert out["rows"] == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert out["total_rows"] == 10
    assert out["truncated"] is True
    assert out["cursor"] == 3


def test_rows_to_dicts_converts_duckdb_relation_to_list_of_dicts() -> None:
    import duckdb

    from data_analyst_mcp.formatting import rows_to_dicts

    con = duckdb.connect()
    rel = con.sql("SELECT 1 AS a, 'x' AS b UNION ALL SELECT 2, 'y' ORDER BY a")

    out = rows_to_dicts(rel)

    assert out == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
