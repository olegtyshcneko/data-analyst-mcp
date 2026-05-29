"""Tests for the ``query`` SQL tool."""

from __future__ import annotations

import os
from typing import Any

import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
MESSY_CSV = os.path.join(FIXTURE_DIR, "messy.csv")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE foo",
        "INSERT INTO foo VALUES (1)",
        "UPDATE foo SET x = 1",
        "DELETE FROM foo WHERE x = 1",
        "CREATE TABLE bar (x INT)",
        "SET memory_limit='1GB'",
    ],
)
def test_query_rejects_write_statements(call_tool: Any, sql: str) -> None:
    result = call_tool("query", {"sql": sql})

    assert result["ok"] is False
    assert result["error"]["type"] == "write_not_allowed"


def test_query_rejects_multistatement_injection(call_tool: Any) -> None:
    """``query`` must reject a multi-statement payload that splices a
    second (potentially destructive) statement after ``;``. The legacy
    ``_first_keyword`` allowlist only inspected the leading SELECT and
    let the injection through; DuckDB then happily executed it.
    """
    from data_analyst_mcp import session as _session

    con = _session.get_connection()
    con.execute('CREATE OR REPLACE TABLE "base" AS SELECT 1 AS x')

    result = call_tool(
        "query",
        {"sql": "SELECT 1; CREATE TABLE evil AS SELECT 42"},
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "write_not_allowed"
    # The injected statement must not have created the evil table.
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "evil" not in tables, "injected CREATE TABLE was executed"


@pytest.mark.parametrize(
    "sql",
    [
        # Semicolon inside a /* block */ comment.
        "SELECT 1 AS x /* ; nope */",
        # Semicolon inside a single-quoted string literal.
        "SELECT 'a;b' AS x",
        # Trailing single semicolon (statement terminator).
        "SELECT 1 AS x;",
    ],
)
def test_query_accepts_benign_semicolons(call_tool: Any, sql: str) -> None:
    """The multi-statement guard must not false-positive on ``;`` inside
    block comments / string literals / trailing whitespace. Trailing line
    comments (``-- ...``) are covered separately by
    ``test_query_accepts_trailing_line_comment`` now that the COUNT(*) /
    auto-LIMIT helpers break onto a fresh line."""
    result = call_tool("query", {"sql": sql})
    assert result["ok"] is True, result


def test_query_rejects_with_cte_injection(call_tool: Any) -> None:
    """A WITH-prefixed query that splices a second statement after ``;``
    is rejected — the scanner is keyword-agnostic."""
    from data_analyst_mcp import session as _session

    con = _session.get_connection()
    con.execute('CREATE OR REPLACE TABLE "base" AS SELECT 1 AS x')

    result = call_tool(
        "query",
        {"sql": "WITH cte AS (SELECT 1) SELECT * FROM cte; CREATE TABLE evil AS SELECT 42"},
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "write_not_allowed"
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "evil" not in tables


def test_query_select_returns_rows_columns_total_and_timing(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool(
        "query",
        {"sql": "SELECT customer_id FROM messy ORDER BY customer_id", "limit": 3},
    )

    assert result["ok"] is True
    assert result["columns"] == ["customer_id"]
    assert len(result["rows"]) == 3
    assert result["total_rows"] == 5000
    assert "execution_time_ms" in result
    assert result["execution_time_ms"] >= 0


def test_query_accepts_trailing_line_comment(call_tool: Any) -> None:
    """A SELECT ending in a ``-- line comment`` must execute, count, and
    auto-LIMIT correctly. The COUNT(*) row-count wrapper spliced the closing
    paren onto the commented-out line (``FROM (SELECT ... -- c)`` → the
    ``)`` is swallowed by the comment → parser error), and the auto-LIMIT
    append put ``LIMIT N`` on the same line too. Both now break onto a fresh
    line."""
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool(
        "query",
        {"sql": "SELECT customer_id FROM messy -- trailing comment", "limit": 3},
    )

    assert result["ok"] is True, result
    assert len(result["rows"]) == 3
    assert result["total_rows"] == 5000


def test_query_accepts_leading_comment_before_keyword(call_tool: Any) -> None:
    """A SELECT prefixed with a ``-- line`` or ``/* block */`` comment must
    be accepted — the leading-keyword guard skips comments to find the real
    verb instead of seeing the comment's first char (``-`` / ``/``) and
    rejecting valid SQL as ``write_not_allowed``."""
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool(
        "query",
        {"sql": "-- fetch ids\nSELECT customer_id FROM messy", "limit": 3},
    )

    assert result["ok"] is True, result
    assert len(result["rows"]) == 3


def test_query_explicit_limit_is_honored(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    # Explicit LIMIT 7 should win over auto-limit of 50.
    result = call_tool(
        "query",
        {"sql": "SELECT customer_id FROM messy LIMIT 7", "limit": 50},
    )

    assert result["ok"] is True
    assert len(result["rows"]) == 7


def test_query_truncated_true_when_rows_hit_limit(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("query", {"sql": "SELECT customer_id FROM messy", "limit": 10})

    assert result["truncated"] is True
