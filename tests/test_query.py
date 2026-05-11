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
