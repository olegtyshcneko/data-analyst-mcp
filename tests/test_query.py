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
