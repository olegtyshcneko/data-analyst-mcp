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
