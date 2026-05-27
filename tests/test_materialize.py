"""Tests for the ``materialize_query`` tool."""

from __future__ import annotations

from typing import Any

import pytest


def test_materialize_query_returns_ok_on_trivial_select(call_tool: Any) -> None:
    result = call_tool(
        "materialize_query",
        {"sql": "SELECT 1 AS x", "name": "trivial"},
    )

    assert result["ok"] is True


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO foo VALUES (1)",
        "UPDATE foo SET x = 1",
        "DROP TABLE foo",
    ],
)
def test_materialize_query_rejects_write_statements(call_tool: Any, sql: str) -> None:
    result = call_tool(
        "materialize_query",
        {"sql": sql, "name": "out"},
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "write_not_allowed"


@pytest.mark.parametrize(
    "sql",
    [
        "DESCRIBE foo",
        "SHOW TABLES",
        "PRAGMA show_tables",
    ],
)
def test_materialize_query_rejects_meta_statements(call_tool: Any, sql: str) -> None:
    result = call_tool(
        "materialize_query",
        {"sql": sql, "name": "out"},
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "write_not_allowed"
