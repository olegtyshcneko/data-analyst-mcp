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


def test_materialize_query_creates_registered_dataset_with_row_count(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session(
        "src",
        pd.DataFrame({"x": [1, 2, 3, 4, 5], "g": ["a", "a", "b", "b", "b"]}),
    )

    result = call_tool(
        "materialize_query",
        {"sql": "SELECT x, g FROM src WHERE g = 'b'", "name": "filtered"},
    )

    assert result["ok"] is True
    assert result["name"] == "filtered"
    assert result["rows"] == 3
    assert result["total_rows"] == 3
    assert "filtered" in _session.get_datasets()
    entry = _session.get_datasets()["filtered"]
    assert entry.format == "derived"
    assert entry.rows == 3
    assert entry.path == "(query)"
    assert entry.read_options == {"sql": "SELECT x, g FROM src WHERE g = 'b'"}
