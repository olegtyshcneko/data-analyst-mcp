"""Tests for the ``materialize_query`` tool."""

from __future__ import annotations

from typing import Any


def test_materialize_query_returns_ok_on_trivial_select(call_tool: Any) -> None:
    result = call_tool(
        "materialize_query",
        {"sql": "SELECT 1 AS x", "name": "trivial"},
    )

    assert result["ok"] is True
