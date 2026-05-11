"""Tests for the session singleton (datasets registry + DuckDB connection)."""

from __future__ import annotations


def test_fresh_session_has_empty_datasets_dict() -> None:
    from data_analyst_mcp import session

    session.reset()

    assert session.get_datasets() == {}
