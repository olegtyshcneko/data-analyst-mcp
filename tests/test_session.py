"""Tests for the session singleton (datasets registry + DuckDB connection)."""

from __future__ import annotations


def test_fresh_session_has_empty_datasets_dict() -> None:
    from data_analyst_mcp import session

    session.reset()

    assert session.get_datasets() == {}


def test_register_inserts_a_dataset_entry() -> None:
    from data_analyst_mcp import session

    session.reset()
    session.register(
        name="orders",
        path="/tmp/orders.csv",
        read_options={"header": True},
        format="csv",
        rows=42,
        columns=[{"name": "id", "dtype": "INTEGER"}],
    )

    entries = session.get_datasets()
    assert "orders" in entries
    entry = entries["orders"]
    assert entry.path == "/tmp/orders.csv"
    assert entry.read_options == {"header": True}
    assert entry.format == "csv"
    assert entry.rows == 42
    assert entry.columns == [{"name": "id", "dtype": "INTEGER"}]
    assert entry.registered_at is not None


def test_get_connection_returns_singleton_duckdb_handle() -> None:
    from data_analyst_mcp import session

    session.reset()
    con_a = session.get_connection()
    con_b = session.get_connection()

    assert con_a is con_b
    # Smoke: the handle is usable as a DuckDB connection.
    result = con_a.sql("SELECT 1 AS x").fetchall()
    assert result == [(1,)]
