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


def test_reset_drops_registered_tables_and_keeps_connection() -> None:
    from data_analyst_mcp import session

    session.reset()
    con = session.get_connection()
    con.execute("CREATE OR REPLACE TABLE keep_me AS SELECT 1 AS x")
    session.register(
        name="keep_me",
        path="(memory)",
        read_options={},
        format="csv",
        rows=1,
        columns=[],
    )

    session.reset()

    # Registry is empty.
    assert session.get_datasets() == {}
    # Connection identity is preserved.
    assert session.get_connection() is con
    # And the table is gone — reset drops registered tables on the way out.
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    assert "keep_me" not in tables


def test_register_records_content_hash_for_file_backed_dataset(tmp_path) -> None:
    import hashlib

    from data_analyst_mcp import session

    csv = tmp_path / "tiny.csv"
    csv.write_bytes(b"a,b\n1,2\n3,4\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    session.reset()
    session.register(name="tiny", path=str(csv), read_options={}, format="csv", rows=2, columns=[])

    assert session.get_datasets()["tiny"].source_hash == expected


def test_register_records_sentinel_hash_for_non_file_paths() -> None:
    from data_analyst_mcp import session

    session.reset()
    session.register(
        name="derived",
        path="(query)",
        read_options={"sql": "SELECT 1"},
        format="derived",
        rows=1,
        columns=[],
    )
    session.register(
        name="mem",
        path="(dataframe)",
        read_options={},
        format="dataframe",
        rows=1,
        columns=[],
    )

    assert session.get_datasets()["derived"].source_hash.startswith("sentinel:")
    assert session.get_datasets()["mem"].source_hash.startswith("sentinel:")
