"""Conformance tests for damcp-digest-v1 (spec v4 §3)."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def con() -> Iterator[Any]:
    from data_analyst_mcp import session

    connection = session.get_connection()
    row = connection.execute("SELECT current_setting('threads')").fetchone()
    original_threads = int(row[0])
    yield connection
    # Tables here are created directly on the shared connection (never
    # registered), so the autouse session reset cannot drop them — clean up
    # locally to keep the catalog empty for later test files.
    names = [
        r[0]
        for r in connection.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
        ).fetchall()
    ]
    for name in names:
        escaped = name.replace('"', '""')
        connection.execute(f'DROP TABLE IF EXISTS "{escaped}"')
    connection.execute(f"SET threads={original_threads}")


def _make(con: Any, name: str, select_sql: str) -> None:
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS {select_sql}')


def test_golden_vector_single_int_column(con: Any) -> None:
    """Pins the exact byte layout: one BIGINT column 'a', rows [1, NULL]."""
    from data_analyst_mcp.digest import digest_table

    _make(con, "g", "SELECT * FROM (VALUES (CAST(1 AS BIGINT)), (NULL)) t(a)")

    h = hashlib.sha256()
    h.update(b"\x01")  # schema part
    h.update((0).to_bytes(8, "little"))  # column position
    h.update(len(b"a").to_bytes(8, "little") + b"a")
    h.update(len(b"BIGINT").to_bytes(8, "little") + b"BIGINT")
    h.update(b"\x02")  # value part
    one = (1).to_bytes(1, "little", signed=True)
    h.update(b"\x11" + len(one).to_bytes(8, "little") + one)  # INT tag
    h.update(b"\x00" + (0).to_bytes(8, "little"))  # NULL tag, empty payload
    assert digest_table(con, "g") == h.hexdigest()


def test_same_table_twice_is_stable(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "s", "SELECT range AS a, range * 1.5 AS b FROM range(1000)")
    assert digest_table(con, "s") == digest_table(con, "s")


def test_row_order_changes_digest(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "o1", "SELECT * FROM (VALUES (1), (2)) t(a)")
    d1 = digest_table(con, "o1")
    _make(con, "o1", "SELECT * FROM (VALUES (2), (1)) t(a)")
    assert digest_table(con, "o1") != d1


def test_chunk_size_invariance(con: Any, monkeypatch: Any) -> None:
    from data_analyst_mcp import digest as digest_mod

    _make(con, "c", "SELECT range AS a FROM range(50)")
    big = digest_mod.digest_table(con, "c")
    monkeypatch.setattr(digest_mod, "CHUNK_ROWS", 3)
    assert digest_mod.digest_table(con, "c") == big


def test_signed_zero_and_nan_are_distinct_and_stable(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "f1", "SELECT CAST(0.0 AS DOUBLE) AS a")
    _make(con, "f2", "SELECT CAST(-0.0 AS DOUBLE) AS a")
    assert digest_table(con, "f1") != digest_table(con, "f2")
    _make(con, "f3", "SELECT CAST('nan' AS DOUBLE) AS a")
    assert digest_table(con, "f3") == digest_table(con, "f3")


def test_null_differs_from_null_string(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "n1", "SELECT CAST(NULL AS VARCHAR) AS a")
    _make(con, "n2", "SELECT '<null>' AS a")
    assert digest_table(con, "n1") != digest_table(con, "n2")


def test_timestamp_ns_nanoseconds_are_not_truncated(con: Any) -> None:
    """The exact collision the review demonstrated: two TIMESTAMP_NS values
    equal at microsecond resolution must digest differently."""
    from data_analyst_mcp.digest import digest_table

    _make(con, "t1", "SELECT TIMESTAMP_NS '2024-01-01 00:00:00.123456789' AS a")
    _make(con, "t2", "SELECT TIMESTAMP_NS '2024-01-01 00:00:00.123456780' AS a")
    assert digest_table(con, "t1") != digest_table(con, "t2")


def test_timestamp_variants_have_distinct_digests(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "v1", "SELECT TIMESTAMP '2024-01-01 00:00:01' AS a")
    _make(con, "v2", "SELECT TIMESTAMP_S '2024-01-01 00:00:01' AS a")
    assert digest_table(con, "v1") != digest_table(con, "v2")  # schema type string differs


def test_decimal_scale_matters(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "d1", "SELECT CAST(1.50 AS DECIMAL(9,2)) AS a")
    _make(con, "d2", "SELECT CAST(1.500 AS DECIMAL(9,3)) AS a")
    assert digest_table(con, "d1") != digest_table(con, "d2")


def test_nested_list_and_struct_supported(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "ls", "SELECT [1, 2, 3] AS a, {'k': 'v'} AS b")
    assert digest_table(con, "ls") is not None
    assert digest_table(con, "ls") == digest_table(con, "ls")


def test_union_type_is_undigestable(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "u", "SELECT union_value(num := 2) AS a")
    assert digest_table(con, "u") is None


def test_single_thread_scan_restores_threads_setting(con: Any) -> None:
    from data_analyst_mcp.digest import single_thread_scan

    con.execute("SET threads=4")
    with single_thread_scan(con):
        row = con.execute("SELECT current_setting('threads')").fetchone()
        assert int(row[0]) == 1
    row = con.execute("SELECT current_setting('threads')").fetchone()
    assert int(row[0]) == 4


def test_single_thread_scan_restores_on_exception(con: Any) -> None:
    from data_analyst_mcp.digest import single_thread_scan

    con.execute("SET threads=4")
    with pytest.raises(RuntimeError):
        with single_thread_scan(con):
            raise RuntimeError("boom")
    row = con.execute("SELECT current_setting('threads')").fetchone()
    assert int(row[0]) == 4


def test_float32_bit_packed(con: Any) -> None:
    """FLOAT column encodes as 4-byte pattern — golden vector."""
    from data_analyst_mcp.digest import digest_table

    _make(con, "f32", "SELECT CAST(1.5 AS FLOAT) AS a")
    h = hashlib.sha256()
    h.update(b"\x01")
    h.update((0).to_bytes(8, "little"))
    h.update(len(b"a").to_bytes(8, "little") + b"a")
    h.update(len(b"FLOAT").to_bytes(8, "little") + b"FLOAT")
    h.update(b"\x02")
    payload = struct.pack("<f", 1.5)
    h.update(b"\x12" + len(payload).to_bytes(8, "little") + payload)
    assert digest_table(con, "f32") == h.hexdigest()
