"""Security regression tests for issue #4 — arbitrary host-file read.

The ``query`` and ``materialize_query`` tools execute agent-supplied SQL on
the session's DuckDB connection. Before the fix, that connection had full
filesystem access, so a payload like ``SELECT * FROM read_csv('/etc/passwd')``
returned the file's contents. The reporter's suggested function denylist is
insufficient: DuckDB also reads files via *replacement scans*
(``SELECT * FROM '/etc/passwd.csv'`` — no function name) and ``glob('/etc/*')``.

The fix disables filesystem access on the query connection
(``enable_external_access=false``), which blocks every vector at once.
Legitimate file loading is delegated to a separate short-lived connection
(see ``session.read_file_as_df``), so ``load_dataset`` is unaffected.
"""

from __future__ import annotations

from typing import Any

import pytest

# Each payload is a distinct DuckDB filesystem-read vector. A function-name
# denylist would miss the replacement scan and glob cases.
_FILE_READ_VECTORS = [
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_csv_auto('/etc/passwd')",
    "SELECT * FROM read_parquet('/etc/passwd')",
    "SELECT * FROM '/etc/passwd.csv'",  # replacement scan — no function name
    "SELECT * FROM glob('/etc/*')",
]


@pytest.mark.parametrize("sql", _FILE_READ_VECTORS)
def test_query_cannot_read_host_files(call_tool: Any, sql: str) -> None:
    result = call_tool("query", {"sql": sql})

    assert result["ok"] is False, f"host file read was NOT blocked: {sql!r}"
    assert result["error"]["type"] == "query_error"


@pytest.mark.parametrize("sql", _FILE_READ_VECTORS)
def test_materialize_query_cannot_read_host_files(call_tool: Any, sql: str) -> None:
    result = call_tool("materialize_query", {"sql": sql, "name": "leaked"})

    assert result["ok"] is False, f"host file read was NOT blocked: {sql!r}"
    assert result["error"]["type"] == "query_error"


def test_query_still_runs_normal_analytics(call_tool: Any) -> None:
    """The sandbox must not break in-memory queries against loaded data."""
    result = call_tool("query", {"sql": "SELECT 21 + 21 AS answer"})

    assert result["ok"] is True, result
    assert result["rows"] == [{"answer": 42}]
