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


def test_materialize_query_columns_match_describe_output(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session(
        "src",
        pd.DataFrame({"i": [1, 2, 3], "f": [1.5, 2.5, 3.5], "s": ["a", "b", "c"]}),
    )

    result = call_tool(
        "materialize_query",
        {"sql": "SELECT i, f, s FROM src", "name": "projected"},
    )

    assert result["ok"] is True
    con = _session.get_connection()
    describe_rows = con.execute('DESCRIBE "projected"').fetchall()
    expected = [{"name": str(r[0]), "dtype": str(r[1])} for r in describe_rows]
    assert result["columns"] == expected
    # Every entry in result["columns"] has the required keys.
    for col in result["columns"]:
        assert set(col.keys()) == {"name", "dtype"}


def test_materialize_query_collision_without_overwrite(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session("src", pd.DataFrame({"x": [1, 2, 3]}))

    first = call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM src", "name": "dup"},
    )
    assert first["ok"] is True

    second = call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM src WHERE x > 1", "name": "dup"},
    )

    assert second["ok"] is False
    assert second["error"]["type"] == "dataset_name_collision"


def test_materialize_query_overwrite_replaces_existing_derived(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session("src", pd.DataFrame({"x": [1, 2, 3, 4, 5]}))

    first = call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM src", "name": "snap"},
    )
    assert first["ok"] is True
    assert first["rows"] == 5

    second = call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM src WHERE x > 3", "name": "snap", "overwrite": True},
    )
    assert second["ok"] is True
    assert second["rows"] == 2

    entry = _session.get_datasets()["snap"]
    assert entry.rows == 2
    assert entry.read_options == {"sql": "SELECT x FROM src WHERE x > 3"}


@pytest.mark.parametrize(
    "name",
    [
        "1leading_digit",
        "has-dash",
        "has space",
        "has.dot",
        "has;semicolon",
        "",
    ],
)
def test_materialize_query_invalid_name(call_tool: Any, name: str) -> None:
    result = call_tool(
        "materialize_query",
        {"sql": "SELECT 1 AS x", "name": name},
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_name"


def test_materialize_query_bad_sql_returns_query_error(call_tool: Any) -> None:
    result = call_tool(
        "materialize_query",
        {"sql": "SELECT * FROM no_such_table_xyz", "name": "out"},
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "query_error"
    # DuckDB surface message mentions the missing identifier.
    assert "no_such_table_xyz" in result["error"]["message"]


def test_materialize_query_recorder_writes_on_success_only(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("src", pd.DataFrame({"x": [1, 2, 3]}))

    # Failure path: bad SQL → no cells added.
    rec = get_recorder()
    before_fail = len(rec.cells)
    bad = call_tool(
        "materialize_query",
        {"sql": "SELECT * FROM no_such_xyz", "name": "out"},
    )
    assert bad["ok"] is False
    assert len(rec.cells) == before_fail  # nothing recorded

    # Success path: one markdown + one code cell appended.
    before_ok = len(rec.cells)
    good = call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM src", "name": "ok"},
    )
    assert good["ok"] is True
    assert len(rec.cells) == before_ok + 2
    md, code = rec.cells[-2], rec.cells[-1]
    assert md["cell_type"] == "markdown"
    assert "ok" in md["source"]
    assert code["cell_type"] == "code"
    assert "CREATE OR REPLACE TABLE" in code["source"]
    assert code["metadata"]["tool_name"] == "materialize_query"


def test_setup_cell_emits_derived_after_base_tables(
    call_tool: Any, tmp_path: Any
) -> None:
    """The derived CREATE OR REPLACE TABLE for a derived dataset must
    appear *after* the base-table CREATE line — otherwise DuckDB would
    fail at replay because the derived SQL references a base table that
    doesn't exist yet."""
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": [1, 2, 3, 4, 5]}).to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "base"})
    assert r["ok"], r

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM base WHERE x > 2", "name": "derived"},
    )
    assert r["ok"], r

    from data_analyst_mcp.recorder import _build_setup_source

    setup = _build_setup_source()
    # Both lines present.
    assert "CREATE OR REPLACE TABLE base" in setup
    assert "CREATE OR REPLACE TABLE \"derived\"" in setup or (
        "CREATE OR REPLACE TABLE derived" in setup
    )
    # Derived must come AFTER base.
    base_idx = setup.index("CREATE OR REPLACE TABLE base")
    # Match either quoted or unquoted derived name in the rehydration line.
    if 'CREATE OR REPLACE TABLE "derived"' in setup:
        derived_idx = setup.index('CREATE OR REPLACE TABLE "derived"')
    else:
        derived_idx = setup.index("CREATE OR REPLACE TABLE derived")
    assert derived_idx > base_idx, (
        f"derived emission must come after base; got base_idx={base_idx}, "
        f"derived_idx={derived_idx}.\nSetup source:\n{setup}"
    )
    # The derived line uses the recorded SQL (from read_options["sql"]).
    assert "SELECT x FROM base WHERE x > 2" in setup
