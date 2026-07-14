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


def test_materialize_query_rejects_multistatement_injection(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """A SELECT containing a trailing ``;`` followed by a destructive
    statement must be rejected with ``write_not_allowed``. The legacy
    ``_first_keyword`` allowlist only inspected the leading token, so the
    payload below silently dropped the baseline table ``base``.
    """
    import pandas as pd

    from data_analyst_mcp import session as _session

    # Pre-load a baseline dataset so we can verify it survives the attack.
    load_df_into_session("base", pd.DataFrame({"x": [1, 2, 3]}))
    assert "base" in _session.get_datasets()

    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT 1; DROP TABLE "base"', "name": "evil"},
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "write_not_allowed"
    # The baseline table must still be present.
    con = _session.get_connection()
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "base" in tables, "baseline table was dropped by the injection payload"


@pytest.mark.parametrize(
    "sql",
    [
        # Semicolon inside a -- line comment.
        "SELECT 1 AS x -- ; nope",
        # Semicolon inside a /* block */ comment.
        "SELECT 1 AS x /* ; nope */",
        # Semicolon inside a single-quoted string literal.
        "SELECT 'a;b' AS x",
        # Trailing single semicolon (statement terminator).
        "SELECT 1 AS x;",
    ],
)
def test_materialize_query_accepts_benign_semicolons(call_tool: Any, sql: str) -> None:
    """The multi-statement guard must not false-positive on ``;`` inside
    comments / string literals / trailing whitespace — those are part of a
    single statement and must continue to materialize successfully."""
    result = call_tool("materialize_query", {"sql": sql, "name": "ok"})
    assert result["ok"] is True, result


def test_materialize_query_rejects_with_cte_injection(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """A WITH-prefixed query that splices a second statement after ``;``
    is rejected too — the scanner must be keyword-agnostic and only cares
    whether an executable second statement follows."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session("base", pd.DataFrame({"x": [1, 2, 3]}))

    result = call_tool(
        "materialize_query",
        {
            "sql": 'WITH cte AS (SELECT 1) SELECT * FROM cte; DROP TABLE "base"',
            "name": "evil",
        },
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "write_not_allowed"
    con = _session.get_connection()
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "base" in tables


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


def test_materialize_query_empty_sql_returns_typed_error(call_tool: Any) -> None:
    """Empty sql must surface a typed error rather than leaking the generic
    `internal` envelope. The pydantic min_length violation was caught only
    for `name` and re-raised for `sql`, so it fell through to the wrapper's
    `except Exception` → `internal`. Empty sql now behaves like `query`:
    the leading-keyword guard rejects it with `write_not_allowed`."""
    result = call_tool("materialize_query", {"sql": "", "name": "out"})

    assert result["ok"] is False
    assert result["error"]["type"] != "internal"
    assert result["error"]["type"] == "write_not_allowed"


def test_materialize_query_accepts_leading_comment_before_keyword(call_tool: Any) -> None:
    """A SELECT/WITH prefixed with a ``/* block */`` or ``-- line`` comment
    must be accepted — the leading-keyword guard skips comments to find the
    real verb instead of rejecting valid SQL as ``write_not_allowed``."""
    result = call_tool(
        "materialize_query",
        {"sql": "/* derive a constant */ SELECT 1 AS x", "name": "out"},
    )

    assert result["ok"] is True, result
    assert result["name"] == "out"


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


def test_materialize_query_recorder_code_cell_handles_triple_quote_in_sql(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Recorder code cell must be valid Python even when the SQL contains
    a ``\"\"\"`` substring (e.g. in a block comment) — otherwise the
    embedded triple-quoted string literal terminates early and
    ``jupyter nbconvert --execute`` breaks on legitimate user input."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("src", pd.DataFrame({"x": [1, 2, 3]}))

    sql_with_triple_quote = 'SELECT x FROM src /* """ */ WHERE x > 0'
    result = call_tool(
        "materialize_query",
        {"sql": sql_with_triple_quote, "name": "tq_out"},
    )
    assert result["ok"] is True

    rec = get_recorder()
    code_cell = rec.cells[-1]
    assert code_cell["cell_type"] == "code"

    # The cell source must parse as valid Python — currently fails when
    # ``payload.sql`` is f-string-interpolated into a ``\"\"\"...\"\"\"`` literal.
    compile(code_cell["source"], "<recorder-cell>", "exec")


def test_setup_cell_emits_derived_after_base_tables(call_tool: Any, tmp_path: Any) -> None:
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
    assert 'CREATE OR REPLACE TABLE "derived"' in setup or (
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


def test_setup_cell_emits_chained_derived_in_dependency_order(
    call_tool: Any, tmp_path: Any
) -> None:
    """When derived B is built from derived A, the setup cell must emit
    A's CREATE OR REPLACE TABLE *before* B's — otherwise B's SQL would
    reference a table that doesn't exist yet at replay time. Registration
    order (dict insertion order) is the source of truth here."""
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"]
    # derived_a: filter base.
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM base WHERE x > 3", "name": "derived_a"},
    )["ok"]
    # derived_b: filter derived_a (chained dependency).
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM derived_a WHERE x < 9", "name": "derived_b"},
    )["ok"]

    from data_analyst_mcp.recorder import _build_setup_source

    setup = _build_setup_source()
    base_idx = setup.index("CREATE OR REPLACE TABLE base")
    a_idx = setup.index('CREATE OR REPLACE TABLE "derived_a"')
    b_idx = setup.index('CREATE OR REPLACE TABLE "derived_b"')
    assert base_idx < a_idx < b_idx, (
        f"order must be base → derived_a → derived_b; got "
        f"base={base_idx}, a={a_idx}, b={b_idx}.\nSetup:\n{setup}"
    )


def test_emitted_notebook_with_materialize_runs_via_nbconvert(
    tmp_path: Any, call_tool: Any
) -> None:
    """End-to-end: load → materialize → emit_notebook → nbconvert --execute
    exits 0. The reproducibility moat — proves the derived-dataset
    rehydration block in the setup cell is well-formed."""
    import os
    import subprocess

    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "g": list("aabbccddee")}).to_csv(
        csv, index=False
    )

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"]
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT x, g FROM base WHERE x > 4", "name": "filtered"},
    )
    assert r["ok"], r
    # Query the derived table to make sure the resulting notebook has at
    # least one cell that references it.
    r = call_tool("query", {"sql": "SELECT COUNT(*) AS n FROM filtered"})
    assert r["ok"], r

    nb_path = tmp_path / "materialize_roundtrip.ipynb"
    assert call_tool("emit_notebook", {"path": str(nb_path)})["ok"]

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


def test_session_reset_clears_derived_entries(call_tool: Any, load_df_into_session: Any) -> None:
    """Characterization: ``session.reset()`` empties the datasets dict and
    drops the DuckDB tables, regardless of whether the entries were loaded
    or derived. Confirms derived datasets don't need any special reset
    logic — they share the same cleanup path as file-backed entries."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    # One file-backed (in-memory shim) + one derived.
    load_df_into_session("base", pd.DataFrame({"x": [1, 2, 3]}))
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT x FROM base WHERE x > 1", "name": "derived"},
    )
    assert r["ok"], r
    assert "base" in _session.get_datasets()
    assert "derived" in _session.get_datasets()
    assert _session.get_datasets()["derived"].format == "derived"

    _session.reset()

    # Both entries cleared, regardless of format.
    assert _session.get_datasets() == {}
    # And the DuckDB tables themselves are gone (CatalogException on
    # SELECT * FROM derived).
    import duckdb

    con = _session.get_connection()
    with pytest.raises(duckdb.CatalogException):
        con.execute('SELECT * FROM "derived"').fetchall()


def test_overwrite_carries_source_hash_into_base_loader(call_tool, tmp_path) -> None:
    """Overwriting a file-backed dataset must retain the original file's
    load-time hash in base_loader so the emitted setup cell can guard the
    base reload; a second (derived-over-derived) overwrite carries it on."""
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "base.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    original_hash = session.get_datasets()["data"].source_hash
    assert not original_hash.startswith("sentinel:")

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 2 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    entry = session.get_datasets()["data"]
    assert entry.base_loader is not None
    assert entry.base_loader["source_hash"] == original_hash

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    entry = session.get_datasets()["data"]
    assert entry.base_loader is not None
    assert entry.base_loader["source_hash"] == original_hash


def test_overwrite_base_loader_records_original_file_revision(call_tool, tmp_path) -> None:
    """base_loader must pin the replaced FILE entry's revision (R0) and keep
    it unchanged across chained derived overwrites — the model guard uses it
    to recognize a fit on the pre-overwrite file-backed state."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    csv = tmp_path / "base.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    r0 = _session.get_datasets()["data"].revision

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    base = _session.get_datasets()["data"].base_loader
    assert base is not None
    assert base["revision"] == r0

    # Second chained overwrite: the carried dict still says R0, never R1.
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    base = _session.get_datasets()["data"].base_loader
    assert base is not None
    assert base["revision"] == r0


def test_overwrite_of_split_entry_records_split_provenance(call_tool, load_df_into_session) -> None:
    """Overwriting a split side must record {side, source} on the derived
    entry itself — the recorder's wrap must not depend on a surviving
    sibling (the double-overwrite case has none)."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True

    assert (
        call_tool(
            "materialize_query",
            {"sql": 'SELECT * FROM "base" WHERE x > 5', "name": "base_train", "overwrite": True},
        )["ok"]
        is True
    )
    assert _session.get_datasets()["base_train"].split_overwrite == {
        "side": "train",
        "source": "base",
    }

    assert (
        call_tool(
            "materialize_query",
            {"sql": 'SELECT * FROM "base" WHERE x <= 2', "name": "base_test", "overwrite": True},
        )["ok"]
        is True
    )
    assert _session.get_datasets()["base_test"].split_overwrite == {
        "side": "test",
        "source": "base",
    }


def test_chained_overwrite_carries_split_provenance_forward(
    call_tool, load_df_into_session
) -> None:
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    assert (
        call_tool(
            "materialize_query",
            {"sql": 'SELECT * FROM "base" WHERE x > 5', "name": "base_train", "overwrite": True},
        )["ok"]
        is True
    )
    assert (
        call_tool(
            "materialize_query",
            {"sql": 'SELECT * FROM "base" WHERE x > 6', "name": "base_train", "overwrite": True},
        )["ok"]
        is True
    )

    assert _session.get_datasets()["base_train"].split_overwrite == {
        "side": "train",
        "source": "base",
    }


def test_plain_overwrites_leave_split_provenance_none(call_tool, tmp_path) -> None:
    import pandas as pd

    from data_analyst_mcp import session as _session

    csv = tmp_path / "base.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "data"})["ok"] is True
    assert (
        call_tool(
            "materialize_query",
            {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
        )["ok"]
        is True
    )
    assert _session.get_datasets()["data"].split_overwrite is None
    assert (
        call_tool(
            "materialize_query",
            {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
        )["ok"]
        is True
    )
    assert _session.get_datasets()["data"].split_overwrite is None
