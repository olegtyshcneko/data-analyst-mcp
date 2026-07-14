"""Tests for the ``split_dataset`` tool."""

from __future__ import annotations

from typing import Any

import pytest


def _load_ten_rows(load_df_into_session: Any) -> None:
    import pandas as pd

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))


def test_split_dataset_returns_ok_and_row_counts(call_tool: Any, load_df_into_session: Any) -> None:
    _load_ten_rows(load_df_into_session)

    result = call_tool("split_dataset", {"name": "base"})

    assert result["ok"] is True
    assert result["source"] == "base"
    assert result["train"] == {"name": "base_train", "rows": 8}
    assert result["test"] == {"name": "base_test", "rows": 2}
    assert result["seed"] == 42
    assert result["test_fraction"] == 0.25
    assert result["stratify_by"] is None
    assert result["strata"] is None
    assert result["warnings"] == []


def test_split_dataset_membership_is_the_pinned_permutation(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """RandomState(42).permutation(10) = [8,1,5,0,7,2,9,4,3,6]; with
    test_fraction=0.25, n_test = int(round(2.5)) = 2 (banker's rounding),
    so rows 8 and 1 land in test."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)

    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True

    con = _session.get_connection()
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    train_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_train"').fetchall())
    assert test_x == [1, 8]
    assert train_x == [0, 2, 3, 4, 5, 6, 7, 9]


def test_split_dataset_registers_both_as_split_format(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True

    datasets = _session.get_datasets()
    for out_name, role in (("base_train", "train"), ("base_test", "test")):
        entry = datasets[out_name]
        assert entry.format == "split"
        assert entry.path == "(split)"
        assert entry.read_options["source"] == "base"
        assert entry.read_options["seed"] == 42
        assert entry.read_options["test_fraction"] == 0.25
        assert entry.read_options["role"] == role
        assert entry.read_options["train_name"] == "base_train"
        assert entry.read_options["test_name"] == "base_test"
    # The test-side entry carries the membership checksum (count:xor:sum hex).
    import re

    checksum = datasets["base_test"].read_options["membership_checksum"]
    assert re.fullmatch(r"[0-9a-f]+:[0-9a-f]{32}:[0-9a-f]{32}", checksum)


def test_split_dataset_same_seed_is_deterministic(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    call_tool("split_dataset", {"name": "base"})
    call_tool(
        "split_dataset",
        {"name": "base", "train_name": "tr2", "test_name": "te2"},
    )

    con = _session.get_connection()
    first = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    second = sorted(r[0] for r in con.execute('SELECT x FROM "te2"').fetchall())
    assert first == second


def test_split_dataset_custom_names_and_fraction_clamps(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """n=10, test_fraction=0.05 → round(0.5)=0 → clamped to 1 test row."""
    _load_ten_rows(load_df_into_session)

    result = call_tool(
        "split_dataset",
        {"name": "base", "test_fraction": 0.05, "train_name": "tr", "test_name": "te"},
    )

    assert result["ok"] is True
    assert result["train"] == {"name": "tr", "rows": 9}
    assert result["test"] == {"name": "te", "rows": 1}


def test_split_dataset_source_column_named_split_rid_survives(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """The internal row-number column must dodge a source column of the
    same name instead of colliding with it."""
    import pandas as pd

    load_df_into_session(
        "tricky", pd.DataFrame({"__split_rid": list(range(10)), "y": list(range(10))})
    )

    result = call_tool("split_dataset", {"name": "tricky"})

    assert result["ok"] is True
    from data_analyst_mcp import session as _session

    cols = [c["name"] for c in _session.get_datasets()["tricky_train"].columns]
    assert cols == ["__split_rid", "y"]


def test_split_dataset_unknown_source(call_tool: Any) -> None:
    result = call_tool("split_dataset", {"name": "nope"})
    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_not_found"


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_dataset_rejects_fraction_endpoints(
    call_tool: Any, load_df_into_session: Any, fraction: float
) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", "test_fraction": fraction})
    assert result["ok"] is False
    assert result["error"]["type"] == "test_fraction_out_of_range"


@pytest.mark.parametrize("bad", ["1train", "has space", "has-dash", ""])
def test_split_dataset_rejects_invalid_output_names(
    call_tool: Any, load_df_into_session: Any, bad: str
) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", "train_name": bad})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_name"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_name": "same", "test_name": "same"},
        {"train_name": "base"},
        {"test_name": "base"},
    ],
)
def test_split_dataset_rejects_name_conflicts(
    call_tool: Any, load_df_into_session: Any, kwargs: dict[str, str]
) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", **kwargs})
    assert result["ok"] is False
    assert result["error"]["type"] == "split_name_conflict"


def test_split_dataset_collision_is_atomic(call_tool: Any, load_df_into_session: Any) -> None:
    """If one output name collides, NEITHER table is created/registered."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    load_df_into_session("taken", pd.DataFrame({"z": [1]}))

    result = call_tool(
        "split_dataset", {"name": "base", "train_name": "fresh", "test_name": "taken"}
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_name_collision"
    assert "fresh" not in _session.get_datasets()
    con = _session.get_connection()
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "fresh" not in tables


def test_split_dataset_overwrite_replaces_existing(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    _load_ten_rows(load_df_into_session)
    load_df_into_session("taken", pd.DataFrame({"z": [1]}))

    result = call_tool(
        "split_dataset",
        {"name": "base", "train_name": "fresh", "test_name": "taken", "overwrite": True},
    )

    assert result["ok"] is True
    assert result["test"]["name"] == "taken"


def test_split_dataset_rejects_single_row_source(call_tool: Any, load_df_into_session: Any) -> None:
    import pandas as pd

    load_df_into_session("tiny", pd.DataFrame({"x": [1]}))
    result = call_tool("split_dataset", {"name": "tiny"})
    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_too_small"


def _load_strata(load_df_into_session: Any) -> None:
    import pandas as pd

    # 4×'a', 3×'b', 3×NULL — 10 rows.
    load_df_into_session(
        "strat",
        pd.DataFrame(
            {
                "g": ["a", "a", "a", "a", "b", "b", "b", None, None, None],
                "x": list(range(10)),
            }
        ),
    )


def test_split_dataset_stratified_counts_and_strata_table(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """fraction=0.25: 'a' (4 rows) → 1 test; 'b' (3) → 1 test; NULL (3) → 1 test."""
    _load_strata(load_df_into_session)

    result = call_tool(
        "split_dataset", {"name": "strat", "stratify_by": "g", "test_fraction": 0.25}
    )

    assert result["ok"] is True
    assert result["test"]["rows"] == 3
    assert result["train"]["rows"] == 7
    strata = result["strata"]
    assert [s["value"] for s in strata] == ["a", "b", None]
    assert [(s["train_rows"], s["test_rows"]) for s in strata] == [(3, 1), (2, 1), (2, 1)]


def test_split_dataset_stratified_is_deterministic(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_strata(load_df_into_session)
    call_tool("split_dataset", {"name": "strat", "stratify_by": "g"})
    call_tool(
        "split_dataset",
        {"name": "strat", "stratify_by": "g", "train_name": "t2", "test_name": "e2"},
    )
    con = _session.get_connection()
    a = sorted(r[0] for r in con.execute('SELECT x FROM "strat_test"').fetchall())
    b = sorted(r[0] for r in con.execute('SELECT x FROM "e2"').fetchall())
    assert a == b


def test_split_dataset_small_strata_go_to_train_with_warning(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session(
        "mixed",
        pd.DataFrame({"g": ["a"] * 8 + ["solo"], "x": list(range(9))}),
    )

    result = call_tool("split_dataset", {"name": "mixed", "stratify_by": "g"})

    assert result["ok"] is True
    assert "small_strata" in result["warnings"]
    solo = [s for s in result["strata"] if s["value"] == "solo"][0]  # noqa: RUF015
    assert solo == {"value": "solo", "train_rows": 1, "test_rows": 0}


def test_split_dataset_all_singleton_strata_rejected(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session("singles", pd.DataFrame({"g": ["a", "b", "c"], "x": [1, 2, 3]}))

    result = call_tool("split_dataset", {"name": "singles", "stratify_by": "g"})

    assert result["ok"] is False
    assert result["error"]["type"] == "stratification_too_sparse"


def test_split_dataset_unknown_stratify_column(call_tool: Any, load_df_into_session: Any) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", "stratify_by": "ghost"})
    assert result["ok"] is False
    assert result["error"]["type"] == "stratify_column_missing"


def _setup_source(call_tool: Any) -> str:
    from data_analyst_mcp.recorder import get_recorder

    nb = get_recorder().to_notebook(include_setup=True)
    return nb.cells[0]["source"]


def test_split_setup_cell_recreates_both_tables_with_checksum_assert(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True

    src = _setup_source(call_tool)
    assert 'CREATE OR REPLACE TABLE "base_train"' in src
    assert 'CREATE OR REPLACE TABLE "base_test"' in src
    assert "RandomState(42)" in src
    checksum = _session.get_datasets()["base_test"].read_options["membership_checksum"]
    assert checksum in src
    assert "drifted at replay" in src
    # The first (file-backed) pass must SKIP split entries — no bogus
    # reload of the "(split)" placeholder path may appear anywhere.
    assert "'(split)'" not in src


def test_split_percall_cell_uses_same_replay_source(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp.recorder import get_recorder

    _load_ten_rows(load_df_into_session)
    call_tool("split_dataset", {"name": "base"})

    code_cells = [
        c
        for c in get_recorder().cells
        if c["cell_type"] == "code" and c["metadata"]["tool_name"] == "split_dataset"
    ]
    assert len(code_cells) == 1
    assert "RandomState(42)" in code_cells[0]["source"]
    assert 'CREATE OR REPLACE TABLE "base_test"' in code_cells[0]["source"]


def test_split_replay_snippet_executes_and_reproduces_membership(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Execute the emitted snippet against a fresh DuckDB connection loaded
    with the same rows — the checksum assert inside the snippet must pass,
    proving the snippet's algorithm matches the live one."""
    import duckdb
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.recorder import split_replay_source

    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True
    entry = _session.get_datasets()["base_test"]

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column=entry.read_options["rid_column"],
        membership_checksum=entry.read_options["membership_checksum"],
    )
    con = duckdb.connect()
    con.register("__base_src", pd.DataFrame({"x": list(range(10))}))
    con.execute('CREATE TABLE "base" AS SELECT * FROM __base_src')
    exec(snippet, {"con": con, "np": np, "pd": pd})  # replay snippet under test
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    assert test_x == [1, 8]


def test_split_stratified_setup_cell_replays(call_tool: Any, load_df_into_session: Any) -> None:
    _load_strata(load_df_into_session)
    result = call_tool("split_dataset", {"name": "strat", "stratify_by": "g"})
    assert result["ok"] is True
    src = _setup_source(call_tool)
    assert 'CREATE OR REPLACE TABLE "strat_test"' in src
    assert "isna" in src  # stratified branch emitted


def test_split_membership_stable_under_multithreaded_scan(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Explicit spec check: threads > 1 must not change file/table scan
    order, hence not membership."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    call_tool("split_dataset", {"name": "base"})
    con = _session.get_connection()
    before = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    con.execute("SET threads=4")
    try:
        call_tool("split_dataset", {"name": "base", "train_name": "mt_tr", "test_name": "mt_te"})
        after = sorted(r[0] for r in con.execute('SELECT x FROM "mt_te"').fetchall())
    finally:
        con.execute("SET threads=1")
    assert before == after


def test_model_fit_on_split_then_overwritten_raises_at_replay(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """A model fit on a split output that is later overwritten by
    materialize_query has no truthful re-fit source — the setup cell must
    emit a hard raise for it, never a silent re-fit on the post-transform
    table."""
    import pandas as pd

    # Two columns: fit_model's OLS diagnostics (Breusch-Pagan) reject
    # intercept-only exog, so the model needs a real regressor.
    load_df_into_session(
        "base",
        pd.DataFrame({"x": list(range(10)), "y": [float(i % 3 + i) for i in range(10)]}),
    )
    call_tool("split_dataset", {"name": "base"})
    r = call_tool(
        "fit_model",
        {"name": "base_train", "formula": "y ~ x", "model_name": "m_split"},
    )
    assert r["ok"] is True
    r = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_train" WHERE x > 2', "name": "base_train", "overwrite": True},
    )
    assert r["ok"] is True

    src = _setup_source(call_tool)
    assert "raise AssertionError" in src
    assert "m_split" in src


def test_materialize_overwrite_of_split_entry_keeps_base_loader_none(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """A split entry has no file loader; materialize overwrite must not
    fabricate one with path '(split)'."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    call_tool("split_dataset", {"name": "base"})
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_train" WHERE x > 2', "name": "base_train", "overwrite": True},
    )
    assert result["ok"] is True
    assert _session.get_datasets()["base_train"].base_loader is None


def test_split_setup_cell_train_overwrite_drops_train_recreation(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Overwriting the train side of a split with materialize_query must drop
    the train table from the split block's JOIN recreation — otherwise the
    setup cell clobbers the derived train table with the original split rows
    (silent replay drift). The test-side recreation and the membership
    checksum assert must survive unchanged."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    # Non-self-referential overwrite: SELECT against the SOURCE table, not the
    # split train, so the derived CREATE stands alone at replay.
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 5', "name": "base_train", "overwrite": True},
    )
    assert result["ok"] is True

    src = _setup_source(call_tool)
    # The split-JOIN recreation is distinguishable from the plain derived
    # CREATE by its ``SELECT s.* EXCLUDE`` shape: the test side is recreated,
    # the train side is not.
    assert '"base_test" AS SELECT s.* EXCLUDE' in src
    assert '"base_train" AS SELECT s.* EXCLUDE' not in src
    # The derived overwrite CREATE for base_train still stands on its own.
    assert 'CREATE OR REPLACE TABLE "base_train" AS SELECT * FROM "base" WHERE x > 5' in src
    # The test-side membership checksum assert is still emitted.
    checksum = _session.get_datasets()["base_test"].read_options["membership_checksum"]
    assert checksum in src
    assert "drifted at replay" in src


def test_split_setup_cell_train_overwrite_replays_overwrite_not_split(
    call_tool: Any, tmp_path: Any
) -> None:
    """Full drift regression (the reviewer's repro, automated): build a split
    plus a train-side overwrite on a real CSV so the emitted setup source is
    self-contained (file reload + hash assert + derived CREATE + split block),
    exec it in a fresh namespace, and assert base_train holds the OVERWRITE
    result — not the original split-train rows the buggy split block used to
    clobber it with."""
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 10', "name": "base_train", "overwrite": True},
    )
    assert result["ok"] is True

    src = _setup_source(call_tool)
    ns: dict[str, Any] = {}
    exec(src, ns)  # the self-contained setup cell under test
    con = ns["con"]
    train_x = sorted(row[0] for row in con.execute('SELECT x FROM "base_train"').fetchall())
    # The overwrite kept rows 11..19 (9 rows); the original split train had 15.
    assert train_x == list(range(11, 20))


def test_split_setup_cell_train_overwrite_self_ref_raises_at_replay(
    call_tool: Any, tmp_path: Any
) -> None:
    """A self-referential train-side overwrite drops the split train recipe and
    leaves a derived CREATE that references a table nothing recreates. Replay
    must fail with a purpose-written ``AssertionError`` that names the
    overwritten split side and says to rematerialize from a table that exists —
    never a silent re-fabrication from the stale split recipe. The original
    DuckDB catalog error stays chained as ``__cause__``."""
    import duckdb
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_train" WHERE x > 10', "name": "base_train", "overwrite": True},
    )
    assert result["ok"] is True

    src = _setup_source(call_tool)
    ns: dict[str, Any] = {}
    with pytest.raises(AssertionError, match="overwriting the train side") as excinfo:
        exec(src, ns)  # self-ref recipe with no table to build from → friendly failure
    assert isinstance(excinfo.value.__cause__, duckdb.CatalogException)


def test_split_setup_cell_test_overwrite_self_ref_raises_at_replay(
    call_tool: Any, tmp_path: Any
) -> None:
    """Symmetric to the train-side case: a self-referential test-side overwrite
    drops that side's split recipe (the split block, keyed off the test entry,
    is then not emitted at all), leaving a derived CREATE for ``base_test`` that
    reads from a table nothing recreates. Replay must fail with the same
    purpose-written ``AssertionError``, this time naming the ``test`` side, with
    the DuckDB catalog error chained as ``__cause__``."""
    import duckdb
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_test" WHERE x > 10', "name": "base_test", "overwrite": True},
    )
    assert result["ok"] is True

    src = _setup_source(call_tool)
    ns: dict[str, Any] = {}
    with pytest.raises(AssertionError, match="overwriting the test side") as excinfo:
        exec(src, ns)  # test-side self-ref recipe with no table to build from → friendly failure
    assert isinstance(excinfo.value.__cause__, duckdb.CatalogException)


def test_plain_derived_create_not_wrapped_beside_split_overwrite(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """The catalog-error wrapper is scoped to split-side overwrites only. In a
    session that has *both* a self-referential split overwrite (whose derived
    CREATE is wrapped in a ``try/except duckdb.CatalogException``) and an
    ordinary ``materialize_query`` (a brand-new derived table), the plain
    derived CREATE must stay a bare, unindented ``con.execute(...)`` with no
    wrapper — ordinary materialize_query notebooks stay byte-identical to
    today."""
    _load_ten_rows(load_df_into_session)
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    # Train-side self-referential overwrite → its derived CREATE is wrapped.
    assert (
        call_tool(
            "materialize_query",
            {
                "sql": 'SELECT * FROM "base_train" WHERE x > 2',
                "name": "base_train",
                "overwrite": True,
            },
        )["ok"]
        is True
    )
    # Ordinary materialize_query → a brand-new derived table, never wrapped.
    assert (
        call_tool(
            "materialize_query",
            {"sql": 'SELECT * FROM "base" WHERE x > 4', "name": "plain"},
        )["ok"]
        is True
    )

    src = _setup_source(call_tool)
    # The split-overwrite path IS wrapped: the base_train CREATE is indented
    # under a try, and exactly one CatalogException handler is emitted.
    assert '\n    con.execute(\'CREATE OR REPLACE TABLE "base_train"' in src
    assert src.count("except duckdb.CatalogException") == 1
    # The plain derived CREATE is bare (unindented, no wrapper attached).
    assert (
        '\ncon.execute(\'CREATE OR REPLACE TABLE "plain" AS SELECT * FROM "base" WHERE x > 4\')'
        in src
    )


def test_double_split_side_overwrite_raises_explained_at_replay(
    call_tool, tmp_path: Any
) -> None:
    """S9 (ROADMAP gap): BOTH sides self-referentially overwritten — no split
    entry survives, so sibling inference finds nothing and replay dies with a
    raw CatalogException today. Recorded provenance must wrap both CREATEs;
    the first failing wrapper (train, earlier revision) halts the cell with
    the pinned friendly message and the catalog error chained."""
    import duckdb
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_train" WHERE x > 10', "name": "base_train", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_test" WHERE x > 1', "name": "base_test", "overwrite": True},
    )["ok"] is True

    src = _setup_source(call_tool)
    # Both derived CREATEs carry the provenance wrap.
    assert src.count("except duckdb.CatalogException") == 2
    ns: dict[str, Any] = {}
    with pytest.raises(AssertionError, match="overwriting the train side") as excinfo:
        exec(src, ns)  # first unreplayable CREATE halts the cell, explained
    assert isinstance(excinfo.value.__cause__, duckdb.CatalogException)


def test_both_sides_replayable_overwrite_replays_transparently(
    call_tool, tmp_path: Any
) -> None:
    """S14 pin (success case): both sides overwritten with SQL that reads
    only from the source table. The wraps must be transparent — replay
    succeeds and both tables hold the overwrite results."""
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 10', "name": "base_train", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x <= 3', "name": "base_test", "overwrite": True},
    )["ok"] is True

    src = _setup_source(call_tool)
    ns: dict[str, Any] = {}
    exec(src, ns)  # wrapped but replayable: must run clean
    con = ns["con"]
    train_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_train"').fetchall())
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    assert train_x == list(range(11, 20))
    assert test_x == [0, 1, 2, 3]


def test_split_stores_membership_checksum_on_both_sides(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Per-side checksums (spec Part 4): each entry stores its OWN side's
    digest under the same 'membership_checksum' key. A split-source drift
    that only changes train rows is invisible to the test checksum (S16) —
    the train side needs its own."""
    import re

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.split import membership_checksum

    _load_ten_rows(load_df_into_session)
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True

    datasets = _session.get_datasets()
    con = _session.get_connection()
    for side in ("base_train", "base_test"):
        stored = datasets[side].read_options["membership_checksum"]
        assert re.fullmatch(r"[0-9a-f]+:[0-9a-f]{32}:[0-9a-f]{32}", stored)
        actual = membership_checksum(con.execute(f'SELECT * FROM "{side}"').df())
        assert stored == actual
    # Different rows on each side -> different digests.
    assert (
        datasets["base_train"].read_options["membership_checksum"]
        != datasets["base_test"].read_options["membership_checksum"]
    )


def test_split_replay_source_emits_per_side_checksum_asserts() -> None:
    from data_analyst_mcp.recorder import split_replay_source

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column="__split_rid",
        membership_checksum="aa",
        train_membership_checksum="bb",
    )
    assert "'aa'" in snippet
    assert "'bb'" in snippet
    assert snippet.count("drifted at replay") == 2
    assert "SELECT * FROM \"base_train\"" in snippet


def test_split_replay_source_train_only_block() -> None:
    """Train-only shape (surviving train side, test overwritten): recreates
    and asserts ONLY the train table."""
    import duckdb
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import split_replay_source
    from data_analyst_mcp.tools.split import _assign_is_test, membership_checksum

    df = pd.DataFrame({"x": list(range(10))})
    is_test, _ = _assign_is_test(10, 0.25, 42, None)
    train_checksum = membership_checksum(df[~is_test])

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column="__split_rid",
        membership_checksum=None,
        train_membership_checksum=train_checksum,
        include_train=True,
        include_test=False,
    )
    con = duckdb.connect()
    con.register("__base_src", df)
    con.execute('CREATE TABLE "base" AS SELECT * FROM __base_src')
    exec(snippet, {"con": con, "np": np, "pd": pd})  # train-only replay under test
    train_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_train"').fetchall())
    assert train_x == [0, 2, 3, 4, 5, 6, 7, 9]
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "base_test" not in tables


def test_split_replay_snippet_train_only_drift_fails_train_checksum(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """S16 core (review r2, reproduced): replay-time source rows differ at a
    TRAIN-side position only. Test rows are byte-identical, so the test
    checksum passes; only the train-side assert can catch the drift."""
    import duckdb
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.recorder import split_replay_source

    load_df_into_session("base", pd.DataFrame({"x": list(range(20))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    datasets = _session.get_datasets()
    test_opts = datasets["base_test"].read_options

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column=str(test_opts["rid_column"]),
        membership_checksum=str(test_opts["membership_checksum"]),
        train_membership_checksum=str(
            datasets["base_train"].read_options["membership_checksum"]
        ),
    )

    # Membership is positional: find a position the seed sends to TRAIN and
    # perturb only that value in the replayed source.
    perm = np.random.RandomState(42).permutation(20)
    test_positions = set(perm[:5].tolist())
    train_pos = min(set(range(20)) - test_positions)
    drifted = pd.DataFrame({"x": list(range(20))})
    drifted.loc[train_pos, "x"] = 999

    con = duckdb.connect()
    con.register("__base_src", drifted)
    con.execute('CREATE TABLE "base" AS SELECT * FROM __base_src')
    with pytest.raises(AssertionError, match="base_train"):
        exec(snippet, {"con": con, "np": np, "pd": pd})  # train drift must be loud
