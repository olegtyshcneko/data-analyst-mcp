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
