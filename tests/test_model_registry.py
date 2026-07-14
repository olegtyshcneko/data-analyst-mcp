"""Tests for the model registry, ``fit_model.model_name``, and ``list_models``.

Slices follow the proposal §TDD slices: 7 registry + 3 fit_model deltas
+ 3 list_models. The session module gains a ``ModelEntry`` dataclass and
a ``register_model`` / ``get_model`` / ``get_models`` trio mirroring the
existing dataset registry.
"""

from __future__ import annotations

import pandas as pd

# ---- Session-level registry slices ---------------------------------------


def test_register_and_lookup_model_round_trip() -> None:
    from data_analyst_mcp import session

    session.reset()
    session.register_model(
        name="m1",
        kind="ols",
        formula="y ~ x",
        fitted_on_dataset="ds",
        n_obs=42,
        training_dataset_hash="deadbeef",
        result=object(),
    )

    entry = session.get_model("m1")
    assert entry is not None
    assert entry.name == "m1"
    assert entry.kind == "ols"
    assert entry.formula == "y ~ x"
    assert entry.fitted_on_dataset == "ds"
    assert entry.n_obs == 42
    assert entry.training_dataset_hash == "deadbeef"


def test_duplicate_model_name_raises_keyerror_at_session_level() -> None:
    """The session-level register_model is the low-level guard; the tool
    layer surfaces a structured error before this path is hit, but the
    raw guard exists so any bypassing code path fails loudly."""
    import pytest

    from data_analyst_mcp import session

    session.reset()
    session.register_model(
        name="m1",
        kind="ols",
        formula="y ~ x",
        fitted_on_dataset="ds",
        n_obs=10,
        training_dataset_hash="h",
        result=object(),
    )
    with pytest.raises(KeyError):
        session.register_model(
            name="m1",
            kind="ols",
            formula="y ~ x",
            fitted_on_dataset="ds",
            n_obs=10,
            training_dataset_hash="h",
            result=object(),
        )


def test_get_models_returns_entries_in_registration_order() -> None:
    from data_analyst_mcp import session

    session.reset()
    for name in ("c", "a", "b"):
        session.register_model(
            name=name,
            kind="ols",
            formula="y ~ x",
            fitted_on_dataset="ds",
            n_obs=10,
            training_dataset_hash="h",
            result=object(),
        )
    assert list(session.get_models().keys()) == ["c", "a", "b"]


def test_session_reset_clears_model_registry() -> None:
    from data_analyst_mcp import session

    session.reset()
    session.register_model(
        name="m1",
        kind="ols",
        formula="y ~ x",
        fitted_on_dataset="ds",
        n_obs=10,
        training_dataset_hash="h",
        result=object(),
    )
    assert session.get_models() != {}

    session.reset()

    assert session.get_models() == {}


def test_get_model_returns_none_when_name_missing() -> None:
    from data_analyst_mcp import session

    session.reset()
    assert session.get_model("typo") is None


# ---- fit_model(model_name=...) slices -----------------------------------


def test_fit_model_with_no_model_name_does_not_register(call_tool, load_df_into_session):
    from data_analyst_mcp import session

    load_df_into_session(
        "tiny", pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    )
    r = call_tool("fit_model", {"name": "tiny", "formula": "y ~ x", "kind": "ols"})
    assert r["ok"] is True
    assert "model_name" not in r
    assert session.get_models() == {}


def test_fit_model_with_model_name_registers_and_echoes(call_tool, load_df_into_session):
    from data_analyst_mcp import session

    load_df_into_session(
        "tiny",
        pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
    )
    r = call_tool(
        "fit_model",
        {"name": "tiny", "formula": "y ~ x", "kind": "ols", "model_name": "m1"},
    )
    assert r["ok"] is True
    assert r["model_name"] == "m1"
    entry = session.get_model("m1")
    assert entry is not None
    assert entry.kind == "ols"
    assert entry.formula == "y ~ x"
    assert entry.fitted_on_dataset == "tiny"
    assert entry.n_obs == 5
    # The live statsmodels Results is present in-process for downstream tools.
    assert entry._result is not None


def test_fit_model_with_duplicate_model_name_returns_collision(call_tool, load_df_into_session):
    from data_analyst_mcp import session

    load_df_into_session(
        "tiny",
        pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
    )
    r1 = call_tool(
        "fit_model",
        {"name": "tiny", "formula": "y ~ x", "kind": "ols", "model_name": "m1"},
    )
    assert r1["ok"] is True
    r2 = call_tool(
        "fit_model",
        {"name": "tiny", "formula": "y ~ x", "kind": "ols", "model_name": "m1"},
    )
    assert r2["ok"] is False
    assert r2["error"]["type"] == "model_name_collision"
    # The original entry stays put — collision should not overwrite.
    assert len(session.get_models()) == 1


def test_fit_model_with_invalid_model_name_returns_validation_error(
    call_tool, load_df_into_session
):
    load_df_into_session(
        "tiny",
        pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
    )
    # Whitespace inside the name → rejected.
    r = call_tool(
        "fit_model",
        {
            "name": "tiny",
            "formula": "y ~ x",
            "kind": "ols",
            "model_name": "bad name",
        },
    )
    assert r["ok"] is False
    assert r["error"]["type"] == "model_name_invalid"


def test_fit_model_with_empty_string_model_name_returns_validation_error(
    call_tool, load_df_into_session
):
    load_df_into_session(
        "tiny",
        pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
    )
    r = call_tool(
        "fit_model",
        {"name": "tiny", "formula": "y ~ x", "kind": "ols", "model_name": ""},
    )
    assert r["ok"] is False
    assert r["error"]["type"] == "model_name_invalid"


# ---- list_models slices --------------------------------------------------


def test_list_models_on_empty_registry_returns_empty_list(call_tool):
    r = call_tool("list_models", {})
    assert r["ok"] is True
    assert r["models"] == []


def test_list_models_after_three_fits_returns_three_entries(call_tool, load_df_into_session):
    load_df_into_session(
        "ds",
        pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
    )
    for name in ("a", "b", "c"):
        r = call_tool(
            "fit_model",
            {"name": "ds", "formula": "y ~ x", "kind": "ols", "model_name": name},
        )
        assert r["ok"] is True
    r = call_tool("list_models", {})
    assert r["ok"] is True
    assert [m["name"] for m in r["models"]] == ["a", "b", "c"]
    # Metadata shape per proposal §list_models output.
    sample = r["models"][0]
    for key in ("name", "kind", "formula", "fitted_on_dataset", "n_obs", "fitted_at"):
        assert key in sample
    assert sample["kind"] == "ols"
    assert sample["formula"] == "y ~ x"
    assert sample["fitted_on_dataset"] == "ds"
    assert sample["n_obs"] == 5


def test_list_models_does_not_emit_a_recorder_cell(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session(
        "ds",
        pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
    )
    call_tool("fit_model", {"name": "ds", "formula": "y ~ x", "kind": "ols", "model_name": "m1"})
    before = len(get_recorder().cells)
    call_tool("list_models", {})
    after = len(get_recorder().cells)
    # list_models is read-only inspection (same as list_datasets) — zero cells.
    assert before == after


def test_fit_model_records_load_time_hash_not_fit_time(call_tool, tmp_path) -> None:
    """fit_model trains on the DuckDB table populated at load time, so its
    provenance hash must be the load-time file hash — not a re-hash of
    whatever the file contains at fit time."""
    from data_analyst_mcp import session

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [1, 2, 3, 4, 5, 6]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    load_time_hash = session.get_datasets()["train"].source_hash

    # Edit the file after load, before fit. The in-session table is unchanged.
    with open(csv, "a") as fh:
        fh.write("7.0,7\n")

    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    assert session.get_models()["m"].training_dataset_hash == load_time_hash


def test_fit_model_stamps_training_revision_and_loader(call_tool, tmp_path) -> None:
    """fit_model must capture the training dataset's registration revision
    and its fit-time loader identity {path, format, read_options} — the
    recorder's replacement and same-semantics-reload guards read both."""
    from data_analyst_mcp import session

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    entry = session.get_datasets()["train"]
    model = session.get_model("m")
    assert model is not None
    assert model.training_dataset_revision == entry.revision
    assert model.training_loader == {
        "path": str(csv),
        "format": "csv",
        "read_options": {},
    }


def test_register_model_defaults_for_direct_constructions() -> None:
    """Direct register_model calls (tests, hypothetical bypass paths) get
    sentinel defaults: revision -1 (matches no real registration) and no
    loader."""
    from data_analyst_mcp import session

    session.reset()
    session.register_model(
        name="m1",
        kind="ols",
        formula="y ~ x",
        fitted_on_dataset="ds",
        n_obs=10,
        training_dataset_hash="h",
        result=object(),
    )

    entry = session.get_model("m1")
    assert entry is not None
    assert entry.training_dataset_revision == -1
    assert entry.training_loader is None
