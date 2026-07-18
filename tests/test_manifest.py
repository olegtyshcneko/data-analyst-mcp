"""Manifest build + strict validation (spec v4 §2)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest


def _emit_nb(call_tool: Any, tmp_path: Any) -> Any:
    """One load + one materialize + one registered fit, then to_notebook."""
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "m.csv"
    pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "m"})["ok"] is True
    assert call_tool("materialize_query", {"sql": "SELECT x, y FROM m", "name": "m2"})["ok"] is True
    assert (
        call_tool(
            "fit_model", {"name": "m2", "formula": "y ~ x", "kind": "ols", "model_name": "mm"}
        )["ok"]
        is True
    )
    return get_recorder().to_notebook(include_setup=True)


def test_notebook_carries_manifest_and_cell_metadata(call_tool: Any, tmp_path: Any) -> None:
    nb = _emit_nb(call_tool, tmp_path)

    meta = nb.metadata["data_analyst_mcp"]
    assert meta["manifest_version"] == 1
    assert meta["digest_algorithm"] == "damcp-digest-v1"
    assert meta["comparison"] == {"rtol": 1e-7, "atol": 1e-12}
    assert meta["resume_supported"] is True
    assert meta["notebook_replayable"] is True
    assert [e["op"] for e in meta["journal"]] == ["load", "materialize", "fit"]
    assert set(meta["producer"]) == {"duckdb", "pandas", "numpy", "statsmodels", "python"}

    assert nb.cells[0].metadata["role"] == "setup"
    body = nb.cells[1:]
    assert len(meta["cells"]) == len(body)
    for desc, cell in zip(meta["cells"], body, strict=True):
        assert desc["cell_type"] == cell.cell_type
        assert desc["source_sha256"] == hashlib.sha256(cell.source.encode("utf-8")).hexdigest()
    assert (
        meta["setup_cell_sha256"] == hashlib.sha256(nb.cells[0].source.encode("utf-8")).hexdigest()
    )
    # Manifest is JSON-serializable (nbformat write requirement).
    json.dumps(meta)


def test_manifest_final_registry_matches_session(call_tool: Any, tmp_path: Any) -> None:
    from data_analyst_mcp import session

    nb = _emit_nb(call_tool, tmp_path)
    meta = nb.metadata["data_analyst_mcp"]
    fr = meta["final_registry"]
    assert {d["name"] for d in fr["datasets"]} == {"m", "m2"}
    m2 = next(d for d in fr["datasets"] if d["name"] == "m2")
    assert m2["rows"] == 30
    assert m2["revision"] == session.get_datasets()["m2"].revision
    assert [m["name"] for m in fr["models"]] == ["mm"]
    assert fr["models"][0]["fit_options"] == {"robust": False}
    assert fr["next_revision"] == max(d["revision"] for d in fr["datasets"]) + 1
    assert set(meta["state_digests"]) == {"m", "m2"}


def test_dataframe_dataset_marks_resume_unsupported(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("mem", pd.DataFrame({"a": [1]}))
    meta = get_recorder().to_notebook(include_setup=True).metadata["data_analyst_mcp"]
    assert meta["resume_supported"] is False
    assert any("mem" in r for r in meta["resume_unsupported_reasons"])


def test_ephemeral_model_marks_notebook_unreplayable(call_tool: Any, tmp_path: Any) -> None:
    """Fit on a table then overwrite it: setup cell raises → replayable false,
    but resume stays supported (journal replay recreates the fit)."""
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "e.csv"
    pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "e"})["ok"] is True
    assert (
        call_tool(
            "materialize_query", {"sql": "SELECT x, y FROM e", "name": "d", "overwrite": False}
        )["ok"]
        is True
    )
    assert (
        call_tool(
            "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "em"}
        )["ok"]
        is True
    )
    assert (
        call_tool(
            "materialize_query", {"sql": "SELECT x, y FROM e", "name": "d", "overwrite": True}
        )["ok"]
        is True
    )

    meta = get_recorder().to_notebook(include_setup=True).metadata["data_analyst_mcp"]
    assert meta["notebook_replayable"] is False
    assert meta["resume_supported"] is True


def test_validate_manifest_round_trips_and_forbids_extras(call_tool: Any, tmp_path: Any) -> None:
    from data_analyst_mcp.manifest import ManifestInvalid, validate_manifest

    meta = _emit_nb(call_tool, tmp_path).metadata["data_analyst_mcp"]
    validate_manifest(json.loads(json.dumps(meta)))  # round-trip clean

    bad = json.loads(json.dumps(meta))
    bad["surprise"] = 1
    with pytest.raises(ManifestInvalid):
        validate_manifest(bad)

    dup = json.loads(json.dumps(meta))
    dup["journal"][1]["op_id"] = dup["journal"][0]["op_id"]
    with pytest.raises(ManifestInvalid):
        validate_manifest(dup)
