"""End-to-end evals for load_session_from_notebook (spec v4).

The point of these scenarios is process death: ``tests/`` exercises resume
in-process, but the feature's contract is surviving a server restart. Each
scenario emits from one ``data-analyst-mcp`` subprocess, kills it, and
resumes in a fresh one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import PROJECT_ROOT, call, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"


def _nbconvert(nb_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
            "--ExecutePreprocessor.timeout=180",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("y,x\n" + "\n".join(rows) + "\n")


@pytest.mark.eval
async def eval_resume_round_trip_replayable():
    """Process A builds load → overwrite chain → split → robust fit → emit.
    Process B resumes, continues with one more op, and re-emits; the second
    notebook must replay cleanly under nbconvert."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "resume_source.csv"
    nb_path = ARTIFACTS / "eval_resume_round_trip.ipynb"
    nb_path2 = ARTIFACTS / "eval_resume_round_trip_continued.ipynb"
    for p in (nb_path, nb_path2):
        if p.exists():
            p.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(60)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "base"})
        assert r["ok"] is True
        # Overwrite chain on a copy; both recipes read the stable file-backed
        # base so the emitted setup cell stays replayable (spec slice 10).
        r = await call(
            s, "materialize_query", {"sql": "SELECT y * 2 AS y, x FROM base", "name": "d"}
        )
        assert r["ok"] is True
        r = await call(
            s,
            "materialize_query",
            {"sql": "SELECT y + 1 AS y, x FROM base", "name": "d", "overwrite": True},
        )
        assert r["ok"] is True
        r = await call(s, "split_dataset", {"name": "base", "seed": 5})
        assert r["ok"] is True
        r = await call(
            s,
            "fit_model",
            {"name": "d", "formula": "y ~ x", "kind": "ols", "robust": True, "model_name": "m"},
        )
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True
    # Process A is dead here — the subprocess exits with the context manager.

    async with mcp_session() as s:
        r = await call(s, "load_session_from_notebook", {"path": str(nb_path)})
        assert r["ok"] is True, r
        assert {d["name"] for d in r["datasets"]} == {"base", "d", "base_train", "base_test"}
        assert r["models"] == ["m"]
        # The chain's final recipe won: d.y == base.y + 1, not base.y * 2.
        r = await call(s, "query", {"sql": "SELECT y FROM d ORDER BY x LIMIT 1"})
        assert r["ok"] is True
        assert r["rows"][0]["y"] == 1.0  # base row 0 has y=0.0 → chain end is 0+1
        # Continue the session and emit a unified notebook.
        r = await call(
            s,
            "materialize_query",
            {"sql": "SELECT y, x FROM base WHERE x < 30", "name": "d2"},
        )
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path2)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path2)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.eval
async def eval_resume_only_ephemeral_model():
    """Fit on a table, then overwrite it: the notebook is deliberately NOT
    replayable (setup cell raises), but resume recreates the fit from the
    journal — the two flags are independent by design."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "resume_ephemeral.csv"
    nb_path = ARTIFACTS / "eval_resume_ephemeral.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(60)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "base"})
        assert r["ok"] is True
        r = await call(s, "materialize_query", {"sql": "SELECT y, x FROM base", "name": "d"})
        assert r["ok"] is True
        r = await call(
            s, "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "em"}
        )
        assert r["ok"] is True
        r = await call(
            s,
            "materialize_query",
            {"sql": "SELECT y, x FROM base WHERE x < 30", "name": "d", "overwrite": True},
        )
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    import nbformat

    meta = nbformat.read(str(nb_path), as_version=4).metadata["data_analyst_mcp"]
    assert meta["notebook_replayable"] is False
    assert meta["resume_supported"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode != 0, "the setup cell must raise for the ephemeral fit"

    # nbconvert --inplace re-wrote the notebook with outputs and execution
    # counts, but cell SOURCES and metadata are untouched — resume must
    # still accept it (integrity hashes cover sources, not outputs).
    async with mcp_session() as s:
        r = await call(s, "load_session_from_notebook", {"path": str(nb_path)})
        assert r["ok"] is True, r
        assert r["models"] == ["em"]
        r = await call(s, "list_models", {})
        assert r["ok"] is True
        assert [m["name"] for m in r["models"]] == ["em"]
        # The restored Results object is live: predict works against it.
        r = await call(s, "predict", {"model_name": "em", "dataset": "d", "limit": 3})
        assert r["ok"] is True, r


@pytest.mark.eval
async def eval_resume_drift_atomicity():
    """Emit, mutate the source, resume in a fresh process: source_drift with
    a completely untouched session — no datasets, no tables."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "resume_drift.csv"
    nb_path = ARTIFACTS / "eval_resume_drift.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    _write_csv(csv_path, [f"{(i * 5) % 11}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_session_from_notebook", {"path": str(nb_path)})
        assert r["ok"] is False
        assert r["error"]["type"] == "source_drift"
        r = await call(s, "list_datasets", {})
        assert r["ok"] is True
        assert r["datasets"] == []
        r = await call(
            s,
            "query",
            {"sql": "SELECT COUNT(*) AS n FROM duckdb_tables() WHERE schema_name = 'main'"},
        )
        assert r["ok"] is True
        assert r["rows"][0]["n"] == 0
