"""End-to-end workflow evals.

These exercise the differentiator: a recorded session emitted as a
notebook and re-executed via ``jupyter nbconvert``. The notebook lives in
``evals/_artifacts/`` (gitignored). If ``nbconvert --execute`` exits
non-zero, the reproducibility promise breaks and the suite fails.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, PROJECT_ROOT, call, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"
MESSY = str(FIXTURES_DIR / "messy.csv")


def _nbconvert(nb_path: Path) -> subprocess.CompletedProcess[str]:
    """Re-execute the notebook in-place via the project's jupyter."""
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


@pytest.mark.eval
async def eval_full_workflow_emits_and_reexecutes():
    """Six-step workflow → emit → nbconvert --execute exits 0."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_path = ARTIFACTS / "full_workflow.ipynb"
    if nb_path.exists():
        nb_path.unlink()

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": MESSY, "name": "raw"})
        assert r["ok"], r
        r = await call(s, "profile_dataset", {"name": "raw"})
        assert r["ok"], r
        r = await call(s, "query", {"sql": "SELECT COUNT(*) AS n FROM raw"})
        assert r["ok"], r
        r = await call(
            s,
            "correlate",
            {"name": "raw", "method": "spearman", "plot": False},
        )
        assert r["ok"], r
        r = await call(s, "plot", {"name": "raw", "kind": "hist", "x": "score"})
        assert r["ok"], r
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"], r

    assert nb_path.exists()
    result = _nbconvert(nb_path)
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


@pytest.mark.eval
async def eval_notebook_contains_expected_state():
    """Emitted setup cell wires `con = duckdb.connect()` and reloads each dataset."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_path = ARTIFACTS / "expected_state.ipynb"
    if nb_path.exists():
        nb_path.unlink()

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": MESSY, "name": "raw"})
        assert r["ok"], r
        r = await call(s, "query", {"sql": "SELECT COUNT(*) AS n FROM raw"})
        assert r["ok"], r
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"], r

    import nbformat as nbf

    nb = nbf.read(str(nb_path), as_version=4)
    setup_src = nb.cells[0].source
    assert "con = duckdb.connect()" in setup_src
    assert "CREATE OR REPLACE TABLE raw" in setup_src


@pytest.mark.eval
async def eval_session_reset_clears_state():
    """Each fresh `mcp_session` starts with an empty dataset registry."""
    # First session: load two datasets.
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": MESSY, "name": "raw"})
        await call(
            s,
            "load_dataset",
            {
                "path": str(FIXTURES_DIR / "synthetic_crm" / "accounts.csv"),
                "name": "accounts",
            },
        )
        r = await call(s, "list_datasets", {})
        assert r["ok"]
        assert {d["name"] for d in r["datasets"]} == {"raw", "accounts"}

    # Second session: subprocess respawn ⇒ no carry-over.
    async with mcp_session() as s2:
        r = await call(s2, "list_datasets", {})
        assert r["ok"]
        assert r["datasets"] == []


@pytest.mark.eval
async def eval_determinism_same_workflow_same_numerics():
    """Running the same compare_groups twice returns bit-identical statistics."""

    async def _run() -> tuple[float, float]:
        async with mcp_session() as s:
            r = await call(
                s,
                "load_dataset",
                {
                    "path": str(FIXTURES_DIR / "synthetic_crm" / "opportunities.csv"),
                    "name": "opp",
                },
            )
            assert r["ok"], r
            r = await call(
                s,
                "compare_groups",
                {
                    "name": "opp",
                    "group_column": "stage",
                    "metric_column": "amount",
                    "groups": ["Closed Won", "Closed Lost"],
                },
            )
            assert r["ok"], r
            return float(r["statistic"]), float(r["p_value"])

    stat_a, p_a = await _run()
    stat_b, p_b = await _run()
    assert abs(stat_a - stat_b) <= 1e-9, (stat_a, stat_b)
    assert abs(p_a - p_b) <= 1e-9, (p_a, p_b)
