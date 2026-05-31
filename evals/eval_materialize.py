"""End-to-end evals for ``materialize_query``.

These exercise the derived-dataset reproducibility moat: a recorded
session that includes a ``materialize_query`` call must round-trip
through ``emit_notebook`` + ``jupyter nbconvert --execute`` cleanly,
because the recorder's setup cell rehydrates derived datasets via the
recorded SQL in a second pass after every file-backed dataset is loaded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import CRM_DIR, PROJECT_ROOT, call, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"


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
async def eval_materialize_join_roundtrips_via_nbconvert():
    """Load → materialize join → query the derived table → emit notebook →
    nbconvert --execute exits 0."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_path = ARTIFACTS / "materialize_join.ipynb"
    if nb_path.exists():
        nb_path.unlink()

    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(CRM_DIR / "accounts.csv"), "name": "accounts"},
        )
        assert r["ok"], r
        r = await call(
            s,
            "load_dataset",
            {"path": str(CRM_DIR / "opportunities.csv"), "name": "opportunities"},
        )
        assert r["ok"], r

        # Derived dataset built from a join across two base tables.
        r = await call(
            s,
            "materialize_query",
            {
                "sql": (
                    "SELECT a.account_id, a.country, o.stage, o.amount "
                    "FROM accounts a JOIN opportunities o "
                    "ON a.account_id = o.account_id "
                    "WHERE o.stage = 'Closed Won'"
                ),
                "name": "won_opps",
            },
        )
        assert r["ok"], r
        assert r["rows"] > 0
        assert r["total_rows"] == r["rows"]
        assert any(c["name"] == "country" for c in r["columns"])

        # Query the derived dataset to prove it's queryable like any other.
        r = await call(
            s,
            "query",
            {"sql": "SELECT COUNT(*) AS n FROM won_opps"},
        )
        assert r["ok"], r

        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"], r

    assert nb_path.exists()
    result = _nbconvert(nb_path)
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


@pytest.mark.eval
async def eval_materialize_invalid_name_returns_structured_error():
    """Pydantic-rejected names surface as ``invalid_name`` through the live
    MCP stdio path (not a raw ValidationError stacktrace)."""
    async with mcp_session() as s:
        r = await call(
            s,
            "materialize_query",
            {"sql": "SELECT 1 AS x", "name": "1leading_digit"},
        )
        assert r["ok"] is False
        assert r["error"]["type"] == "invalid_name"


@pytest.mark.eval
async def eval_materialize_collision_then_overwrite():
    """Second call with the same name + overwrite=False fails; with
    overwrite=True succeeds and replaces the entry."""
    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(CRM_DIR / "accounts.csv"), "name": "accounts"},
        )
        assert r["ok"], r

        r1 = await call(
            s,
            "materialize_query",
            {"sql": "SELECT account_id FROM accounts", "name": "snap"},
        )
        assert r1["ok"], r1

        r2 = await call(
            s,
            "materialize_query",
            {"sql": "SELECT account_id FROM accounts LIMIT 10", "name": "snap"},
        )
        assert r2["ok"] is False
        assert r2["error"]["type"] == "dataset_name_collision"

        r3 = await call(
            s,
            "materialize_query",
            {
                "sql": "SELECT account_id FROM accounts LIMIT 10",
                "name": "snap",
                "overwrite": True,
            },
        )
        assert r3["ok"], r3
        assert r3["rows"] == 10
