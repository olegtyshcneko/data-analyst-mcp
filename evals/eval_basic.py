"""Basic-coverage evals: load → list → profile → query → describe round trip.

Each eval drives a freshly spawned ``data-analyst-mcp`` subprocess through
``mcp.client.stdio``. The ``mcp_session`` context manager creates one
subprocess per test, so no eval relies on state from another.
"""

from __future__ import annotations

import pytest
from conftest import CRM_DIR, FIXTURES_DIR, call, mcp_session


@pytest.mark.eval
async def eval_load_messy_csv():
    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "messy.csv"), "name": "raw"},
        )
        assert r["ok"] is True
        assert r["name"] == "raw"
        assert r["rows"] == 5000


@pytest.mark.eval
async def eval_load_accounts_parquet():
    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(CRM_DIR / "accounts.csv"), "name": "accounts"},
        )
        assert r["ok"] is True
        assert r["rows"] == 2000


@pytest.mark.eval
async def eval_list_datasets_lists_loaded():
    async with mcp_session() as s:
        await call(
            s,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "messy.csv"), "name": "raw"},
        )
        await call(
            s,
            "load_dataset",
            {"path": str(CRM_DIR / "accounts.csv"), "name": "accounts"},
        )
        r = await call(s, "list_datasets", {})
        assert r["ok"] is True
        names = {d["name"] for d in r["datasets"]}
        assert names == {"raw", "accounts"}


@pytest.mark.eval
async def eval_profile_messy():
    async with mcp_session() as s:
        await call(
            s,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "messy.csv"), "name": "raw"},
        )
        r = await call(s, "profile_dataset", {"name": "raw"})
        assert r["ok"] is True
        assert "summary" in r
        assert r["summary"]["total_rows"] == 5000
        assert len(r["columns"]) == 12


@pytest.mark.eval
async def eval_query_count():
    async with mcp_session() as s:
        await call(
            s,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "messy.csv"), "name": "raw"},
        )
        r = await call(s, "query", {"sql": "SELECT COUNT(*) AS n FROM raw"})
        assert r["ok"] is True
        assert r["rows"] == [{"n": 5000}]


@pytest.mark.eval
async def eval_describe_column_score():
    async with mcp_session() as s:
        await call(
            s,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "messy.csv"), "name": "raw"},
        )
        r = await call(s, "describe_column", {"name": "raw", "column": "score"})
        assert r["ok"] is True
        assert "quantiles" in r
        # quantile keys may be ints or strings depending on JSON serialization
        keys = set(r["quantiles"].keys())
        assert 25 in keys or "25" in keys
        assert "iqr" in r
        assert r["iqr"] >= 0
