"""End-to-end evals for ``find_outliers`` on ``fixtures/messy.csv``.

The messy CSV plants a clear extreme outlier in the ``score`` column at
``score = 123456`` (row CUST-002394-9866). Every supported method must
surface that row through the live MCP stdio transport.
"""

from __future__ import annotations

import pytest
from conftest import FIXTURES_DIR, call, mcp_session

MESSY = str(FIXTURES_DIR / "messy.csv")

# The fixture plants score=123456 as a deliberate extreme outlier.
PLANTED_SCORE = 123456.0


async def _load_clean(s) -> None:
    """Load messy.csv and materialize a clean numeric-only view as ``clean``.

    The raw ``score`` column auto-detects as DOUBLE but ``revenue`` parses
    as VARCHAR; ``find_outliers`` requires numeric columns, so we
    materialize a small view containing just the numeric columns we
    intend to score on.
    """
    r = await call(s, "load_dataset", {"path": MESSY, "name": "raw"})
    assert r["ok"], r
    r = await call(
        s,
        "materialize_query",
        {
            "sql": "SELECT score, age FROM raw WHERE score IS NOT NULL",
            "name": "clean",
        },
    )
    assert r["ok"], r


@pytest.mark.eval
async def eval_iqr_surfaces_planted_outlier():
    """IQR on the ``score`` column must flag the score=123456 row."""
    async with mcp_session() as s:
        await _load_clean(s)
        r = await call(
            s,
            "find_outliers",
            {"name": "clean", "columns": ["score"], "method": "iqr"},
        )
        assert r["ok"], r
        scores = {o["values"]["score"] for o in r["outliers"]}
        assert PLANTED_SCORE in scores, (
            f"IQR did not surface the planted {PLANTED_SCORE} outlier; got scores={scores}"
        )


@pytest.mark.eval
async def eval_zscore_surfaces_planted_outlier():
    """Z-score on ``score`` must flag the score=123456 row."""
    async with mcp_session() as s:
        await _load_clean(s)
        r = await call(
            s,
            "find_outliers",
            {"name": "clean", "columns": ["score"], "method": "zscore"},
        )
        assert r["ok"], r
        scores = {o["values"]["score"] for o in r["outliers"]}
        assert PLANTED_SCORE in scores, (
            f"Z-score did not surface the planted {PLANTED_SCORE} outlier; got scores={scores}"
        )


@pytest.mark.eval
async def eval_mahalanobis_surfaces_planted_outlier():
    """Mahalanobis over (score, age) must flag the score=123456 row.

    The planted extreme dwarfs every other point on the ``score`` axis,
    so D² for that row is enormous regardless of where ``age`` lands.
    """
    async with mcp_session() as s:
        await _load_clean(s)
        r = await call(
            s,
            "find_outliers",
            {
                "name": "clean",
                "columns": ["score", "age"],
                "method": "mahalanobis",
            },
        )
        assert r["ok"], r
        scores = {o["values"]["score"] for o in r["outliers"]}
        assert PLANTED_SCORE in scores, (
            f"Mahalanobis did not surface the planted {PLANTED_SCORE} outlier; got scores={scores}"
        )


@pytest.mark.eval
async def eval_isolation_forest_surfaces_planted_outlier():
    """Isolation Forest over (score, age) must flag the score=123456 row."""
    async with mcp_session() as s:
        await _load_clean(s)
        r = await call(
            s,
            "find_outliers",
            {
                "name": "clean",
                "columns": ["score", "age"],
                "method": "isolation_forest",
            },
        )
        assert r["ok"], r
        scores = {o["values"]["score"] for o in r["outliers"]}
        assert PLANTED_SCORE in scores, (
            f"IsolationForest did not surface the planted {PLANTED_SCORE} outlier; "
            f"got scores={scores}"
        )
