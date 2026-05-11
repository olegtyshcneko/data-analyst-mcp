"""Evals for the planted issues in ``fixtures/messy.csv``.

One eval per failure mode the fixture script (`fixtures/_build_messy.py`)
deliberately introduces. The point is to confirm that the live server,
driven over stdio, surfaces each issue through ``profile_dataset``,
``describe_column``, and ``query`` rather than silently coercing it away.
"""

from __future__ import annotations

import pytest
from conftest import FIXTURES_DIR, call, mcp_session

MESSY = str(FIXTURES_DIR / "messy.csv")


async def _load_raw(s):
    r = await call(s, "load_dataset", {"path": MESSY, "name": "raw"})
    assert r["ok"], r
    return r


@pytest.mark.eval
async def eval_email_flagged_mostly_null():
    """78% null > 50% threshold ⇒ profile flags `email` as mostly_null."""
    async with mcp_session() as s:
        await _load_raw(s)
        prof = await call(s, "profile_dataset", {"name": "raw"})
        assert prof["ok"]
        email = next(c for c in prof["columns"] if c["name"] == "email")
        assert email["flags"]["mostly_null"] is True


@pytest.mark.eval
async def eval_country_high_distinct_count():
    """Dirty country variants inflate distinct count beyond the 4 canonical codes."""
    async with mcp_session() as s:
        await _load_raw(s)
        r = await call(s, "describe_column", {"name": "raw", "column": "country"})
        assert r["ok"]
        # categorical describe returns value_counts; count the entries
        assert len(r["value_counts"]) >= 6, r["value_counts"]


@pytest.mark.eval
async def eval_score_outliers_detected():
    """The 20 planted IQR outliers in `score` must be surfaced as outliers."""
    async with mcp_session() as s:
        await _load_raw(s)
        r = await call(s, "describe_column", {"name": "raw", "column": "score"})
        assert r["ok"]
        assert r["outliers"]["iqr_count"] >= 15, r["outliers"]


@pytest.mark.eval
async def eval_duplicates_detectable_via_query():
    """The 2 planted exact-duplicate rows show up as a row-vs-distinct delta."""
    async with mcp_session() as s:
        await _load_raw(s)
        r = await call(
            s,
            "query",
            {
                "sql": (
                    "SELECT (SELECT COUNT(*) FROM raw) "
                    "- (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw)) "
                    "AS dup_excess"
                )
            },
        )
        assert r["ok"], r
        assert r["rows"][0]["dup_excess"] >= 1


@pytest.mark.eval
async def eval_signup_date_is_varchar():
    """Mixed date formats ⇒ DuckDB must leave `signup_date` as VARCHAR."""
    async with mcp_session() as s:
        await _load_raw(s)
        prof = await call(s, "profile_dataset", {"name": "raw"})
        col = next(c for c in prof["columns"] if c["name"] == "signup_date")
        assert col["dtype"] == "VARCHAR", col


@pytest.mark.eval
async def eval_revenue_is_varchar():
    """`revenue` contains 'N/A' / '' strings ⇒ must stay VARCHAR, not coerced."""
    async with mcp_session() as s:
        await _load_raw(s)
        prof = await call(s, "profile_dataset", {"name": "raw"})
        col = next(c for c in prof["columns"] if c["name"] == "revenue")
        assert col["dtype"] == "VARCHAR", col


@pytest.mark.eval
async def eval_header_whitespace_columns_accessible():
    """All 12 columns (incl. those with whitespace headers) must be queryable."""
    async with mcp_session() as s:
        await _load_raw(s)
        prof = await call(s, "profile_dataset", {"name": "raw"})
        assert len(prof["columns"]) == 12
        # DuckDB strips trailing whitespace from auto-detected headers, so
        # `revenue` and `last_login` should be reachable by their stripped names.
        names = {c["name"] for c in prof["columns"]}
        for expected in ("revenue", "last_login"):
            assert expected in names, names
        # Final sanity check: a query against the stripped name returns a row.
        r = await call(s, "query", {"sql": "SELECT revenue, last_login FROM raw LIMIT 1"})
        assert r["ok"], r
        assert len(r["rows"]) == 1


@pytest.mark.eval
async def eval_bom_does_not_corrupt_first_header():
    """The leading UTF-8 BOM must be stripped before the first header name."""
    async with mcp_session() as s:
        await _load_raw(s)
        prof = await call(s, "profile_dataset", {"name": "raw"})
        first_name = prof["columns"][0]["name"]
        # The BOM byte sequence is U+FEFF (﻿) — must not survive parse.
        assert "﻿" not in first_name, repr(first_name)
        assert first_name == "customer_id"
