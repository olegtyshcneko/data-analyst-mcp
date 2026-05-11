"""Statistical evals against the synthetic CRM fixtures.

These confirm the test-selection logic, assumption-check population, and
end-to-end model fits work through the live stdio server. We use
``CORR`` over a join via the ``query`` tool to validate cross-table
correlations because the read-only ``query`` allowlist deliberately rejects
write statements — so there is no protocol-supported way to register a
join result as a new dataset.
"""

from __future__ import annotations

import pytest
from conftest import CRM_DIR, call, load_crm, mcp_session


@pytest.mark.eval
async def eval_compare_groups_anova_for_three_stage():
    """Opportunities have 6 stages, so compare_groups picks ANOVA or Kruskal."""
    async with mcp_session() as s:
        await load_crm(s)
        r = await call(
            s,
            "compare_groups",
            {
                "name": "opportunities",
                "group_column": "stage",
                "metric_column": "amount",
            },
        )
        assert r["ok"], r
        assert r["test"] in {"anova", "kruskal_wallis"}, r["test"]


@pytest.mark.eval
async def eval_correlation_arr_amount_positive():
    """ARR and opportunity amount are positively correlated (built into fixture)."""
    async with mcp_session() as s:
        await load_crm(s)
        # `correlate` requires a registered dataset; the read-only query path
        # can't materialize a join. Use DuckDB's CORR aggregate to extract the
        # cross-table Pearson r over the live protocol.
        r = await call(
            s,
            "query",
            {
                "sql": (
                    "SELECT CORR(a.arr, o.amount) AS r "
                    "FROM opportunities o JOIN accounts a USING(account_id)"
                )
            },
        )
        assert r["ok"], r
        pearson = r["rows"][0]["r"]
        assert pearson > 0.25, pearson
        # Sanity: also exercise the `correlate` tool on a single-table case
        # (accounts has both `employees` and `arr` — two numeric columns).
        r2 = await call(
            s,
            "correlate",
            {
                "name": "accounts",
                "columns": ["employees", "arr"],
                "method": "pearson",
                "plot": False,
            },
        )
        assert r2["ok"], r2
        # 2x2 self-correlation matrix; the off-diagonal is the pair we want.
        off = r2["matrix"][0][1]
        assert isinstance(off, float)


@pytest.mark.eval
async def eval_fit_model_logistic_won():
    """Logistic fit of `won ~ amount` over the live server."""
    async with mcp_session() as s:
        await load_crm(s)
        r = await call(
            s,
            "fit_model",
            {
                "name": "opportunities",
                "formula": "won ~ amount",
                "kind": "logistic",
            },
        )
        assert r["ok"], r
        assert "aic" in r["fit"]
        by_name = {c["name"]: c for c in r["coefficients"]}
        assert "amount" in by_name
        assert by_name["amount"]["estimate"] is not None


@pytest.mark.eval
async def eval_chi_square_industry_won():
    """Chi-square independence of `industry` × `won` joined across tables."""
    async with mcp_session() as s:
        await load_crm(s)
        rows = await call(
            s,
            "query",
            {
                "sql": (
                    "SELECT a.industry, o.won, COUNT(*) c "
                    "FROM opportunities o JOIN accounts a USING(account_id) "
                    "WHERE o.won IS NOT NULL "
                    "GROUP BY a.industry, o.won "
                    "ORDER BY a.industry, o.won"
                ),
                "limit": 100,
            },
        )
        assert rows["ok"]
        # Pivot the long-format query result into an industry × won contingency.
        industries: list[str] = []
        cells: dict[tuple[str, bool], int] = {}
        for row in rows["rows"]:
            ind, won, c = row["industry"], row["won"], int(row["c"])
            if ind not in industries:
                industries.append(ind)
            cells[(ind, bool(won))] = c
        table = [
            [cells.get((ind, False), 0), cells.get((ind, True), 0)]
            for ind in industries
        ]

        r = await call(
            s,
            "test_hypothesis",
            {"kind": "chi_square", "table": table},
        )
        assert r["ok"], r
        assert r["statistic"] is not None
        assert r["p_value"] is not None
        assert r["test"] == "chi_square"


@pytest.mark.eval
async def eval_compare_groups_switches_for_non_normal():
    """Heavy-tail amounts on Closed Won vs Closed Lost ⇒ Mann-Whitney."""
    async with mcp_session() as s:
        await load_crm(s)
        r = await call(
            s,
            "compare_groups",
            {
                "name": "opportunities",
                "group_column": "stage",
                "metric_column": "amount",
                "groups": ["Closed Won", "Closed Lost"],
            },
        )
        assert r["ok"], r
        # Amount is heavily right-skewed (lognormal-ish), so Shapiro rejects
        # and the decision tree must drop t-test for the rank-based test.
        norm = r["assumption_checks"]["normality_test"]
        assert norm["violated"] is True, norm
        assert r["test"] == "mann_whitney", r["test"]


@pytest.mark.eval
async def eval_assumption_checks_always_populated():
    """Every compare_groups response must surface its assumption_checks block."""
    async with mcp_session() as s:
        # Load just one CSV — this eval doesn't need all three tables.
        r = await call(
            s,
            "load_dataset",
            {"path": str(CRM_DIR / "opportunities.csv"), "name": "opp"},
        )
        assert r["ok"], r
        r = await call(
            s,
            "compare_groups",
            {
                "name": "opp",
                "group_column": "stage",
                "metric_column": "amount",
            },
        )
        assert r["ok"], r
        assert "assumption_checks" in r
        assert isinstance(r["assumption_checks"], dict)
        assert len(r["assumption_checks"]) >= 1
        # Each entry has a `name`, `violated`, and `consequence`.
        for entry in r["assumption_checks"].values():
            assert "violated" in entry
            assert "consequence" in entry
