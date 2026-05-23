"""Evals against the public Titanic fixture.

These exercise the same tool surface as `eval_stats.py` (compare_groups,
fit_model, test_hypothesis, query) but on a real-world dataset whose
results are textbook-known. The assertions check substantive outcomes —
female survival rate is much higher than male, 1st class survival rate
is much higher than 3rd, Sex × Survived is wildly significant, and the
logit coefficient on `Sex[male]` is large and negative — rather than
just response shape.

The fixture has two columns whose names contain a space and a slash
(`Siblings/Spouses Aboard`, `Parents/Children Aboard`), which forces the
`query` tool's identifier-quoting path to actually be exercised.
"""

from __future__ import annotations

import pytest
from conftest import FIXTURES_DIR, call, mcp_session

TITANIC = FIXTURES_DIR / "titanic.csv"


@pytest.mark.eval
async def eval_load_titanic():
    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(TITANIC), "name": "titanic"},
        )
        assert r["ok"] is True
        assert r["rows"] == 887


@pytest.mark.eval
async def eval_profile_titanic_has_no_nulls():
    """The Stanford variant is pre-cleaned; profiler should report zero nulls."""
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": str(TITANIC), "name": "titanic"})
        r = await call(s, "profile_dataset", {"name": "titanic"})
        assert r["ok"] is True
        assert r["summary"]["total_rows"] == 887
        assert len(r["columns"]) == 8
        # Every column should report zero missing.
        for col in r["columns"]:
            assert col.get("null_count", 0) == 0, col


@pytest.mark.eval
async def eval_survival_rate_by_sex():
    """Female survival rate (~0.74) is much higher than male (~0.19)."""
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": str(TITANIC), "name": "titanic"})
        r = await call(
            s,
            "query",
            {
                "sql": (
                    "SELECT Sex, AVG(Survived) AS rate, COUNT(*) AS n "
                    "FROM titanic GROUP BY Sex ORDER BY Sex"
                )
            },
        )
        assert r["ok"], r
        by_sex = {row["Sex"]: row for row in r["rows"]}
        assert by_sex["female"]["n"] == 314
        assert by_sex["male"]["n"] == 573
        assert by_sex["female"]["rate"] > 0.70
        assert by_sex["male"]["rate"] < 0.25
        assert by_sex["female"]["rate"] - by_sex["male"]["rate"] > 0.5


@pytest.mark.eval
async def eval_survival_rate_monotone_in_pclass():
    """1st-class survival > 2nd > 3rd — a textbook monotone gradient."""
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": str(TITANIC), "name": "titanic"})
        r = await call(
            s,
            "query",
            {
                "sql": (
                    "SELECT Pclass, AVG(Survived) AS rate FROM titanic "
                    "GROUP BY Pclass ORDER BY Pclass"
                )
            },
        )
        assert r["ok"], r
        rates = [row["rate"] for row in r["rows"]]
        assert len(rates) == 3
        assert rates[0] > rates[1] > rates[2]
        assert rates[0] > 0.60  # 1st class ~ 0.63
        assert rates[2] < 0.30  # 3rd class ~ 0.24


@pytest.mark.eval
async def eval_chi_square_sex_survived():
    """Sex × Survived contingency is overwhelmingly non-independent."""
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": str(TITANIC), "name": "titanic"})
        rows = await call(
            s,
            "query",
            {
                "sql": (
                    "SELECT Sex, Survived, COUNT(*) AS c FROM titanic "
                    "GROUP BY Sex, Survived ORDER BY Sex, Survived"
                )
            },
        )
        assert rows["ok"], rows
        cells: dict[tuple[str, int], int] = {
            (row["Sex"], int(row["Survived"])): int(row["c"]) for row in rows["rows"]
        }
        # 2x2 table: rows = [female, male], cols = [died, survived]
        table = [
            [cells[("female", 0)], cells[("female", 1)]],
            [cells[("male", 0)], cells[("male", 1)]],
        ]
        r = await call(s, "test_hypothesis", {"kind": "chi_square", "table": table})
        assert r["ok"], r
        assert r["test"] == "chi_square"
        assert r["p_value"] < 1e-50  # χ² ≈ 260 on this table — vanishing p


@pytest.mark.eval
async def eval_compare_fare_across_pclass():
    """Fare differs across Pclass; skew should push selection off plain ANOVA."""
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": str(TITANIC), "name": "titanic"})
        r = await call(
            s,
            "compare_groups",
            {
                "name": "titanic",
                "group_column": "Pclass",
                "metric_column": "Fare",
            },
        )
        assert r["ok"], r
        # Fare is heavily right-skewed (skew ≈ 4.78), so Shapiro should
        # reject normality and the selector should switch to Kruskal-Wallis.
        norm = r["assumption_checks"]["normality_test"]
        assert norm["violated"] is True, norm
        assert r["test"] == "kruskal_wallis", r["test"]
        assert r["p_value"] < 1e-50


@pytest.mark.eval
async def eval_fit_logistic_survived_on_sex_age_pclass():
    """Logit `Survived ~ Sex + Age + C(Pclass)` produces the expected signs."""
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": str(TITANIC), "name": "titanic"})
        r = await call(
            s,
            "fit_model",
            {
                "name": "titanic",
                "formula": "Survived ~ Sex + Age + C(Pclass)",
                "kind": "logistic",
            },
        )
        assert r["ok"], r
        assert "aic" in r["fit"]
        by_name = {c["name"]: c for c in r["coefficients"]}
        # statsmodels names the dummy as `Sex[T.male]` (alphabetical baseline = female).
        sex_male = by_name.get("Sex[T.male]")
        assert sex_male is not None, list(by_name)
        # Being male is a huge negative on survival; coefficient ≈ -2.59.
        assert sex_male["estimate"] < -1.5
        # 3rd-class coefficient should be more negative than 2nd-class.
        pclass2 = by_name["C(Pclass)[T.2]"]["estimate"]
        pclass3 = by_name["C(Pclass)[T.3]"]["estimate"]
        assert pclass3 < pclass2 < 0
        # Age effect is small and negative (older slightly less likely).
        assert by_name["Age"]["estimate"] < 0


@pytest.mark.eval
async def eval_query_handles_slash_in_column_name():
    """`Siblings/Spouses Aboard` requires double-quoting in DuckDB SQL."""
    async with mcp_session() as s:
        await call(s, "load_dataset", {"path": str(TITANIC), "name": "titanic"})
        r = await call(
            s,
            "query",
            {
                "sql": (
                    'SELECT AVG("Siblings/Spouses Aboard") AS avg_sibs, '
                    'AVG("Parents/Children Aboard") AS avg_par FROM titanic'
                )
            },
        )
        assert r["ok"], r
        assert len(r["rows"]) == 1
        assert r["rows"][0]["avg_sibs"] > 0
        assert r["rows"][0]["avg_par"] >= 0
