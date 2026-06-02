"""End-to-end evals for ``find_outliers``.

Two fixtures, two regimes:

- ``fixtures/messy.csv`` plants a clear extreme outlier in the ``score``
  column at ``score = 123456`` (row CUST-002394-9866). Every supported
  method must surface that row through the live MCP stdio transport.
- ``fixtures/breast_cancer.csv`` (569 × 30 numeric) exercises the genuinely
  high-dimensional paths the synthetic fixtures never reach: Mahalanobis
  over k=10 columns, Isolation Forest over k=10, and the singular-covariance
  pseudoinverse fallback.
"""

from __future__ import annotations

import pytest
from conftest import FIXTURES_DIR, call, mcp_session
from scipy.stats import chi2

MESSY = str(FIXTURES_DIR / "messy.csv")

# The fixture plants score=123456 as a deliberate extreme outlier.
PLANTED_SCORE = 123456.0

BREAST_CANCER = str(FIXTURES_DIR / "breast_cancer.csv")

# The 10 "mean_" features — a genuinely high-dimensional manifold (k=10),
# unlike the <=3-column synthetic fixtures.
MEAN_FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
]
N_CANCER_ROWS = 569

# The largest tumor in the set (max mean_area); a stable extreme that every
# distance/density method must surface.
MAX_MEAN_AREA = 2501.0


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


async def _load_cancer(s) -> None:
    """Load the wide WDBC reference fixture as ``cancer``."""
    r = await call(s, "load_dataset", {"path": BREAST_CANCER, "name": "cancer"})
    assert r["ok"], r


@pytest.mark.eval
async def eval_mahalanobis_highdim_breast_cancer():
    """Mahalanobis over 10 features exercises the k>>2 covariance path.

    The χ² cutoff must track the dimensionality (df=10, not a 2-D default),
    every one of the 569 rows must be scored, and the single largest tumor
    in the set must come back flagged.
    """
    async with mcp_session() as s:
        await _load_cancer(s)
        r = await call(
            s,
            "find_outliers",
            {"name": "cancer", "columns": MEAN_FEATURES, "method": "mahalanobis"},
        )
        assert r["ok"], r
        assert r["n_rows_scored"] == N_CANCER_ROWS, r
        expected_threshold = float(chi2.ppf(0.975, df=len(MEAN_FEATURES)))
        assert r["threshold_used"] == pytest.approx(expected_threshold, rel=1e-6), r
        assert r["n_outliers"] > 0, r
        areas = {o["values"]["mean_area"] for o in r["outliers"]}
        assert MAX_MEAN_AREA in areas, (
            f"largest tumor (mean_area={MAX_MEAN_AREA}) not flagged; top areas={sorted(areas)[-3:]}"
        )


@pytest.mark.eval
async def eval_isolation_forest_highdim_breast_cancer():
    """Isolation Forest over 10 features; flagged count tracks contamination."""
    async with mcp_session() as s:
        await _load_cancer(s)
        r = await call(
            s,
            "find_outliers",
            {
                "name": "cancer",
                "columns": MEAN_FEATURES,
                "method": "isolation_forest",
                "contamination": 0.05,
            },
        )
        assert r["ok"], r
        assert r["n_rows_scored"] == N_CANCER_ROWS, r
        # contamination=0.05 over 569 rows ≈ 28 flagged (random_state is fixed).
        assert abs(r["n_outliers"] - round(0.05 * N_CANCER_ROWS)) <= 2, r


@pytest.mark.eval
async def eval_mahalanobis_singular_covariance_falls_back():
    """An exactly collinear column makes Σ singular → pseudoinverse fallback.

    The server must not crash: it emits a ``covariance_singular`` warning and
    still scores every row via the Moore-Penrose pseudoinverse.
    """
    async with mcp_session() as s:
        await _load_cancer(s)
        r = await call(
            s,
            "materialize_query",
            {
                "sql": "SELECT mean_radius, mean_texture, mean_radius * 2.0 AS radius_dup FROM cancer",
                "name": "cancer_collinear",
            },
        )
        assert r["ok"], r
        r = await call(
            s,
            "find_outliers",
            {
                "name": "cancer_collinear",
                "columns": ["mean_radius", "mean_texture", "radius_dup"],
                "method": "mahalanobis",
            },
        )
        assert r["ok"], r
        assert "covariance_singular" in r["warnings"], r
        assert r["n_rows_scored"] == N_CANCER_ROWS, r
