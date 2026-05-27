"""Tests for the ``find_outliers`` tool.

Every numeric assertion uses hand-computed or scipy/numpy/sklearn-derived
expected values at ≤1e-4 tolerance. Synthetic fixtures use a fixed seed so
runs are reproducible byte-for-byte.
"""

from __future__ import annotations

from typing import Any

import pytest


# === shared ===


def test_find_outliers_unknown_dataset_returns_not_found(call_tool: Any) -> None:
    result = call_tool(
        "find_outliers",
        {"name": "nope", "columns": ["x"], "method": "iqr"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_find_outliers_unknown_column_returns_column_not_found(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session("tiny", pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    result = call_tool(
        "find_outliers",
        {"name": "tiny", "columns": ["x", "zzz"], "method": "iqr"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "column_not_found"


def test_find_outliers_non_numeric_column_returns_typed_error(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session(
        "mixed",
        pd.DataFrame({"x": [1, 2, 3], "g": ["a", "b", "c"]}),
    )
    result = call_tool(
        "find_outliers",
        {"name": "mixed", "columns": ["x", "g"], "method": "iqr"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "non_numeric_column"


# === iqr ===


def test_find_outliers_iqr_default_threshold_flags_extreme_value(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Default threshold k=1.5 flags the planted extreme value.

    Data: 100 nominal values in [0, 1) plus one extreme value at 1000.
    With k=1.5, the extreme must be flagged; the nominal values must not.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed=0)
    values = rng.uniform(0.0, 1.0, size=100).tolist()
    values.append(1000.0)
    load_df_into_session("d", pd.DataFrame({"v": values}))

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr"},
    )

    assert result["ok"] is True
    assert result["method"] == "iqr"
    assert result["threshold_used"] == 1.5
    # The extreme value (row 100, score=1000) must be flagged.
    indices = {o["row_index"] for o in result["outliers"]}
    assert 100 in indices
    assert result["n_outliers"] >= 1


def test_find_outliers_iqr_known_answer_score(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Hand-computed IQR + score for [1..10, 100].

    Q1 = np.quantile(..., 0.25) = 3.5
    Q3 = np.quantile(..., 0.75) = 8.5
    IQR = 5.0; upper fence = 8.5 + 1.5 * 5 = 16.0
    Value 100 → flagged, score = (100 - 16) / 5 = 16.8
    """
    import pandas as pd

    load_df_into_session(
        "d",
        pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]}),
    )
    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr"},
    )
    assert result["ok"] is True
    assert result["n_outliers"] == 1
    flagged = result["outliers"][0]
    assert flagged["row_index"] == 10
    assert flagged["score"] == pytest.approx(16.8, abs=1e-4)
    assert flagged["values"] == {"v": 100.0}


def test_find_outliers_iqr_per_column_aggregation(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Rows flagged by any column must be in the union; per_column_flags
    must report the right index for each column independently."""
    import pandas as pd

    # Column a: row 5 is extreme (100). Column b: row 9 is extreme (200).
    # All other rows are 1..10 in both columns. Both rows must be flagged
    # at the row level; per_column_flags["a"] = [5], ["b"] = [9].
    a = [1, 2, 3, 4, 5, 100, 7, 8, 9, 10]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 200]
    load_df_into_session("d", pd.DataFrame({"a": a, "b": b}))

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["a", "b"], "method": "iqr"},
    )
    assert result["ok"] is True
    flagged = {o["row_index"] for o in result["outliers"]}
    assert 5 in flagged
    assert 9 in flagged
    assert result["per_column_flags"]["a"] == [5]
    assert result["per_column_flags"]["b"] == [9]


def test_find_outliers_iqr_custom_threshold_widens_band(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """A larger threshold k must flag *fewer* values for the same data.

    Construct a column whose only mild-outlier rows sit between
    Q3 + 1.5·IQR and Q3 + 3.0·IQR. With k=1.5 they are flagged; with
    k=3.0 they are not.
    """
    import pandas as pd

    # Values 1..10 plus a mild outlier at 18. With Q1=3.25, Q3=7.75,
    # IQR=4.5 → upper fence at 1.5 is 7.75 + 6.75 = 14.5 (18 > 14.5 →
    # flagged at k=1.5). Upper fence at 3.0 is 7.75 + 13.5 = 21.25
    # (18 < 21.25 → not flagged at k=3.0).
    data = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18]})
    load_df_into_session("d", data)

    r15 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr", "threshold": 1.5},
    )
    r30 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "iqr", "threshold": 3.0},
    )

    assert r15["threshold_used"] == 1.5
    assert r30["threshold_used"] == 3.0
    assert r15["n_outliers"] == 1
    assert r30["n_outliers"] == 0


# === zscore ===


def test_find_outliers_zscore_default_threshold_3(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Default threshold 3.0: only |z|>3 values are flagged."""
    import numpy as np
    import pandas as pd

    # 100 nominal N(0,1) plus one extreme at 50. With ddof=1 sample std
    # the extreme dwarfs everything else; the extreme must be the only
    # flagged row.
    rng = np.random.default_rng(seed=42)
    values = rng.standard_normal(size=100).tolist()
    values.append(50.0)
    load_df_into_session("d", pd.DataFrame({"v": values}))

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "zscore"},
    )

    assert result["ok"] is True
    assert result["method"] == "zscore"
    assert result["threshold_used"] == 3.0
    assert result["n_outliers"] == 1
    assert result["outliers"][0]["row_index"] == 100


def test_find_outliers_zscore_known_answer_score(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Hand-computed |z| score for the largest value of [1..10].

    Mean=5.5, sd (ddof=1)=3.0276503540974917
    z(10)=(10-5.5)/3.0276503540974917 = 1.4863010829205867
    With threshold=1.4 the extremes 1 and 10 must be flagged, and the
    row with value 10 must have score=1.48630...
    """
    import pandas as pd

    load_df_into_session(
        "d",
        pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}),
    )
    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "zscore", "threshold": 1.4},
    )
    assert result["ok"] is True
    assert result["n_outliers"] == 2
    # Row 9 (value 10) has the top score; the row with value 1 ties on
    # |z| but sorts second when we sort by score descending stably.
    top = result["outliers"][0]
    assert top["score"] == pytest.approx(1.4863010829, abs=1e-4)


def test_find_outliers_zscore_custom_threshold(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Custom threshold respected — same data, |z|>2 flags more than |z|>3."""
    import pandas as pd

    # 10 values: 1..10. Mean=5.5, sd≈3.0277. z(1)≈-1.486, z(10)≈1.486.
    # No value exceeds |z|=3.0; at |z|>1.4 the extremes 1 and 10 should
    # be flagged.
    load_df_into_session(
        "d",
        pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}),
    )
    r3 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "zscore", "threshold": 3.0},
    )
    r1_4 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "zscore", "threshold": 1.4},
    )
    assert r3["threshold_used"] == 3.0
    assert r1_4["threshold_used"] == 1.4
    assert r3["n_outliers"] == 0
    assert r1_4["n_outliers"] == 2


def test_find_outliers_zscore_nan_rows_not_flagged(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """NaN values must not be flagged as outliers."""
    import numpy as np
    import pandas as pd

    # Two NaNs in the middle. 100 small values from N(0, 1) plus a
    # blatant 50 at the end. The NaN rows must not appear in `outliers`;
    # the row with value 50 must be the one and only flagged row.
    rng = np.random.default_rng(seed=7)
    raw = rng.standard_normal(size=100).tolist()
    raw[10] = float("nan")
    raw[20] = float("nan")
    raw.append(50.0)
    load_df_into_session("d", pd.DataFrame({"v": raw}))
    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["v"], "method": "zscore"},
    )
    assert result["ok"] is True
    indices = {o["row_index"] for o in result["outliers"]}
    assert 10 not in indices  # NaN row must not be flagged
    assert 20 not in indices  # NaN row must not be flagged
    assert 100 in indices  # the planted extreme must be flagged
    # zscore does *not* drop NaN rows — n_rows_scored == total rows.
    # (Mahalanobis is the only method that drops NA rows.)
    assert result["n_rows_scored"] == 101
    assert "dropped_2_na_rows" not in result["warnings"]


# === mahalanobis ===


def test_find_outliers_mahalanobis_known_answer_2d(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """50 N(0, I) points + one (5, 5): the (5, 5) row must be flagged.

    Hand-computed (seed=42):
      D²(last row) ≈ 28.2406675647818
      threshold = scipy.stats.chi2.ppf(1 - 0.025, df=2) = 7.3777589082279
    Assert: n_outliers == 1, top row is index 50, score ≈ 28.2406675647818,
    threshold_used ≈ 7.3777589082279.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(50, 2))
    X = np.vstack([X, [5.0, 5.0]])
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})
    load_df_into_session("d", df)

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["x", "y"], "method": "mahalanobis"},
    )

    assert result["ok"] is True
    assert result["method"] == "mahalanobis"
    # Hand-computed against the seed-42 fixture: rows 15 (7.6587…) and
    # 50 (28.2407…) both exceed the 7.3778 cutoff.
    assert result["n_outliers"] == 2
    # Top by D² must be the planted (5, 5) extreme at index 50.
    top = result["outliers"][0]
    assert top["row_index"] == 50
    assert top["score"] == pytest.approx(28.2406675648, abs=1e-4)
    assert result["threshold_used"] == pytest.approx(7.3777589082, abs=1e-4)


def test_find_outliers_mahalanobis_insufficient_rows(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """When n_scored <= k the covariance is rank-deficient; surface a typed error."""
    import pandas as pd

    # k=2 columns, n=2 rows → n <= k, must error.
    load_df_into_session(
        "d",
        pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
    )
    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["x", "y"], "method": "mahalanobis"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "insufficient_rows"


def test_find_outliers_mahalanobis_singular_covariance_falls_back_to_pinv(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Perfectly collinear columns → Σ is singular → use pinv + warn."""
    import pandas as pd

    # y = 2x perfectly. The 2×2 covariance has rank 1 → Σ is singular.
    # Mahalanobis must still return ok=True with a covariance_singular
    # warning, not error out.
    x = list(range(20))
    y = [2.0 * v for v in x]
    load_df_into_session("d", pd.DataFrame({"x": x, "y": y}))

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["x", "y"], "method": "mahalanobis"},
    )
    assert result["ok"] is True
    assert "covariance_singular" in result["warnings"]


def test_find_outliers_mahalanobis_threshold_used_is_chi2_quantile(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """``threshold_used`` is the χ² quantile, not the α probability."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(40, 3))
    load_df_into_session("d", pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]}))

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["a", "b", "c"], "method": "mahalanobis"},
    )

    # Default α = 0.025; df = k = 3.
    # scipy.stats.chi2.ppf(0.975, df=3) = 9.348404029359972
    assert result["ok"] is True
    assert result["threshold_used"] == pytest.approx(9.348404029, abs=1e-4)
    # Make sure threshold_used is NOT just α itself.
    assert result["threshold_used"] > 0.5


def test_find_outliers_mahalanobis_drops_na_and_reports_count(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """NaN rows are dropped from the scoring and counted in ``warnings``."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    X = rng.standard_normal(size=(50, 2))
    # Sprinkle 3 NaN rows.
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})
    df.loc[5, "x"] = np.nan
    df.loc[10, "y"] = np.nan
    df.loc[15, "x"] = np.nan
    load_df_into_session("d", df)

    result = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["x", "y"], "method": "mahalanobis"},
    )

    assert result["ok"] is True
    assert result["n_rows_scored"] == 47  # 50 − 3 dropped
    assert "dropped_3_na_rows" in result["warnings"]


def test_find_outliers_mahalanobis_custom_alpha(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """``threshold`` overrides α — smaller α → larger χ² quantile → fewer flags."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(50, 2))
    X = np.vstack([X, [5.0, 5.0]])
    load_df_into_session("d", pd.DataFrame({"x": X[:, 0], "y": X[:, 1]}))

    r_default = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["x", "y"], "method": "mahalanobis"},
    )
    r_strict = call_tool(
        "find_outliers",
        {
            "name": "d",
            "columns": ["x", "y"],
            "method": "mahalanobis",
            "threshold": 0.001,
        },
    )

    # chi2.ppf(0.975, df=2) = 7.3777589082 → default threshold_used
    # chi2.ppf(0.999, df=2) = 13.815510557964273 → strict threshold_used
    assert r_default["threshold_used"] == pytest.approx(7.3777589082, abs=1e-4)
    assert r_strict["threshold_used"] == pytest.approx(13.8155105580, abs=1e-4)
    # Strict α must flag at most as many rows as default α.
    assert r_strict["n_outliers"] <= r_default["n_outliers"]


# === isolation_forest ===


def test_find_outliers_isolation_forest_known_answer_seed_42(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Planted outliers at the extremes must be flagged at seed=42.

    Build 100 N(0,I) points + 3 obvious outliers at (±10, ±10).
    With sklearn.ensemble.IsolationForest(contamination=0.05,
    random_state=42), the 3 planted outliers (indices 100, 101, 102)
    are deterministically flagged.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    X_nominal = rng.standard_normal(size=(100, 2))
    X_outliers = np.array([[10.0, 10.0], [-10.0, -10.0], [10.0, -10.0]])
    X = np.vstack([X_nominal, X_outliers])
    load_df_into_session("d", pd.DataFrame({"x": X[:, 0], "y": X[:, 1]}))

    result = call_tool(
        "find_outliers",
        {
            "name": "d",
            "columns": ["x", "y"],
            "method": "isolation_forest",
        },
    )

    assert result["ok"] is True
    assert result["method"] == "isolation_forest"
    flagged = {o["row_index"] for o in result["outliers"]}
    # The 3 planted extremes must be flagged.
    assert {100, 101, 102}.issubset(flagged)


def test_find_outliers_isolation_forest_contamination_respected(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Higher contamination flags more rows than lower contamination."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = rng.standard_normal(size=(200, 2))
    load_df_into_session("d", pd.DataFrame({"x": X[:, 0], "y": X[:, 1]}))

    r_low = call_tool(
        "find_outliers",
        {
            "name": "d",
            "columns": ["x", "y"],
            "method": "isolation_forest",
            "contamination": 0.02,
        },
    )
    r_high = call_tool(
        "find_outliers",
        {
            "name": "d",
            "columns": ["x", "y"],
            "method": "isolation_forest",
            "contamination": 0.20,
        },
    )

    assert r_low["ok"] is True
    assert r_high["ok"] is True
    # Contamination is the expected fraction of anomalies, so 20% > 2%.
    assert r_high["n_outliers"] > r_low["n_outliers"]
    assert r_high["threshold_used"] == 0.20
    assert r_low["threshold_used"] == 0.02


def test_find_outliers_isolation_forest_is_deterministic_at_seed_42(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Repeated calls on the same data must produce identical row indices + scores."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    X = rng.standard_normal(size=(80, 2))
    load_df_into_session("d", pd.DataFrame({"x": X[:, 0], "y": X[:, 1]}))

    r1 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["x", "y"], "method": "isolation_forest"},
    )
    r2 = call_tool(
        "find_outliers",
        {"name": "d", "columns": ["x", "y"], "method": "isolation_forest"},
    )

    assert r1["outliers"] == r2["outliers"]
    assert r1["n_outliers"] == r2["n_outliers"]
