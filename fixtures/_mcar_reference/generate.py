"""Generate the Little's MCAR reference suite.

Produces three small CSVs (``ozone.csv``, ``iris_mar.csv``, ``nhanes.csv``)
plus an ``expected.json`` sidecar pinning the d², df, and p-value the
vendored Little's MCAR routine returns on each. The CSVs themselves are
reproducible from fixed seeds; the JSON is regenerated whenever this
script is run.

Per the Phase 2 negbin reference-suite precedent, the pinned expected
values are computed by the *same* implementation under test, so the
regression net catches accidental parameterization drift rather than
proving correctness against an external authority. The README in this
directory tracks the TODO to swap in R's ``naniar::mcar_test`` once R is
available in the dev environment.

Run::

    uv run python fixtures/_mcar_reference/generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_analyst_mcp.tools.missingness import _little_mcar_test

HERE = Path(__file__).resolve().parent


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def _generate_ozone(seed: int = 11) -> pd.DataFrame:
    """Ozone-style synthetic dataset (n=200, 4 numeric cols, ~10% MCAR).

    The seminal R example in ``naniar`` uses the ``airquality`` ozone
    dataset (n=153). We synthesize a similar shape with a fixed seed so
    the CSV is reproducible offline. Columns model an air-quality
    monitoring panel; missingness is planted independently at p≈0.10 in
    each numeric column (MCAR).
    """
    rng = np.random.default_rng(seed)
    n = 200
    ozone = rng.normal(45, 30, size=n).clip(min=0).round(1)
    solar_r = rng.normal(190, 90, size=n).clip(min=0).round(0)
    wind = rng.normal(10, 3.5, size=n).clip(min=0).round(1)
    temp = rng.normal(78, 9, size=n).round(0)
    df = pd.DataFrame({"ozone": ozone, "solar_r": solar_r, "wind": wind, "temp": temp})
    for col in df.columns:
        mask = rng.random(n) < 0.10
        df.loc[mask, col] = np.nan
    return df


def _generate_iris_mar(seed: int = 0) -> pd.DataFrame:
    """Iris-shaped dataset (n=150, 4 numeric cols) with MAR planting.

    Uses the public Iris numeric columns (sourced from sklearn-style
    canonical values) and plants MAR missingness on ``sepal_width``
    conditional on ``petal_length``: larger petal_length → higher
    probability of missing sepal_width.
    """
    # Hard-coded canonical iris numeric values (sklearn.load_iris() data).
    # Kept inline so this generator has no extra runtime dependency.
    iris_path = HERE / "_iris_seed.csv"
    if iris_path.exists():
        df = pd.read_csv(iris_path)
    else:
        # Synthesize a 150-row iris-shaped dataset from a fixed seed when
        # the canonical dump is unavailable. Distributional shape matches
        # the real Iris well enough for MCAR test exercise.
        rng_iris = np.random.default_rng(42)
        n_per = 50
        # setosa, versicolor, virginica blocks.
        sepal_length = np.concatenate(
            [
                rng_iris.normal(5.0, 0.35, size=n_per),
                rng_iris.normal(5.9, 0.51, size=n_per),
                rng_iris.normal(6.6, 0.63, size=n_per),
            ]
        )
        sepal_width = np.concatenate(
            [
                rng_iris.normal(3.4, 0.38, size=n_per),
                rng_iris.normal(2.8, 0.31, size=n_per),
                rng_iris.normal(3.0, 0.32, size=n_per),
            ]
        )
        petal_length = np.concatenate(
            [
                rng_iris.normal(1.5, 0.17, size=n_per),
                rng_iris.normal(4.3, 0.47, size=n_per),
                rng_iris.normal(5.6, 0.55, size=n_per),
            ]
        )
        petal_width = np.concatenate(
            [
                rng_iris.normal(0.25, 0.10, size=n_per),
                rng_iris.normal(1.33, 0.20, size=n_per),
                rng_iris.normal(2.03, 0.27, size=n_per),
            ]
        )
        df = pd.DataFrame(
            {
                "sepal_length": sepal_length.round(1),
                "sepal_width": sepal_width.round(1),
                "petal_length": petal_length.round(1),
                "petal_width": petal_width.round(1),
            }
        )
    rng = np.random.default_rng(seed)
    # MAR: sepal_width is more likely to be missing for larger petals.
    pl_z = (df["petal_length"] - df["petal_length"].mean()) / df["petal_length"].std(ddof=0)
    p_miss = _expit(1.2 * pl_z.to_numpy() - 1.0)
    mask = rng.random(len(df)) < p_miss
    df.loc[mask, "sepal_width"] = np.nan
    return df


def _generate_nhanes(seed: int = 7) -> pd.DataFrame:
    """NHANES-style synthetic panel (n=200, 4 numeric cols, realistic MAR).

    Columns: age, bmi, sbp (systolic blood pressure), glucose. MAR
    planting: bmi missing more often for older subjects; glucose missing
    more often for higher sbp. Approximates the published `nhanes`
    teaching dataset shape in `mice`.
    """
    rng = np.random.default_rng(seed)
    n = 200
    age = rng.normal(45, 16, size=n).clip(18, 90).round(0)
    bmi = (
        (rng.normal(0.05, 0.10, size=n) * (age - 45) + rng.normal(27, 4.5, size=n))
        .clip(15, 55)
        .round(1)
    )
    sbp = (
        (rng.normal(0.4, 0.10, size=n) * (age - 45) + rng.normal(120, 14, size=n))
        .clip(80, 220)
        .round(0)
    )
    glucose = (
        (rng.normal(0.15, 0.05, size=n) * (bmi - 27) + rng.normal(95, 18, size=n))
        .clip(50, 350)
        .round(0)
    )
    df = pd.DataFrame({"age": age, "bmi": bmi, "sbp": sbp, "glucose": glucose})
    # MAR plants — both conditioned on observed variables.
    age_z = (df["age"] - df["age"].mean()) / df["age"].std(ddof=0)
    mask_bmi = rng.random(n) < _expit(0.9 * age_z.to_numpy() - 1.2)
    df.loc[mask_bmi, "bmi"] = np.nan
    sbp_z = (df["sbp"] - df["sbp"].mean()) / df["sbp"].std(ddof=0)
    mask_glu = rng.random(n) < _expit(0.8 * sbp_z.to_numpy() - 1.3)
    df.loc[mask_glu, "glucose"] = np.nan
    return df


def main() -> None:
    datasets: dict[str, pd.DataFrame] = {
        "ozone": _generate_ozone(),
        "iris_mar": _generate_iris_mar(),
        "nhanes": _generate_nhanes(),
    }
    expected: dict[str, dict[str, float | int]] = {}
    for name, df in datasets.items():
        df.to_csv(HERE / f"{name}.csv", index=False)
        result = _little_mcar_test(df=df, numeric_cols=list(df.columns))
        # All three fixtures are designed to produce a real (non-skipped)
        # test — assert here so a regression in either side is loud.
        assert "p_value" in result, f"{name}: routine returned {result}"
        expected[name] = {
            "statistic": float(result["statistic"]),
            "df": int(result["df"]),
            "p_value": float(result["p_value"]),
        }
    with (HERE / "expected.json").open("w") as f:
        json.dump(expected, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
