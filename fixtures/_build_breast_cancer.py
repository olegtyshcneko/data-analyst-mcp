"""Generate `fixtures/breast_cancer.csv` — a wide real-world reference dataset.

This is the high-dimensional counterpart to `titanic.csv`: a small, real,
public dataset committed verbatim so evals can assert on substantive,
textbook-stable results. Its specific job is to give the multivariate
outlier detectors (`find_outliers` with `mahalanobis` / `isolation_forest`)
a genuinely high-dimensional manifold to work on — the synthetic fixtures
top out at 3 numeric columns, which never exercises the k≫2 covariance and
isolation-forest paths.

Source: scikit-learn's bundled Wisconsin Diagnostic Breast Cancer (WDBC)
dataset (`sklearn.datasets.load_breast_cancer`), itself the UCI WDBC set
(Wolberg, Street, Mangasarian). 569 rows × 30 numeric features + target.
scikit-learn is already a runtime dependency (Isolation Forest), so no new
dep is introduced. The bundled data is static, so the export is fully
deterministic given a fixed scikit-learn build.

Transforms applied (all deterministic):

- Feature names sanitized: spaces → underscores (`mean radius` →
  `mean_radius`) so they are clean SQL identifiers and patsy terms.
- A string `diagnosis` column is appended next to the integer `target`.
  scikit-learn's convention is **0 = malignant, 1 = benign** (counter-
  intuitive — recorded here and in `fixtures/README.md`).

Run:
    uv run python fixtures/_build_breast_cancer.py

Re-running must produce a byte-identical file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

OUTPUT = Path(__file__).parent / "breast_cancer.csv"

# scikit-learn labels benign=1, malignant=0. Spelled out so the categorical
# `diagnosis` column never silently inverts if upstream ever changes.
TARGET_LABELS = {0: "malignant", 1: "benign"}


def build_frame() -> pd.DataFrame:
    """Load the bundled WDBC set and apply the deterministic transforms."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [col.replace(" ", "_") for col in df.columns]
    df["diagnosis"] = df["target"].map(TARGET_LABELS)
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV: full float precision, \\n line endings."""
    df.to_csv(path, index=False, lineterminator="\n")


def _verify(df: pd.DataFrame) -> None:
    """Assert the shape and class balance the evals and README rely on."""
    assert df.shape == (569, 32), f"unexpected shape {df.shape}"
    assert not df.isna().any().any(), "WDBC reference set must have no NaNs"
    assert not any(" " in col for col in df.columns), "column names must be sanitized"
    counts = df["target"].value_counts().to_dict()
    assert counts == {1: 357, 0: 212}, f"unexpected class balance {counts}"
    assert (df["diagnosis"] == df["target"].map(TARGET_LABELS)).all()


def main() -> None:
    df = build_frame()
    _verify(df)
    _write_csv(df, OUTPUT)
    print(f"wrote {OUTPUT} — {df.shape[0]} rows × {df.shape[1]} cols")


if __name__ == "__main__":
    main()
