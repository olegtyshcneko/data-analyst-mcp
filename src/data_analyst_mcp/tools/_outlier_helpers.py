"""Shared outlier-detection helpers — IQR + z-score primitives.

Used by both ``describe_column`` (single column, SQL-driven over a DuckDB
table) and ``find_outliers`` (multi-column, pandas-driven over a
materialized DataFrame). The two callers have different shapes — the
SQL path counts rows + grabs examples; the DataFrame path returns
per-row boolean masks and per-row scores — so each helper is exposed
separately rather than glued into a single mega-function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def iqr_outliers_sql(
    con: Any,
    table: str,
    quoted: str,
    *,
    p25: Any,
    p75: Any,
    json_safe: Callable[[Any], Any],
) -> dict[str, Any]:
    """IQR-rule outliers + z>3 outliers, with up to 5 example raw values.

    SQL path used by ``describe_column`` over the live DuckDB table —
    returns counts and examples without materializing the full column.
    ``json_safe`` is injected by the caller to coerce DuckDB scalars to a
    JSON-serializable shape (kept out of this module to avoid an import
    cycle with ``datasets.py``).
    """
    lo = p25 - 1.5 * (p75 - p25)
    hi = p75 + 1.5 * (p75 - p25)
    iqr_row = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {quoted} IS NOT NULL "
        f"AND ({quoted} < {lo!r} OR {quoted} > {hi!r})"
    ).fetchone()
    assert iqr_row is not None
    iqr_count = int(iqr_row[0])
    z_row = con.execute(
        f"""
        WITH s AS (
            SELECT AVG({quoted}) AS m, STDDEV_SAMP({quoted}) AS sd FROM {table}
        )
        SELECT COUNT(*) FROM {table}, s
        WHERE {quoted} IS NOT NULL AND s.sd > 0
          AND ABS(({quoted} - s.m) / s.sd) > 3
        """
    ).fetchone()
    assert z_row is not None
    z_count = int(z_row[0])
    example_rows = con.execute(
        f"SELECT {quoted} FROM {table} WHERE {quoted} IS NOT NULL "
        f"AND ({quoted} < {lo!r} OR {quoted} > {hi!r}) LIMIT 5"
    ).fetchall()
    examples = [json_safe(r[0]) for r in example_rows]
    return {"iqr_count": iqr_count, "zscore_count": z_count, "examples": examples}


def iqr_column_mask(
    values: Any,
    *,
    threshold: float,
) -> tuple[Any, Any]:
    """Per-row IQR flag + normalized excess score for one numeric column.

    ``values`` is any 1-D array-like coerced to a numpy float array with
    NaN-aware quantiles. Returns ``(mask, score)`` where ``mask`` is a
    boolean array of length ``n`` (True = flagged) and ``score`` is a
    float array of the *signed* normalized excess. The excess is 0 for
    rows inside the band, ``(x - hi) / IQR`` above the upper bound, and
    ``(lo - x) / IQR`` below the lower bound — both flavors clamp to
    non-negative so ``score >= 0`` everywhere. NaN values are not flagged
    and contribute score ``0`` so the row-level aggregator can take a
    plain ``max`` without special-casing.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return (np.zeros_like(arr, dtype=bool), np.zeros_like(arr))
    q1 = float(np.quantile(finite, 0.25))
    q3 = float(np.quantile(finite, 0.75))
    iqr = q3 - q1
    if iqr == 0.0:
        # Degenerate IQR — no row can exceed ±k·0; nothing flagged.
        return (np.zeros_like(arr, dtype=bool), np.zeros_like(arr))
    lo = q1 - threshold * iqr
    hi = q3 + threshold * iqr
    with np.errstate(invalid="ignore"):
        below = (arr < lo) & ~np.isnan(arr)
        above = (arr > hi) & ~np.isnan(arr)
        mask = below | above
        excess = np.zeros_like(arr)
        excess[above] = (arr[above] - hi) / iqr
        excess[below] = (lo - arr[below]) / iqr
    return (mask, excess)


def zscore_column_mask(
    values: Any,
    *,
    threshold: float,
) -> tuple[Any, Any]:
    """Per-row |z| flag + |z| score for one numeric column.

    Same NaN convention as :func:`iqr_column_mask`. Uses the sample
    standard deviation (``ddof=1``) for parity with the SQL helper's
    ``STDDEV_SAMP``. Returns ``(mask, score)``; ``score`` is ``|z|``,
    NaN-safe (NaNs contribute 0).
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    finite_mask = ~np.isnan(arr)
    finite = arr[finite_mask]
    if finite.size < 2:
        return (np.zeros_like(arr, dtype=bool), np.zeros_like(arr))
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1))
    if sd == 0.0:
        return (np.zeros_like(arr, dtype=bool), np.zeros_like(arr))
    z = np.zeros_like(arr)
    z[finite_mask] = np.abs((arr[finite_mask] - mean) / sd)
    mask = (z > threshold) & finite_mask
    return (mask, z)
