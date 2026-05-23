"""Missingness diagnostics — ``analyze_missingness`` (v1.1).

Reports per-column null stats, top missingness patterns, pairwise
φ-correlation between null indicators, Little's MCAR test, and
actionable suggestions.

See ``docs/SPEC.md`` §5.4 for the contract.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder
from data_analyst_mcp.tools.datasets import looks_like_categorical

logger = logging.getLogger(__name__)


_MAX_SUGGESTIONS = 6
# Severity ranks: structural > MCAR-violation > high-null > co-missing >
# imputation-OK > no-nulls. Lower is more severe (sorted ascending). Rank
# 1 is reserved for the MCAR-violation suggestion per the proposal.
_SEV_STRUCTURAL = 0
_SEV_MCAR_VIOLATION = 1
_SEV_HIGH_NULL = 2
_SEV_COMISSING = 3
_SEV_IMPUTATION_OK = 4
_SEV_NO_NULLS = 5

# Suggestion thresholds.
_MCAR_NULL_PCT_MIN = 1.0
_MCAR_NULL_PCT_MAX = 30.0
_MCAR_ALPHA = 0.05

_MCAR_CONSEQUENCE_REJECTED = (
    "Reject MCAR — missingness depends on observed data; mean-imputation will bias."
)
_MCAR_CONSEQUENCE_NOT_REJECTED = (
    "Fail to reject MCAR; missingness is consistent with random absence."
)


class AnalyzeMissingnessInput(BaseModel):
    """Inputs for ``analyze_missingness`` (v1, descriptive only)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name to analyze.")
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Restrict the analysis to a subset of columns. When omitted, "
            "every column in the dataset is considered."
        ),
    )
    pattern_top_k: int = Field(
        default=10,
        description=(
            "How many distinct missingness patterns to return, ordered by "
            "descending count. Range [1, 100]; default 10."
        ),
    )
    pairwise_corr_threshold: float = Field(
        default=0.1,
        description=(
            "Hide pairwise φ entries whose absolute value is below this "
            "threshold. Range [0.0, 1.0]; default 0.1."
        ),
    )
    run_mcar_test: bool = Field(
        default=True,
        description=(
            "Run Little's MCAR test on the numeric columns. Pass False to "
            "skip the test and return ``mcar_test: null``."
        ),
    )


def analyze_missingness(payload: AnalyzeMissingnessInput) -> dict[str, Any]:
    """Per-column nulls + pattern aggregation + pairwise φ + suggestions."""
    # Range validation up-front so error type is deterministic.
    if not (1 <= payload.pattern_top_k <= 100):
        return build_error(
            type="pattern_top_k_out_of_range",
            message=f"pattern_top_k must be in [1, 100]; got {payload.pattern_top_k}.",
            hint="pattern_top_k must be in [1, 100].",
        )
    if not (0.0 <= payload.pairwise_corr_threshold <= 1.0):
        return build_error(
            type="pairwise_corr_threshold_out_of_range",
            message=(
                "pairwise_corr_threshold must be in [0.0, 1.0]; "
                f"got {payload.pairwise_corr_threshold}."
            ),
            hint="pairwise_corr_threshold must be in [0.0, 1.0].",
        )

    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="dataset_not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )

    entry = entries[payload.name]
    available = {c["name"]: c["dtype"] for c in entry.columns}
    if payload.columns is not None:
        unknown = [c for c in payload.columns if c not in available]
        if unknown:
            return build_error(
                type="unknown_columns",
                message=f"Columns not in dataset {payload.name!r}: {unknown}.",
                hint=f"Available: {sorted(available)}",
            )

    # Materialize the chosen column set as a pandas frame.
    all_cols = [c["name"] for c in entry.columns]
    analysis_cols = list(payload.columns) if payload.columns is not None else all_cols
    df = _materialize_df(payload.name, analysis_cols)
    n_rows = len(df)

    null_mask = df.isna()
    # Per-column null counts across the *analyzed* subset.
    null_counts_per_col: dict[str, int] = {col: int(null_mask[col].sum()) for col in analysis_cols}

    # ---- per-column block: only columns with ≥1 null appear ------------
    per_column = _build_per_column(
        analysis_cols=analysis_cols,
        null_counts=null_counts_per_col,
        n_rows=n_rows,
        available=available,
        dataset_name=payload.name,
        df=df,
        null_mask=null_mask,
    )

    # ---- pairwise φ -----------------------------------------------------
    pairwise = _build_pairwise(
        analysis_cols=analysis_cols,
        null_counts=null_counts_per_col,
        n_rows=n_rows,
        null_mask=null_mask,
        threshold=payload.pairwise_corr_threshold,
    )

    # ---- patterns block -------------------------------------------------
    patterns = _build_patterns(
        analysis_cols=analysis_cols, null_mask=null_mask, top_k=payload.pattern_top_k
    )

    # ---- summary --------------------------------------------------------
    rows_complete = int((~null_mask.any(axis=1)).sum()) if n_rows else 0
    summary = {
        "n_rows": int(n_rows),
        "columns_analyzed": len(analysis_cols),
        "columns_with_missing": len(per_column),
        "rows_complete": rows_complete,
    }

    # ---- Little's MCAR test --------------------------------------------
    mcar_test: dict[str, Any] | None
    if payload.run_mcar_test:
        # Numeric-only per proposal §Open question 1. Build the numeric
        # subset from the analyzed columns by dtype.
        numeric_cols = _numeric_columns(df=df, analysis_cols=analysis_cols, available=available)
        mcar_test = _little_mcar_test(df=df, numeric_cols=numeric_cols)
    else:
        mcar_test = None

    # ---- suggestions ----------------------------------------------------
    suggestions = _build_suggestions(
        per_column=per_column,
        pairwise=pairwise,
        any_nulls=any(null_counts_per_col[c] > 0 for c in analysis_cols),
        mcar_test=mcar_test,
    )

    out: dict[str, Any] = {
        "ok": True,
        "summary": summary,
        "per_column": per_column,
        "patterns": patterns,
        "pairwise_correlation": pairwise,
        "mcar_test": mcar_test,
        "suggestions": suggestions,
    }

    _record(payload=payload, out=out)
    return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _materialize_df(dataset_name: str, columns: list[str]) -> Any:
    """Fetch the chosen columns of ``dataset_name`` as a pandas DataFrame."""
    con = session.get_connection()
    select_cols = ", ".join(_quote(c) for c in columns) if columns else "*"
    return con.execute(f"SELECT {select_cols} FROM {_quote(dataset_name)}").df()


def _scipy_stats() -> Any:
    """Return ``scipy.stats`` as untyped to keep strict pyright clean."""
    from scipy import stats as _sps  # type: ignore[reportMissingTypeStubs]

    return _sps


def _scipy_linalg() -> Any:
    """Return ``scipy.linalg`` as untyped to keep strict pyright clean."""
    from scipy import linalg as _sla  # type: ignore[reportMissingTypeStubs]

    return _sla


def _numeric_columns(*, df: Any, analysis_cols: list[str], available: dict[str, str]) -> list[str]:
    """Filter ``analysis_cols`` down to numeric columns.

    Source of truth is the pandas dtype (post-materialization); the
    DuckDB-reported ``available`` mapping is fallback for cases where
    pandas inferred ``object`` for an all-null column. Little's MCAR is
    defined on numeric variables only — categorical/string columns are
    silently excluded.
    """
    numeric_kinds = {"i", "u", "f", "b"}  # int, uint, float, bool
    out: list[str] = []
    for c in analysis_cols:
        if c not in df.columns:
            continue
        kind = df[c].dtype.kind
        if kind in numeric_kinds:
            out.append(c)
            continue
        # DuckDB-reported type as fallback for object-typed all-null cols.
        duck_type = (available.get(c) or "").upper()
        if any(
            tok in duck_type
            for tok in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL", "HUGEINT")
        ):
            out.append(c)
    return out


def _little_mcar_test(*, df: Any, numeric_cols: list[str]) -> dict[str, Any]:
    """Little's (1988) MCAR test on the numeric subset of ``df``.

    Procedure
    ---------
    1. Restrict to ``numeric_cols``; coerce to float; build the
       missingness-pattern key per row as a tuple of column→is_null
       booleans.
    2. Group rows by pattern; drop any pattern with ``n_j < 2`` (cannot
       estimate variance from a single row).
    3. Under MCAR with jointly Gaussian data, the EM solution for the
       full-data mean and covariance converges in **one step** to the
       sample mean / covariance computed on the pairwise-available
       complete cases (i.e. the standard "available-case" estimators).
       The proposal's outline notes this; we implement it as a
       closed-form single-pass computation rather than a loop.
    4. d² = Σ_j n_j · (μ̂_j − μ̂)ᵀ · Σ̂_j⁻¹ · (μ̂_j − μ̂), where μ̂_j /
       Σ̂_j are the pooled mean / covariance restricted to the
       variables observed in pattern j. df = Σ p_j − p where p_j is the
       count of observed vars in pattern j and p is the total.
    5. p-value = 1 − chi2.cdf(d², df).

    Singular Σ̂_j (degenerate patterns where a variable is constant in
    the available rows) fall back to ``scipy.linalg.pinv`` so the test
    still produces a finite statistic.

    Returns ``{"name": "little", "skipped": true, "reason": ...}`` when
    fewer than 2 patterns survive the n_j ≥ 2 filter or when there are
    no numeric columns to test.
    """
    if len(numeric_cols) == 0:
        return {"name": "little", "skipped": True, "reason": "insufficient_patterns"}

    # Coerce to a numeric numpy matrix; preserve NaN.
    arr = df[numeric_cols].to_numpy(dtype=float, copy=True)
    n, p = arr.shape
    if n == 0:
        return {"name": "little", "skipped": True, "reason": "insufficient_patterns"}

    null_mask = np.isnan(arr)

    # Pattern key = tuple of column→is_null booleans. Two rows with the
    # same pattern share the same set of observed variables.
    pattern_keys = [tuple(bool(b) for b in null_mask[i]) for i in range(n)]
    patterns: dict[tuple[bool, ...], list[int]] = {}
    for i, key in enumerate(pattern_keys):
        patterns.setdefault(key, []).append(i)

    # Drop patterns with n_j < 2 — cannot estimate variance from one row.
    kept = {k: idx for k, idx in patterns.items() if len(idx) >= 2}
    if len(kept) < 2:
        return {"name": "little", "skipped": True, "reason": "insufficient_patterns"}

    # Pooled (available-case) mean and covariance — closed-form Gaussian
    # EM solution under MCAR. Use pandas-style pairwise estimators by
    # computing each entry from rows where both variables are observed.
    mu_hat = np.empty(p, dtype=float)
    for j in range(p):
        col = arr[:, j]
        col_obs = col[~np.isnan(col)]
        # Guard: if a variable is fully missing it cannot contribute.
        mu_hat[j] = float(col_obs.mean()) if col_obs.size else 0.0

    sigma_hat = np.zeros((p, p), dtype=float)
    for j in range(p):
        for k in range(p):
            both = ~np.isnan(arr[:, j]) & ~np.isnan(arr[:, k])
            n_both = int(both.sum())
            if n_both < 2:
                # Patch missing entries with 0 covariance; pinv handles
                # the resulting rank deficiency in the per-pattern inverse.
                sigma_hat[j, k] = 0.0
                continue
            xj = arr[both, j] - mu_hat[j]
            xk = arr[both, k] - mu_hat[k]
            # Unbiased sample covariance, ddof=1 to match scipy/R convention.
            sigma_hat[j, k] = float(np.dot(xj, xk) / (n_both - 1))

    # Accumulate d² over patterns; df = Σ p_j - p.
    sla = _scipy_linalg()
    d2 = 0.0
    df_total = 0
    for key, idx in kept.items():
        # Observed-variable index list for this pattern.
        observed = [j for j, is_null in enumerate(key) if not is_null]
        p_j = len(observed)
        if p_j == 0:
            # All-missing pattern contributes nothing.
            continue
        df_total += p_j
        n_j = len(idx)

        sub = arr[np.ix_(idx, observed)]
        mu_j_hat = sub.mean(axis=0)
        mu_pool = mu_hat[observed]
        diff = mu_j_hat - mu_pool

        sigma_sub = sigma_hat[np.ix_(observed, observed)]
        try:
            sigma_inv = sla.inv(sigma_sub)
        except Exception:
            # Singular Σ̂_j fallback — pseudoinverse keeps d² finite.
            sigma_inv = sla.pinv(sigma_sub)

        d2 += float(n_j * diff @ sigma_inv @ diff)

    df_final = df_total - p
    if df_final <= 0:
        # Degenerate — not enough excess observed variables to form a
        # chi-square. Treat as insufficient.
        return {"name": "little", "skipped": True, "reason": "insufficient_patterns"}

    sps = _scipy_stats()
    p_value = float(1.0 - sps.chi2.cdf(d2, df_final))
    violated = p_value < _MCAR_ALPHA
    return {
        "name": "little",
        "statistic": float(d2),
        "df": int(df_final),
        "p_value": p_value,
        "violated": bool(violated),
        "consequence": (_MCAR_CONSEQUENCE_REJECTED if violated else _MCAR_CONSEQUENCE_NOT_REJECTED),
    }


def _build_per_column(
    *,
    analysis_cols: list[str],
    null_counts: dict[str, int],
    n_rows: int,
    available: dict[str, str],
    dataset_name: str,
    df: Any,
    null_mask: Any,
) -> list[dict[str, Any]]:
    """Per-column null block; only columns with ≥1 null are emitted.

    Each entry carries ``column``, ``null_count``, ``null_pct``,
    ``null_grouping``, and ``variance_zero``. ``null_grouping`` searches
    for a ``looks_like_categorical`` column where this column's
    missingness is 0% or 100% within every group; ties broken by fewest
    groups (most informative).
    """
    per_column: list[dict[str, Any]] = []

    grouping_candidates = _grouping_candidates(
        dataset_name=dataset_name,
        analysis_cols=analysis_cols,
        available=available,
        n_rows=n_rows,
    )

    for col in analysis_cols:
        nc = null_counts[col]
        if nc == 0:
            continue
        # variance_zero on the null indicator: all-null (nc == n_rows) — by
        # definition we have at least one null here, so the all-present case
        # is excluded by the `nc == 0` guard above; only all-null trips it.
        variance_zero = nc == n_rows
        null_pct = (nc / n_rows * 100.0) if n_rows else 0.0
        per_column.append(
            {
                "column": col,
                "null_count": int(nc),
                "null_pct": round(float(null_pct), 4),
                "null_grouping": _null_grouping(
                    col=col,
                    df=df,
                    null_mask=null_mask,
                    candidates=[g for g in grouping_candidates if g != col],
                ),
                "variance_zero": bool(variance_zero),
            }
        )
    return per_column


def _grouping_candidates(
    *,
    dataset_name: str,
    analysis_cols: list[str],
    available: dict[str, str],
    n_rows: int,
) -> list[str]:
    """Columns to test as ``null_grouping.aligned_with`` candidates.

    Restricted to columns the shared ``profile_dataset`` heuristic flags
    as ``looks_like_categorical``. High-cardinality candidates
    (``group_count > sqrt(n_rows)``) are skipped per the proposal's
    §Open question 2 — they trivially "align" with anything.
    """
    if n_rows == 0:
        return []
    con = session.get_connection()
    table = _quote(dataset_name)
    threshold = math.sqrt(n_rows)
    out: list[str] = []
    # Examine every column in the dataset (not just analyzed ones) — the
    # grouping signal may live on a column the caller didn't ask about.
    for cname, dtype in available.items():
        # Per-column null/distinct stats — same shape profile_dataset uses.
        agg_row = con.execute(
            f"SELECT COUNT(*) - COUNT({_quote(cname)}), COUNT(DISTINCT {_quote(cname)}) "
            f"FROM {table}"
        ).fetchone()
        assert agg_row is not None
        non_null = n_rows - int(agg_row[0])
        distinct_count = int(agg_row[1])
        if not looks_like_categorical(dtype, distinct_count=distinct_count, non_null=non_null):
            continue
        # Skip high-cardinality candidates.
        if distinct_count > threshold:
            continue
        # Skip columns with no usable groups (everything null).
        if distinct_count == 0:
            continue
        out.append(cname)
    return out


def _null_grouping(
    *,
    col: str,
    df: Any,
    null_mask: Any,
    candidates: list[str],
) -> dict[str, Any]:
    """Search ``candidates`` for a categorical whose groups partition nulls.

    A candidate ``g`` aligns with ``col`` iff every distinct value of
    ``g`` has either 0% or 100% nulls for ``col``. Among aligned
    candidates, pick the one with the fewest groups (most informative
    tiebreaker).
    """
    aligned: list[tuple[str, int]] = []
    col_nulls = null_mask[col]
    for g in candidates:
        if g not in df.columns:
            continue
        # Aggregate per-group: mean of the null indicator is the null
        # fraction. Aligned iff every fraction is 0.0 or 1.0.
        grouped = col_nulls.groupby(df[g], dropna=False).mean()
        if len(grouped) == 0:
            continue
        # NaN groups (i.e. the candidate column itself has nulls) are
        # included via dropna=False; treat them as a regular bucket.
        fractions = grouped.to_numpy()
        if not all(_is_zero_or_one(float(f)) for f in fractions):
            continue
        aligned.append((g, len(grouped)))

    if not aligned:
        return {"aligned_with": None, "all_or_nothing": False, "group_count": None}

    # Fewest groups wins; stable secondary sort by column name keeps it
    # deterministic across equal-group-count candidates.
    aligned.sort(key=lambda t: (t[1], t[0]))
    best_name, best_count = aligned[0]
    return {
        "aligned_with": best_name,
        "all_or_nothing": True,
        "group_count": best_count,
    }


def _is_zero_or_one(value: float) -> bool:
    """True when ``value`` is essentially 0.0 or 1.0 (with float epsilon)."""
    if math.isnan(value):
        return False
    return math.isclose(value, 0.0, abs_tol=1e-12) or math.isclose(value, 1.0, abs_tol=1e-12)


def _build_pairwise(
    *,
    analysis_cols: list[str],
    null_counts: dict[str, int],
    n_rows: int,
    null_mask: Any,
    threshold: float,
) -> list[dict[str, Any]]:
    """Pairwise φ-correlation between null indicators.

    Columns with zero variance on the null indicator (all-null or
    no-nulls) are skipped silently — Pearson is undefined and pandas
    returns NaN. The result is sorted by descending |φ|; pairs below
    ``threshold`` are dropped.
    """
    if len(analysis_cols) < 2 or n_rows == 0:
        return []

    # Variance is zero iff the column is all-null (nc == n_rows) or
    # all-present (nc == 0).
    usable = [c for c in analysis_cols if 0 < null_counts[c] < n_rows]
    if len(usable) < 2:
        return []

    # Pearson on 0/1 == φ.
    int_mask = null_mask[usable].astype(int)
    corr = int_mask.corr(method="pearson")

    out: list[dict[str, Any]] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            phi = float(corr.iat[i, j])
            if math.isnan(phi):
                continue
            if abs(phi) < threshold:
                continue
            out.append({"col_a": a, "col_b": b, "phi": round(phi, 4)})

    out.sort(key=lambda r: abs(float(r["phi"])), reverse=True)
    return out


def _build_patterns(
    *, analysis_cols: list[str], null_mask: Any, top_k: int
) -> list[dict[str, Any]]:
    """Top-K distinct (col → is_null) tuples by descending count.

    Empty input or zero rows still returns at most one pattern (the
    all-present baseline) so the recorder cell always has something to
    show.
    """
    if not analysis_cols or len(null_mask) == 0:
        return []

    # Group by the entire null-indicator tuple. ``groupby(list_of_cols)``
    # treats each unique row as a key.
    grouped = null_mask.groupby(list(analysis_cols)).size().sort_values(ascending=False)

    out: list[dict[str, Any]] = []
    for key, count in grouped.head(top_k).items():
        # ``key`` is a tuple of bools when len(analysis_cols) > 1, else a
        # scalar bool — normalize into a list so iteration is uniform.
        key_list: list[Any] = (
            list(key)  # type: ignore[reportUnknownArgumentType]
            if isinstance(key, tuple)
            else [key]
        )
        pattern: dict[str, bool] = {
            col: bool(val) for col, val in zip(analysis_cols, key_list, strict=True)
        }
        out.append({"pattern": pattern, "count": int(count)})
    return out


def _build_suggestions(
    *,
    per_column: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    any_nulls: bool,
    mcar_test: dict[str, Any] | None = None,
) -> list[str]:
    """Severity-sorted suggestions, capped at ``_MAX_SUGGESTIONS``."""
    items: list[tuple[int, str]] = []

    if not any_nulls:
        items.append((_SEV_NO_NULLS, "No missingness detected."))

    for c in per_column:
        grouping: dict[str, Any] = c.get("null_grouping") or {}
        if grouping.get("all_or_nothing"):
            items.append(
                (
                    _SEV_STRUCTURAL,
                    f"Column `{c['column']}` is missing entirely within "
                    f"`{grouping['aligned_with']}` groups — likely a "
                    "structural / join issue, not random.",
                )
            )

    # MCAR-dependent rules fire once per qualifying column (null_pct in
    # [1, 30]) when an MCAR p-value is available. Skipped tests
    # (insufficient_patterns) and run_mcar_test=False suppress these.
    if mcar_test is not None and not mcar_test.get("skipped") and "p_value" in mcar_test:
        p_value = float(mcar_test["p_value"])
        rejected = bool(mcar_test.get("violated"))
        qualifying = [
            c
            for c in per_column
            if _MCAR_NULL_PCT_MIN <= float(c["null_pct"]) <= _MCAR_NULL_PCT_MAX
        ]
        for c in qualifying:
            if rejected:
                items.append(
                    (
                        _SEV_MCAR_VIOLATION,
                        f"Missingness is not random (Little's MCAR p={p_value:.3g}); "
                        f"mean-imputation will bias `{c['column']}`. "
                        "Consider model-based imputation or include a missingness indicator.",
                    )
                )
            else:
                items.append(
                    (
                        _SEV_IMPUTATION_OK,
                        f"Missingness on `{c['column']}` is consistent with MCAR; "
                        "mean/median imputation acceptable.",
                    )
                )

    for c in per_column:
        if float(c["null_pct"]) > 50.0:
            items.append(
                (
                    _SEV_HIGH_NULL,
                    f"Column `{c['column']}` is {c['null_pct']:.0f}% null — "
                    f"consider dropping or recoding as a binary `has_{c['column']}`.",
                )
            )

    for p in pairwise:
        if abs(float(p["phi"])) > 0.5:
            items.append(
                (
                    _SEV_COMISSING,
                    f"`{p['col_a']}` and `{p['col_b']}` are co-missing "
                    f"(φ={float(p['phi']):.2f}) — likely shared upstream cause.",
                )
            )

    items.sort(key=lambda t: t[0])
    return [text for _, text in items[:_MAX_SUGGESTIONS]]


def _record(*, payload: AnalyzeMissingnessInput, out: dict[str, Any]) -> None:
    """Append the recorder markdown + code cell pair for one analysis."""
    if not out.get("ok"):
        return

    summary = out["summary"]
    per_column = out["per_column"]
    patterns = out["patterns"]

    md_lines = [f"### Missingness analysis — `{payload.name}`"]
    if per_column:
        parts = ", ".join(f"`{c['column']}` {float(c['null_pct']):.1f}%" for c in per_column[:5])
        md_lines.append(f"- {len(per_column)} columns have nulls ({parts})")
    else:
        md_lines.append("- No missingness detected")
    md_lines.append(f"- {summary['rows_complete']} / {summary['n_rows']} rows complete")
    if patterns:
        top = patterns[0]
        present_cols = [c for c, v in top["pattern"].items() if not v]
        missing_cols = [c for c, v in top["pattern"].items() if v]
        if missing_cols:
            md_lines.append(
                "- Top pattern: "
                f"{', '.join(f'`{c}`' for c in missing_cols)} missing ({top['count']} rows)"
            )
        else:
            md_lines.append(
                f"- Top pattern: all {len(present_cols)} columns present ({top['count']} rows)"
            )
    mcar_dict: dict[str, Any] | None = out.get("mcar_test")  # type: ignore[assignment]
    if mcar_dict is not None and not mcar_dict.get("skipped") and "p_value" in mcar_dict:
        verdict = "rejected" if mcar_dict.get("violated") else "not rejected"
        md_lines.append(f"- Little's MCAR {verdict} (p={float(mcar_dict['p_value']):.3g})")
    for s in out["suggestions"][:1]:
        md_lines.append(f"- Top suggestion: {s}")
    md = "\n".join(md_lines)

    code = (
        f'raw_df = con.sql("SELECT * FROM {payload.name}").df()\n'
        "nulls = raw_df.isna()\n"
        'print("Null percentages:")\n'
        "print(nulls.mean().sort_values(ascending=False))\n\n"
        'print("\\nTop missingness patterns:")\n'
        "print(nulls.groupby(list(nulls.columns)).size()"
        ".sort_values(ascending=False).head(10))\n\n"
        "# Pairwise phi via Pearson on 0/1:\n"
        'print("\\nPairwise missingness correlation:")\n'
        "print(nulls.astype(int).corr().round(2))"
    )
    get_recorder().record(markdown=md, code=code, tool_name="analyze_missingness")
