"""Multi-column outlier detection — IQR / z-score / Mahalanobis / Isolation Forest."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


_NUMERIC_DTYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
}


def _is_numeric_dtype(dtype: str) -> bool:
    """True if a DuckDB dtype represents a numeric column."""
    base = dtype.split("(")[0].strip().upper()
    return base in _NUMERIC_DTYPES


class FindOutliersInput(BaseModel):
    """Inputs for ``find_outliers``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    columns: list[str] = Field(min_length=1)
    method: Literal["iqr", "zscore", "mahalanobis", "isolation_forest"]
    threshold: float | None = None
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)
    limit: int = Field(default=50, ge=1, le=10_000)


def find_outliers(payload: FindOutliersInput) -> dict[str, Any]:
    """Detect outliers via one of four methods over a numeric column subset."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    available = {c["name"] for c in entry.columns}
    missing = [c for c in payload.columns if c not in available]
    if missing:
        return build_error(
            type="column_not_found",
            message=f"Columns not in dataset {payload.name!r}: {missing}.",
            hint=f"Available: {sorted(available)}",
        )
    dtype_by_name = {c["name"]: c["dtype"] for c in entry.columns}
    for col in payload.columns:
        if not _is_numeric_dtype(dtype_by_name[col]):
            return build_error(
                type="non_numeric_column",
                message=(
                    f"Column {col!r} has non-numeric dtype "
                    f"{dtype_by_name[col]!r}; find_outliers requires numeric columns."
                ),
                hint="Cast the column to a numeric type or pick a different column.",
            )

    if payload.method == "iqr":
        result = _iqr_method(payload)
    elif payload.method == "zscore":
        result = _zscore_method(payload)
    elif payload.method == "mahalanobis":
        result = _mahalanobis_method(payload)
    elif payload.method == "isolation_forest":
        result = _isolation_forest_method(payload)
    else:  # pragma: no cover - Literal in the input model excludes this branch
        return build_error(type="internal", message="method dispatch not yet implemented")

    if result.get("ok"):
        _record_cell(payload, result)
    return result


def _record_cell(payload: FindOutliersInput, result: dict[str, Any]) -> None:
    """Append one markdown + one code cell that reproduces the method call."""
    md = (
        f"### Outlier detection on `{payload.name}` ({payload.method})\n"
        f"- Columns: {', '.join(payload.columns)}\n"
        f"- Flagged: {result['n_outliers']} / {result['n_rows_scored']} rows"
    )
    code = _code_snippet(payload)
    get_recorder().record(markdown=md, code=code, tool_name="find_outliers")


def _code_snippet(payload: FindOutliersInput) -> str:
    """Method-specific code cell that re-runs the detection from the DataFrame."""
    cols_repr = ", ".join(f'"{c}"' for c in payload.columns)
    fetch = f'df = con.sql("SELECT {cols_repr} FROM {payload.name}").df()'
    if payload.method == "iqr":
        k = payload.threshold if payload.threshold is not None else 1.5
        return (
            f"{fetch}\n"
            f"# IQR outliers (k={k}) — per-column union over the selected cols\n"
            f"q1 = df.quantile(0.25); q3 = df.quantile(0.75); iqr = q3 - q1\n"
            f"lower = q1 - {k} * iqr; upper = q3 + {k} * iqr\n"
            f"mask = ((df < lower) | (df > upper)).any(axis=1)\n"
            f"df[mask]"
        )
    if payload.method == "zscore":
        t = payload.threshold if payload.threshold is not None else 3.0
        return (
            f"{fetch}\n"
            f"# Z-score outliers (|z| > {t}) — per-column union over the selected cols\n"
            f"z = (df - df.mean()) / df.std(ddof=1)\n"
            f"mask = (z.abs() > {t}).any(axis=1)\n"
            f"df[mask]"
        )
    if payload.method == "mahalanobis":
        alpha = payload.threshold if payload.threshold is not None else 0.025
        return (
            f"{fetch}\n"
            f"import numpy as np\n"
            f"from scipy.stats import chi2\n"
            f"# Mahalanobis joint outliers — D² > chi2.ppf(1−α, df=k)\n"
            f"X = df.dropna().to_numpy(dtype=float)\n"
            f"mu = X.mean(axis=0); Sigma = np.cov(X, rowvar=False)\n"
            f"try:\n"
            f"    inv = np.linalg.inv(Sigma)\n"
            f"except np.linalg.LinAlgError:\n"
            f"    inv = np.linalg.pinv(Sigma)\n"
            f"diff = X - mu\n"
            f'd2 = np.einsum("ij,jk,ik->i", diff, inv, diff)\n'
            f"cutoff = chi2.ppf({1 - alpha!r}, df=X.shape[1])\n"
            f"df.dropna().iloc[d2 > cutoff]"
        )
    # isolation_forest
    return (
        f"{fetch}\n"
        f"from sklearn.ensemble import IsolationForest\n"
        f"X = df.dropna().to_numpy(dtype=float)\n"
        f"m = IsolationForest(contamination={payload.contamination!r}, random_state=42).fit(X)\n"
        f"mask = m.predict(X) == -1\n"
        f"# Higher score = more anomalous.\n"
        f"scores = -m.decision_function(X)\n"
        f"df.dropna().iloc[mask]"
    )


def _chi2_quantile(q: float, *, df: int) -> float:
    """Return ``scipy.stats.chi2.ppf(q, df=df)`` as a plain float.

    Wrapped to centralize the ``# type: ignore`` for the noisy scipy stubs.
    """
    from scipy import stats as _sps  # type: ignore[reportMissingTypeStubs]

    chi2: Any = _sps.chi2
    val: Any = chi2.ppf(q, df=df)
    return float(val)


def _sklearn_iforest() -> Any:
    """Return ``sklearn.ensemble.IsolationForest`` as an untyped symbol.

    Mirrors the ``_sklearn_metrics`` pattern in ``evaluate.py``: sklearn
    stubs are noisy under strict pyright, so we centralize the
    ``# type: ignore`` here.
    """
    from sklearn.ensemble import IsolationForest  # type: ignore[reportMissingTypeStubs]

    return IsolationForest


def _isolation_forest_method(payload: FindOutliersInput) -> dict[str, Any]:
    """sklearn IsolationForest with ``random_state=42``.

    Predictions of -1 mark anomalies; ``score = -decision_function(X)`` so
    higher score means more anomalous. NaN rows are dropped before fitting
    (sklearn rejects them) and counted in ``warnings``.
    """
    import numpy as np

    df = _materialize_columns_df(payload.name, list(payload.columns))
    n_total = len(df)
    valid = df[list(payload.columns)].dropna()
    n_scored = len(valid)
    dropped = n_total - n_scored
    k = len(payload.columns)
    min_n = max(10, 2 * k)
    if n_scored < min_n:
        return build_error(
            type="insufficient_rows",
            message=(
                f"Isolation Forest needs n >= max(10, 2*k) = {min_n}; "
                f"got n={n_scored} after dropping NA rows, k={k}."
            ),
            hint="Add more rows or pick a different method.",
        )
    warnings: list[str] = []
    if dropped > 0:
        warnings.append(f"dropped_{dropped}_na_rows")

    X: Any = valid.to_numpy(dtype=float)
    IsolationForest = _sklearn_iforest()
    model: Any = IsolationForest(
        contamination=payload.contamination,
        random_state=42,
    ).fit(X)
    preds: Any = model.predict(X)
    scores: Any = -model.decision_function(X)
    src_indices: Any = valid.index.to_numpy()
    mask: Any = preds == -1
    flagged_src: Any = src_indices[mask]
    flagged_scores: Any = scores[mask]
    order: Any = np.argsort(-flagged_scores)
    n_outliers = int(flagged_src.size)
    truncated = n_outliers > payload.limit
    chosen = order[: payload.limit]
    outliers = [
        {
            "row_index": int(flagged_src[i]),
            "score": float(flagged_scores[i]),
            "values": {
                col: _json_value(df[col].iloc[int(flagged_src[i])]) for col in payload.columns
            },
        }
        for i in chosen
    ]
    return {
        "ok": True,
        "method": "isolation_forest",
        "n_outliers": n_outliers,
        "n_rows_scored": n_scored,
        "outliers": outliers,
        "truncated": truncated,
        "threshold_used": float(payload.contamination),
        "warnings": warnings,
    }


def _mahalanobis_method(payload: FindOutliersInput) -> dict[str, Any]:
    """Joint outlier detection via D² > χ²(k, 1−α).

    Drops rows with any NaN in the selected columns. ``payload.threshold``
    overrides ``α`` (the upper-tail probability used in the chi² quantile);
    the default ``α`` is 0.025. The echoed ``threshold_used`` is the
    chi² quantile itself (not ``α``). If ``np.linalg.inv(Σ)`` raises
    ``LinAlgError`` we fall back to ``np.linalg.pinv(Σ)`` and append
    ``covariance_singular`` to ``warnings``; if the pseudoinverse also
    fails we surface a ``singular_covariance`` error.
    """
    import numpy as np

    df = _materialize_columns_df(payload.name, list(payload.columns))
    n_total = len(df)
    valid = df[list(payload.columns)].dropna()
    n_scored = len(valid)
    dropped = n_total - n_scored
    k = len(payload.columns)
    if n_scored <= k:
        return build_error(
            type="insufficient_rows",
            message=(f"Mahalanobis needs n > k; got n={n_scored} after dropping NA rows, k={k}."),
            hint="Add more rows or pick fewer columns.",
        )
    X = valid.to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = np.cov(X, rowvar=False)
    warnings: list[str] = []
    if dropped > 0:
        warnings.append(f"dropped_{dropped}_na_rows")
    try:
        inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        try:
            inv = np.linalg.pinv(sigma)
        except np.linalg.LinAlgError:
            return build_error(
                type="singular_covariance",
                message="Covariance matrix is singular and pseudo-inverse failed.",
                hint="Drop perfectly collinear columns and retry.",
            )
        warnings.append("covariance_singular")

    diff = X - mu
    # D²_i = (x_i − μ)ᵀ Σ⁻¹ (x_i − μ)
    d2 = np.einsum("ij,jk,ik->i", diff, inv, diff)

    alpha = payload.threshold if payload.threshold is not None else 0.025
    cutoff = _chi2_quantile(1.0 - alpha, df=k)

    mask = d2 > cutoff
    # Map back to source-dataset row indices (valid keeps original index).
    src_indices = valid.index.to_numpy()
    flagged_src = src_indices[mask]
    flagged_scores = d2[mask]
    order = np.argsort(-flagged_scores)
    n_outliers = int(flagged_src.size)
    truncated = n_outliers > payload.limit
    chosen = order[: payload.limit]
    outliers = [
        {
            "row_index": int(flagged_src[i]),
            "score": float(flagged_scores[i]),
            "values": {
                col: _json_value(df[col].iloc[int(flagged_src[i])]) for col in payload.columns
            },
        }
        for i in chosen
    ]
    return {
        "ok": True,
        "method": "mahalanobis",
        "n_outliers": n_outliers,
        "n_rows_scored": n_scored,
        "outliers": outliers,
        "truncated": truncated,
        "threshold_used": cutoff,
        "warnings": warnings,
    }


def _materialize_columns_df(name: str, columns: list[str]) -> Any:
    """Pull the column subset of ``name`` as a pandas DataFrame via DuckDB."""
    con = session.get_connection()
    quoted_cols = ", ".join('"' + c.replace('"', '""') + '"' for c in columns)
    quoted_table = '"' + name.replace('"', '""') + '"'
    return con.execute(f"SELECT {quoted_cols} FROM {quoted_table}").df()


def _iqr_method(payload: FindOutliersInput) -> dict[str, Any]:
    """IQR per-column flag + union row score = max normalized excess."""
    from data_analyst_mcp.tools._outlier_helpers import iqr_column_mask

    threshold = payload.threshold if payload.threshold is not None else 1.5

    def _per_col(values: Any) -> tuple[Any, Any]:
        return iqr_column_mask(values, threshold=threshold)

    return _per_column_union(
        payload,
        method_label="iqr",
        threshold=threshold,
        per_column_fn=_per_col,
    )


def _zscore_method(payload: FindOutliersInput) -> dict[str, Any]:
    """Z-score per-column flag + union row score = max |z|."""
    from data_analyst_mcp.tools._outlier_helpers import zscore_column_mask

    threshold = payload.threshold if payload.threshold is not None else 3.0

    def _per_col(values: Any) -> tuple[Any, Any]:
        return zscore_column_mask(values, threshold=threshold)

    return _per_column_union(
        payload,
        method_label="zscore",
        threshold=threshold,
        per_column_fn=_per_col,
    )


def _per_column_union(
    payload: FindOutliersInput,
    *,
    method_label: str,
    threshold: float,
    per_column_fn: Any,
) -> dict[str, Any]:
    """Apply ``per_column_fn`` to every selected column and union the flags.

    Each column contributes a boolean mask and a per-row score; the row
    is flagged when any column flags it, and its row-level score is the
    max across columns. ``per_column_flags`` records the indices flagged
    by each column independently.
    """
    import numpy as np

    df = _materialize_columns_df(payload.name, list(payload.columns))
    n_rows = len(df)
    row_mask = np.zeros(n_rows, dtype=bool)
    row_score = np.zeros(n_rows, dtype=float)
    per_column_flags: dict[str, list[int]] = {}
    for col in payload.columns:
        mask, score = per_column_fn(df[col].to_numpy())
        row_mask |= mask
        row_score = np.maximum(row_score, score)
        per_column_flags[col] = [int(i) for i in np.nonzero(mask)[0].tolist()]

    flagged_indices = np.nonzero(row_mask)[0]
    n_outliers = int(flagged_indices.size)
    order = sorted(flagged_indices.tolist(), key=lambda i: row_score[i], reverse=True)
    truncated = n_outliers > payload.limit
    chosen = order[: payload.limit]
    outliers = [
        {
            "row_index": int(i),
            "score": float(row_score[i]),
            "values": {col: _json_value(df[col].iloc[i]) for col in payload.columns},
        }
        for i in chosen
    ]

    return {
        "ok": True,
        "method": method_label,
        "n_outliers": n_outliers,
        "n_rows_scored": n_rows,
        "outliers": outliers,
        "truncated": truncated,
        "threshold_used": float(threshold),
        "warnings": [],
        "per_column_flags": per_column_flags,
    }


def _json_value(value: Any) -> Any:
    """Coerce a pandas scalar to a JSON-serializable Python primitive."""
    import math

    import numpy as np

    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        f: float = float(value)  # type: ignore[reportUnknownArgumentType]
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, np.integer):
        return int(value)  # type: ignore[reportUnknownArgumentType]
    if isinstance(value, np.bool_):
        return bool(value)  # type: ignore[reportUnknownArgumentType]
    return value
