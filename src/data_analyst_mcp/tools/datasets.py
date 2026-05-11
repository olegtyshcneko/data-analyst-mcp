"""Dataset registration + EDA tools."""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


class LoadDatasetInput(BaseModel):
    """Inputs for ``load_dataset``."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "Local filesystem path or s3:// URL to a CSV, TSV, Parquet, "
            "Excel, JSON, or JSONL file. The extension determines the "
            "DuckDB reader used."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Name to register the dataset under so other tools can refer to "
            "it. When omitted a slugified version of the filename stem is used."
        ),
    )
    read_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional DuckDB reader options forwarded into the read_* call, "
            "e.g. {\"header\": false, \"delim\": \";\"} for CSV. Use this "
            "only when auto-detection produces wrong results."
        ),
    )


_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".json", ".jsonl"}

_EXT_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".xlsx": "xlsx",
    ".json": "json",
    ".jsonl": "jsonl",
}


def _extension(path: str) -> str:
    """Lowercase extension including the leading dot."""
    return os.path.splitext(path)[1].lower()


def _build_read_call(path: str, fmt: str) -> str:
    """Render the DuckDB read_* call used in the CREATE TABLE statement."""
    if fmt == "parquet":
        return f"read_parquet('{path}')"
    if fmt in {"json", "jsonl"}:
        return f"read_json('{path}')"
    return f"read_csv_auto('{path}', SAMPLE_SIZE=-1)"


class DescribeColumnInput(BaseModel):
    """Inputs for ``describe_column``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    column: str = Field(..., description="Column name to describe.")
    bins: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of histogram bins for numeric columns.",
    )


def describe_column(payload: DescribeColumnInput) -> dict[str, Any]:
    """Single-column deep dive (numeric/categorical/temporal)."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
        )
    entry = entries[payload.name]
    col_meta = next((c for c in entry.columns if c["name"] == payload.column), None)
    if col_meta is None:
        return build_error(
            type="column_not_found",
            message=f"Column {payload.column!r} is not in dataset {payload.name!r}.",
            hint=f"Available columns: {', '.join(c['name'] for c in entry.columns)}",
        )

    con = session.get_connection()
    table = _quote(payload.name)
    quoted = _quote(payload.column)
    result: dict[str, Any] = {"ok": True, "column": payload.column, "dtype": col_meta["dtype"]}

    if _is_numeric(col_meta["dtype"]):
        quantiles_row = con.execute(
            f"""
            SELECT
                QUANTILE_CONT({quoted}, 0.01),
                QUANTILE_CONT({quoted}, 0.05),
                QUANTILE_CONT({quoted}, 0.10),
                QUANTILE_CONT({quoted}, 0.25),
                QUANTILE_CONT({quoted}, 0.50),
                QUANTILE_CONT({quoted}, 0.75),
                QUANTILE_CONT({quoted}, 0.90),
                QUANTILE_CONT({quoted}, 0.95),
                QUANTILE_CONT({quoted}, 0.99),
                SKEWNESS({quoted}),
                KURTOSIS({quoted})
            FROM {table}
            """
        ).fetchone()
        assert quantiles_row is not None  # DuckDB aggregate always returns one row
        keys_pct = (1, 5, 10, 25, 50, 75, 90, 95, 99)
        quantiles = {p: _json_safe(quantiles_row[i]) for i, p in enumerate(keys_pct)}
        result["quantiles"] = quantiles
        result["skewness"] = _json_safe(quantiles_row[9])
        result["kurtosis"] = _json_safe(quantiles_row[10])
        p25 = quantiles[25]
        p75 = quantiles[75]
        iqr = (p75 - p25) if (p25 is not None and p75 is not None) else 0.0
        result["iqr"] = iqr
        result["histogram"] = _histogram(con, table, quoted, bins=payload.bins)
        result["outliers"] = _outliers(con, table, quoted, p25=p25, p75=p75)
    elif _is_temporal(col_meta["dtype"]):
        result.update(_temporal_describe(con, table, quoted))
    elif _is_string(col_meta["dtype"]):
        result.update(_categorical_describe(con, table, quoted))

    return result


class ProfileDatasetInput(BaseModel):
    """Inputs for ``profile_dataset``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name to profile.")
    sample_rows: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of head rows to include in the response sample.",
    )


def _quote(name: str) -> str:
    """Quote a SQL identifier (column or table) safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _json_safe(value: Any) -> Any:
    """Coerce DuckDB-returned scalars into a JSON-serializable shape."""
    import datetime as _dt

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    return str(value)


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


def _is_numeric(dtype: str) -> bool:
    """True if a DuckDB dtype is numeric (we strip decimal precision)."""
    base = dtype.split("(")[0].strip().upper()
    return base in _NUMERIC_DTYPES


_STRING_DTYPES = {"VARCHAR", "CHAR", "TEXT", "BLOB", "STRING"}


def _build_suggestions(column_profiles: list[dict[str, Any]]) -> list[str]:
    """Pick up to three actionable next-step strings from the profile."""
    suggestions: list[str] = []
    null_col = next(
        (c for c in column_profiles if c.get("flags", {}).get("mostly_null")),
        None,
    )
    if null_col is not None:
        suggestions.append(
            f"Column `{null_col['name']}` is mostly null — consider dropping or imputing."
        )
    cat_col = next(
        (
            c
            for c in column_profiles
            if c.get("flags", {}).get("looks_like_categorical")
            and c.get("distinct_count", 0) <= 10
        ),
        None,
    )
    if cat_col is not None:
        suggestions.append(
            f"Column `{cat_col['name']}` looks categorical — try compare_groups across it."
        )
    num_col = next((c for c in column_profiles if c.get("numeric")), None)
    if num_col is not None:
        suggestions.append(
            f"Numeric column `{num_col['name']}` is available — "
            "describe_column would surface outliers."
        )
    return suggestions[:3]


def _temporal_describe(con: Any, table: str, quoted: str) -> dict[str, Any]:
    """Bucketed counts by year / month / weekday / hour."""

    def _grouped(expr: str) -> list[dict[str, Any]]:
        rows = con.execute(
            f"""
            SELECT {expr} AS bucket, COUNT(*) AS c
            FROM {table}
            WHERE {quoted} IS NOT NULL
            GROUP BY bucket
            ORDER BY bucket
            """
        ).fetchall()
        return [{"bucket": _json_safe(r[0]), "count": int(r[1])} for r in rows]

    return {
        "by_year": _grouped(f"EXTRACT(year FROM {quoted})"),
        "by_month": _grouped(f"EXTRACT(month FROM {quoted})"),
        "by_weekday": _grouped(f"DAYOFWEEK({quoted})"),
        "by_hour": _grouped(f"EXTRACT(hour FROM {quoted})"),
    }


def _categorical_describe(con: Any, table: str, quoted: str) -> dict[str, Any]:
    """Value-counts (capped at 50 + "other" bucket) and Shannon entropy."""
    import math

    rows = con.execute(
        f"""
        SELECT {quoted} AS value, COUNT(*) AS c
        FROM {table}
        WHERE {quoted} IS NOT NULL
        GROUP BY value
        ORDER BY c DESC, value ASC
        """
    ).fetchall()
    total = sum(int(r[1]) for r in rows)
    entropy = 0.0
    if total > 0:
        for r in rows:
            p = int(r[1]) / total
            if p > 0:
                entropy -= p * math.log2(p)
    counts = [{"value": _json_safe(r[0]), "count": int(r[1])} for r in rows[:50]]
    if len(rows) > 50:
        other_count = sum(int(r[1]) for r in rows[50:])
        counts.append({"value": "<other>", "count": other_count})
    return {"value_counts": counts, "entropy": entropy}


def _outliers(
    con: Any, table: str, quoted: str, *, p25: Any, p75: Any
) -> dict[str, Any]:
    """IQR-rule outliers + z>3 outliers, with up to 5 example raw values."""
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
    examples = [_json_safe(r[0]) for r in example_rows]
    return {"iqr_count": iqr_count, "zscore_count": z_count, "examples": examples}


def _histogram(con: Any, table: str, quoted: str, bins: int) -> dict[str, Any]:
    """Equal-width histogram with ``bins`` bins for one numeric column."""
    range_row = con.execute(
        f"SELECT MIN({quoted}), MAX({quoted}) FROM {table} WHERE {quoted} IS NOT NULL"
    ).fetchone()
    assert range_row is not None
    lo = float(range_row[0])
    hi = float(range_row[1])
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    edges[-1] = hi  # avoid float-drift cutoff
    bucket_rows = con.execute(
        f"""
        SELECT bucket, COUNT(*) FROM (
            SELECT LEAST(CAST(FLOOR(({quoted} - {lo!r}) / {width!r}) AS INTEGER), {bins - 1}) AS bucket
            FROM {table}
            WHERE {quoted} IS NOT NULL
        ) GROUP BY bucket ORDER BY bucket
        """
    ).fetchall()
    counts = [0] * bins
    for bucket, count in bucket_rows:
        idx = max(0, min(int(bucket), bins - 1))
        counts[idx] = int(count)
    return {"bin_edges": edges, "counts": counts}


def _top_values(con: Any, table: str, quoted: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return the top ``limit`` most-frequent values for a column."""
    rows = con.execute(
        f"""
        SELECT {quoted} AS value, COUNT(*) AS c
        FROM {table}
        WHERE {quoted} IS NOT NULL
        GROUP BY value
        ORDER BY c DESC, value ASC
        LIMIT {int(limit)}
        """
    ).fetchall()
    return [{"value": _json_safe(r[0]), "count": int(r[1])} for r in rows]

_TEMPORAL_DTYPES = {"DATE", "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP_NS", "TIMESTAMP_MS",
                    "TIMESTAMP_S", "TIME", "TIMETZ", "DATETIME"}

_WEEKDAY_NAMES = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")


def _is_temporal(dtype: str) -> bool:
    """True if the DuckDB dtype is a date/time-shaped value."""
    base = dtype.split("(")[0].strip().upper()
    return base in _TEMPORAL_DTYPES


def _temporal_stats(con: Any, table: str, quoted: str) -> dict[str, Any]:
    """min/max/range_days/null_count/modal_weekday for a temporal column."""
    row = con.execute(
        f"""
        SELECT
            MIN({quoted}),
            MAX({quoted}),
            DATE_DIFF('day', MIN({quoted}), MAX({quoted})),
            COUNT(*) - COUNT({quoted})
        FROM {table}
        """
    ).fetchone()
    modal_row = con.execute(
        f"""
        SELECT DAYOFWEEK({quoted}) AS wd, COUNT(*) AS c
        FROM {table}
        WHERE {quoted} IS NOT NULL
        GROUP BY wd
        ORDER BY c DESC, wd ASC
        LIMIT 1
        """
    ).fetchone()
    modal_weekday: str | None = None
    if modal_row is not None:
        wd = int(modal_row[0])
        modal_weekday = _WEEKDAY_NAMES[wd]
    assert row is not None
    return {
        "min": str(row[0]) if row[0] is not None else None,
        "max": str(row[1]) if row[1] is not None else None,
        "range_days": int(row[2]) if row[2] is not None else 0,
        "null_count": int(row[3]),
        "modal_weekday": modal_weekday,
    }


def _is_string(dtype: str) -> bool:
    """True if a DuckDB dtype represents a string-shaped value."""
    base = dtype.split("(")[0].strip().upper()
    return base in _STRING_DTYPES


def _string_stats(con: Any, table: str, quoted: str) -> dict[str, Any]:
    """Per-column length stats + empty/whitespace counts."""
    row = con.execute(
        f"""
        SELECT
            MIN(LENGTH({quoted})),
            MAX(LENGTH({quoted})),
            AVG(LENGTH({quoted})),
            COUNT(*) FILTER (WHERE {quoted} = ''),
            COUNT(*) FILTER (WHERE {quoted} IS NOT NULL AND TRIM({quoted}) = '')
        FROM {table}
        """
    ).fetchone()
    assert row is not None  # DuckDB aggregate always returns one row
    keys = ("min_length", "max_length", "mean_length", "empty_count", "whitespace_count")
    return {key: _json_safe(row[i]) for i, key in enumerate(keys)}


def _numeric_stats(con: Any, table: str, quoted: str) -> dict[str, Any]:
    """Return min/max/mean/median/std/p25/p75/p99/zeros/negatives for one column."""
    row = con.execute(
        f"""
        SELECT
            MIN({quoted}),
            MAX({quoted}),
            AVG({quoted}),
            MEDIAN({quoted}),
            STDDEV_SAMP({quoted}),
            QUANTILE_CONT({quoted}, 0.25),
            QUANTILE_CONT({quoted}, 0.75),
            QUANTILE_CONT({quoted}, 0.99),
            COUNT(*) FILTER (WHERE {quoted} = 0),
            COUNT(*) FILTER (WHERE {quoted} < 0)
        FROM {table}
        """
    ).fetchone()
    assert row is not None  # DuckDB aggregate always returns one row
    keys = ("min", "max", "mean", "median", "std", "p25", "p75", "p99", "zeros", "negatives")
    return {key: _json_safe(row[i]) for i, key in enumerate(keys)}


def profile_dataset(payload: ProfileDatasetInput) -> dict[str, Any]:
    """Produce a full EDA profile for the named dataset."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    con = session.get_connection()
    table = _quote(payload.name)

    column_profiles: list[dict[str, Any]] = []
    total_rows = entry.rows
    for col in entry.columns:
        quoted = _quote(col["name"])
        agg_row = con.execute(
            f"""
            SELECT
                COUNT(*) - COUNT({quoted}),
                COUNT(DISTINCT {quoted})
            FROM {table}
            """
        ).fetchone()
        null_count = int(agg_row[0]) if agg_row else 0
        distinct_count = int(agg_row[1]) if agg_row else 0
        null_frac = null_count / total_rows if total_rows else 0.0
        non_null = total_rows - null_count
        distinct_frac = distinct_count / non_null if non_null else 0.0
        dtype_is_string = _is_string(col["dtype"])
        dtype_is_temporal = _is_temporal(col["dtype"])
        flags = {
            "mostly_null": null_frac > 0.5,
            "looks_like_id": (
                dtype_is_string
                and distinct_frac >= 0.95
                and non_null > 100
            ),
            "looks_like_categorical": (
                dtype_is_string
                and distinct_count <= 50
                and non_null > distinct_count
            ),
            "looks_like_timestamp": dtype_is_temporal,
            "high_cardinality": distinct_frac > 0.9 and non_null > 100,
            "constant": distinct_count == 1,
            "mixed_dtype_suspect": False,
        }
        entry_dict: dict[str, Any] = {
            "name": col["name"],
            "dtype": col["dtype"],
            "null_count": null_count,
            "distinct_count": distinct_count,
            "flags": flags,
        }
        if _is_numeric(col["dtype"]):
            entry_dict["numeric"] = _numeric_stats(con, table, quoted)
        elif _is_temporal(col["dtype"]):
            entry_dict["temporal"] = _temporal_stats(con, table, quoted)
        elif _is_string(col["dtype"]):
            entry_dict["string"] = _string_stats(con, table, quoted)
        entry_dict["top_values"] = _top_values(con, table, quoted)
        column_profiles.append(entry_dict)

    head_rows = con.execute(
        f"SELECT * FROM {table} LIMIT {int(payload.sample_rows)}"
    ).fetchall()
    head_cols = [c["name"] for c in entry.columns]
    head = [
        {col: _json_safe(value) for col, value in zip(head_cols, row, strict=True)}
        for row in head_rows
    ]

    suggestions = _build_suggestions(column_profiles)

    md_lines = [f"### Profiled `{payload.name}`"]
    md_lines.append(f"- {entry.rows} rows x {len(entry.columns)} columns")
    flagged = [
        c["name"]
        for c in column_profiles
        if c.get("flags", {}).get("mostly_null")
    ]
    if flagged:
        md_lines.append(f"- Mostly-null columns: {', '.join(flagged)}")
    for s in suggestions[:3]:
        md_lines.append(f"- {s}")
    md = "\n".join(md_lines)
    code = (
        f"profile_df = con.sql(\"SELECT * FROM {payload.name}\").df()\n"
        f"profile_df.describe(include='all')"
    )
    get_recorder().record(markdown=md, code=code, tool_name="profile_dataset")

    return {
        "ok": True,
        "summary": {
            "total_rows": entry.rows,
            "total_columns": len(entry.columns),
        },
        "columns": column_profiles,
        "head": head,
        "suggestions": suggestions,
    }


def list_datasets() -> dict[str, Any]:
    """Return every registered dataset with name, rows, columns, registered_at."""
    items = [
        {
            "name": name,
            "rows": entry.rows,
            "columns": len(entry.columns),
            "registered_at": entry.registered_at.isoformat(),
        }
        for name, entry in session.get_datasets().items()
    ]
    return {"ok": True, "datasets": items}


def load_dataset(payload: LoadDatasetInput) -> dict[str, Any]:
    """Register a file as a DuckDB table in the session."""
    ext = _extension(payload.path)
    if ext not in _SUPPORTED_EXTENSIONS:
        return build_error(
            type="unsupported_format",
            message=f"Extension {ext!r} is not a supported tabular format.",
            hint="Use one of .csv, .tsv, .parquet, .xlsx, .json, .jsonl.",
        )
    is_remote = payload.path.startswith(("s3://", "http://", "https://"))
    if not is_remote and not os.path.exists(payload.path):
        return build_error(
            type="file_not_found",
            message=f"No file at {payload.path!r}.",
            hint="Check the path is absolute or relative to the server's cwd.",
        )

    fmt = _EXT_TO_FORMAT[ext]
    name = payload.name or os.path.splitext(os.path.basename(payload.path))[0]
    con = session.get_connection()
    read_call = _build_read_call(payload.path, fmt)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM {read_call}')

    describe_rows = con.execute(f'DESCRIBE "{name}"').fetchall()
    columns = [{"name": str(row[0]), "dtype": str(row[1])} for row in describe_rows]
    rows = int(con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])  # type: ignore[index]

    session.register(
        name=name,
        path=payload.path,
        read_options=payload.read_options or {},
        format=fmt,
        rows=rows,
        columns=columns,
    )

    md = (
        f"### Loaded dataset `{name}`\n"
        f"- Source: `{payload.path}`\n"
        f"- {rows} rows x {len(columns)} columns"
    )
    code = (
        f'con.execute("""\n'
        f"    CREATE OR REPLACE TABLE {name} AS\n"
        f"    SELECT * FROM {read_call}\n"
        f'""")\n'
        f'{name}_df = con.sql("SELECT * FROM {name}").df()\n'
        f"{name}_df.head()"
    )
    get_recorder().record(markdown=md, code=code, tool_name="load_dataset")

    return {
        "ok": True,
        "name": name,
        "rows": rows,
        "columns": columns,
        "warnings": [],
    }
