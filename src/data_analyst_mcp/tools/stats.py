"""Statistical tools — correlate, compare_groups, test_hypothesis."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

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


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'

logger = logging.getLogger(__name__)


class CorrelateInput(BaseModel):
    """Inputs for ``correlate``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of column names to correlate. When omitted, every "
            "numeric column in the dataset is used."
        ),
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description="Correlation method: pearson, spearman, or kendall.",
    )
    plot: bool = Field(
        default=True,
        description="When true, include a base64-encoded PNG heatmap in the response.",
    )


def correlate(payload: CorrelateInput) -> dict[str, Any]:
    """Compute a correlation matrix across numeric columns."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    available = {c["name"] for c in entry.columns}
    if payload.columns is not None:
        missing = [c for c in payload.columns if c not in available]
        if missing:
            return build_error(
                type="column_not_found",
                message=f"Columns not in dataset {payload.name!r}: {missing}.",
                hint=f"Available: {sorted(available)}",
            )
        chosen = list(payload.columns)
    else:
        chosen = [c["name"] for c in entry.columns if _is_numeric_dtype(c["dtype"])]
        if not chosen:
            return build_error(
                type="no_numeric_columns",
                message=f"Dataset {payload.name!r} has no numeric columns.",
                hint="Pass an explicit `columns` list, or cast columns to numeric first.",
            )

    matrix = _build_corr_matrix(payload.name, chosen, payload.method)

    out: dict[str, Any] = {
        "ok": True,
        "method": payload.method,
        "labels": list(chosen),
        "matrix": matrix,
    }
    if payload.plot:
        out["heatmap_png_base64"] = _render_heatmap_png(chosen, matrix)

    md = (
        f"### Correlation matrix on `{payload.name}` ({payload.method})\n"
        f"- Columns: {', '.join(chosen)}"
    )
    cols_list = ", ".join(f'"{c}"' for c in chosen)
    code = (
        f"from scipy import stats\n"
        f"df = con.sql('SELECT {cols_list} FROM {payload.name}').df()\n"
        f"df.corr(method='{payload.method}')"
    )
    get_recorder().record(markdown=md, code=code, tool_name="correlate")
    return out


def _render_heatmap_png(labels: list[str], matrix: list[list[float]]) -> str:
    """Render the correlation matrix as a base64-encoded PNG heatmap."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data_analyst_mcp.formatting import png_to_base64

    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(labels), 4 + 0.4 * len(labels)))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return png_to_base64(buf.getvalue())


def _pairwise_corr(a: Any, b: Any, method: str) -> float:
    """Return the requested pairwise correlation coefficient as a float."""
    from scipy import stats as _sps

    if method == "spearman":
        return float(_sps.spearmanr(a, b).statistic)
    if method == "kendall":
        return float(_sps.kendalltau(a, b).statistic)
    return float(_sps.pearsonr(a, b).statistic)


def _build_corr_matrix(
    dataset_name: str, columns: list[str], method: str
) -> list[list[float]]:
    """Materialize columns then compute the correlation matrix."""
    con = session.get_connection()
    table = _quote(dataset_name)
    select_cols = ", ".join(_quote(c) for c in columns)
    df = con.execute(f"SELECT {select_cols} FROM {table}").df()
    n = len(columns)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            r = _pairwise_corr(df[columns[i]].to_numpy(), df[columns[j]].to_numpy(), method)
            matrix[i][j] = r
            matrix[j][i] = r
    return matrix
