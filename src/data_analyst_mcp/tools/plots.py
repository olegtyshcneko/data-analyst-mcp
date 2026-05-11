"""Plot tool — render matplotlib charts as base64-encoded PNGs.

Uses the object-oriented matplotlib API (``Figure`` + ``FigureCanvasAgg``)
instead of ``pyplot`` because pyplot keeps global state that is fragile
inside the MCP stdio server loop.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


PlotKind = Literal["hist", "bar", "line", "scatter", "box", "violin", "heatmap"]


class PlotInput(BaseModel):
    """Inputs for ``plot``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    kind: PlotKind = Field(
        ...,
        description=(
            "Chart kind: hist, bar, line, scatter, box, violin, heatmap. "
            "Each kind has its own required columns — see the x/y/hue fields."
        ),
    )
    x: str | None = Field(default=None, description="Column for the x-axis.")
    y: str | None = Field(default=None, description="Column for the y-axis.")
    hue: str | None = Field(
        default=None,
        description="Optional column used to color-group multi-series plots.",
    )
    title: str | None = Field(default=None, description="Optional chart title.")
    bins: int | None = Field(default=None, description="Histogram bin count.")


def plot(payload: PlotInput) -> dict[str, Any]:
    """Render a chart to a base64-encoded PNG."""
    result = _plot_impl(payload)
    _record_plot(payload, result)
    return result


def _plot_impl(payload: PlotInput) -> dict[str, Any]:
    """Validate inputs and dispatch to the kind-specific renderer."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    available = {c["name"] for c in entry.columns}
    missing = _missing_required_params(payload.kind, x=payload.x, y=payload.y)
    if missing:
        return build_error(
            type="missing_required_param",
            message=f"Plot kind {payload.kind!r} requires parameter(s): {missing}.",
            hint=f"Pass {missing} as well as `name` and `kind`.",
        )
    for label, col in (("x", payload.x), ("y", payload.y), ("hue", payload.hue)):
        if col is not None and col not in available:
            return build_error(
                type="column_not_found",
                message=f"Column {col!r} ({label}) is not in dataset {payload.name!r}.",
                hint=f"Available columns: {sorted(available)}",
            )
    if payload.kind == "hist":
        return _plot_hist(payload)
    if payload.kind == "bar":
        return _plot_bar(payload)
    if payload.kind == "line":
        return _plot_line(payload)
    if payload.kind == "scatter":
        return _plot_scatter(payload)
    if payload.kind == "box":
        return _plot_box(payload)
    if payload.kind == "violin":
        return _plot_violin(payload)
    # heatmap (Literal validation in PlotInput guarantees no other kinds reach here)
    return _plot_heatmap_tool(payload)


def _record_plot(payload: PlotInput, result: dict[str, Any]) -> None:
    """Append a markdown+code cell pair describing the plot call (on success)."""
    if not result.get("ok"):
        return
    md = _plot_markdown(payload)
    code = _plot_code(payload)
    get_recorder().record(markdown=md, code=code, tool_name="plot")


def _plot_markdown(payload: PlotInput) -> str:
    """Human-readable description of the chart for the notebook recorder."""
    bits = [f"### Plotted `{payload.kind}` of dataset `{payload.name}`"]
    if payload.x is not None:
        bits.append(f"- x: `{payload.x}`")
    if payload.y is not None:
        bits.append(f"- y: `{payload.y}`")
    if payload.hue is not None:
        bits.append(f"- hue: `{payload.hue}`")
    if payload.title is not None:
        bits.append(f"- title: {payload.title!r}")
    if payload.bins is not None:
        bits.append(f"- bins: {payload.bins}")
    return "\n".join(bits)


def _plot_code(payload: PlotInput) -> str:
    """Render a reproducible matplotlib snippet for the given payload."""
    table = payload.name
    cols: list[str] = [c for c in (payload.x, payload.y, payload.hue) if c is not None]
    select_cols = ", ".join(cols) if cols else "*"
    base = (
        "import matplotlib.pyplot as plt\n"
        f'_df = con.sql("SELECT {select_cols} FROM {table}").df()\n'
        "fig, ax = plt.subplots(figsize=(8, 6), dpi=100)\n"
    )
    title_line = f"ax.set_title({payload.title!r})\n" if payload.title is not None else ""
    bins = payload.bins if payload.bins is not None else 20
    if payload.kind == "hist":
        body = f"ax.hist(_df['{payload.x}'], bins={bins})\nax.set_xlabel('{payload.x}')\n"
    elif payload.kind == "bar":
        if payload.y is None:
            body = (
                f"_counts = _df.groupby('{payload.x}').size()\n"
                "ax.bar(_counts.index.astype(str), _counts.values)\n"
                f"ax.set_xlabel('{payload.x}')\n"
            )
        else:
            body = (
                f"_means = _df.groupby('{payload.x}')['{payload.y}'].mean()\n"
                "ax.bar(_means.index.astype(str), _means.values)\n"
                f"ax.set_xlabel('{payload.x}')\nax.set_ylabel('avg({payload.y})')\n"
            )
    elif payload.kind == "line":
        body = (
            f"_sorted = _df.sort_values('{payload.x}')\n"
            f"ax.plot(_sorted['{payload.x}'], _sorted['{payload.y}'])\n"
            f"ax.set_xlabel('{payload.x}')\nax.set_ylabel('{payload.y}')\n"
        )
    elif payload.kind == "scatter":
        if payload.hue is None:
            body = (
                f"ax.scatter(_df['{payload.x}'], _df['{payload.y}'])\n"
                f"ax.set_xlabel('{payload.x}')\nax.set_ylabel('{payload.y}')\n"
            )
        else:
            body = (
                f"for _lab, _sub in _df.groupby('{payload.hue}'):\n"
                f"    ax.scatter(_sub['{payload.x}'], _sub['{payload.y}'], label=str(_lab))\n"
                f"ax.legend(title='{payload.hue}')\n"
                f"ax.set_xlabel('{payload.x}')\nax.set_ylabel('{payload.y}')\n"
            )
    elif payload.kind == "box":
        if payload.x is None:
            body = (
                f"ax.boxplot([_df['{payload.y}']], tick_labels=['{payload.y}'])\n"
                f"ax.set_ylabel('{payload.y}')\n"
            )
        else:
            body = (
                f"_groups = _df.groupby('{payload.x}')['{payload.y}'].apply(list)\n"
                "ax.boxplot(_groups.values, tick_labels=_groups.index.astype(str).tolist())\n"
                f"ax.set_xlabel('{payload.x}')\nax.set_ylabel('{payload.y}')\n"
            )
    elif payload.kind == "violin":
        if payload.x is None:
            body = (
                f"ax.violinplot([_df['{payload.y}']])\n"
                f"ax.set_xticks([1])\nax.set_xticklabels(['{payload.y}'])\n"
                f"ax.set_ylabel('{payload.y}')\n"
            )
        else:
            body = (
                f"_groups = _df.groupby('{payload.x}')['{payload.y}'].apply(list)\n"
                "ax.violinplot(_groups.values)\n"
                "ax.set_xticks(range(1, len(_groups) + 1))\n"
                "ax.set_xticklabels(_groups.index.astype(str).tolist())\n"
                f"ax.set_xlabel('{payload.x}')\nax.set_ylabel('{payload.y}')\n"
            )
    else:  # heatmap
        body = (
            f"_corr = con.sql('SELECT * FROM {table}').df().corr(numeric_only=True)\n"
            "im = ax.imshow(_corr.values, vmin=-1, vmax=1, cmap='RdBu_r')\n"
            "ax.set_xticks(range(len(_corr.columns)))\n"
            "ax.set_yticks(range(len(_corr.columns)))\n"
            "ax.set_xticklabels(_corr.columns, rotation=45, ha='right')\n"
            "ax.set_yticklabels(_corr.columns)\nfig.colorbar(im, ax=ax)\n"
        )
    return base + body + title_line + "plt.show()"


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _fetch_column(table: str, column: str) -> Any:
    """Materialize a single column as a 1-D numpy array via DuckDB."""
    con = session.get_connection()
    df: Any = con.execute(f"SELECT {_quote(column)} FROM {_quote(table)}").df()
    return df[column].to_numpy()


_FIGSIZE: tuple[float, float] = (8.0, 6.0)
_DPI: int = 100


def _apply_style(fig: Any, ax: Any) -> None:
    """Apply the project's consistent matplotlib style to a Figure + Axes.

    Stock matplotlib only — no ``seaborn`` import. Sets a white face color,
    a light-grey major grid (under the data), spine de-emphasis, and a
    sans-serif tick font. Called by every kind so all rendered images look
    like they came out of the same notebook.
    """
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(visible=True, which="major", linestyle="-", linewidth=0.6, color="#dddddd")
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#888888")
    ax.tick_params(colors="#444444", labelsize=9)


def _apply_title(ax: Any, title: str | None) -> None:
    """Set an axis title when one was provided; no-op otherwise."""
    if title is not None:
        ax.set_title(title)


def _make_figure() -> tuple[Any, Any]:
    """Construct a fresh ``Figure`` + ``Axes`` with the project's standard style."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(111)
    _apply_style(fig, ax)
    return fig, ax


def render_to_base64(fig: Any) -> dict[str, Any]:
    """Render ``fig`` to a PNG and return the ``{ok, png_base64, width, height}`` envelope.

    Uses the object-oriented ``FigureCanvasAgg`` so we never touch
    ``pyplot``'s global state. The width and height are measured after
    ``bbox_inches="tight"`` runs so they reflect what the client actually
    receives.
    """
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    from data_analyst_mcp.formatting import png_to_base64

    canvas = FigureCanvasAgg(fig)
    _ = canvas  # keep the canvas alive for savefig
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    png_bytes = buf.getvalue()
    with Image.open(io.BytesIO(png_bytes)) as im:
        w, h = im.size
    return {
        "ok": True,
        "png_base64": png_to_base64(png_bytes),
        "width": int(w),
        "height": int(h),
    }


def _grouped_arrays(name: str, group_col: str, value_col: str) -> tuple[list[str], list[Any]]:
    """Materialize per-group value arrays sorted by group label."""
    con = session.get_connection()
    table = _quote(name)
    g_q = _quote(group_col)
    label_rows = con.execute(
        f"SELECT DISTINCT {g_q} FROM {table} WHERE {g_q} IS NOT NULL ORDER BY {g_q}"
    ).fetchall()
    labels = [str(r[0]) for r in label_rows]
    arrays: list[Any] = []
    for lab in labels:
        df: Any = con.execute(
            f"SELECT {_quote(value_col)} FROM {table} WHERE {g_q} = ?", [lab]
        ).df()
        arrays.append(df[value_col].to_numpy())
    return labels, arrays


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
    """True when a DuckDB dtype represents a numeric column."""
    base = dtype.split("(")[0].strip().upper()
    return base in _NUMERIC_DTYPES


def _plot_heatmap_tool(payload: PlotInput) -> dict[str, Any]:
    """Render a correlation-matrix heatmap across every numeric column."""
    entries = session.get_datasets()
    entry = entries[payload.name]
    chosen = [c["name"] for c in entry.columns if _is_numeric_dtype(c["dtype"])]
    if not chosen:
        return build_error(
            type="no_numeric_columns",
            message=f"Dataset {payload.name!r} has no numeric columns.",
            hint="Heatmap visualises a correlation matrix — add at least one numeric column.",
        )
    from data_analyst_mcp.tools.stats import _build_corr_matrix  # type: ignore[reportPrivateUsage]

    matrix = _build_corr_matrix(payload.name, chosen, "pearson")
    fig = build_heatmap_figure(chosen, matrix)
    _apply_title(fig.axes[0], payload.title)
    return render_to_base64(fig)


def build_heatmap_figure(labels: list[str], matrix: list[list[float]]) -> Any:
    """Construct a correlation-heatmap figure (shared between ``plot`` and ``correlate``)."""
    fig, ax = _make_figure()
    # heatmaps don't want the data grid behind cells
    ax.grid(visible=False)
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _plot_violin(payload: PlotInput) -> dict[str, Any]:
    """Violin plot of ``y``. With ``x`` it groups violins by category."""
    assert payload.y is not None
    fig, ax = _make_figure()
    if payload.x is None:
        values = _fetch_column(payload.name, payload.y)
        ax.violinplot([values], showmeans=True, showmedians=False)
        ax.set_xticks([1])
        ax.set_xticklabels([payload.y])
    else:
        labels, arrays = _grouped_arrays(payload.name, payload.x, payload.y)
        ax.violinplot(arrays, showmeans=True, showmedians=False)
        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels)
        ax.set_xlabel(payload.x)
    ax.set_ylabel(payload.y)
    _apply_title(ax, payload.title)
    return render_to_base64(fig)


def _plot_box(payload: PlotInput) -> dict[str, Any]:
    """Box plot of ``y``. With ``x`` it groups boxes by category."""
    assert payload.y is not None
    fig, ax = _make_figure()
    if payload.x is None:
        values = _fetch_column(payload.name, payload.y)
        ax.boxplot([values], tick_labels=[payload.y])
    else:
        labels, arrays = _grouped_arrays(payload.name, payload.x, payload.y)
        ax.boxplot(arrays, tick_labels=labels)
        ax.set_xlabel(payload.x)
    ax.set_ylabel(payload.y)
    _apply_title(ax, payload.title)
    return render_to_base64(fig)


def _plot_scatter(payload: PlotInput) -> dict[str, Any]:
    """Scatter of ``y`` against ``x``. ``hue`` colors points by group."""
    assert payload.x is not None and payload.y is not None
    con = session.get_connection()
    cols = [payload.x, payload.y] + ([payload.hue] if payload.hue is not None else [])
    select = ", ".join(_quote(c) for c in cols)
    df: Any = con.execute(f"SELECT {select} FROM {_quote(payload.name)}").df()
    fig, ax = _make_figure()
    if payload.hue is None:
        ax.scatter(df[payload.x].to_numpy(), df[payload.y].to_numpy(), s=18, alpha=0.75)
    else:
        for label, sub in df.groupby(payload.hue):
            ax.scatter(
                sub[payload.x].to_numpy(),
                sub[payload.y].to_numpy(),
                s=18,
                alpha=0.75,
                label=str(label),
            )
        ax.legend(title=payload.hue, frameon=False)
    ax.set_xlabel(payload.x)
    ax.set_ylabel(payload.y)
    _apply_title(ax, payload.title)
    return render_to_base64(fig)


def _plot_line(payload: PlotInput) -> dict[str, Any]:
    """Line plot of ``y`` against ``x``, sorted by x."""
    assert payload.x is not None and payload.y is not None
    con = session.get_connection()
    df: Any = con.execute(
        f"SELECT {_quote(payload.x)}, {_quote(payload.y)} FROM {_quote(payload.name)} "
        f"ORDER BY {_quote(payload.x)}"
    ).df()
    fig, ax = _make_figure()
    ax.plot(df[payload.x].to_numpy(), df[payload.y].to_numpy())
    ax.set_xlabel(payload.x)
    ax.set_ylabel(payload.y)
    _apply_title(ax, payload.title)
    return render_to_base64(fig)


def _plot_bar(payload: PlotInput) -> dict[str, Any]:
    """Bar chart: count rows per ``payload.x`` (or aggregate ``y`` by mean)."""
    assert payload.x is not None
    con = session.get_connection()
    table = _quote(payload.name)
    x_q = _quote(payload.x)
    if payload.y is None:
        rows = con.execute(
            f"SELECT {x_q} AS k, COUNT(*) AS v FROM {table} "
            f"WHERE {x_q} IS NOT NULL GROUP BY {x_q} ORDER BY {x_q}"
        ).fetchall()
        y_label = "count"
    else:
        y_q = _quote(payload.y)
        rows = con.execute(
            f"SELECT {x_q} AS k, AVG({y_q}) AS v FROM {table} "
            f"WHERE {x_q} IS NOT NULL GROUP BY {x_q} ORDER BY {x_q}"
        ).fetchall()
        y_label = f"avg({payload.y})"
    labels = [str(r[0]) for r in rows]
    heights = [float(r[1]) for r in rows]
    fig, ax = _make_figure()
    ax.bar(labels, heights)
    ax.set_xlabel(payload.x)
    ax.set_ylabel(y_label)
    _apply_title(ax, payload.title)
    return render_to_base64(fig)


def _plot_hist(payload: PlotInput) -> dict[str, Any]:
    """Histogram of ``payload.x`` (numeric) with optional ``bins``."""
    assert payload.x is not None  # guarded by _missing_required_params
    values = _fetch_column(payload.name, payload.x)
    fig, ax = _make_figure()
    bins = payload.bins if payload.bins is not None else 20
    ax.hist(values, bins=bins)
    ax.set_xlabel(payload.x)
    ax.set_ylabel("count")
    _apply_title(ax, payload.title)
    return render_to_base64(fig)


_REQUIRES_X: frozenset[str] = frozenset({"hist", "bar", "line", "scatter"})
_REQUIRES_Y: frozenset[str] = frozenset({"line", "scatter", "box", "violin"})


def _missing_required_params(kind: str, *, x: str | None, y: str | None) -> list[str]:
    """Return the list of required-but-missing param names for ``kind``."""
    missing: list[str] = []
    if kind in _REQUIRES_X and x is None:
        missing.append("x")
    if kind in _REQUIRES_Y and y is None:
        missing.append("y")
    return missing
