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
    return {"ok": True, "png_base64": "", "width": 0, "height": 0}


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


def _make_figure() -> tuple[Any, Any]:
    """Construct a fresh ``Figure`` + ``Axes`` with the project's standard style."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(111)
    _apply_style(fig, ax)
    return fig, ax


def _render_to_base64(fig: Any) -> dict[str, Any]:
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
    if payload.title is not None:
        ax.set_title(payload.title)
    return _render_to_base64(fig)


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
    if payload.title is not None:
        ax.set_title(payload.title)
    return _render_to_base64(fig)


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
    if payload.title is not None:
        ax.set_title(payload.title)
    return _render_to_base64(fig)


def _plot_hist(payload: PlotInput) -> dict[str, Any]:
    """Histogram of ``payload.x`` (numeric) with optional ``bins``."""
    assert payload.x is not None  # guarded by _missing_required_params
    values = _fetch_column(payload.name, payload.x)
    fig, ax = _make_figure()
    bins = payload.bins if payload.bins is not None else 20
    ax.hist(values, bins=bins)
    ax.set_xlabel(payload.x)
    ax.set_ylabel("count")
    if payload.title is not None:
        ax.set_title(payload.title)
    return _render_to_base64(fig)


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
