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
    return {"ok": True, "png_base64": "", "width": 0, "height": 0}


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _fetch_column(table: str, column: str) -> Any:
    """Materialize a single column as a 1-D numpy array via DuckDB."""
    con = session.get_connection()
    df: Any = con.execute(f"SELECT {_quote(column)} FROM {_quote(table)}").df()
    return df[column].to_numpy()


def _make_figure() -> Any:
    """Construct a fresh ``Figure`` with the project's standard size + DPI."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 6), dpi=100)
    fig.set_facecolor("white")
    return fig


def _encode_figure(fig: Any) -> dict[str, Any]:
    """Render ``fig`` to a PNG and return the ``{ok, png_base64, width, height}`` envelope."""
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
    fig = _make_figure()
    ax = fig.add_subplot(111)
    ax.bar(labels, heights)
    ax.set_xlabel(payload.x)
    ax.set_ylabel(y_label)
    if payload.title is not None:
        ax.set_title(payload.title)
    return _encode_figure(fig)


def _plot_hist(payload: PlotInput) -> dict[str, Any]:
    """Histogram of ``payload.x`` (numeric) with optional ``bins``."""
    assert payload.x is not None  # guarded by _missing_required_params
    values = _fetch_column(payload.name, payload.x)
    fig = _make_figure()
    ax = fig.add_subplot(111)
    bins = payload.bins if payload.bins is not None else 20
    ax.hist(values, bins=bins)
    ax.set_xlabel(payload.x)
    ax.set_ylabel("count")
    if payload.title is not None:
        ax.set_title(payload.title)
    return _encode_figure(fig)


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
