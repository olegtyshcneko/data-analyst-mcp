"""Tests for the plot tool — characterization style.

Per spec §3 "Characterization tests, not pure TDD": pixel-perfect TDD on
matplotlib is wasted effort. Each plot kind asserts:
  1. ``result["ok"] is True``
  2. PNG magic-byte prefix on ``base64.b64decode(result["png_base64"])``
  3. ``len(decoded_bytes) >= 5000`` (non-trivial image)
  4. ``result["width"]`` and ``result["height"]`` are positive ints

Error paths get strict red/green TDD — they are behavior, not visual.
"""

from __future__ import annotations

import base64

import pandas as pd

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(result: dict) -> bytes:
    """Shared characterization assertion: ok + PNG header + size + dims."""
    assert result["ok"] is True, result
    raw = base64.b64decode(result["png_base64"])
    assert raw[:8] == _PNG_MAGIC
    assert len(raw) >= 5000
    assert isinstance(result["width"], int) and result["width"] > 0
    assert isinstance(result["height"], int) and result["height"] > 0
    return raw


def test_plot_unknown_dataset_returns_not_found(call_tool):
    result = call_tool("plot", {"name": "nope", "kind": "hist", "x": "x"})
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_plot_invalid_kind_returns_invalid_kind(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"x": [1, 2, 3]}))
    result = call_tool("plot", {"name": "tiny", "kind": "pie", "x": "x"})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_kind"


def test_plot_missing_column_returns_column_not_found(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"x": [1, 2, 3]}))
    result = call_tool("plot", {"name": "tiny", "kind": "hist", "x": "nope"})
    assert result["ok"] is False
    assert result["error"]["type"] == "column_not_found"


def test_plot_scatter_without_y_returns_missing_required_param(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    result = call_tool("plot", {"name": "tiny", "kind": "scatter", "x": "a"})
    assert result["ok"] is False
    assert result["error"]["type"] == "missing_required_param"


# === per-kind characterization ===


_NUMERIC_DF = pd.DataFrame(
    {
        "x": list(range(100)),
        "y": [i * 0.5 + 1.0 for i in range(100)],
        "g": ["A", "B"] * 50,
    }
)


def test_plot_hist_returns_valid_png(call_tool, load_df_into_session):
    load_df_into_session("d", _NUMERIC_DF)
    result = call_tool("plot", {"name": "d", "kind": "hist", "x": "x"})
    _assert_valid_png(result)
