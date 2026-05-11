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

import pandas as pd


def test_plot_unknown_dataset_returns_not_found(call_tool):
    result = call_tool("plot", {"name": "nope", "kind": "hist", "x": "x"})
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"
