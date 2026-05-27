"""Tests for the ``power_analysis`` tool.

Numeric expected values come from statsmodels reference computations
(see the helper notebook commented inline) and are pinned to ≥4 decimal
places per SPEC §3.
"""

from __future__ import annotations

from typing import Any


# === shared ===


def test_power_analysis_zero_unknowns_returns_invalid_inputs(call_tool: Any) -> None:
    """All three of effect_size/n/power provided → invalid_inputs."""
    result = call_tool(
        "power_analysis",
        {
            "test": "two_sample_t",
            "effect_size": 0.5,
            "n": 64,
            "power": 0.8,
        },
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_inputs"


def test_power_analysis_two_unknowns_returns_invalid_inputs(call_tool: Any) -> None:
    """Two of effect_size/n/power omitted → invalid_inputs."""
    result = call_tool(
        "power_analysis",
        {"test": "two_sample_t", "effect_size": 0.5},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_inputs"
