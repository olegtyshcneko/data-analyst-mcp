"""Tests for the ``power_analysis`` tool.

Numeric expected values come from statsmodels reference computations
(see the helper notebook commented inline) and are pinned to ≥4 decimal
places per SPEC §3.
"""

from __future__ import annotations

from typing import Any

import pytest


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


# === two_sample_t ===


def test_two_sample_t_known_answer_solves_n(call_tool: Any) -> None:
    """Cohen's d=0.5, alpha=0.05, power=0.8, two-sided → n_1 ≈ 63.7656.

    Reference (statsmodels):
        from statsmodels.stats.power import TTestIndPower
        TTestIndPower().solve_power(effect_size=0.5, alpha=0.05,
            power=0.8, ratio=1.0, alternative='two-sided')
        # → 63.76561058891169
    """
    result = call_tool(
        "power_analysis",
        {
            "test": "two_sample_t",
            "effect_size": 0.5,
            "power": 0.8,
            "alpha": 0.05,
        },
    )
    assert result["ok"] is True
    assert result["test"] == "two_sample_t"
    assert result["solved_for"] == "n"
    assert result["n"] == pytest.approx(63.7656, abs=1e-4)
    assert result["effect_size_metric"] == "cohens_d"
