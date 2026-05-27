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


def test_two_sample_t_solve_for_n_reports_n_total_and_interpretation(call_tool: Any) -> None:
    """Solving for n must echo n_total = ceil(n) + ceil(ratio*n) and an interpretation string."""
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
    # n is the per-group sample (statsmodels nobs1); n_total = ceil(n) + ceil(ratio*n)
    assert "n_total" in result
    assert result["n_total"] == 128  # ceil(63.7656) * 2 = 64 * 2
    assert "interpretation" in result
    assert "0.5" in result["interpretation"]
    assert "80%" in result["interpretation"] or "0.8" in result["interpretation"]


def test_two_sample_t_solves_for_effect_size(call_tool: Any) -> None:
    """Given n=64, alpha=0.05, power=0.8 → effect_size ≈ 0.49907 (MDE).

    Reference: TTestIndPower().solve_power(effect_size=None, nobs1=64,
        alpha=0.05, power=0.8, ratio=1.0, alternative='two-sided')
        → 0.499069177194551
    """
    result = call_tool(
        "power_analysis",
        {"test": "two_sample_t", "n": 64, "power": 0.8, "alpha": 0.05},
    )
    assert result["ok"] is True
    assert result["solved_for"] == "effect_size"
    assert result["effect_size"] == pytest.approx(0.4991, abs=1e-4)
    assert result["n"] == 64
    assert result["power"] == pytest.approx(0.8, abs=1e-12)
    # Interpretation must report the *solved* MDE, not the placeholder.
    assert "0.4991" in result["interpretation"] or "0.499" in result["interpretation"]
    # When solving for effect_size, phrasing should highlight the MDE/detectability.
    assert "detectable" in result["interpretation"].lower()


def test_two_sample_t_solves_for_power(call_tool: Any) -> None:
    """Given effect=0.5, n=64, alpha=0.05 → power ≈ 0.8015.

    Reference: TTestIndPower().solve_power(effect_size=0.5, nobs1=64,
        alpha=0.05, power=None, ratio=1.0, alternative='two-sided')
        → 0.8014595579222542
    """
    result = call_tool(
        "power_analysis",
        {"test": "two_sample_t", "effect_size": 0.5, "n": 64, "alpha": 0.05},
    )
    assert result["ok"] is True
    assert result["solved_for"] == "power"
    assert result["power"] == pytest.approx(0.8015, abs=1e-4)
    # Interpretation should mention achieved power.
    assert "achieved power" in result["interpretation"].lower()


def test_two_sample_t_ratio_respected(call_tool: Any) -> None:
    """Asymmetric ratio=2 (n2 = 2*n1) shrinks n1 vs ratio=1.

    Reference: TTestIndPower().solve_power(effect_size=0.5, alpha=0.05,
        power=0.8, ratio=2.0, alternative='two-sided')
        → 47.741920646056975
    """
    result = call_tool(
        "power_analysis",
        {
            "test": "two_sample_t",
            "effect_size": 0.5,
            "power": 0.8,
            "alpha": 0.05,
            "ratio": 2.0,
        },
    )
    assert result["ok"] is True
    assert result["n"] == pytest.approx(47.7419, abs=1e-4)
    # n_total = ceil(n1) + ceil(2 * n1) = 48 + 96 = 144
    assert result["n_total"] == 144


