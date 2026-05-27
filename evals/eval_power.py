"""End-to-end eval for ``power_analysis``.

The integration scenario the proposal mandates:

1. Use ``power_analysis`` to compute the per-group sample size required
   to detect a known Cohen's d at 80% power, alpha=0.05.
2. Generate a synthetic dataset of that size with a known planted effect.
3. Run ``compare_groups`` on the synthetic dataset and verify the test
   correctly rejects (i.e., recovers the planted effect).
4. Sanity-check the Cohen's-h worked example from the proposal
   (``p1=0.10, p2=0.12, power=0.8`` → n ≈ 3835 per group).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from conftest import call, mcp_session


@pytest.mark.eval
async def eval_two_sample_t_solve_for_n_then_recover_effect(tmp_path_factory):
    """Power-driven sample size, then compare_groups recovers the planted d=0.5."""
    tmp_path: Path = tmp_path_factory.mktemp("power_eval")
    async with mcp_session() as s:
        r = await call(
            s,
            "power_analysis",
            {
                "test": "two_sample_t",
                "effect_size": 0.5,
                "power": 0.8,
                "alpha": 0.05,
            },
        )
        assert r["ok"], r
        n_per_group = int(np.ceil(r["n"]))  # ≈ 64
        # Build a synthetic CSV with a planted d ≈ 0.5: control N(0,1),
        # treatment N(0.5, 1). The fixed seed makes the eval deterministic.
        rng = np.random.default_rng(seed=12345)
        control = rng.normal(loc=0.0, scale=1.0, size=n_per_group)
        treatment = rng.normal(loc=0.5, scale=1.0, size=n_per_group)
        csv_path = tmp_path / "planted_effect.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["group", "metric"])
            for v in control:
                writer.writerow(["A", float(v)])
            for v in treatment:
                writer.writerow(["B", float(v)])

        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "planted"})
        assert r["ok"], r

        r = await call(
            s,
            "compare_groups",
            {
                "name": "planted",
                "group_column": "group",
                "metric_column": "metric",
            },
        )
        assert r["ok"], r
        # At 80% power we expect rejection most of the time; with this
        # specific fixed seed the planted effect is clearly recovered.
        assert r["p_value"] < 0.05, r


@pytest.mark.eval
async def eval_two_proportion_z_worked_example_matches_proposal():
    """Pin the proposal's worked example: p1=0.10, p2=0.12 → n ≈ 3835/group."""
    async with mcp_session() as s:
        r = await call(
            s,
            "power_analysis",
            {
                "test": "two_proportion_z",
                "p1": 0.10,
                "p2": 0.12,
                "power": 0.8,
                "alpha": 0.05,
            },
        )
        assert r["ok"], r
        # statsmodels: 3834.595739884031. The proposal says "≈ 3800ish".
        assert 3800.0 < r["n"] < 3900.0, r
        assert r["effect_size_metric"] == "cohens_h"
        # n_total = 2 * ceil(n) = 2 * 3835 = 7670
        assert r["n_total"] == 2 * int(np.ceil(r["n"]))


@pytest.mark.eval
async def eval_anova_oneway_known_answer_through_protocol():
    """Cohen's f=0.25, k_groups=4, power=0.8 → total n ≈ 178 across the stdio path."""
    async with mcp_session() as s:
        r = await call(
            s,
            "power_analysis",
            {
                "test": "anova_oneway",
                "effect_size": 0.25,
                "power": 0.8,
                "alpha": 0.05,
                "k_groups": 4,
            },
        )
        assert r["ok"], r
        assert 170.0 < r["n"] < 185.0, r
        assert "alternative" not in r  # F-test → no alternative field.
