"""Tests for ``pairwise_comparisons`` — post-hoc pairwise comparisons.

The TDD slices in ``docs/proposals/pairwise_comparisons.md`` map one-to-one
to the test functions below. This module covers slices 1–9, the validation
surface (proposal "Behavior" steps 1–8 and the "Errors" table): dataset /
column / dtype / alpha checks, label resolution (duplicate + missing-label
rejection, the 3–20 group bounds), and the ``method="tukey"`` +
``p_adjust`` conflict. The Tukey / Dunn engines (slices 10+) land in a
later task; until then a passing validation path returns an ``internal``
stub, so error-type assertions here never rely on ``ok is True``.
"""

from __future__ import annotations

# === slice 1: pairwise_comparisons returns not_found for unregistered dataset ===


def test_slice01_not_found_for_unregistered_dataset(call_tool):
    result = call_tool(
        "pairwise_comparisons",
        {"name": "nope", "group_column": "grp", "metric_column": "val"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"
