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


# === slice 2: pairwise_comparisons returns column_not_found for missing group or metric column ===


def test_slice02_column_not_found_for_missing_column(call_tool, load_df_into_session):
    import pandas as pd

    df = pd.DataFrame({"grp": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
    load_df_into_session("ds", df)

    missing_group = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "nope", "metric_column": "val"},
    )
    assert missing_group["ok"] is False
    assert missing_group["error"]["type"] == "column_not_found"

    missing_metric = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "nope"},
    )
    assert missing_metric["ok"] is False
    assert missing_metric["error"]["type"] == "column_not_found"


# === slice 3: pairwise_comparisons returns metric_not_numeric for a VARCHAR metric column ===


def test_slice03_metric_not_numeric_for_varchar_metric(call_tool, load_df_into_session):
    import pandas as pd

    df = pd.DataFrame({"grp": ["A", "B", "C"], "val": ["x", "y", "z"]})
    load_df_into_session("ds", df)

    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "metric_not_numeric"


# === slice 4: pairwise_comparisons returns too_few_groups below three groups and hints at compare_groups ===


def test_slice04_too_few_groups_hints_compare_groups(call_tool, load_df_into_session):
    import pandas as pd

    df = pd.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0]})
    load_df_into_session("ds", df)

    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "too_few_groups"
    assert "compare_groups" in result["error"]["hint"]


# === slice 5: pairwise_comparisons returns invalid_alpha outside the open unit interval ===


def _three_group_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B", "C", "C"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_slice05_alpha_zero_returns_invalid_alpha(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())
    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "alpha": 0.0},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_alpha"


def test_slice05_alpha_one_returns_invalid_alpha(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())
    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "alpha": 1.0},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_alpha"


# === slice 6: pairwise_comparisons returns duplicate_groups for repeated labels in groups ===


def test_slice06_duplicate_groups_for_repeated_labels(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())
    result = call_tool(
        "pairwise_comparisons",
        {
            "name": "ds",
            "group_column": "grp",
            "metric_column": "val",
            "groups": ["A", "A", "B"],
        },
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "duplicate_groups"
