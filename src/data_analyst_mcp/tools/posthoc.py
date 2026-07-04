"""Post-hoc pairwise comparisons — ``pairwise_comparisons``.

The follow-up to ``compare_groups``: once the omnibus test (one-way ANOVA
or Kruskal–Wallis) reports that groups differ, this tool answers *which
pairs differ* — Tukey HSD after ANOVA, Dunn's test (tie-corrected) after
Kruskal–Wallis, gated by the same Shapiro normality auto-selection.

This module (task T2) implements the validation surface only; the Tukey /
Dunn engines land in a later task. A payload that clears every validation
returns a deterministic ``internal`` stub until then.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.tools.stats import (
    _all_labels,  # type: ignore[reportPrivateUsage]
    _is_numeric_dtype,  # type: ignore[reportPrivateUsage]
    _quote,  # type: ignore[reportPrivateUsage]
)

logger = logging.getLogger(__name__)

_MAX_GROUPS = 20


def _materialize_group_nonnull(name: str, group_col: str, metric_col: str, label: str) -> Any:
    """Metric array for ``group_col == label`` rows whose metric is non-null.

    Deliberately diverges from ``stats._materialize_group`` by adding
    ``AND <metric> IS NOT NULL``: Tukey and Dunn both require complete
    numeric vectors, and a silent NaN would corrupt the rank pooling. This
    divergence is documented in ``docs/proposals/pairwise_comparisons.md``
    (behavior step 6) so it is not "fixed" back to match the shared helper.
    """
    con = session.get_connection()
    table = _quote(name)
    rel = con.execute(
        f"SELECT {_quote(metric_col)} FROM {table} "
        f"WHERE {_quote(group_col)} = ? AND {_quote(metric_col)} IS NOT NULL",
        [label],
    )
    df: Any = rel.df()
    return df[metric_col].to_numpy()


class PairwiseComparisonsInput(BaseModel):
    """Inputs for ``pairwise_comparisons``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    group_column: str = Field(..., description="Column holding the group labels.")
    metric_column: str = Field(..., description="Numeric metric column to compare across groups.")
    groups: list[str] | None = Field(
        default=None,
        description=(
            "Subset of group labels to compare (>=3 labels, no duplicates). "
            "When omitted, every distinct label in group_column is used."
        ),
    )
    method: Literal["auto", "tukey", "dunn"] = Field(
        default="auto",
        description=(
            "Post-hoc engine. 'auto' mirrors compare_groups' Shapiro gate "
            "(normality holds -> Tukey HSD, violated -> Dunn's test); "
            "'tukey' / 'dunn' force the engine."
        ),
    )
    p_adjust: Literal["holm", "bonferroni", "sidak", "bh", "by"] | None = Field(
        default=None,
        description=(
            "Family-wise correction applied to Dunn's raw p-values only. "
            "Tukey controls FWER internally, so a correction is meaningless "
            "there — an explicit p_adjust with method='tukey' is rejected. "
            "When Dunn runs and p_adjust is None it resolves to 'holm'."
        ),
    )
    alpha: float = Field(
        default=0.05,
        description="Significance threshold in the open interval (0, 1).",
    )


def pairwise_comparisons(payload: PairwiseComparisonsInput) -> dict[str, Any]:
    """Post-hoc pairwise comparisons after a significant omnibus test.

    Thin entry point: validation + dispatch live in the impl, recording is
    attached at this wrapper after a successful result (T5).
    """
    return _pairwise_comparisons_impl(payload)


def _pairwise_comparisons_impl(payload: PairwiseComparisonsInput) -> dict[str, Any]:
    """Validate the request, then dispatch to the engine (engines land in T3)."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )

    entry = entries[payload.name]
    available = {c["name"]: c["dtype"] for c in entry.columns}
    for col in (payload.group_column, payload.metric_column):
        if col not in available:
            return build_error(
                type="column_not_found",
                message=f"Column {col!r} is not in dataset {payload.name!r}.",
                hint=f"Available columns: {', '.join(sorted(available))}",
            )

    metric_dtype = available[payload.metric_column]
    if not _is_numeric_dtype(metric_dtype):
        return build_error(
            type="metric_not_numeric",
            message=(
                f"Metric column {payload.metric_column!r} has non-numeric dtype {metric_dtype!r}."
            ),
            hint="Tukey and Dunn need a numeric metric; pick a numeric column.",
        )

    if not (0.0 < payload.alpha < 1.0):
        return build_error(
            type="invalid_alpha",
            message=f"alpha must be in (0, 1); got {payload.alpha}.",
            hint="Use 0.05 (default), 0.01, or any value strictly between 0 and 1.",
        )

    if payload.groups is not None:
        labels = list(payload.groups)
        seen: set[str] = set()
        duplicates: list[str] = []
        for lab in labels:
            if lab in seen:
                duplicates.append(lab)
            seen.add(lab)
        if duplicates:
            return build_error(
                type="duplicate_groups",
                message=f"Duplicate labels in groups: {duplicates}.",
                hint="List each group label at most once in `groups`.",
            )
    else:
        labels = _all_labels(payload.name, payload.group_column)
    if len(labels) < 3:
        return build_error(
            type="too_few_groups",
            message=f"Need at least 3 groups; resolved {len(labels)}.",
            hint="Use compare_groups for a two-group comparison.",
        )
    if len(labels) > _MAX_GROUPS:
        return build_error(
            type="too_many_groups",
            message=f"Resolved {len(labels)} groups; the cap is {_MAX_GROUPS}.",
            hint=(
                "Pass a `groups` subset of at most 20 labels — the cap bounds "
                "the quadratic n·(n−1)/2 comparison output."
            ),
        )

    arrays: list[Any] = []
    for lab in labels:
        arr = _materialize_group_nonnull(
            payload.name, payload.group_column, payload.metric_column, lab
        )
        if len(arr) == 0:
            return build_error(
                type="group_not_found",
                message=f"Group label {lab!r} matched no rows in {payload.name!r}.",
                hint="Check the label spelling, or omit `groups` to use every label.",
            )
        arrays.append(arr)

    # All validations passed — the Tukey / Dunn engines land in T3.
    return build_error(type="internal", message="pairwise engines land in T3")
