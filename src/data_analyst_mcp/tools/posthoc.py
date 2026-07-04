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

import itertools
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.tools.stats import (
    _all_labels,  # type: ignore[reportPrivateUsage]
    _is_numeric_dtype,  # type: ignore[reportPrivateUsage]
    _levene_p,  # type: ignore[reportPrivateUsage]
    _quote,  # type: ignore[reportPrivateUsage]
    _scipy_stats,  # type: ignore[reportPrivateUsage]
    _shapiro_p,  # type: ignore[reportPrivateUsage]
)

logger = logging.getLogger(__name__)

_MAX_GROUPS = 20


def _sm_multicomp() -> Any:
    """Return ``statsmodels.stats.multicomp`` as an untyped module.

    Mirrors ``multitest._sm_multitest``: an ``Any`` annotation keeps strict
    pyright quiet without an inline ignore at every ``pairwise_tukeyhsd``
    call site.
    """
    import statsmodels.stats.multicomp as _mc  # type: ignore[reportMissingTypeStubs]

    return _mc


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

    if payload.method == "tukey" and payload.p_adjust is not None:
        return build_error(
            type="p_adjust_not_applicable",
            message=(
                "method='tukey' controls the family-wise error rate internally, "
                "so an explicit p_adjust does not apply."
            ),
            hint="Drop p_adjust for Tukey, or use method='dunn' to apply a correction.",
        )

    # Sort resolved labels ascending before materialization and pairing, so
    # pairs enumerate in itertools.combinations order (which matches
    # statsmodels' own Tukey table order) even when `groups` was unsorted.
    labels = sorted(labels)
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
        if len(arr) < 2:
            # Guard BEFORE any scipy/statsmodels call: Tukey needs within-group
            # variance and Dunn's rank pooling degenerates on a singleton group.
            return build_error(
                type="insufficient_group_size",
                message=(
                    f"Group {lab!r} has only {len(arr)} non-null row(s); at least 2 are required."
                ),
                hint="Every group needs >=2 non-null metric values to compare.",
            )
        arrays.append(arr)

    if payload.method == "tukey":
        return _run_tukey(labels, arrays, payload)
    # Dunn and the auto Shapiro gate land in later slices.
    return build_error(type="internal", message="auto method resolution lands in T4")


def _run_tukey(
    labels: list[str], arrays: list[Any], payload: PairwiseComparisonsInput
) -> dict[str, Any]:
    """Tukey HSD engine: statsmodels ``pairwise_tukeyhsd`` behind ``_sm_multicomp``.

    Tukey controls the family-wise error rate internally via the
    studentized-range distribution, so each row's ``statistic`` and ``p_raw``
    are null and ``p_adjust`` is echoed null. The omnibus is one-way ANOVA.
    """
    import numpy as np

    endog = np.concatenate(arrays)
    group_labels: list[str] = []
    for lab, arr in zip(labels, arrays, strict=True):
        group_labels.extend([lab] * len(arr))

    res = _sm_multicomp().pairwise_tukeyhsd(endog, group_labels, payload.alpha)
    n_by = {lab: len(arr) for lab, arr in zip(labels, arrays, strict=True)}

    comparisons: list[dict[str, Any]] = []
    # statsmodels orders its rows by itertools.combinations over the sorted
    # unique labels, which is exactly `labels` here — so row i aligns.
    for i, (a, b) in enumerate(itertools.combinations(labels, 2)):
        comparisons.append(
            {
                "group_a": a,
                "group_b": b,
                "n_a": n_by[a],
                "n_b": n_by[b],
                "estimate": float(res.meandiffs[i]),  # group_b - group_a
                "statistic": None,
                "p_raw": None,
                "p_adj": float(res.pvalues[i]),
                "reject": bool(res.reject[i]),
                "ci_low": float(res.confint[i][0]),
                "ci_high": float(res.confint[i][1]),
            }
        )

    omni = _scipy_stats().f_oneway(*arrays)
    omnibus = {
        "test": "anova",
        "statistic": float(omni.statistic),
        "p_value": float(omni.pvalue),
        "significant": float(omni.pvalue) < payload.alpha,
    }
    return _build_response(
        engine="tukey",
        payload=payload,
        p_adjust=None,
        estimate_name="mean_diff",
        omnibus=omnibus,
        comparisons=comparisons,
        labels=labels,
        arrays=arrays,
    )


def _build_response(
    *,
    engine: str,
    payload: PairwiseComparisonsInput,
    p_adjust: str | None,
    estimate_name: str,
    omnibus: dict[str, Any],
    comparisons: list[dict[str, Any]],
    labels: list[str],
    arrays: list[Any],
) -> dict[str, Any]:
    """Assemble the stable pairwise-comparisons envelope shared by both engines."""
    n_rejected = sum(1 for c in comparisons if c["reject"])
    p_norm = [_shapiro_p(a) for a in arrays]
    p_lev = _levene_p(*arrays)
    return {
        "ok": True,
        "method": engine,
        "method_requested": payload.method,
        "p_adjust": p_adjust,
        "alpha": payload.alpha,
        "estimate_name": estimate_name,
        "omnibus": omnibus,
        "comparisons": comparisons,
        "n_comparisons": len(comparisons),
        "n_rejected": n_rejected,
        "groups": [{"name": lab, "n": len(arr)} for lab, arr in zip(labels, arrays, strict=True)],
        "assumption_checks": _assumption_checks(engine, p_norm, p_lev),
        "interpretation": _interpretation(
            engine=engine,
            p_adjust=p_adjust,
            estimate_name=estimate_name,
            comparisons=comparisons,
            alpha=payload.alpha,
            omnibus=omnibus,
        ),
    }


def _assumption_checks(
    engine: str, p_norm: list[float | None], p_lev: float
) -> list[dict[str, Any]]:
    """compare_groups-style shapiro + levene blocks, narrated for the engine."""
    p_norm_min = min((p for p in p_norm if p is not None), default=None)
    norm_violated = any(p is not None and p < 0.05 for p in p_norm)
    var_violated = p_lev < 0.05
    if engine == "tukey":
        norm_consequence = "Normality holds; Tukey HSD applies."
        var_consequence = "Levene's test informs the equal-variance assumption Tukey relies on."
    else:
        norm_consequence = "Non-normal residuals — switched to Dunn's test."
        var_consequence = "Rank-based Dunn's test handles unequal variances."
    return [
        {
            "name": "shapiro",
            "p": p_norm_min,
            "violated": norm_violated,
            "consequence": norm_consequence,
        },
        {
            "name": "levene",
            "p": p_lev,
            "violated": var_violated,
            "consequence": var_consequence,
        },
    ]


def _interpretation(
    *,
    engine: str,
    p_adjust: str | None,
    estimate_name: str,
    comparisons: list[dict[str, Any]],
    alpha: float,
    omnibus: dict[str, Any],
) -> str:
    """Plain-English summary: engine, adjustment, N-of-M pairs, largest difference."""
    n = len(comparisons)
    n_rejected = sum(1 for c in comparisons if c["reject"])
    if engine == "tukey":
        engine_name = "Tukey HSD"
        adjustment = "controls the family-wise error rate internally"
    else:
        engine_name = "Dunn's test"
        adjustment = f"{(p_adjust or 'holm').capitalize()}-adjusted"
    largest = max(comparisons, key=lambda c: abs(c["estimate"]))
    text = (
        f"{engine_name} ({adjustment}): {n_rejected} of {n} pairs differ at "
        f"α={alpha:g}. Largest difference: {largest['group_a']} vs "
        f"{largest['group_b']} ({estimate_name}={largest['estimate']:.4f})."
    )
    if not omnibus["significant"]:
        text += (
            f" Note: the omnibus {omnibus['test']} is not significant "
            f"(p={omnibus['p_value']:.4f}), so treat the pairwise findings cautiously."
        )
    return text
