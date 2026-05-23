"""Multiple-testing correction — ``adjust_pvalues``.

A stateless wrapper over :func:`statsmodels.stats.multitest.multipletests`.
The tool does no session lookup, returns input rows in input order, and
emits the standard recorder cell pair.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp.errors import build_error
from data_analyst_mcp.formatting import format_adjust_pvalues_markdown
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


_METHOD_TO_STATSMODELS: dict[str, str] = {
    "bonferroni": "bonferroni",
    "sidak": "sidak",
    "holm": "holm",
    "bh": "fdr_bh",
    "by": "fdr_by",
}


def _sm_multitest() -> Any:
    """Return ``statsmodels.stats.multitest`` as an untyped module.

    Returning the module under an ``Any`` annotation is the project pattern
    for statsmodels — see the sibling helpers in ``tools/models.py``. Doing
    it this way keeps strict pyright quiet without inline ignores at every
    call site.
    """
    import statsmodels.stats.multitest as _mt  # type: ignore[reportMissingTypeStubs]

    return _mt


class AdjustPvaluesInput(BaseModel):
    """Inputs for ``adjust_pvalues``."""

    model_config = ConfigDict(extra="forbid")

    p_values: list[float] = Field(
        ...,
        description=(
            "Raw p-values to correct. Each must lie in [0, 1] — NaN, inf, "
            "negative, and >1 are rejected (no silent dropping). An empty "
            "list is valid and returns an empty result set."
        ),
    )
    method: Literal["bonferroni", "sidak", "holm", "bh", "by"] = Field(
        default="bh",
        description=(
            "Correction procedure: bonferroni / sidak / holm (FWER) or "
            "bh / by (FDR). Default 'bh' (Benjamini–Hochberg) is the "
            "EDA-friendly choice; switch to 'holm' for strict FWER control."
        ),
    )
    alpha: float = Field(
        default=0.05,
        description=(
            "Significance threshold in (0, 1). Affects only the 'rejected' "
            "column — adjusted p-values themselves are method-determined."
        ),
    )
    labels: list[str] | None = Field(
        default=None,
        description=(
            "Optional row labels, echoed back into each result row. When "
            "provided, length must equal p_values. Duplicates are allowed "
            "and not deduplicated."
        ),
    )


def adjust_pvalues(payload: AdjustPvaluesInput) -> dict[str, Any]:
    """Adjust a family of p-values for multiple testing.

    Validates inputs (range, alpha, label-length) *before* invoking
    statsmodels so error types stay deterministic. Returns rows in input
    order — do not pass ``returnsorted=True`` to statsmodels.
    """
    p_values = payload.p_values
    labels = payload.labels
    method = payload.method
    alpha = payload.alpha

    # Empty input → early return; no statsmodels call.
    if len(p_values) == 0:
        out: dict[str, Any] = {
            "ok": True,
            "method": method,
            "alpha": alpha,
            "results": [],
            "n_tests": 0,
            "n_rejected": 0,
        }
        _record(out, method=method, alpha=alpha)
        return out

    # Validate p-value range (NaN/inf/negative/>1) — name the first offender.
    for idx, val in enumerate(p_values):
        if not _is_valid_p(val):
            return build_error(
                type="invalid_p_value",
                message=f"p_values[{idx}] = {val} is outside [0, 1].",
                hint="Every p-value must be a finite number in [0, 1].",
            )

    # Validate alpha range — strict open interval.
    if not (0.0 < alpha < 1.0):
        return build_error(
            type="invalid_alpha",
            message=f"alpha must be in (0, 1); got {alpha}.",
            hint="Use 0.05 (default), 0.01, or any other value strictly between 0 and 1.",
        )

    # Validate labels length match.
    if labels is not None and len(labels) != len(p_values):
        return build_error(
            type="length_mismatch",
            message=(f"labels has {len(labels)} entries but p_values has {len(p_values)}."),
            hint="Supply one label per p-value, or omit labels entirely.",
        )

    sm_method = _METHOD_TO_STATSMODELS.get(method)
    if sm_method is None:
        # Defense-in-depth — Pydantic should have caught this already.
        return build_error(
            type="unknown_method",
            message=f"Unknown method {method!r}.",
            hint=f"Allowed methods: {sorted(_METHOD_TO_STATSMODELS)}.",
        )

    rejected, p_adj, _alphacSidak, _alphacBonf = _sm_multitest().multipletests(
        p_values, alpha=alpha, method=sm_method
    )

    results: list[dict[str, Any]] = []
    for i, (p_raw, p_adj_i, rej) in enumerate(zip(p_values, p_adj, rejected, strict=True)):
        label = labels[i] if labels is not None else None
        results.append(
            {
                "label": label,
                "p_raw": float(p_raw),
                "p_adj": float(p_adj_i),
                "rejected": bool(rej),
            }
        )

    out = {
        "ok": True,
        "method": method,
        "alpha": alpha,
        "results": results,
        "n_tests": len(results),
        "n_rejected": sum(1 for r in results if r["rejected"]),
    }
    _record(out, method=method, alpha=alpha)
    return out


def _is_valid_p(v: float) -> bool:
    """True iff ``v`` is a finite real number in [0, 1]."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    if math.isnan(f) or math.isinf(f):
        return False
    return 0.0 <= f <= 1.0


def _record(out: dict[str, Any], *, method: str, alpha: float) -> None:
    """Append the standard markdown+code cell pair for an ``adjust_pvalues`` call."""
    if not out.get("ok"):
        return
    md = format_adjust_pvalues_markdown(out, method=method, alpha=alpha)
    p_raw_list = [r["p_raw"] for r in out["results"]]
    labels = [r["label"] for r in out["results"]]
    sm_method = _METHOD_TO_STATSMODELS.get(method, method)
    code = (
        "from statsmodels.stats.multitest import multipletests\n"
        f"p_raw = {p_raw_list!r}\n"
        f"labels = {labels!r}\n"
        f"rejected, p_adj, _, _ = multipletests(p_raw, alpha={alpha!r}, "
        f"method={sm_method!r})\n"
        "for lbl, raw, adj, rej in zip(labels, p_raw, p_adj, rejected):\n"
        '    print(f"{lbl}  raw={raw:.4g}  adj={adj:.4g}  reject={rej}")'
    )
    get_recorder().record(markdown=md, code=code, tool_name="adjust_pvalues")
