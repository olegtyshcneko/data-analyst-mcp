"""Nonfinite-safe model-evidence serialization + comparison (spec v4 §1)."""

from __future__ import annotations

import math
from typing import Any


def tag_nonfinite(x: float) -> float | str:
    """JSON-safe encoding: nonfinite floats become tagged strings."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return float(x)


def untag(v: float | str) -> float:
    """Inverse of tag_nonfinite."""
    if v == "NaN":
        return float("nan")
    if v == "Infinity":
        return float("inf")
    if v == "-Infinity":
        return float("-inf")
    return float(v)  # type: ignore[arg-type]


def evidence_equal(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-12,
) -> bool:
    """Exact key sets; NaN==NaN; infinities sign-exact; finites via isclose."""
    if set(expected) != set(actual):
        return False
    for key, exp in expected.items():
        act = actual[key]
        e, a = untag(exp), untag(act)
        if math.isnan(e) or math.isnan(a):
            if not (math.isnan(e) and math.isnan(a)):
                return False
        elif math.isinf(e) or math.isinf(a):
            if e != a:
                return False
        elif not math.isclose(e, a, rel_tol=rtol, abs_tol=atol):
            return False
    return True
