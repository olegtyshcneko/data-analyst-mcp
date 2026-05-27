"""Statistical power / sample-size / MDE analysis — ``power_analysis``.

A stateless wrapper over the ``statsmodels.stats.power`` family. The tool
does no session lookup, returns numbers only, and emits a recorder cell
pair on success.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp.errors import build_error

logger = logging.getLogger(__name__)


_EFFECT_SIZE_METRIC: dict[str, str] = {
    "two_sample_t": "cohens_d",
    "one_sample_t": "cohens_d",
    "paired_t": "cohens_d",
    "two_proportion_z": "cohens_h",
    "anova_oneway": "cohens_f",
}


def _sm_power() -> Any:
    """Return ``statsmodels.stats.power`` as an untyped module."""
    import statsmodels.stats.power as _power  # type: ignore[reportMissingTypeStubs]

    return _power


class PowerAnalysisInput(BaseModel):
    """Inputs for ``power_analysis``."""

    model_config = ConfigDict(extra="forbid")

    test: Literal[
        "two_sample_t",
        "one_sample_t",
        "paired_t",
        "two_proportion_z",
        "anova_oneway",
    ]
    effect_size: float | None = None
    n: int | float | None = None
    power: float | None = Field(default=None, ge=0.0, le=1.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    p1: float | None = None
    p2: float | None = None
    k_groups: int | None = Field(default=None, ge=2)
    ratio: float = 1.0
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided"


def power_analysis(payload: PowerAnalysisInput) -> dict[str, Any]:
    """Solve for the unknown among ``{effect_size, n, power}``."""
    unknowns = [
        name
        for name, val in (
            ("effect_size", payload.effect_size),
            ("n", payload.n),
            ("power", payload.power),
        )
        if val is None
    ]
    if len(unknowns) != 1:
        return build_error(
            type="invalid_inputs",
            message=(
                f"Exactly one of effect_size/n/power must be omitted (None); "
                f"got {len(unknowns)} omitted: {unknowns}."
            ),
            hint=(
                "Provide two of {effect_size, n, power} and leave the third "
                "unset — that is the quantity the solver will return."
            ),
        )
    solved_for = unknowns[0]

    if payload.test == "two_sample_t":
        solver = _sm_power().TTestIndPower()
        value = solver.solve_power(
            effect_size=payload.effect_size,
            nobs1=payload.n,
            alpha=payload.alpha,
            power=payload.power,
            ratio=payload.ratio,
            alternative=payload.alternative,
        )
        result: dict[str, Any] = {
            "ok": True,
            "test": payload.test,
            "solved_for": solved_for,
            "effect_size_metric": _EFFECT_SIZE_METRIC[payload.test],
            "alpha": payload.alpha,
            "alternative": payload.alternative,
        }
        if solved_for == "n":
            result["n"] = float(value)
            result["effect_size"] = float(payload.effect_size)  # type: ignore[arg-type]
            result["power"] = float(payload.power)  # type: ignore[arg-type]
        elif solved_for == "effect_size":
            result["effect_size"] = float(value)
            result["n"] = float(payload.n)  # type: ignore[arg-type]
            result["power"] = float(payload.power)  # type: ignore[arg-type]
        else:  # power
            result["power"] = float(value)
            result["effect_size"] = float(payload.effect_size)  # type: ignore[arg-type]
            result["n"] = float(payload.n)  # type: ignore[arg-type]
        return result

    return build_error(type="internal", message="not implemented")
