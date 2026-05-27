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
from data_analyst_mcp.recorder import get_recorder

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


def _sm_proportion_effectsize(p1: float, p2: float) -> float:
    """Cohen's h via ``statsmodels.stats.proportion.proportion_effectsize``."""
    from statsmodels.stats.proportion import (  # type: ignore[reportMissingTypeStubs]
        proportion_effectsize,
    )

    val: Any = proportion_effectsize(p1, p2)
    return float(val)


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
    # For two_proportion_z, p1+p2 implicitly provide effect_size — treat as
    # not-an-unknown for the count check (the actual derivation happens
    # downstream). If neither effect_size nor (p1, p2) is provided we emit
    # the family-specific missing_proportions error so callers don't get
    # the more generic invalid_inputs.
    if payload.test == "two_proportion_z" and (
        payload.effect_size is None and (payload.p1 is None or payload.p2 is None)
    ):
        return build_error(
            type="missing_proportions",
            message=(
                "two_proportion_z needs either an explicit effect_size "
                "(Cohen's h) or both p1 and p2 to derive it; got "
                f"effect_size={payload.effect_size}, p1={payload.p1}, p2={payload.p2}."
            ),
            hint=(
                "Supply effect_size directly, or pass p1 and p2 so the "
                "tool can compute h via proportion_effectsize(p1, p2)."
            ),
        )
    effect_size_known = payload.effect_size is not None or (
        payload.test == "two_proportion_z" and payload.p1 is not None and payload.p2 is not None
    )
    unknowns = [
        name
        for name, val in (
            ("effect_size", None if effect_size_known else payload.effect_size),
            ("n", payload.n),
            ("power", payload.power),
        )
        if val is None and not (name == "effect_size" and effect_size_known)
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
    # The single None among {effect_size, n, power} is the field to solve
    # for — echoed back as ``solved_for`` in the output envelope.
    solved_for = unknowns[0]

    try:
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
            result = _build_two_sample_t_result(payload, solved_for, float(value))
        elif payload.test == "two_proportion_z":
            result = _solve_two_proportion_z(payload, solved_for)
        elif payload.test in ("one_sample_t", "paired_t"):
            # The default-solve-for-n case (only effect_size + power supplied)
            # routes through this branch unchanged — solved_for is just
            # ``unknowns[0]`` and is "n" by construction. Paired-t shares the
            # TTestPower solver (it operates on the n pairwise differences).
            result = _solve_one_or_paired_t(payload, solved_for)
        elif payload.test == "anova_oneway":
            if payload.k_groups is None:
                return build_error(
                    type="missing_k_groups",
                    message="anova_oneway requires k_groups (>= 2); none was provided.",
                    hint=(
                        "Pass k_groups equal to the number of groups in the design "
                        "(e.g. 3 for a three-arm trial)."
                    ),
                )
            result = _solve_anova_oneway(payload, solved_for)
        else:  # pragma: no cover - Literal in input model excludes this branch
            return build_error(type="internal", message="not implemented")
    except (ValueError, FloatingPointError) as exc:
        # statsmodels raises ValueError on contradictory inputs (e.g.
        # effect_size=0). Surface the underlying message rather than the
        # generic ``internal`` envelope.
        return build_error(
            type="infeasible_solution",
            message=f"Solver could not converge: {exc}",
            hint="Re-check the inputs — typical causes are effect_size=0 or ratio<=0.",
        )

    # NaN results are statsmodels' other infeasible-input signal.
    solved_value = result.get(solved_for) if result.get("ok") else None
    if isinstance(solved_value, float) and (solved_value != solved_value):  # NaN
        return build_error(
            type="infeasible_solution",
            message=f"Solver returned NaN for {solved_for}; inputs are contradictory.",
            hint="Re-check the inputs — common causes: too-small effect_size or alpha near 1.",
        )

    if result.get("ok"):
        _record_cell(payload, result, solved_for=solved_for)
    return result


def _record_cell(payload: PowerAnalysisInput, result: dict[str, Any], *, solved_for: str) -> None:
    """Append one markdown + one code cell that reproduces the solve."""
    value = result.get(solved_for)
    markdown = (
        f"### Power analysis ({payload.test})\n"
        f"Solved for `{solved_for}` = {value:.4g}.\n\n"
        f"{result.get('interpretation', '')}"
    )
    code = _code_snippet(payload, solved_for=solved_for)
    get_recorder().record(markdown=markdown, code=code, tool_name="power_analysis")


def _code_snippet(payload: PowerAnalysisInput, *, solved_for: str) -> str:
    """Reproduce the statsmodels call faithfully for the chosen test family."""
    es_arg = "None" if solved_for == "effect_size" else f"{payload.effect_size!r}"
    pw_arg = "None" if solved_for == "power" else f"{payload.power!r}"

    if payload.test == "two_sample_t":
        n_arg = "None" if solved_for == "n" else f"{payload.n!r}"
        return (
            "from statsmodels.stats.power import TTestIndPower\n"
            f"TTestIndPower().solve_power(effect_size={es_arg}, nobs1={n_arg}, "
            f"alpha={payload.alpha!r}, power={pw_arg}, "
            f"ratio={payload.ratio!r}, alternative={payload.alternative!r})"
        )
    if payload.test in ("one_sample_t", "paired_t"):
        n_arg = "None" if solved_for == "n" else f"{payload.n!r}"
        return (
            "from statsmodels.stats.power import TTestPower\n"
            f"TTestPower().solve_power(effect_size={es_arg}, nobs={n_arg}, "
            f"alpha={payload.alpha!r}, power={pw_arg}, "
            f"alternative={payload.alternative!r})"
        )
    if payload.test == "two_proportion_z":
        n_arg = "None" if solved_for == "n" else f"{payload.n!r}"
        # If p1+p2 were given (and effect_size was not), preface the call
        # with the proportion_effectsize derivation so the cell reads as a
        # faithful reproducer of what the tool computed.
        es_block = ""
        if payload.effect_size is None and payload.p1 is not None and payload.p2 is not None:
            es_block = (
                "from statsmodels.stats.proportion import proportion_effectsize\n"
                f"h = abs(proportion_effectsize({payload.p1!r}, {payload.p2!r}))\n"
            )
            es_arg_inner = "h"
        else:
            es_arg_inner = es_arg
        return (
            f"{es_block}"
            "from statsmodels.stats.power import NormalIndPower\n"
            f"NormalIndPower().solve_power(effect_size={es_arg_inner}, nobs1={n_arg}, "
            f"alpha={payload.alpha!r}, power={pw_arg}, "
            f"ratio={payload.ratio!r}, alternative={payload.alternative!r})"
        )
    # anova_oneway
    n_arg = "None" if solved_for == "n" else f"{payload.n!r}"
    return (
        "from statsmodels.stats.power import FTestAnovaPower\n"
        f"FTestAnovaPower().solve_power(effect_size={es_arg}, nobs={n_arg}, "
        f"alpha={payload.alpha!r}, power={pw_arg}, k_groups={payload.k_groups!r})"
    )


def _solve_one_or_paired_t(payload: PowerAnalysisInput, solved_for: str) -> dict[str, Any]:
    """Dispatch the TTestPower solver — shared by ``one_sample_t`` and ``paired_t``.

    Paired-t is statistically a one-sample t over the n pairwise
    differences, so the same statsmodels ``TTestPower`` solver handles
    both. ``n`` is the single (or paired) sample size; no ``n_total`` is
    emitted because n itself is the total observation count.
    """
    solver = _sm_power().TTestPower()
    value = float(
        solver.solve_power(
            effect_size=payload.effect_size,
            nobs=payload.n,
            alpha=payload.alpha,
            power=payload.power,
            alternative=payload.alternative,
        )
    )
    if solved_for == "n":
        n = value
        es = float(payload.effect_size)  # type: ignore[arg-type]
        pw = float(payload.power)  # type: ignore[arg-type]
    elif solved_for == "effect_size":
        n = float(payload.n)  # type: ignore[arg-type]
        es = value
        pw = float(payload.power)  # type: ignore[arg-type]
    else:  # power
        n = float(payload.n)  # type: ignore[arg-type]
        es = float(payload.effect_size)  # type: ignore[arg-type]
        pw = value
    # ``test`` is echoed verbatim so callers can distinguish
    # ``one_sample_t`` from ``paired_t`` even though both share the same
    # underlying solver / arithmetic.
    return {
        "ok": True,
        "test": payload.test,
        "solved_for": solved_for,
        "effect_size_metric": _EFFECT_SIZE_METRIC[payload.test],
        "alpha": payload.alpha,
        "alternative": payload.alternative,
        "effect_size": es,
        "n": n,
        "power": pw,
        "interpretation": _interpret_one_or_paired_t(
            test=payload.test,
            solved_for=solved_for,
            n=n,
            es=es,
            pw=pw,
            alpha=payload.alpha,
            alternative=payload.alternative,
        ),
    }


def _interpret_one_or_paired_t(
    *,
    test: str,
    solved_for: str,
    n: float,
    es: float,
    pw: float,
    alpha: float,
    alternative: str,
) -> str:
    """Plain-English interpretation for ``one_sample_t`` / ``paired_t``."""
    import math

    family = "paired t" if test == "paired_t" else "one-sample t"
    if solved_for == "n":
        return (
            f"Need {math.ceil(n)} observations at α={alpha} to detect d={es:.4g} "
            f"with {pw * 100:.0f}% power ({family}, alternative={alternative})."
        )
    if solved_for == "effect_size":
        return (
            f"With n={int(n)} at α={alpha} and {pw * 100:.0f}% power, "
            f"the minimum detectable effect is d={es:.4g} "
            f"({family}, alternative={alternative})."
        )
    return (
        f"Achieved power = {pw:.4g} (i.e. {pw * 100:.1f}%) with n={int(n)}, "
        f"d={es:.4g}, α={alpha} ({family}, alternative={alternative})."
    )


def _solve_anova_oneway(payload: PowerAnalysisInput, solved_for: str) -> dict[str, Any]:
    """Dispatch the FTestAnovaPower solver. ``n`` is the *total* sample size."""
    import math

    solver = _sm_power().FTestAnovaPower()
    value = float(
        solver.solve_power(
            effect_size=payload.effect_size,
            nobs=payload.n,
            alpha=payload.alpha,
            power=payload.power,
            k_groups=payload.k_groups,
        )
    )
    if solved_for == "n":
        n_total = value
        es = float(payload.effect_size)  # type: ignore[arg-type]
        pw = float(payload.power)  # type: ignore[arg-type]
    elif solved_for == "effect_size":
        n_total = float(payload.n)  # type: ignore[arg-type]
        es = value
        pw = float(payload.power)  # type: ignore[arg-type]
    else:  # power
        n_total = float(payload.n)  # type: ignore[arg-type]
        es = float(payload.effect_size)  # type: ignore[arg-type]
        pw = value
    return {
        "ok": True,
        "test": payload.test,
        "solved_for": solved_for,
        "effect_size_metric": _EFFECT_SIZE_METRIC[payload.test],
        "alpha": payload.alpha,
        # No `alternative` field — F-test for one-way ANOVA is two-sided
        # by construction.
        "effect_size": es,
        "n": n_total,
        "n_total": math.ceil(n_total),
        "k_groups": payload.k_groups,
        "power": pw,
        "interpretation": _interpret_anova(
            solved_for=solved_for,
            n_total=n_total,
            es=es,
            pw=pw,
            alpha=payload.alpha,
            k_groups=int(payload.k_groups),  # type: ignore[arg-type]
        ),
    }


def _interpret_anova(
    *,
    solved_for: str,
    n_total: float,
    es: float,
    pw: float,
    alpha: float,
    k_groups: int,
) -> str:
    """Plain-English interpretation for one-way ANOVA power."""
    import math

    if solved_for == "n":
        return (
            f"Need {math.ceil(n_total)} total observations across {k_groups} groups "
            f"at α={alpha} to detect f={es:.4g} with {pw * 100:.0f}% power "
            f"(one-way ANOVA)."
        )
    if solved_for == "effect_size":
        return (
            f"With n={int(n_total)} total across {k_groups} groups at α={alpha} "
            f"and {pw * 100:.0f}% power, the minimum detectable Cohen's f is "
            f"{es:.4g} (one-way ANOVA)."
        )
    return (
        f"Achieved power = {pw:.4g} (i.e. {pw * 100:.1f}%) with n={int(n_total)} total, "
        f"f={es:.4g}, {k_groups} groups, α={alpha} (one-way ANOVA)."
    )


def _solve_two_proportion_z(payload: PowerAnalysisInput, solved_for: str) -> dict[str, Any]:
    """Dispatch the NormalIndPower solver with Cohen's-h effect size."""
    import math

    es = payload.effect_size
    if es is None and payload.p1 is not None and payload.p2 is not None:
        # Derive Cohen's h. Sign is irrelevant to power; use absolute value.
        # An explicit effect_size wins over p1/p2 — the user is being deliberate.
        es = abs(_sm_proportion_effectsize(payload.p1, payload.p2))

    solver = _sm_power().NormalIndPower()
    value = float(
        solver.solve_power(
            effect_size=es,
            nobs1=payload.n,
            alpha=payload.alpha,
            power=payload.power,
            ratio=payload.ratio,
            alternative=payload.alternative,
        )
    )
    if solved_for == "n":
        n1 = value
        es_out = float(es)  # type: ignore[arg-type]
        pw = float(payload.power)  # type: ignore[arg-type]
    elif solved_for == "effect_size":
        n1 = float(payload.n)  # type: ignore[arg-type]
        es_out = value
        pw = float(payload.power)  # type: ignore[arg-type]
    else:  # power
        n1 = float(payload.n)  # type: ignore[arg-type]
        es_out = float(es)  # type: ignore[arg-type]
        pw = value
    n_total = math.ceil(n1) + math.ceil(n1 * payload.ratio)
    return {
        "ok": True,
        "test": payload.test,
        "solved_for": solved_for,
        "effect_size_metric": _EFFECT_SIZE_METRIC[payload.test],
        "alpha": payload.alpha,
        "alternative": payload.alternative,
        "effect_size": es_out,
        "n": n1,
        "power": pw,
        "n_total": n_total,
        "interpretation": _interpret_two_proportion_z(
            solved_for=solved_for,
            n1=n1,
            es=es_out,
            pw=pw,
            alpha=payload.alpha,
            n_total=n_total,
            alternative=payload.alternative,
        ),
    }


def _interpret_two_proportion_z(
    *,
    solved_for: str,
    n1: float,
    es: float,
    pw: float,
    alpha: float,
    n_total: int,
    alternative: str,
) -> str:
    """Plain-English interpretation for the two-proportion-z family."""
    import math

    if solved_for == "n":
        return (
            f"Need {math.ceil(n1)} per group ({n_total} total) at α={alpha} "
            f"to detect h={es:.4g} with {pw * 100:.0f}% power "
            f"(two-proportion z, alternative={alternative})."
        )
    if solved_for == "effect_size":
        return (
            f"With n={int(n1)} per group at α={alpha} and {pw * 100:.0f}% power, "
            f"the minimum detectable Cohen's h is {es:.4g} "
            f"(two-proportion z, alternative={alternative})."
        )
    return (
        f"Achieved power = {pw:.4g} (i.e. {pw * 100:.1f}%) with n={int(n1)} per group, "
        f"h={es:.4g}, α={alpha} (two-proportion z, alternative={alternative})."
    )


def _build_two_sample_t_result(
    payload: PowerAnalysisInput, solved_for: str, value: float
) -> dict[str, Any]:
    """Compose the two_sample_t output envelope including n_total + interpretation."""
    import math

    result: dict[str, Any] = {
        "ok": True,
        "test": payload.test,
        "solved_for": solved_for,
        "effect_size_metric": _EFFECT_SIZE_METRIC[payload.test],
        "alpha": payload.alpha,
        "alternative": payload.alternative,
    }
    if solved_for == "n":
        n1 = value
        es = float(payload.effect_size)  # type: ignore[arg-type]
        pw = float(payload.power)  # type: ignore[arg-type]
    elif solved_for == "effect_size":
        n1 = float(payload.n)  # type: ignore[arg-type]
        es = value
        pw = float(payload.power)  # type: ignore[arg-type]
    else:  # power
        n1 = float(payload.n)  # type: ignore[arg-type]
        es = float(payload.effect_size)  # type: ignore[arg-type]
        pw = value
    n_total = math.ceil(n1) + math.ceil(n1 * payload.ratio)
    result["effect_size"] = es
    result["n"] = n1
    result["power"] = pw
    result["n_total"] = n_total
    result["interpretation"] = _interpret_two_sample_t(
        solved_for=solved_for,
        n1=n1,
        es=es,
        pw=pw,
        alpha=payload.alpha,
        n_total=n_total,
        alternative=payload.alternative,
    )
    return result


def _interpret_two_sample_t(
    *,
    solved_for: str,
    n1: float,
    es: float,
    pw: float,
    alpha: float,
    n_total: int,
    alternative: str,
) -> str:
    """Compose a plain-English interpretation that names the solved-for quantity."""
    import math

    if solved_for == "n":
        return (
            f"Need {math.ceil(n1)} per group ({n_total} total) at α={alpha} "
            f"to detect d={es:.4g} with {pw * 100:.0f}% power "
            f"(two-sample t, alternative={alternative})."
        )
    if solved_for == "effect_size":
        return (
            f"With n={int(n1)} per group at α={alpha} and {pw * 100:.0f}% power, "
            f"the minimum detectable effect is d={es:.4g} "
            f"(two-sample t, alternative={alternative})."
        )
    # power
    return (
        f"Achieved power = {pw:.4g} (i.e. {pw * 100:.1f}%) with n={int(n1)} per group, "
        f"d={es:.4g}, α={alpha} (two-sample t, alternative={alternative})."
    )
