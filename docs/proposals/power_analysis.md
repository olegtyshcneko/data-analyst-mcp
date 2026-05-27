# `power_analysis`

## Purpose

Most A/B-test workflows need sample-size, MDE, or achieved-power
estimates *before* `compare_groups` runs. The server has no answer
today, so agents either hand-compute (wrong) or skip the question
(worse). `power_analysis` ships a stateless solver covering the five
test families that map onto the existing `compare_groups` / `test_hypothesis`
surface: two-sample t, one-sample t, paired t, two-proportion z, and
one-way ANOVA. Solves for whichever of `effect_size` / `n` / `power` is
omitted.

## Input

```python
class PowerAnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test: Literal[
        "two_sample_t",
        "one_sample_t",
        "paired_t",
        "two_proportion_z",
        "anova_oneway",
    ]
    # Solve for whichever of these is None — exactly one must be None.
    effect_size: float | None = None
    n: int | float | None = None       # per-group for two-sample / anova; total for one-sample
    power: float | None = Field(default=None, ge=0.0, le=1.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    # Two-proportion-specific:
    p1: float | None = None
    p2: float | None = None
    # ANOVA-specific:
    k_groups: int | None = Field(default=None, ge=2)
    # Two-sample t-specific:
    ratio: float = 1.0                 # n2 / n1
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided"
```

## Behavior

1. Validate that exactly one of `{effect_size, n, power}` is `None` —
   the unknown to solve for. Zero or ≥2 unknowns → `invalid_inputs`.
2. For `two_proportion_z`: if `p1` and `p2` are provided, derive Cohen's h
   via `statsmodels.stats.proportion.proportion_effectsize(p1, p2)` and
   use it as `effect_size`. If neither `p1`+`p2` nor an explicit
   `effect_size` is provided → `missing_proportions`.
3. For `anova_oneway`: require `k_groups`; otherwise `missing_k_groups`.
4. Dispatch to the statsmodels solver:
   - `two_sample_t` → `TTestIndPower().solve_power(effect_size, nobs1=n,
     alpha, power, ratio, alternative)`. Effect-size metric:
     **Cohen's d**.
   - `one_sample_t` / `paired_t` → `TTestPower().solve_power(...)`.
     Effect-size metric: **Cohen's d**.
   - `two_proportion_z` → `NormalIndPower().solve_power(...)` with the
     `proportion_effectsize`-derived h. Effect-size metric: **Cohen's h**.
   - `anova_oneway` → `FTestAnovaPower().solve_power(effect_size,
     nobs=total_n, alpha, power, k_groups)`. `n` represents **total**
     observations across groups (document clearly in the
     `interpretation`). Effect-size metric: **Cohen's f**.
5. If statsmodels returns `nan` or raises → `infeasible_solution` with
   the underlying message in `details`. Typical cause: contradictory
   inputs (e.g. `power=0.99` with absurd `effect_size`).
6. Build the `interpretation` string and emit the recorder cell
   (success only).

## Output

```python
{
    "ok": True,
    "test": "two_proportion_z",
    "solved_for": "n",                  # one of "effect_size" / "n" / "power"
    "effect_size": 0.20,
    "effect_size_metric": "cohens_h",   # "cohens_d" | "cohens_h" | "cohens_f"
    "n": 393,                           # solved value, per-group for two-sample / anova
    "n_total": 786,                     # convenience field, when applicable
    "power": 0.80,
    "alpha": 0.05,
    "alternative": "two-sided",
    "interpretation": "Need 393 per group (786 total) at α=0.05 to detect h=0.20 with 80% power.",
}
```

## Errors

- `invalid_inputs` — zero or ≥2 of `{effect_size, n, power}` are
  `None`. Hint names which fields must be provided / omitted.
- `missing_proportions` — `test="two_proportion_z"` without `p1`+`p2`
  and without an explicit `effect_size`.
- `missing_k_groups` — `test="anova_oneway"` without `k_groups`.
- `infeasible_solution` — statsmodels returned `nan` or raised;
  typically contradictory inputs.

## Recorder cells

```python
get_recorder().record(
    markdown=(
        f"### Power analysis ({payload.test})\n"
        f"Solved for `{solved_for}` = {value}."
    ),
    code=(
        "from statsmodels.stats.power import TTestIndPower\n"
        "TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8, ratio=1.0)"
    ),
    tool_name="power_analysis",
)
```

The code body is dispatched per test family — `TTestIndPower` for
`two_sample_t`, `TTestPower` for `one_sample_t` / `paired_t`,
`NormalIndPower` (plus `proportion_effectsize` when `p1`+`p2` were
given) for `two_proportion_z`, and `FTestAnovaPower` for
`anova_oneway`.

## TDD slices

~24 cycles, grouped:

**Shared (4 cycles):**
1. `invalid_inputs` when zero of `{effect_size, n, power}` is `None`.
2. `invalid_inputs` when two of them are `None`.
3. Recorder cell written exactly once on success, zero on error.
4. `solved_for` field reflects whichever input was `None`.

**`two_sample_t` (5 cycles):**
5. Known-answer (compared against statsmodels reference values to ≥4
   decimal places).
6. Solve for `n`.
7. Solve for `effect_size`.
8. Solve for `power`.
9. `ratio` respected.

**`two_proportion_z` (5 cycles):**
10. Known-answer (Cohen's h calculation against the worked example
    `p1=0.10, p2=0.12, power=0.8, alpha=0.05` → `n ≈ 3800`-ish per group).
11. `p1`+`p2` → `effect_size` auto-derived.
12. Explicit `effect_size` overrides `p1`+`p2`.
13. `missing_proportions` when neither pair nor `effect_size` provided.
14. `alternative` respected.

**`anova_oneway` (4 cycles):**
15. Known-answer.
16. `missing_k_groups` when `k_groups=None`.
17. `n` documented and reported as **total** (not per-group); `n_total`
    populated.
18. `effect_size_metric == "cohens_f"`.

**`one_sample_t` (3 cycles):**
19. Known-answer.
20. `alternative` respected.
21. Default `solved_for == "n"` when only `effect_size` and `power`
    given.

**`paired_t` (3 cycles):**
22. Known-answer.
23. Same underlying solver as `one_sample_t` (`TTestPower`).
24. Separate test for clarity (`test="paired_t"` echoed in output).

## Acceptance criteria

- All ~24 TDD cycles green with `red:` / `green:` / `refactor:`
  commits.
- `tests/test_power.py` passes under `uv run pytest -q`.
- `evals/eval_power.py` passes: sample-size calc → load synthetic
  dataset of that size → `compare_groups` recovers the planted
  effect.
- `ruff check .`, `ruff format --check .`, `pyright src/`, and
  `scripts/check_tdd_commits.py` all green.

## ROADMAP impact

- No existing `ROADMAP.md` line removed (this is a new addition
  outside the prior tooling list).
- Adds a new SPEC §5 subsection alongside the test-family tools.
- Adds a line to `ROADMAP.md` "Shipped" recording that
  `power_analysis` landed in this bundle.
- Bumps the tool count in `README.md` and SPEC §1.
