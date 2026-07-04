# `pairwise_comparisons`

## Purpose

`compare_groups` runs the omnibus test — one-way ANOVA or Kruskal–Wallis —
and stops. When it reports "the groups differ," the immediate follow-up
question, *which pairs differ?*, has no tool today. `pairwise_comparisons`
is that post-hoc follow-up: **Tukey HSD** after ANOVA, **Dunn's test**
(tie-corrected) after Kruskal–Wallis, gated by the same auto-selection
logic (`_select_test`'s Shapiro normality gate) that `compare_groups`
already uses. It is **stateful** — it reads a registered dataset by
`name`, materializes the requested groups, and recomputes the omnibus
inline on the filtered groups so the interpretation can caveat a
non-significant family.

Dunn's test is **vendored** as hand-rolled rank arithmetic rather than
pulled from `scikit-posthocs`. This keeps the runtime dependency set
frozen (Tukey already ships in statsmodels; Dunn is a few lines of
`scipy.stats.rankdata` plus a normal tail), and it follows the
established precedent of vendoring a small closed-form statistic instead
of taking a new dependency — the Little's-MCAR test in
`analyze_missingness` (SPEC §5.4) is vendored the same way. The vendored
formula is cross-checked against `scikit-posthocs.posthoc_dunn` in a
one-off, out-of-repo script (never added as a dependency).

## Input

```python
class PairwiseComparisonsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    group_column: str
    metric_column: str
    # ≥3 labels, no duplicates; omitted → every distinct label in the column.
    groups: list[str] | None = None
    # "auto" mirrors compare_groups: normality holds → tukey, violated → dunn.
    method: Literal["auto", "tukey", "dunn"] = "auto"
    # Dunn only. Resolves to "holm" when Dunn runs and p_adjust is None.
    # Explicit p_adjust + method="tukey" → p_adjust_not_applicable.
    # Under auto, an explicit p_adjust is never an error (echoed null if Tukey wins).
    p_adjust: Literal["holm", "bonferroni", "sidak", "bh", "by"] | None = None
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)  # open (0, 1)
```

- `method="auto"` reuses `_select_test`'s Shapiro–Wilk gate: if per-group
  normality holds → Tukey, if violated → Dunn. `method="tukey"` /
  `method="dunn"` force the engine.
- `p_adjust` controls the family-wise correction applied to Dunn's raw
  z-test p-values only. Tukey HSD controls FWER internally via the
  studentized-range distribution, so a correction is meaningless there.
  When Dunn runs and `p_adjust is None`, it **resolves to `"holm"`**.
- `alpha` must satisfy `0 < alpha < 1`.

## Behavior

1. Resolve the dataset by `name`; missing → `not_found`.
2. Validate `group_column` and `metric_column` both exist; first miss →
   `column_not_found`.
3. Validate `metric_column` is numeric (`_is_numeric_dtype`); non-numeric
   → `metric_not_numeric`.
4. Validate `alpha` in `(0, 1)` → `invalid_alpha` (belt-and-suspenders
   alongside the pydantic bound, so the error type is deterministic).
5. **Label resolution.** If `groups` is omitted, resolve to every distinct
   label via `_all_labels` (imported from `tools.stats`). If provided:
   reject duplicates → `duplicate_groups`; every label must have rows →
   `group_not_found` (names the first missing label). After resolution,
   `< 3` labels → `too_few_groups` (hint: use `compare_groups` for a
   two-group comparison); `> 20` labels → `too_many_groups` (hint: pass a
   `groups` subset — the 20-group cap bounds the quadratic
   `n·(n−1)/2` comparison output).
6. **Group materialization** via a new `_materialize_group_nonnull` helper:
   the same bound-parameter SQL as the existing `_materialize_group`
   (`stats.py:238-247`) **plus `AND <metric> IS NOT NULL`**. This is a
   **deliberate divergence** from `_materialize_group`, which does *not*
   filter NULLs — Tukey and Dunn both require complete numeric vectors,
   and a silent NaN would corrupt the rank pooling. Documented here so it
   is not "fixed" back to match the shared helper. Any group with `n < 2`
   after the NULL filter → `insufficient_group_size`.
7. **Assumption checks.** Compute `_shapiro_p` per group and `_levene_p`
   across groups (imported from `tools.stats`, deterministic
   `default_rng(0)`). These populate the `compare_groups`-style
   `assumption_checks` envelope. `_select_test` decides the auto engine
   from the same Shapiro gate.
8. **`p_adjust_not_applicable` guard.** Error **only** when
   `method="tukey"` *and* an explicit `p_adjust` was supplied. Under
   `method="auto"` an explicit `p_adjust` is *never* an error: if Tukey
   wins the auto gate the `p_adjust` field is simply echoed as `null`.
9. **Engine dispatch** (labels sorted ascending; pairs enumerated in
   `itertools.combinations` order, which matches statsmodels' own pair
   ordering; `estimate` reported in **b − a** orientation):
   - **Tukey** — `statsmodels.stats.multicomp.pairwise_tukeyhsd(endog,
     groups, alpha)` behind an `Any`-returning `_sm_multicomp()` wrapper
     (pyright-strict pattern, `multitest.py:32-42`). `estimate_name =
     "mean_diff"`; each row carries `statistic=null`, `p_raw=null`,
     `p_adj` = Tukey's adjusted p, and `ci_low`/`ci_high` from the Tukey
     confint. Omnibus = `scipy.stats.f_oneway`.
   - **Dunn** (vendored) — pooled average ranks via
     `scipy.stats.rankdata` over the concatenated groups; tie term
     `T = Σ(t³ − t)` over tie-group sizes `t`;
     `SE_ij = sqrt((N(N+1)/12 − T/(12(N−1)))·(1/nᵢ + 1/nⱼ))`;
     `z = (R̄ⱼ − R̄ᵢ)/SE_ij` (signed, **b − a**);
     `p_raw = 2·norm.sf(|z|)`. Family correction via statsmodels
     `multipletests`, using `_METHOD_TO_STATSMODELS` imported from
     `tools.multitest` (never re-declared here). `estimate_name =
     "mean_rank_diff"`; `estimate = R̄_b − R̄_a`; `statistic = z`;
     `ci_low`/`ci_high = null`. Omnibus = `scipy.stats.kruskal`.
10. Recompute the omnibus inline on the filtered groups; a
    non-significant omnibus (`p ≥ alpha`) drives a caveat in the
    `interpretation`.
11. Emit the recorder cell (success only), then return.

Module shape mirrors the `stats.py:703-707` pattern: a thin
`pairwise_comparisons()` entry point → `_pairwise_comparisons_impl()` →
`_record()` after a successful result.

## Output

```python
{
    "ok": True,
    "method": "dunn",                 # engine actually run
    "method_requested": "auto",       # echoes payload.method
    "p_adjust": "holm",               # resolved correction; null when Tukey ran
    "alpha": 0.05,
    "estimate_name": "mean_rank_diff",  # "mean_diff" (tukey) | "mean_rank_diff" (dunn)
    "omnibus": {                      # recomputed inline on the filtered groups
        "test": "kruskal",            # "f_oneway" | "kruskal"
        "statistic": 0.3239344262,
        "p_value": 0.8504690883,
        "significant": False,         # p < alpha
    },
    "comparisons": [                  # sorted labels, itertools.combinations order
        {
            "group_a": "A", "group_b": "B",
            "n_a": 20, "n_b": 20,
            "estimate": 1.34,         # b − a (mean_diff or mean_rank_diff)
            "statistic": 1.3416407865,  # dunn z; null for tukey
            "p_raw": 0.1797124949,    # dunn raw; null for tukey
            "p_adj": 0.3594249898,
            "reject": False,
            "ci_low": None,           # tukey confint; null for dunn
            "ci_high": None,
        },
        # ...
    ],
    "n_comparisons": int,             # n·(n−1)/2
    "n_rejected": int,
    "groups": [{"name": "A", "n": 20}, ...],  # post-NULL-filter counts
    "assumption_checks": [            # compare_groups envelope
        {"name": "shapiro", "p": 0.01, "violated": True,
         "consequence": "Non-normal residuals — switched to Dunn's test."},
        {"name": "levene", "p": 0.44, "violated": False, "consequence": "..."},
    ],
    "interpretation": "…",            # plain-English; caveats a non-significant omnibus
}
```

Every comparison row carries the **same keys** regardless of engine —
`statistic`, `p_raw`, `ci_low`, `ci_high` are present as `null` on the
engine that does not produce them. Labels are sorted ascending and pairs
follow `itertools.combinations`, so row order is stable and matches
statsmodels' Tukey table order.

## Errors

- `not_found` — `name` is not in the dataset registry.
- `column_not_found` — `group_column` or `metric_column` is absent;
  names the first miss.
- `metric_not_numeric` — `metric_column` has a non-numeric dtype.
- `invalid_alpha` — `alpha` outside the open interval `(0, 1)`.
- `too_few_groups` — fewer than 3 resolved labels; hint names
  `compare_groups` for the two-group case.
- `too_many_groups` — more than 20 resolved labels; hint to pass a
  `groups` subset (the cap bounds quadratic output).
- `duplicate_groups` — a label repeats in an explicit `groups`.
- `group_not_found` — an explicit `groups` label matches no rows; names
  the first missing label.
- `insufficient_group_size` — a group has `n < 2` after the NULL filter.
- `p_adjust_not_applicable` — `method="tukey"` **and** an explicit
  `p_adjust` was supplied (Tukey controls FWER internally). Never raised
  under `method="auto"`.
- `invalid_method` / `invalid_p_adjust` — server-wrapper mapping of the
  pydantic `ValidationError` on the two `Literal` fields (loc-mapping per
  `server.py:345-353`).
- `internal` — unexpected engine failure.

## Recorder cells

Markdown is produced by a new `format_pairwise_comparisons_markdown()` in
`formatting.py`, reusing the existing `_METHOD_PRETTY` map. The code cell
is **fully runnable** (the `adjust_pvalues` precedent,
`multitest.py:181-198` — *not* the `compare_groups` stub): it rehydrates
the group vectors from the notebook's `con` DuckDB connection, imports its
own `pairwise_tukeyhsd` / `multipletests`, and reproduces the reported
table.

```python
get_recorder().record(
    markdown=format_pairwise_comparisons_markdown(out, payload=payload),
    code=_pairwise_code_snippet(out, payload),  # engine-specific reproducer
    tool_name="pairwise_comparisons",
)
```

The Tukey branch emits:

```python
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Labels escaped for the SQL IN-list — single quotes doubled.
df = con.execute(
    "SELECT \"group\", \"value\" FROM tiny "
    "WHERE \"group\" IN ('A', 'B', 'O''Brien') AND \"value\" IS NOT NULL"
).df()
res = pairwise_tukeyhsd(df["value"], df["group"], alpha=0.05)
print(res.summary())
```

The Dunn branch inlines the vendored rank math and calls
`multipletests(p_raw, alpha=..., method=<sm_method>)`, with the same
NULL-filtered, quote-escaped `IN (...)` rehydration.

**Label escaping is mandatory.** Labels interpolated into the SQL
`IN (...)` list have every single quote doubled (`'` → `''`) before
interpolation — do **not** copy `_code_for_kind`'s unescaped
interpolation, which would break (or inject) on a label like `O'Brien`.

## TDD slices

Each slice is a `red:` commit followed by an identical-text `green:`
commit (adjacency enforced by `scripts/check_tdd_commits.py`, which forces
these to land sequentially). Per SPEC §3 (lines 83–107), the statistical
slices are **known-answer tests**: every pinned numeric gets an inline
comment naming the independent reference call that produced it, asserted
with `pytest.approx(abs=1e-4)`. A test that asserts only `ok is True` does
not prove correctness; ≥4-decimal known-answer assertions do.

Commit subjects (verbatim):

1. `pairwise_comparisons returns not_found for unregistered dataset`
   (green also scaffolds the module + `server.py` registration)
2. `pairwise_comparisons returns column_not_found for missing group or metric column`
3. `pairwise_comparisons returns metric_not_numeric for a VARCHAR metric column`
4. `pairwise_comparisons returns too_few_groups below three groups and hints at compare_groups`
5. `pairwise_comparisons returns invalid_alpha outside the open unit interval`
6. `pairwise_comparisons returns duplicate_groups for repeated labels in groups`
7. `pairwise_comparisons returns group_not_found for a label with no rows`
8. `pairwise_comparisons returns too_many_groups above twenty labels`
9. `pairwise_comparisons rejects explicit p_adjust with method tukey as p_adjust_not_applicable`
   (+ counter-assert: auto + explicit p_adjust does **not** error)
10. `pairwise_comparisons tukey matches statsmodels known answer with confidence intervals`
11. `pairwise_comparisons dunn matches hand-computed ranks on untied data`
12. `pairwise_comparisons dunn applies the tie correction`
13. `pairwise_comparisons dunn passes p_adjust through to bonferroni`
14. `pairwise_comparisons auto selects tukey when normality holds`
15. `pairwise_comparisons auto selects dunn when normality is violated`
16. `pairwise_comparisons restricts pairs to the requested groups subset`
17. `pairwise_comparisons excludes null metric rows from group counts`
18. `pairwise_comparisons flags a non-significant omnibus in the interpretation`
19. `pairwise_comparisons records a runnable cell pair for tukey and dunn`
- extra (unnumbered, `test_multitest.py:344-388` round-trip template):
  `pairwise_comparisons recorded notebook round-trips through nbconvert`

`refactor:` commits are allowed after any green. Docs commits use `docs:`;
eval-only commits use `test:` (both ignored by the checker).

### Pinned reference values

Re-verified with a one-off script before the tests are written; each is
reproduced inline in the corresponding slice as a reference-call comment.

- **Tukey fixture** — A=`[1..5]`, B=`[3..7]`, C=`[5..9]`
  (`pairwise_tukeyhsd`, `alpha=0.05`): A–B meandiff `2.0`, `p_adj`
  `0.1545799684`, CI `[−0.6678636566, 4.6678636566]`, reject `False`;
  A–C `4.0`, `0.0046340806`, `[1.3321363434, 6.6678636566]`, reject
  `True`; B–C = A–B. Omnibus `f_oneway` F=`8.0`, p=`0.0061963978`.
  Per-group Shapiro p=`0.967174` (so `auto` also picks Tukey on this
  fixture — slice 14).
- **Dunn, no ties** — A=`[1,2,3]`, B=`[4,5,6]`, C=`[7,8,9]`: mean ranks
  `2/5/8`, `T=0`, `SE=√5`; `z(A,B)=+1.3416407865` (signed, **b − a** —
  slice 11 asserts the *signed* z, not `|z|`), `p_raw=0.1797124949`;
  `z(A,C)=+2.6832815730`, `p_raw=0.0072903581`; Holm `p_adj`
  `[0.3594249898, 0.0218710743, 0.3594249898]`, reject `[F, T, F]`.
- **Dunn, tied** — A=`[1,2,2]`, B=`[2,3,4]`, C=`[5,5,6]`: `T=30`,
  `SE=2.1889875894`, mean ranks `7/3, 14/3, 8`;
  `z=[1.0659417827, 2.5887157579, 1.5227739753]`,
  `p_raw=[0.2864499597, 0.0096334576, 0.1278152631]`; Holm
  `[0.2864499597, 0.0289003729, 0.2556305262]`; Bonferroni
  `[0.8593498790, 0.0289003729, 0.3834457893]`; reject `[F, T, F]` under
  both (slices 12–13).
- **Auto → dunn** — reuse `test_stats.py`'s
  `RandomState(10/11/12).standard_cauchy(20)` fixture: Shapiro p all
  `< 0.05` → Dunn; omnibus Kruskal H=`0.3239344262`, p=`0.8504690883`
  (already pinned in `test_stats.py`); all Holm `p_adj=1.0`,
  `n_rejected=0` → the omnibus caveat is present (slices 15, 18).

Cross-check (out-of-repo, before implementing): run
`uvx --with scikit-posthocs python -c "...posthoc_dunn..."` on the two
small fixtures to confirm the vendored formula for Holm and Bonferroni;
`scikit-posthocs` is **never** added as a dependency.

## Acceptance criteria

- All 19 numbered slices plus the nbconvert round-trip extra green, with
  paired `red:` / `green:` (and any `refactor:`) commits; every known-
  answer assertion holds at `≤ 1e-4`.
- `tests/test_posthoc.py` (and the formatting slices in
  `tests/test_formatting.py`) pass under `uv run pytest -q`.
- `uv run pyright src/` clean under strict mode; `uv run ruff check .` and
  `uv run ruff format --check .` clean.
- `uv run python scripts/check_tdd_commits.py` passes (every `green:`
  immediately preceded by its identical-text `red:`).
- Line coverage `≥ 90%` (`--cov=data_analyst_mcp --cov-fail-under=90`).
- `evals/eval_pairwise.py` added and green under `-m eval`.
- **statsmodels floor.** The Tukey pins assume `statsmodels >= 0.14`,
  which uses the exact studentized-range distribution for Tukey p-values;
  the floor is noted so the `1e-4` pins stay valid. Verified: the pinned
  values reproduce exactly on the current lockfile, and the vendored Dunn
  formula was cross-checked against `scikit-posthocs.posthoc_dunn` to 6
  decimal places for both Holm and Bonferroni on both fixtures.

## ROADMAP impact

- Grows the tool surface **21 → 22** through the documented proposal flow
  (this doc → design conversation → fold into SPEC §5 as §5.9a, alongside
  `compare_groups` → delete the proposal file, the `adjust_pvalues`
  precedent). The 21-tool boundary is stated in `ROADMAP.md:3` and
  `SPEC.md:753`.
- **Not previously parked.** `pairwise_comparisons` is a new addition, not
  one of the `ROADMAP.md` "parked" lines — so no parked line is removed;
  instead the boundary note is updated to record the 21 → 22 bump.
- Zero new runtime dependencies: Tukey ships in statsmodels, Dunn is
  vendored rank arithmetic (the SPEC §5.4 Little's-MCAR vendoring
  precedent).
- Fold-in bumps the tool count in `README.md` (×3), `SPEC.md` §1 / §11,
  `evals/README.md`, and `CHANGELOG.md` (`## [1.1.0]` Added, "Tool surface
  21 → 22").
