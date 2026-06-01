# Design — structured `perfect_separation` error for logistic `fit_model`

**Date:** 2026-06-01
**Status:** approved (brainstorming), pending implementation plan
**Scope:** `fit_model(kind="logistic")` only. OLS / Poisson / negbin paths are untouched.
**ROADMAP item:** *Statistics & modeling → "Logistic-separation handling"* (resolved by this work).

## 1. Problem

When the outcome of a logistic regression is perfectly (or quasi-perfectly)
separated by the predictors, the maximum-likelihood estimates diverge and the
fit is meaningless. On the installed statsmodels (0.14.6) this surfaces in **two
distinct ways**, both currently mishandled:

| Case | statsmodels behavior (measured, 0.14.6) | Current `fit_model` result |
|------|------------------------------------------|----------------------------|
| Complete separation (continuous predictor) | **raises** `numpy.linalg.LinAlgError: Singular matrix` | caught by the generic `except Exception` and mislabeled `formula_error: "Singular matrix"` |
| Quasi-complete separation | **returns** a fit with `ConvergenceWarning`, `converged=False`, SE ≈ 2.7e5 | `ok: true` with garbage inference |
| Categorical level perfectly predicts | **returns** a fit with `ConvergenceWarning`, `converged=False`, SE ≈ 4.6e6 | `ok: true` with garbage inference |
| Well-behaved logit (control) | returns cleanly, `converged=True`, max SE ≈ 0.87 | `ok: true` (correct) |

The roadmap text ("emits a warning when `PerfectSeparationError` is raised … carries
on with `NaN` standard errors") is **version-drifted**: on 0.14.6 the real
signatures are a raised exception and a silent return with astronomically large
(finite, non-NaN) standard errors. The silent-return case is the worse bug — the
agent receives meaningless inference presented as success.

## 2. Goal

Detect both failure modes and return a single structured error:

```json
{"ok": false, "error": {
  "type": "perfect_separation",
  "message": "Logistic fit failed: the outcome is perfectly (or quasi-perfectly) separated by the predictors, so the maximum-likelihood estimates diverge.",
  "hint": "Drop or collapse the predictor/level that perfectly predicts the outcome; cross-tab the outcome against each categorical predictor to find it. Penalized (Firth/L2) logistic — not yet offered by this server — is the standard remedy."
}}
```

No `coefficients` / `fit` / `diagnostics` blocks are emitted. Because `fit_model`
already gates model-registration and the recorder cell on `result.get("ok")`, an
error result **automatically** skips registration and emits no notebook cell — no
extra code is required in those paths.

## 3. Detection design (Approach 1: post-hoc fit-signature inspection)

Restructure **only** the logistic branch of `_fit_dispatch`. The structure mirrors
the existing `negbin` precedent in the same file (catch exception → structured
error; then post-fit `mle_retvals['converged']` check → structured error).

### 3.1 Two-stage logistic branch (construction guard + fit guard) — REQUIRED

The logistic branch **must** be split into two separately-guarded stages. statsmodels
builds the patsy design matrix **eagerly at construction** (`smf.logit(formula, data=df)`),
so column/formula errors raise *there* as `PatsyError`/`NameError` (verified on 0.14.6:
`smf.logit("y ~ nonexistent_col", df)` raises `PatsyError` before any `.fit()`). Numeric
failures (separation, singular Hessian) raise during `.fit()`. The current code wraps
construction and fit in one `try` (`models.py:450`); that conflates the two and risks a
missing-column error being considered alongside fit-stage numeric errors. Splitting also
keeps the constructed `model` reference alive so its design matrix is available for the
rank discriminator when `.fit()` raises (`m` does not exist on a failed fit).

```python
import patsy
from numpy.linalg import LinAlgError, matrix_rank
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# stage 1 — construction (formula/column errors)
try:
    model = smf.logit(payload.formula, data=df)
except (patsy.PatsyError, NameError) as exc:
    raise _FormulaError(str(exc)) from exc                 # -> formula_error

# stage 2 — fit (numeric/separation errors)
try:
    m = model.fit(disp=0)
except PerfectSeparationError:
    return _separation_error()                             # -> perfect_separation (unambiguous)
except LinAlgError as exc:
    # ambiguous: separation (full-rank design) vs perfect collinearity (rank-deficient)
    exog = np.asarray(model.exog)
    if matrix_rank(exog) == exog.shape[1]:
        return _separation_error()                         # full rank  -> perfect_separation
    raise _FormulaError(f"perfectly collinear predictors: {exc}") from exc  # -> formula_error
except Exception as exc:
    raise _FormulaError(str(exc)) from exc                 # unchanged catch-all

# stage 3 — returned-but-degenerate inspection (see 3.2)
degenerate = _detect_logistic_separation(m)
if degenerate is not None:
    return degenerate
```

### 3.2 Returned-but-degenerate path

After a fit that returns, a new helper `_detect_logistic_separation(m)` applies:

```
converged  = bool(m.mle_retvals.get("converged", True))
degenerate = (any non-finite std error in m.bse)
             OR (max|coef| > COEF_CEILING)
             OR (max|SE|  > SE_CEILING)

→ perfect_separation   iff   (not converged) AND degenerate
→ convergence_failed   iff   (not converged) AND NOT degenerate
→ None (clean fit)     iff   converged
```

- Thresholds are module-level constants with comments citing the measured values:
  `SE_CEILING = 1e3`, `COEF_CEILING = 1e3`. The control logit's max SE was ≈ 0.87
  and separated fits were ≥ 1e5, so 1e3 sits in a wide empty gap.
- The `not converged` gate makes false positives on well-behaved data essentially
  impossible (a clean fit converges and is never magnitude-tested).
- `convergence_failed` reuses the type negbin already emits, so plain
  non-convergence is never silently returned as `ok: true`.

## 4. Surface & side-effects

- `src/data_analyst_mcp/tools/models.py` — detection logic in the logistic branch of
  `_fit_dispatch`, one new helper `_detect_logistic_separation`, two threshold
  constants, and one separation/convergence error-builder.
- `docs/SPEC.md` §5.11 (`fit_model`) — document `perfect_separation` and
  `convergence_failed` as logistic error types. (5.11 lists error types inline in
  prose rather than in a dedicated `Errors:` line; extend the prose accordingly.)
- `ROADMAP.md` — strike the now-resolved "Logistic-separation handling" bullet under
  *Statistics & modeling*.
- No new dependencies. No change to `predict` / `evaluate_model` (a separated model is
  never registered, so they can never observe one).

## 5. Testing (strict red→green per repo discipline)

Each `src/` change lands as a failing `red:` test then a `green:` implementation,
enforced by `scripts/check_tdd_commits.py`.

### 5.0 Fixtures (known-answer, all signatures verified on statsmodels 0.14.6)

statsmodels' returned-vs-raised behavior is sensitive to the exact data, so the
fixtures are pinned here rather than left to the implementer. "design rank" is
`matrix_rank(model.exog)` vs the column count — the discriminator the LinAlgError
branch uses.

| Case | formula | data | design rank | statsmodels 0.14.6 signature | expected `fit_model` |
|------|---------|------|-------------|------------------------------|----------------------|
| Complete sep | `y ~ x` | `y=[0,0,0,0,1,1,1,1]`, `x=[1,2,3,4,5,6,7,8]` | 2/2 **full** | **raises** `LinAlgError` | `perfect_separation` |
| Quasi-complete sep | `y ~ x` | `y=[0,0,0,1,0,1,1,1]`, `x=[1,2,3,4,4,5,6,7]` | 2/2 **full** | **returns** `converged=False`, max\|SE\|≈2.7e5 | `perfect_separation` |
| Categorical perfect | `y ~ C(g)` | `y=[0,0,1,1,1,1]`, `g=[a,a,a,b,b,b]` | 2/2 **full** | **returns** `converged=False`, max\|SE\|≈4.6e6 | `perfect_separation` |
| Perfect collinearity | `y ~ x1 + x2` | `y=[0,1]*6`, `x1=range(12)`, `x2=2*x1` | 2/3 **deficient** | **raises** `LinAlgError` | `formula_error` (NOT separation) |
| Well-behaved control | `y ~ x` | `y=[0,1]*10`, `x=range(20)` | 2/2 **full** | **returns** `converged=True`, max\|SE\|≈0.87 | `ok: true` |

### 5.1 Integration tests (`tests/test_models.py`, through the public `fit_model`)

1. Complete separation → `perfect_separation`, **not** `formula_error`.
2. Quasi-complete separation → `perfect_separation`, **not** `ok: true`.
3. Categorical perfect predictor → `perfect_separation`.
4. Regression guards (no false positives): well-behaved control still `ok: true`;
   missing column / bad formula still `formula_error`.
5. Perfect collinearity → `formula_error`, explicitly asserting **not** `perfect_separation`
   (exercises the rank discriminator).
6. Side-effects on separation: no model registered when `model_name` is supplied; no
   recorder cell appended.

### 5.2 Helper-level unit test (`_detect_logistic_separation`)

7. `convergence_failed` branch. **No robust public-API fixture reaches it**: the only
   natural non-converged logistic fixtures also blow up (the quasi-sep fixture above is
   `converged=False` *with* max\|coef\|≈92 / max\|SE\|≈2.7e5, so it classifies as
   separation), and `fit_model` exposes no `maxiter` to force clean non-convergence.
   Cover the branch deterministically by calling `_detect_logistic_separation` with a
   lightweight stub result (`mle_retvals={"converged": False}`, small finite `bse`/`params`)
   and assert it returns `convergence_failed`. The branch is retained as a **defensive
   guard** so a non-converged-but-not-degenerate fit is never returned as `ok: true`.

## 6. Acceptance criteria

- All three separation fixtures return `{"ok": false, "error": {"type": "perfect_separation", ...}}`.
- Well-behaved logit and genuine formula errors are unchanged.
- A separated fit registers no model and emits no recorder cell.
- `ruff format/check`, `pyright`, full `pytest`, and `check_tdd_commits.py` all green.

## 7. Out of scope

- Penalized / Firth / exact logistic (a separate tool/proposal; named only in the hint).
- Naming the specific separating predictor/level in the error (possible follow-up; v1
  hint tells the user how to find it via cross-tabs).
- OLS / Poisson / negbin behavior.
- The version-drift in the roadmap text beyond striking the resolved bullet.
