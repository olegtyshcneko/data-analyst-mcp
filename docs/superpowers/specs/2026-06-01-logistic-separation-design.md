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

### 3.1 Raised path

Wrap the logit `.fit()` and catch specifically:

- `statsmodels.tools.sm_exceptions.PerfectSeparationError` → `perfect_separation`
  (unambiguous).
- `numpy.linalg.LinAlgError` (e.g. "Singular matrix") → **ambiguous**: separation
  *or* perfectly collinear predictors. Discriminate by design-matrix rank:
  - `matrix_rank(exog) == n_cols` (full rank) → `perfect_separation`.
  - rank-deficient → it is collinearity, **not** separation; re-raise as
    `_FormulaError` with a "perfectly collinear predictors" message so it is not
    mislabeled.
- Any other exception → `_FormulaError` (unchanged). Patsy / missing-column errors
  raise at model **construction** (`smf.logit(formula, data=df)`), before `.fit()`,
  so they remain `formula_error`.

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
enforced by `scripts/check_tdd_commits.py`. New tests in `tests/test_models.py`:

1. Complete separation (continuous) → `perfect_separation`, **not** `formula_error`.
2. Quasi-complete separation → `perfect_separation`, **not** `ok: true`.
3. Categorical perfect predictor → `perfect_separation`.
4. Regression guards (no false positives):
   - well-behaved logit still returns `ok: true`;
   - missing column / bad formula still returns `formula_error`.
5. Perfectly collinear predictors → `formula_error` (the rank discriminator routes a
   rank-deficient design here), explicitly asserting it is **not** `perfect_separation`.
6. Side-effects on separation: no model registered when `model_name` is supplied; no
   recorder cell appended.
7. Plain non-convergence (non-separated) → `convergence_failed`.

Known-answer fixtures (per SPEC §"stats tools" obligation) are the small hand-built
separation tables measured in §1.

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
