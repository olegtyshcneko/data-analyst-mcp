# Logistic `perfect_separation` Error — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fit_model(kind="logistic")` return a structured `perfect_separation` error for both the raised (complete-separation) and silently-returned (quasi/categorical separation) failure modes, instead of mislabeling the first as `formula_error` and presenting the second as `ok: true` with garbage inference.

**Architecture:** Restructure only the logistic branch of `_fit_dispatch` in `src/data_analyst_mcp/tools/models.py` into a two-stage guarded helper, mirroring the existing `negbin` precedent. Stage 1 (construction) maps patsy/column errors to `formula_error`. Stage 2 (fit) maps a raised `PerfectSeparationError`/`LinAlgError` to `perfect_separation` when the design matrix is full-rank, and to `formula_error` (perfect collinearity) when rank-deficient. A returned-but-degenerate fit (`converged=False` plus a coefficient/SE blow-up) is reclassified by a new `_detect_logistic_separation` helper; non-convergence without that signature becomes `convergence_failed`. Error results already skip registration and recorder cells because `fit_model` gates both on `result.get("ok")`.

**Tech Stack:** Python 3.13/3.14, statsmodels 0.14.6 (`smf.logit`), patsy, numpy, pytest (fixtures `call_tool`, `load_df_into_session`), ruff, pyright, coverage (`--cov-fail-under=90`, branch on), `scripts/check_tdd_commits.py`.

**Spec:** `docs/superpowers/specs/2026-06-01-logistic-separation-design.md`. One refinement vs spec §3.1: the two `except` clauses (`PerfectSeparationError`, `LinAlgError`) are merged into a single `except (PerfectSeparationError, LinAlgError)` with the rank discriminator inside — on 0.14.6 complete separation raises `LinAlgError`, not `PerfectSeparationError`, so merging keeps the branch covered while preserving identical semantics.

---

## File Structure

- **Modify:** `src/data_analyst_mcp/tools/models.py`
  - New module-level constants `_SEP_SE_CEILING`, `_SEP_COEF_CEILING`.
  - New helpers `_perfect_separation_error()`, `_logistic_convergence_error()`, `_detect_logistic_separation(m)`, `_fit_logistic_or_error(smf, payload, df)`.
  - Edit the logistic dispatch inside `_fit_dispatch` (currently `models.py:449-460`).
- **Modify:** `tests/test_models.py` — new tests appended near the existing logistic block (helpers `_logistic_df`, `_logistic_collinear_df` already exist there).
- **Modify:** `docs/SPEC.md` §5.11 (`fit_model`) — document `perfect_separation` / `convergence_failed`.
- **Modify:** `ROADMAP.md` — remove the resolved "Logistic-separation handling" bullet.

All test fixtures are the known-answer tables pinned in spec §5.0 (verified on statsmodels 0.14.6).

---

## Task 1: Complete separation (raised path) → `perfect_separation`

Introduces the two-stage helper, the `perfect_separation` error builder, and the rank discriminator. Also lands two regression guards (perfect collinearity → `formula_error`; logistic missing column → `formula_error`) that exercise the new construction/rank branches.

**Files:**
- Modify: `src/data_analyst_mcp/tools/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test + two regression guards**

Append to `tests/test_models.py`:

```python
# === Logistic perfect separation ===========================================
# Fixtures + signatures verified on statsmodels 0.14.6; see
# docs/superpowers/specs/2026-06-01-logistic-separation-design.md §5.0.


def _logistic_complete_sep_df() -> pd.DataFrame:
    """Complete separation: y is exactly determined by x. fit() RAISES LinAlgError."""
    return pd.DataFrame(
        {"y": [0, 0, 0, 0, 1, 1, 1, 1], "x": [1, 2, 3, 4, 5, 6, 7, 8]}
    )


def _logistic_perfect_collinear_df() -> pd.DataFrame:
    """Perfectly collinear design (x2 == 2*x1), outcome NOT separated.

    fit() RAISES LinAlgError too, but the design is rank-deficient (rank 2 of 3),
    so it must be reported as formula_error, NOT perfect_separation.
    """
    rng = list(range(12))
    return pd.DataFrame({"y": [0, 1] * 6, "x1": rng, "x2": [2 * v for v in rng]})


def test_fit_model_logistic_complete_separation_returns_perfect_separation(
    call_tool, load_df_into_session
):
    load_df_into_session("sep", _logistic_complete_sep_df())
    result = call_tool("fit_model", {"name": "sep", "formula": "y ~ x", "kind": "logistic"})
    assert result["ok"] is False, result
    assert result["error"]["type"] == "perfect_separation"


def test_fit_model_logistic_perfect_collinearity_is_formula_error_not_separation(
    call_tool, load_df_into_session
):
    load_df_into_session("coll", _logistic_perfect_collinear_df())
    result = call_tool(
        "fit_model", {"name": "coll", "formula": "y ~ x1 + x2", "kind": "logistic"}
    )
    assert result["ok"] is False, result
    # Rank-deficient design ⇒ collinearity, routed to formula_error.
    assert result["error"]["type"] == "formula_error"
    assert result["error"]["type"] != "perfect_separation"


def test_fit_model_logistic_missing_column_stays_formula_error(
    call_tool, load_df_into_session
):
    load_df_into_session("logi", _logistic_df())
    result = call_tool(
        "fit_model", {"name": "logi", "formula": "y ~ nope", "kind": "logistic"}
    )
    assert result["ok"] is False, result
    assert result["error"]["type"] == "formula_error"
```

- [ ] **Step 2: Run tests to verify the driver fails**

Run: `uv run pytest tests/test_models.py -k "complete_separation or perfect_collinearity or logistic_missing_column" -q`
Expected: `test_..._complete_separation_returns_perfect_separation` FAILS (currently returns `formula_error: "Singular matrix"`); the two guard tests PASS (already `formula_error`).

- [ ] **Step 3: Add the constants, error builder, and two-stage helper**

In `src/data_analyst_mcp/tools/models.py`, insert immediately **before** `def _fit_dispatch` (currently line 422, after the `_NEGBIN_CONVERGENCE_HINT` block):

```python
_PERFECT_SEPARATION_HINT = (
    "Drop or collapse the predictor/level that perfectly predicts the outcome; "
    "cross-tab the outcome against each categorical predictor to find it. "
    "Penalized (Firth/L2) logistic — not yet offered by this server — is the "
    "standard remedy."
)


def _perfect_separation_error() -> dict[str, Any]:
    """Structured error for a logistic fit defeated by (quasi-)perfect separation."""
    return build_error(
        type="perfect_separation",
        message=(
            "Logistic fit failed: the outcome is perfectly (or quasi-perfectly) "
            "separated by the predictors, so the maximum-likelihood estimates diverge."
        ),
        hint=_PERFECT_SEPARATION_HINT,
    )


def _fit_logistic_or_error(smf: Any, payload: FitModelInput, df: Any) -> Any:
    """Fit a logistic model, returning the fitted Results OR a ``build_error`` dict.

    Two stages with distinct guards:
      * construction (``smf.logit``) — patsy builds the design matrix eagerly, so
        formula / missing-column errors raise here as ``PatsyError`` / ``NameError``
        and become ``formula_error``;
      * fit (``.fit``) — a degenerate logit raises ``PerfectSeparationError`` or
        ``LinAlgError``. A full-rank design means the MLE diverged → ``perfect_separation``;
        a rank-deficient design means perfect collinearity → ``formula_error`` (so it is
        not mislabeled).
    """
    import numpy as np
    import patsy  # type: ignore[reportMissingTypeStubs]
    from numpy.linalg import LinAlgError
    from statsmodels.tools.sm_exceptions import (  # type: ignore[reportMissingTypeStubs]
        PerfectSeparationError,
    )

    try:
        model = smf.logit(payload.formula, data=df)
    except (patsy.PatsyError, NameError) as exc:
        raise _FormulaError(str(exc)) from exc
    try:
        m = model.fit(disp=0)
    except (PerfectSeparationError, LinAlgError) as exc:
        # Both signal a degenerate logit. Full-rank design ⇒ true separation;
        # rank-deficient design ⇒ perfect collinearity (a formula problem).
        exog: Any = np.asarray(model.exog)
        if int(np.linalg.matrix_rank(exog)) == exog.shape[1]:
            return _perfect_separation_error()
        raise _FormulaError(f"perfectly collinear predictors: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive: unexpected logit fit failure
        raise _FormulaError(str(exc)) from exc
    return m
```

- [ ] **Step 4: Wire the helper into `_fit_dispatch`**

In `src/data_analyst_mcp/tools/models.py`, replace the shared dispatch block (currently `models.py:449-460`):

```python
    else:
        try:
            if payload.kind == "ols":
                cov_type = "HC3" if payload.robust else "nonrobust"
                m = smf.ols(payload.formula, data=df).fit(cov_type=cov_type)
            elif payload.kind == "logistic":
                m = smf.logit(payload.formula, data=df).fit(disp=0)
            else:  # poisson
                m = smf.poisson(payload.formula, data=df).fit(disp=0)
        except Exception as exc:
            # Patsy / NameError / column-binding failures all bubble up here.
            raise _FormulaError(str(exc)) from exc
```

with:

```python
    elif payload.kind == "logistic":
        logit_result = _fit_logistic_or_error(smf, payload, df)
        if isinstance(logit_result, dict):
            return logit_result
        m = logit_result
    else:
        try:
            if payload.kind == "ols":
                cov_type = "HC3" if payload.robust else "nonrobust"
                m = smf.ols(payload.formula, data=df).fit(cov_type=cov_type)
            else:  # poisson
                m = smf.poisson(payload.formula, data=df).fit(disp=0)
        except Exception as exc:
            # Patsy / NameError / column-binding failures all bubble up here.
            raise _FormulaError(str(exc)) from exc
```

- [ ] **Step 5: Run tests to verify all three pass**

Run: `uv run pytest tests/test_models.py -k "complete_separation or perfect_collinearity or logistic_missing_column" -q`
Expected: 3 passed.

- [ ] **Step 6: Run the full logistic suite to check for regressions**

Run: `uv run pytest tests/test_models.py -k logistic -q`
Expected: all pass (existing pinned-coefficient, pseudo-R², diagnostics, multicollinearity, interpretation, recorder, bool-coercion tests unchanged).

- [ ] **Step 7: Commit (red then green in two commits)**

```bash
git add tests/test_models.py
git commit -m "red: logistic perfect separation returns perfect_separation error"
git add src/data_analyst_mcp/tools/models.py
git commit -m "green: logistic perfect separation returns perfect_separation error"
```

> Note for the worker: the gate (`check_tdd_commits.py`) pairs `green:` with the immediately preceding `red:` by identical suffix. Stage the test file in the `red:` commit and the source file in the `green:` commit, exactly as above.

---

## Task 2: Quasi / categorical separation (returned path) → `perfect_separation`, with no registration or recorder cell

Adds `_detect_logistic_separation` (perfect-separation branch only) and wires it into the helper. Covers the silently-returned degenerate fits and proves the side-effects (no model registered, no recorder cell) that follow from returning `ok: false`.

**Files:**
- Modify: `src/data_analyst_mcp/tools/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def _logistic_quasi_sep_df() -> pd.DataFrame:
    """Quasi-complete separation: one overlapping point at x=4. fit() RETURNS
    converged=False with max|SE| ~ 2.7e5."""
    return pd.DataFrame(
        {"y": [0, 0, 0, 1, 0, 1, 1, 1], "x": [1, 2, 3, 4, 4, 5, 6, 7]}
    )


def _logistic_categorical_sep_df() -> pd.DataFrame:
    """A categorical level perfectly predicts the outcome. fit() RETURNS
    converged=False with max|SE| ~ 4.6e6."""
    return pd.DataFrame(
        {"y": [0, 0, 1, 1, 1, 1], "g": ["a", "a", "a", "b", "b", "b"]}
    )


def test_fit_model_logistic_quasi_separation_returns_perfect_separation(
    call_tool, load_df_into_session
):
    load_df_into_session("quasi", _logistic_quasi_sep_df())
    result = call_tool("fit_model", {"name": "quasi", "formula": "y ~ x", "kind": "logistic"})
    assert result["ok"] is False, result
    assert result["error"]["type"] == "perfect_separation"


def test_fit_model_logistic_categorical_perfect_predictor_returns_perfect_separation(
    call_tool, load_df_into_session
):
    load_df_into_session("cat", _logistic_categorical_sep_df())
    result = call_tool("fit_model", {"name": "cat", "formula": "y ~ C(g)", "kind": "logistic"})
    assert result["ok"] is False, result
    assert result["error"]["type"] == "perfect_separation"


def test_fit_model_logistic_separation_skips_registration_and_recorder(
    call_tool, load_df_into_session
):
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("quasi", _logistic_quasi_sep_df())
    result = call_tool(
        "fit_model",
        {"name": "quasi", "formula": "y ~ x", "kind": "logistic", "model_name": "m_sep"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "perfect_separation"
    # A separated fit must not be registered and must not emit a notebook cell.
    assert "m_sep" not in session.get_models()
    assert get_recorder().cells == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -k "quasi_separation or categorical_perfect_predictor or skips_registration" -q`
Expected: all 3 FAIL — currently the quasi/categorical fits return `ok: true` (garbage SEs), so `m_sep` IS registered and a recorder cell IS appended.

- [ ] **Step 3: Add the detector and call it from the helper**

In `src/data_analyst_mcp/tools/models.py`, add these constants and helper immediately **before** `_fit_logistic_or_error`:

```python
# Separation magnitude ceilings for the logistic returned-but-degenerate check.
# Measured on statsmodels 0.14.6: a well-behaved logit's max |SE| was ~0.87,
# while perfectly/quasi-separated fits returned |SE| >= 1e5. 1e3 sits in the
# wide empty gap between the two regimes.
_SEP_SE_CEILING = 1e3
_SEP_COEF_CEILING = 1e3


def _detect_logistic_separation(m: Any) -> dict[str, Any] | None:
    """Classify a *returned* logistic fit.

    A logit that fails to converge AND shows the separation magnitude signature
    (non-finite, or astronomically large, standard errors / coefficients) is
    reported as ``perfect_separation``. A converged fit is clean (``None``).
    """
    import numpy as np

    converged = bool(m.mle_retvals.get("converged", True))
    if converged:
        return None
    bse: Any = np.asarray(m.bse, dtype=float)
    params: Any = np.asarray(m.params, dtype=float)
    nonfinite = not bool(np.all(np.isfinite(bse)))
    huge = bool(
        np.nanmax(np.abs(params)) > _SEP_COEF_CEILING
        or np.nanmax(np.abs(bse)) > _SEP_SE_CEILING
    )
    if nonfinite or huge:
        return _perfect_separation_error()
    return None
```

Then, in `_fit_logistic_or_error`, replace the final `return m` with:

```python
    degenerate = _detect_logistic_separation(m)
    if degenerate is not None:
        return degenerate
    return m
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models.py -k "quasi_separation or categorical_perfect_predictor or skips_registration" -q`
Expected: 3 passed.

- [ ] **Step 5: Confirm the well-behaved control is unaffected**

Run: `uv run pytest tests/test_models.py::test_fit_model_logistic_returns_pinned_coefficients -q`
Expected: PASS (a converged fit returns `None` from the detector and flows through to `ok: true`).

- [ ] **Step 6: Commit (red then green)**

```bash
git add tests/test_models.py
git commit -m "red: logistic quasi-separation returns perfect_separation and skips registration"
git add src/data_analyst_mcp/tools/models.py
git commit -m "green: logistic quasi-separation returns perfect_separation and skips registration"
```

---

## Task 3: Non-convergence without the separation signature → `convergence_failed`

Closes the remaining detector branch as a defensive guard. No public-API fixture reaches clean non-convergence (every natural non-converged logit also blows up, and `fit_model` exposes no `maxiter`), so this is tested at the helper level with stub results. The stub cases also pin the `nonfinite` magnitude sub-branch for branch coverage.

**Files:**
- Modify: `src/data_analyst_mcp/tools/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing helper-level test**

Append to `tests/test_models.py`:

```python
class _StubLogitResult:
    """Minimal stand-in for a statsmodels Logit Results, for _detect_logistic_separation.

    The public fit_model API exposes no maxiter, and every natural non-converged
    logistic fixture also exhibits the separation blow-up — so the convergence_failed
    branch (and the non-finite-SE branch) are exercised here with constructed inputs.
    """

    def __init__(self, converged: bool, bse: Any, params: Any) -> None:
        self.mle_retvals = {"converged": converged}
        self.bse = np.asarray(bse, dtype=float)
        self.params = np.asarray(params, dtype=float)


def test_detect_logistic_separation_classifies_each_regime():
    from data_analyst_mcp.tools.models import _detect_logistic_separation

    # Converged → clean fit, no error.
    assert _detect_logistic_separation(_StubLogitResult(True, [0.3, 0.4], [0.5, -0.2])) is None
    # Not converged + blown-up SE → perfect_separation.
    huge = _detect_logistic_separation(_StubLogitResult(False, [0.3, 5e5], [1.0, 90.0]))
    assert huge is not None and huge["error"]["type"] == "perfect_separation"
    # Not converged + non-finite SE → perfect_separation.
    inf = _detect_logistic_separation(_StubLogitResult(False, [0.3, np.inf], [1.0, 2.0]))
    assert inf is not None and inf["error"]["type"] == "perfect_separation"
    # Not converged + small finite SE (no separation signature) → convergence_failed.
    conv = _detect_logistic_separation(_StubLogitResult(False, [0.3, 0.4], [0.5, -0.2]))
    assert conv is not None and conv["error"]["type"] == "convergence_failed"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_detect_logistic_separation_classifies_each_regime -q`
Expected: FAIL on the last assertion — the detector currently returns `None` for non-converged-but-not-degenerate input (no `convergence_failed` branch yet), so `conv is not None` fails.

- [ ] **Step 3: Add the `convergence_failed` builder and flip the detector fall-through**

In `src/data_analyst_mcp/tools/models.py`, add this builder immediately **after** `_perfect_separation_error`:

```python
def _logistic_convergence_error() -> dict[str, Any]:
    """Structured error for a logistic MLE that failed to converge *without* the
    coefficient/SE divergence signature of perfect separation."""
    return build_error(
        type="convergence_failed",
        message="Logistic MLE did not converge.",
        hint=(
            "The optimizer did not converge, but without the coefficient/SE blow-up "
            "of perfect separation. Check for near-collinear predictors or rescale "
            "numeric covariates."
        ),
    )
```

Then, in `_detect_logistic_separation`, change the final `return None` (reached only when `not converged` and the signature is absent) to:

```python
    return _logistic_convergence_error()
```

The leading `if converged: return None` guard is unchanged — a converged fit still returns `None`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_models.py::test_detect_logistic_separation_classifies_each_regime -q`
Expected: PASS.

- [ ] **Step 5: Commit (red then green)**

```bash
git add tests/test_models.py
git commit -m "red: logistic non-convergence without separation returns convergence_failed"
git add src/data_analyst_mcp/tools/models.py
git commit -m "green: logistic non-convergence without separation returns convergence_failed"
```

---

## Task 4: Documentation — SPEC §5.11 and ROADMAP

**Files:**
- Modify: `docs/SPEC.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Document the new logistic errors in SPEC §5.11**

In `docs/SPEC.md`, find the `model_name` bullet in the §5.11 `fit_model` **Input** list (currently line 436, ending `When stored, the response echoes "model_name": "..."`.) and insert a new bullet immediately after it:

```markdown
- **Logistic separation/convergence:** for `kind="logistic"`, a perfectly or quasi-perfectly separated outcome returns `{"ok": false, "error": {"type": "perfect_separation", ...}}` — no `coefficients`/`fit`/`diagnostics`, and the model is not registered even if `model_name` was supplied. A logit that fails to converge *without* that coefficient/SE divergence signature returns `convergence_failed`. Perfect collinearity in the design is reported as `formula_error`, not separation.
```

- [ ] **Step 2: Remove the resolved ROADMAP bullet**

In `ROADMAP.md`, delete the entire "Logistic-separation handling" bullet (currently line 19) under *## Statistics & modeling*:

```markdown
- **Logistic-separation handling.** `fit_model(kind="logistic")` currently emits a statsmodels warning when `PerfectSeparationError` is raised, and the response carries on with `NaN` standard errors. Should translate to a structured `{"error": {"type": "perfect_separation", "hint": "..."}}` and skip the diagnostic block. Surfaced in Phase 7.
```

Delete the whole line (and its trailing newline) so the *Statistics & modeling* list starts at the next bullet (`Mixed-effects models`).

- [ ] **Step 3: Commit**

```bash
git add docs/SPEC.md ROADMAP.md
git commit -m "docs: document logistic perfect_separation/convergence_failed; close roadmap item"
```

---

## Task 5: Full verification and branch finish

**Files:** none (verification only).

- [ ] **Step 1: Formatting and lint**

Run: `uv run ruff format --check . && uv run ruff check .`
Expected: clean.

- [ ] **Step 2: Types**

Run: `uv run pyright src/`
Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 3: Full test suite with coverage gate**

Run: `uv run pytest tests/ --cov=data_analyst_mcp --cov-branch --cov-report=term-missing --cov-fail-under=90`
Expected: all pass, coverage ≥ 90%. If `models.py` shows an uncovered new line, check it is the `# pragma: no cover` catch-all; every other new branch is exercised by Tasks 1–3.

- [ ] **Step 4: TDD discipline audit**

Run: `uv run python scripts/check_tdd_commits.py`
Expected: `TDD discipline OK` (the three red/green pairs from Tasks 1–3 each match by suffix; the Task 4 `docs:` commit is ignored).

- [ ] **Step 5: Evals (no regression)**

Run: `uv run pytest evals/ -q`
Expected: 44 passed.

- [ ] **Step 6: Finish the branch**

Invoke the `superpowers:finishing-a-development-branch` skill to push `logistic-separation-error` and open a PR against `main`. The PR description should summarize the two failure modes closed, cite spec entry, and paste the verification results from Steps 1–5.

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** §2 goal → Tasks 1+2; §3.1 two-stage branch → Task 1 Step 3; §3.2 returned-degenerate detector → Tasks 2+3; §4 surface (SPEC/ROADMAP) → Task 4; §5.0 fixtures → embedded in Task 1–2 test helpers; §5.1 integration tests 1–6 → Tasks 1–2; §5.2 helper test 7 → Task 3; §6 acceptance → Task 5.
- **No model registered / no recorder cell** is asserted directly (Task 2 Step 1) rather than assumed from the `ok`-gating, so a future regression in those gates is caught.
- **Coverage:** the only intentionally-uncovered new line is the `except Exception` catch-all in `_fit_logistic_or_error`, marked `# pragma: no cover` per repo convention (`src/data_analyst_mcp/server.py` uses the same). The `PerfectSeparationError` arm shares a branch with `LinAlgError`, so it is covered by the complete-separation fixture even though 0.14.6 raises `LinAlgError`.
