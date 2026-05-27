# `regression_line` + `residual_diagnostic`

Two new plot kinds for the model registry. They ship together because
they share the OLS-only gate, the model-lookup pattern from
`tools/predict.py:154-161`, and the figure-rendering helpers from
`tools/plots.py:218-285` (`_make_figure`, `_apply_style`,
`render_to_base64`).

## Purpose

`fit_model` returns Breusch–Pagan, Durbin–Watson, and Jarque–Bera
numerically, but agents systematically under-use those numbers without
the corresponding picture. `regression_line` renders the
predictor-vs-response scatter with the fitted line and 95 % mean-CI
band; `residual_diagnostic` renders the four canonical residual plots
(residuals vs fitted, Q-Q, scale-location, residuals vs leverage with
Cook's distance). Both are OLS-only — logistic / Poisson / negbin have
different canonical diagnostics and are out of scope for v1.

## Input

```python
class RegressionLineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    predictor: str              # which numeric predictor's axis to plot against
    title: str | None = None


class ResidualDiagnosticInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    kind: Literal["resid_vs_fitted", "qq", "scale_location", "all"] = "all"
    title: str | None = None
```

## Behavior

### `regression_line`

1. Look up the model: `entry = session.get_model(payload.model_name)`
   (pattern from `tools/predict.py:154-161`). Missing →
   `model_not_found`.
2. **OLS-only guard.** If `entry.kind != "ols"` →
   `regression_diagnostics_ols_only`.
3. Pull the training DataFrame via
   `con.execute(f'SELECT * FROM "{entry.fitted_on_dataset}"').df()`.
4. Validate that `payload.predictor` appears in
   `entry._result.model.exog_names` after dropping `Intercept`. Missing
   → `column_not_found`. Non-numeric in the training DataFrame →
   `non_numeric_predictor`.
5. Build the scatter of `(predictor, entry._result.model.endog)` via
   `_make_figure()` (`tools/plots.py:218-254`).
6. Overlay the fitted line by predicting across a dense grid of
   `predictor` values while holding every other predictor at its mean.
   For OLS, `get_prediction(exog).summary_frame()` returns
   `mean_ci_lower` / `mean_ci_upper` — render the 95 % mean-CI band as
   a shaded region.
7. Apply `_apply_style(...)` for axis labels and the optional `title`,
   then `render_to_base64(fig)` (`tools/plots.py:257-284`).

### `residual_diagnostic`

1. Same lookup + OLS guard as above.
2. Compute the diagnostic series once:
   - `fitted = entry._result.fittedvalues`
   - `resid = entry._result.resid`
   - `infl = entry._result.get_influence()`
   - `standardized_resid = infl.resid_studentized_internal`
3. Render based on `kind`:
   - **`resid_vs_fitted`**: scatter `(fitted, resid)` with horizontal
     `y=0` line plus a LOWESS overlay
     (`statsmodels.nonparametric.smoothers_lowess.lowess`).
   - **`qq`**: `scipy.stats.probplot(resid)` on the supplied axes.
   - **`scale_location`**: scatter
     `(fitted, sqrt(|standardized_resid|))` plus LOWESS overlay.
   - **`all`**: 2×2 grid; the fourth panel is residuals vs leverage
     with Cook's distance contours via `infl.cooks_distance`.
4. Use `Figure(figsize=(12, 9))` for `kind="all"` and the default
   `(8, 6)` otherwise. Apply `_apply_style(...)` per panel, then
   `render_to_base64(fig)`.

## Output

Identical shape across both tools, mirroring existing `plot` output:

```python
{
    "ok": True,
    "png_base64": str,        # base64-encoded PNG bytes
    "width": int,             # pixels
    "height": int,            # pixels
    "model_name": str,        # echoes payload.model_name
    "plot_kind": str,         # "regression_line" | "resid_vs_fitted" | "qq" | "scale_location" | "all"
}
```

## Errors

- `model_not_found` — `model_name` not in the model registry.
- `regression_diagnostics_ols_only` — `entry.kind in {"logistic",
  "poisson", "negbin"}`. **Non-negotiable** — logistic / Poisson /
  negbin have separate diagnostic patterns and are out of scope for v1.
  This is a new error type added to `src/data_analyst_mcp/errors.py`.
- `column_not_found` — `regression_line` only; `payload.predictor`
  is not among the model's predictors.
- `non_numeric_predictor` — `regression_line` only; the named
  predictor exists but has a non-numeric dtype in the training
  DataFrame (e.g. it was wrapped in `C(...)` in the formula).
- `internal` — unexpected exception during plotting.

## Recorder cells

Match the existing `plot` recorder pattern (`tools/plots.py:96-102`):
markdown describes the plot, code reproduces it from the model object
and training DataFrame.

```python
# regression_line
get_recorder().record(
    markdown=(
        f"### Regression line for `{payload.model_name}` "
        f"on predictor `{payload.predictor}`"
    ),
    code=(
        f'{payload.model_name}_df = con.sql('
        f'"SELECT * FROM {entry.fitted_on_dataset}").df()\n'
        f"_pred = {payload.model_name}.get_prediction(...)\n"
        f"# scatter + fitted line + 95% mean-CI band\n"
    ),
    tool_name="regression_line",
)

# residual_diagnostic
get_recorder().record(
    markdown=(
        f"### Residual diagnostic ({payload.kind}) for `{payload.model_name}`"
    ),
    code=(
        f"_resid = {payload.model_name}.resid\n"
        f"_fitted = {payload.model_name}.fittedvalues\n"
        f"# render {payload.kind} panel(s)\n"
    ),
    tool_name="residual_diagnostic",
)
```

## TDD slices

~16 cycles total.

**`regression_line` (8 cycles):**
1. `model_not_found`.
2. OLS-only guard: logistic raises `regression_diagnostics_ols_only`.
3. OLS-only guard: poisson raises `regression_diagnostics_ols_only`.
4. `column_not_found` when `predictor` is not in the model.
5. `non_numeric_predictor` when the predictor is categorical /
   string-typed in the training DataFrame.
6. Known-output fixture: PNG dims > 0, non-empty base64 bytes.
7. Scatter contains the expected number of points
   (`len(entry._result.model.endog)`).
8. Fitted-line slope matches `entry._result.params[predictor]`; 95 %
   mean-CI band is rendered (non-None patch on the axes).

**`residual_diagnostic` (8 cycles):**
9. `model_not_found`.
10. OLS-only guard fires for non-OLS kinds.
11. `kind="resid_vs_fitted"` produces a single-axes figure.
12. `kind="qq"` produces a single-axes figure.
13. `kind="scale_location"` produces a single-axes figure.
14. `kind="all"` produces a 2×2 grid (4 axes on the figure).
15. LOWESS overlay present on `resid_vs_fitted` (axes have ≥ 2 lines).
16. Cook's distance computed and rendered for the `all` 4th panel.

## Acceptance criteria

- All ~16 TDD cycles green with `red:` / `green:` / `refactor:`
  commits.
- `tests/test_plots.py` extended cases pass under `uv run pytest -q`.
- `evals/eval_diagnostic_plots.py` passes: fit OLS on `opps`, call
  both new tools, verify PNG headers + dimensions.
- Existing `plot` characterization tests unchanged (no regression in
  `_make_figure` / `_apply_style` / `render_to_base64` helpers).
- `ruff check .`, `ruff format --check .`, `pyright src/`, and
  `scripts/check_tdd_commits.py` all green.

## ROADMAP impact

- Removes the `regression_line` / `residual_diagnostic` lines from
  `ROADMAP.md` "Statistics & modeling". `partial_regression` stays on
  the roadmap (deferred — separate proposal).
- Adds both tools to SPEC §5 (alongside existing `plot`). The new
  error type `regression_diagnostics_ols_only` is added to SPEC §7
  (error catalogue) and to `src/data_analyst_mcp/errors.py`.
- Bumps the tool count in `README.md` and SPEC §1.
