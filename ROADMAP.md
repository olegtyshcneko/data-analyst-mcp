# Roadmap

The v1.x tool surface is **21 tools** — this is the v2 boundary. v1 closed at 11 (spec §5); `adjust_pvalues` (Phase 1), `analyze_missingness` (Phase 3, descriptive + Phase 4 Little's MCAR), and the model registry trio (Phase 5: `list_models`, `predict`, `evaluate_model`, plus the additive `fit_model(model_name=...)` storage path) shipped as waivered additions. The tier-1 bundle (`materialize_query`, `find_outliers`, `power_analysis`, `regression_line`, `residual_diagnostic`) brought the count from 16 → 21. Everything below is parked. New tools go through an issue and a design conversation first — see `README.md` "Contributing".

**Reproducibility caveat (Phase 5).** Statsmodels Results objects are not reliably picklable across kernel boundaries, so the model registry holds them in-process only. The emitted notebook works around this by re-fitting every registered model in its setup cell, guarded by a hard SHA-256 assert on the training file (above the 100 MB ceiling we fall back to `(path, mtime, size)` and accept the weaker guarantee). If the training CSV is edited between session and notebook replay, the setup cell raises `AssertionError` rather than silently producing different numbers.

No active proposals (the tier-1 bundle closed the current cohort; see `docs/proposals/README.md`).

## Tooling

- **Streaming-source support.** Spec §1 calls streaming out of scope, but a thin adapter that buffers a fixed window from a streaming source and registers it as a dataset would unlock "live tail" use-cases without rewriting the engine.
- **SSE / HTTP transport.** v1 is stdio-only. SSE would let the server run remotely and serve multiple clients.
- **`partial_regression` plot kind.** Today `plot` covers the 7 EDA-shape kinds and the tier-1 bundle added `regression_line` + `residual_diagnostic` for OLS diagnostics; `partial_regression` (added-variable plots) is the remaining canonical visual not yet shipped.
- **`summarize_correlations` convenience.** Sits on top of `correlate` and picks the top-N significant pairs with effect-size threshold. Useful when the user has 30+ columns and an N×N matrix is unreadable.
- **Time-series helpers.** Decomposition (STL), autocorrelation, stationarity (ADF) — only if a clear user surfaces. Spec §1 deliberately excludes time-series-first workflows.

## Statistics & modeling

- **Logistic-separation handling.** `fit_model(kind="logistic")` currently emits a statsmodels warning when `PerfectSeparationError` is raised, and the response carries on with `NaN` standard errors. Should translate to a structured `{"error": {"type": "perfect_separation", "hint": "..."}}` and skip the diagnostic block. Surfaced in Phase 7.
- **Mixed-effects models** (`MixedLM`). Currently only OLS / logistic / Poisson / negbin are supported. (The Poisson `overdispersion` warning now has an in-server remedy: `fit_model(kind="negbin")`.)
- **Bayesian alternatives.** A `fit_bayesian` tool over a thin PyMC wrapper, for the cases where p-values aren't the right deliverable.
- **Bootstrap CIs everywhere.** `compare_groups` and `correlate` could return bootstrap CIs alongside the parametric ones, with a `bootstrap_iters` knob.

## Reproducibility

- **`load_session_from_notebook`.** The inverse of `emit_notebook` — read a previously-emitted `.ipynb` and rehydrate the dataset registry from the setup cell. Useful for "pick up where I left off."
- **Notebook diff.** `compare_notebooks(a, b)` — compares two emitted sessions, highlights numeric drift.
- **Provenance hashes.** Each dataset entry could record a content hash of the source file so the emitted notebook can fail loudly if the file was edited between session and replay.

## Polish / DX

- **Better default plot styling.** Matplotlib's default style sheet is acceptable but uninspired; ship a single curated style.
- **Cursor-based pagination on `query`.** The `cursor` field is plumbed in but the implementation paginates by re-running with `OFFSET`. A true cursor (DuckDB result-set streaming) would handle 10M-row result inspection better.
- **More plot accessibility.** Embed a text-summary alongside the PNG (`{"png_base64": "...", "text_summary": "histogram of amount, peaked at..."}`) so non-visual clients still get something.

## Out of scope, forever

- Causal inference. (Spec §1.)
- BI / dashboard replacement. (Spec §1.)
- Streaming-first workflows beyond the buffered-window idea above. (Spec §1.)
- Datasets > 10 GB. (Spec §1.) If you're there, you want Spark / Trino, not us.
- Calling an LLM from inside the server. (Spec §11.) The server returns structured data; the agent does the reasoning.
- **`delete_model` tool.** Agent calls `session.reset()` to recover from a `model_name_collision`; the registry surface stays minimal.
- **Model export to disk.** Pickled statsmodels Results are fragile across versions; the emitted notebook is the export path.
- **GLM prediction intervals.** OLS gets `include_se=True` (delta-method-clean); proper GLM intervals (delta vs simulation) are their own proposal, deferred indefinitely.
