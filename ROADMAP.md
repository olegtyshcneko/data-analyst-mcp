# Roadmap

The v1 tool surface is **closed at 11 tools** (spec §5). Everything below is parked. New tools go through an issue and a design conversation first — see `README.md` "Contributing".

## Tooling

- **Register-query-as-dataset.** Right now `query` rejects any non-SELECT, which means a user who wants to keep a join-derived table around has to re-run the join inline every time. A `materialize_query(sql, name)` tool would `CREATE OR REPLACE TABLE name AS <sql>` and register it as a first-class dataset. Surfaced in Phase 7 evals.
- **Streaming-source support.** Spec §1 calls streaming out of scope, but a thin adapter that buffers a fixed window from a streaming source and registers it as a dataset would unlock "live tail" use-cases without rewriting the engine.
- **SSE / HTTP transport.** v1 is stdio-only. SSE would let the server run remotely and serve multiple clients.
- **More plot kinds.** `regression_line` (scatter + fitted line from a `fit_model` result), `residual_diagnostic` (residual-vs-fitted, QQ-plot, scale-location), and `partial_regression` for OLS. Today `plot` covers the 7 EDA-shape kinds but does not visualize model output.
- **`summarize_correlations` convenience.** Sits on top of `correlate` and picks the top-N significant pairs with effect-size threshold. Useful when the user has 30+ columns and an N×N matrix is unreadable.
- **Multi-column outlier detection.** `describe_column` flags per-column outliers; a `find_outliers(name, columns=[...])` would combine Mahalanobis distance and isolation-forest scoring for joint outliers.
- **Time-series helpers.** Decomposition (STL), autocorrelation, stationarity (ADF) — only if a clear user surfaces. Spec §1 deliberately excludes time-series-first workflows.

## Statistics & modeling

- **Logistic-separation handling.** `fit_model(kind="logistic")` currently emits a statsmodels warning when `PerfectSeparationError` is raised, and the response carries on with `NaN` standard errors. Should translate to a structured `{"error": {"type": "perfect_separation", "hint": "..."}}` and skip the diagnostic block. Surfaced in Phase 7.
- **Mixed-effects models** (`MixedLM`). Currently only OLS / logistic / Poisson are supported.
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
