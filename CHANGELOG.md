# Changelog

All notable changes to **data-analyst-mcp** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-07-04

### Security
- **Fix arbitrary host-file read via DuckDB file functions ([#4]).** The `query`
  and `materialize_query` tools execute agent-supplied SQL, which could reach the
  host filesystem — `SELECT * FROM read_csv('/etc/passwd')`, the
  `SELECT * FROM '/etc/passwd.csv'` replacement scan (no function name), or
  `glob('/etc/*')`. The session's DuckDB connection now runs with
  `enable_external_access=false`, a one-way latch that blocks every file-read
  vector at once and cannot be re-enabled by agent SQL. A blocked read returns a
  structured `query_error` instead of leaking file contents.
- **`load_dataset` is unaffected.** File reading now happens on a separate
  short-lived connection with filesystem access; the loaded rows are handed to
  the sandboxed session connection in memory, so local paths, `s3://`, and
  `http(s)://` sources all keep working. The exported notebook still reproduces
  the load from the original file.

### Changed
- `query` now wraps execution errors (previously an unhandled path) in the
  structured `query_error` envelope, matching `materialize_query`.

[#4]: https://github.com/olegtyshcneko/data-analyst-mcp/issues/4

## [1.1.0] - 2026-07-04

Post-hoc pairwise comparisons — the follow-up to `compare_groups`. Tool surface **21 → 22**.

### Added
- **`pairwise_comparisons`** — post-hoc answer to *which pairs differ* after a
  significant omnibus test, over all `n·(n−1)/2` pairs of up to 20 groups:
  - **Tukey HSD** (via statsmodels `pairwise_tukeyhsd`) after one-way ANOVA, with
    confidence intervals; controls the family-wise error rate internally.
  - **Dunn's test** — a **vendored**, tie-corrected rank statistic (no new
    dependency; the §5.4 Little's-MCAR vendoring precedent) after Kruskal–Wallis,
    Holm-adjusted by default (`p_adjust` selects Bonferroni / Šidák / BH / BY).
  - `method="auto"` reuses `compare_groups`' Shapiro normality gate to pick the
    engine (normality holds → Tukey, violated → Dunn); the omnibus is recomputed
    inline so a non-significant family is caveated in the interpretation.
  - A 20-group cap bounds the quadratic comparison output, and every group is
    NULL-filtered so a leaked NaN can't corrupt the rank pooling.
  - Fully-runnable recorder cells: the emitted code rehydrates the group vectors
    from the notebook's DuckDB connection (single quotes `''`-doubled for the SQL
    `IN (...)` list), imports its own `pairwise_tukeyhsd` / `multipletests`, and
    reproduces the reported table.

### Documentation
- `docs/SPEC.md` §5.9a specifies `pairwise_comparisons`; the accepted proposal was
  folded in and `docs/proposals/pairwise_comparisons.md` deleted per the
  contributing convention.

## [1.0.2] - 2026-06-02

Feature set 2 — robust logistic separation handling (PR #2), plus follow-up polish.

### Changed
- `fit_model(kind="logistic")` now returns a structured **`perfect_separation`**
  error for both separation failure modes (statsmodels 0.14.6) instead of
  mislabeling them or returning garbage inference:
  - **Raised path** (complete separation → `LinAlgError: "Singular matrix"`):
    was mislabeled `formula_error`; now `perfect_separation`.
  - **Silent-return path** (quasi / categorical separation → `converged=False`
    with astronomically large standard errors): was returned as `ok: true`
    with meaningless inference (and registered the garbage model); now
    `perfect_separation`, with the model **not** registered and **no**
    recorder cell emitted.
- A **rank discriminator** keeps perfect collinearity (rank-deficient design)
  classified as `formula_error`, distinct from true separation.
- The logistic separation guards were aligned with the existing `negbin`
  two-stage precedent for a consistent construction-guard + fit-guard shape.

### Added
- Defensive **`convergence_failed`** error for a non-converged logit that
  lacks the divergence signature.
- Wide WDBC fixture for high-dimensional `find_outliers` evals.

### Fixed
- `power_analysis` interpretation prose now reflects an unequal allocation
  `ratio` for `two_sample_t` / `two_proportion_z` (reads `n1=A, n2=B
  (T total)`), instead of mislabeling an unbalanced design as "n per group"
  while `n_total` said otherwise. Computed values were always correct;
  only the human-readable string was wrong (PR #3).

### Documentation
- `docs/SPEC.md` §5.11 documents the new logistic error types.
- README "known gotchas" synced with the new `perfect_separation` behavior.

## [1.0.1] - 2026-05-31

Feature set 1 — Tier-1 tool bundle (PR #1). Tool surface **16 → 21**.

### Added
- **`materialize_query`** — persist a `SELECT`/`WITH` result as a named
  derived dataset (unblocks cohort / funnel / multi-step join workflows).
- **`find_outliers`** — multi-column outlier detection: IQR, z-score,
  Mahalanobis, and Isolation Forest.
- **`power_analysis`** — sample-size / power / MDE for five test families
  (two-sample t, one-sample t, paired t, two-proportion z, one-way ANOVA).
- **`regression_line`** — OLS fit with confidence band.
- **`residual_diagnostic`** — OLS residual panels (residuals-vs-fitted, Q-Q,
  scale-location, residuals-vs-leverage with Cook's distance contours).

### Fixed
- Cook's distance contour formula in `residual_diagnostic` was off by orders
  of magnitude (now `sqrt(D·p·(1-h)/h)`).

### Security
- SQL-injection guard rejecting multi-statement (semicolon) payloads in both
  `query` and `materialize_query`.
- Recorder cells now quote user SQL/paths via `repr()` so values containing
  quotes no longer break `jupyter nbconvert --execute`.

## [1.0.0] - 2026-05-25

Initial release — a reproducible, notebook-emitting data-analysis MCP server
with **16 tools** over a DuckDB session.

### Added
- **Data I/O & exploration:** `load_dataset`, `list_datasets`,
  `profile_dataset`, `describe_column`, `query` (read-only SQL with
  auto-`LIMIT`).
- **Statistics:** `correlate` (Pearson/Spearman/Kendall), `test_hypothesis`
  (t / Welch / Mann-Whitney / chi-square / Fisher / ANOVA / Kruskal-Wallis / KS),
  `compare_groups` (auto test selection with assumption checks),
  `adjust_pvalues` (Bonferroni / Šidák / Holm / BH / BY), `analyze_missingness`.
- **Modeling:** `fit_model` (OLS, logistic, Poisson, negative binomial — with
  diagnostics and warnings), `predict`, `evaluate_model`, `list_models`.
- **Visualization & reproducibility:** `plot` (hist/bar/line/scatter/box/
  violin/heatmap), `emit_notebook` — every tool call records a markdown+code
  cell pair so a session replays as an executable Jupyter notebook.

[1.1.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.1.0
[1.0.2]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.0.2
[1.0.1]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.0.1
[1.0.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.0.0
