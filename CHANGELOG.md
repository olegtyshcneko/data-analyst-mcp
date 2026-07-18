# Changelog

All notable changes to **data-analyst-mcp** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - Unreleased

Session resume: `emit_notebook` now embeds a journal manifest
(`nb.metadata["data_analyst_mcp"]`) and the new tool 25,
`load_session_from_notebook`, replays it transactionally in a fresh
process — datasets, derived tables, splits, registered models, and the
recorder history all come back, and the next emit produces one unified
replayable notebook. Any divergence from the recorded evidence (source
hashes, table digests, split membership, model params/SEs) aborts
atomically with the live session untouched. Design spec:
`docs/superpowers/specs/2026-07-18-load-session-from-notebook-design.md`
(spec v4, three adversarial review passes).

### Added
- `load_session_from_notebook(path)` — three-phase resume: strict manifest
  validation (pydantic `extra="forbid"`, caps, per-cell SHA-256, source
  preflight), transactional journal replay with per-op evidence comparison
  (`source_drift`, `split_drift`, `model_drift`, `state_digest_mismatch`,
  `registry_mismatch`, …), atomic publish. 16-type error taxonomy; 300 s
  cooperative replay budget; degraded-evidence warnings for remote and
  `fallback:`-hashed sources.
- Operation journal: `load_dataset`, `materialize_query`, `split_dataset`,
  and registered `fit_model` now append structured journal entries at call
  time inside an op-transaction (BEGIN → mutate/digest → COMMIT → publish),
  each bound to its recorder cell pair via `op_id`.
- `damcp-digest-v1` (`digest.py`): order-sensitive SHA-256 table digest
  with a frozen byte layout; temporal columns projected to integer epochs
  in SQL so `TIMESTAMP_NS` survives at full resolution; `UNION`/`VARIANT`
  are undigestable and mark the session `resume_supported: false`.
- Emit-time manifest (`manifest.py`): journal + per-cell SHA-256
  descriptors + final-state digests + final-registry descriptor +
  independent `resume_supported` / `notebook_replayable` flags.

### Fixed
- Robust (`robust=True`) OLS models re-fit as *plain* OLS in the emitted
  setup cell, silently shrinking standard errors at replay. `ModelEntry`
  now records `fit_options` and the setup cell emits
  `.fit(cov_type="HC3")` for robust fits.

### Changed
- DuckDB minimum raised from 1.1.0 to **1.5.2** (transactional-DDL and
  type-surface envelope verified there; spec §5.14 version envelope).
- Tool surface 24 → 25 (waivered expansion — SPEC §11, ROADMAP ¶1).

## [1.5.0] - 2026-07-18

Prefix replay guards: emitted notebooks replay setup **then** the full
recorded history, and historical `load_dataset` cells re-read files
unguarded — the last way to make a notebook silently recompute on drifted
data. Every load cell now asserts its own load-time hash, and
`cross_validate` / unregistered `fit_model` cells on in-memory datasets
open with an explanatory raise. Tool surface unchanged (24).

### Fixed
- A source file mutated and reloaded mid-session replayed the pre-mutation
  cells (`cross_validate`, ephemeral and registered `fit_model` inputs,
  every analytic cell) against the new bytes with exit code 0 — the setup
  cell only asserts each dataset's *latest* registration. Each
  `load_dataset` cell now carries its own content assert (fallback digest
  above 100 MB; explanatory comment for remote sources), so replay fails
  loudly at the first load that saw the old bytes.
- `cross_validate` and unregistered `fit_model` cells recorded against
  in-memory (dataframe-registered) datasets failed replay with a bare
  `duckdb.CatalogException`; they now open with a purpose-written
  `AssertionError` naming the tool and dataset.

## [1.4.0] - 2026-07-14

Replay-guard hardening: registration revisions + fit-time loader identity +
recorded split-overwrite provenance + symmetric per-side split replay. Every
known way to make an emitted notebook silently re-fit a model on the wrong
table now fails loudly, and the known loud-but-unexplained split failure now
explains itself. Tool surface unchanged (24).

### Fixed
- A model fit on a pure-query derived dataset later overwritten by
  `materialize_query` silently re-fit on the post-transform table at replay
  (constant `(query)` sentinel hashes were indistinguishable). Replay now
  raises a purpose-written `AssertionError`. Same fix covers: fit on the
  middle materialization of an overwrite chain, fit on a base-carrying
  derived state then overwritten again, split-over-split under the same
  names, `load_dataset` over a derived/split/dataframe name after a fit,
  and same-name reloads with changed content or changed `read_options`
  (identical bytes re-parsed differently are now caught by fit-time loader
  identity — a content hash alone cannot see loading semantics).
- Overwriting **both** sides of a split with recipes that read the missing
  pre-overwrite split tables failed replay with a raw
  `duckdb.CatalogException`. Split-overwrite provenance is now recorded on
  the derived entry at `materialize_query` time (no sibling inference), so
  both CREATEs carry the 1.3.1-style explained `AssertionError`.
- Overwriting the **test** side of a split left the surviving train table
  without any recreation at replay (raw `CatalogException` downstream). The
  surviving train entry now emits a train-only split block guarded by its
  own membership checksum.
- A split-source drift that changed only train-side rows passed the
  test-side membership checksum and replayed silently drifted numbers. Both
  sides now store and assert their own checksum.
- The setup cell's second pass emitted in dict insertion order, which is
  wrong for overwrites (a re-assigned name keeps its old position): with
  pre-registered output names + `split_dataset(overwrite=True)` a derived
  CREATE could emit before the split block it reads from. The second pass
  now emits in registration-revision order.

### Changed
- `DatasetEntry` gains `revision` (monotonic per-session registration
  counter) and `split_overwrite` (recorded overwrite provenance);
  `ModelEntry` gains `training_dataset_revision` and `training_loader`
  (fit-time `{path, format, read_options}`); `base_loader` records the
  replaced file entry's revision. Registry metadata only — no tool-response
  changes.
- Emitted notebooks: split blocks now assert a per-side membership checksum
  (train and test), and one-sided blocks exist for a surviving train side.
- Deliberate conservatism: re-running a byte-identical recipe (or
  re-splitting with the same seed) over an unchanged source still *replaces*
  the dataset, so a model fit before the replacement now fails replay loudly
  even though a faithful re-fit might have been possible. Determinism of the
  recipe is not verifiable; loud beats silently-maybe-right.

## [1.3.1] - 2026-07-08

### Fixed
- Overwriting one side of a still-live train/test split with a
  *self-referential* `materialize_query` (e.g. `SELECT ... FROM "base_train"`
  written back to `base_train`) now fails notebook replay with a
  purpose-written `AssertionError` instead of a raw `duckdb.CatalogException`.
  The pre-overwrite split table is deliberately not recreated at replay (that
  would clobber the derived table), so its self-referencing SQL has no table to
  read from; the setup cell's derived CREATE for the overwritten side is now
  wrapped to explain which split side was overwritten and to rematerialize it
  from a table that exists at replay, with the original catalog error chained.
  Ordinary `materialize_query` notebooks are unchanged.

## [1.3.0] - 2026-07-08

Model-workflow bundle — train/test splits and cross-validation, closing the loop the Phase-5 model registry opened. Tool surface **22 → 24**.

### Added
- **`split_dataset`** — a seeded, optionally stratified train/test split
  registered as two first-class datasets (`{name}_train` / `{name}_test` by
  default) that every other tool can target by name:
  - Membership is a deterministic function of (source rows, seed) via
    `np.random.RandomState` — NumPy's NEP 19-frozen MT19937 stream, not DuckDB
    `hash()` / `USING SAMPLE` — so the same seed always reproduces the same
    split across NumPy and DuckDB versions.
  - The emitted notebook recreates each split behind an **order-independent
    membership checksum** (row count + XOR and sum of per-row hashes, so
    duplicate rows can't cancel out): deterministic replays pass, row-order
    drift in a derived source fails loudly at the assert, mirroring the
    SHA-256 file guards.
  - Outputs register with a dedicated `format="split"`; the name preflight is
    atomic (both collisions checked before either table is created) so a
    half-applied split can't occur.
- **`cross_validate`** — k-fold cross-validated metrics for a model formula;
  the re-fitting complement to `evaluate_model`, with fits ephemeral and the
  model registry untouched:
  - A full-data preflight fit surfaces `fit_model`'s error taxonomy
    (`formula_error`, `perfect_separation`, `convergence_failed`,
    `outcome_dtype_mismatch`) before any fold work, and its patsy design
    matrices are reused so a categorical level appearing in only one fold
    can't crash held-out scoring.
  - Logistic folds are auto-stratified by outcome class (requires ≥ k rows per
    class); metrics match `evaluate_model`'s families, reported as mean,
    std (ddof=1), and per-fold values.

### Changed
- `materialize_query` overwrites and the recorder's setup-cell rehydration now
  recognize the dedicated `format="split"`, so a split output is never misread
  as SQL-derived. Overwriting a split output drops that side's split recipe
  from the setup cell, so replay fails loudly (missing table or checksum)
  rather than recomputing silently.

### Internal
- Extracted `fit_prepared` from `fit_model`: the shared fit / validation /
  coercion path (boolean-outcome coercion, negbin non-negative-integer count
  checks, separation detection) is now a single entry point that
  `cross_validate`'s preflight fit reuses, so CV can't drift from `fit_model`'s
  semantics.

### Documentation
- `docs/SPEC.md` §5.6b (`split_dataset`) and §5.11d (`cross_validate`) specify
  the two tools; the design record is
  `docs/superpowers/specs/2026-07-08-model-workflow-bundle-design.md` and the
  folded proposal stub is `docs/proposals/2026-07-08-model-workflow-bundle.md`.

## [1.2.1] - 2026-07-06

### Fixed
- A model fitted on a dataset that was later overwritten by
  `materialize_query` no longer crashes notebook replay: the setup cell's
  hash guard now targets the original source file carried in `base_loader`
  (not the `"(query)"` placeholder), and the model re-fits on a dedicated
  `<model_name>_train_df` frame loaded from that file instead of the
  post-transform table. Scoring cells keep seeing the current table.

## [1.2.0] - 2026-07-05

### Added
- **Dataset provenance hashes.** Every dataset records a SHA-256 content hash
  of its source file at `load_dataset` time (`(path, mtime, size)` fallback
  above 100 MB; sentinels for s3/http/in-memory/derived sources). The emitted
  notebook's setup cell asserts on it before each reload — including base
  files behind `materialize_query` overwrites — so editing a source file
  between session and replay fails loudly instead of silently recomputing
  different numbers.
- Setup-cell reloads now render the live load's `read_options`, so a passing
  hash also implies the same parse at replay.

### Changed
- `fit_model` no longer re-hashes the training file at fit time; the model's
  provenance hash is copied from the dataset entry's load-time hash (the
  model trains on the table loaded then, not the file as it exists at fit
  time). A same-name dataset reload after a fit still fails replay loudly
  via the model-block assert.

### Fixed
- `session.reset()` no longer raises on datasets whose names contain double
  quotes (reachable via `load_dataset` name defaulting from the file
  basename) — table names are now escaped in the `DROP TABLE` statement.

### Internal
- New `provenance.py` (shared hash) and `read_options.py` (shared DuckDB
  reader-option rendering) leaf modules.

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

[1.4.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.4.0
[1.3.1]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.3.1
[1.3.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.3.0
[1.2.1]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.2.1
[1.2.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.2.0
[1.1.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.1.0
[1.0.2]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.0.2
[1.0.1]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.0.1
[1.0.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.0.0
