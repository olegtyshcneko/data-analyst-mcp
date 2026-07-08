# Model workflow bundle: `split_dataset` + `cross_validate`

**Date:** 2026-07-08
**Status:** approved design, pre-implementation
**Tool count:** 22 → 24 (waivered cohort, same precedent as tier-1)
**Target release:** 1.3.0

## Why

The Phase-5 model registry trio (`fit_model(model_name=...)` / `predict` / `evaluate_model`)
assumes a held-out dataset exists, but the server offers no way to create one: the README's
own worked example (§4a) scores `titanic_test` against a model fit on `titanic_train`, and
those files had to be pre-split outside the server. `evaluate_model` is also single-holdout
only. This bundle closes both gaps with two single-purpose tools that compose with the
existing registry and recorder:

- **`split_dataset`** — seeded, optionally stratified train/test partition registered as
  derived datasets, replayable in the emitted notebook.
- **`cross_validate`** — k-fold cross-validated metrics for a formula; fits are ephemeral
  and never touch the model registry.

Rejected alternatives: a `cv_folds` parameter on `evaluate_model` (muddies its contract —
the same tool would sometimes score the registered fit and sometimes re-fit per fold);
shipping `split_dataset` alone (holdout-only evaluation stays the acknowledged weak spot).

## `split_dataset` (SPEC §5.6b)

**Purpose:** partition a registered dataset into seeded train/test derived datasets.

**Input:**

- `name: str` — source dataset (file-backed or derived); must exist.
- `test_fraction: float = 0.25` — open interval `(0, 1)`; endpoints rejected.
- `seed: int = 42`.
- `stratify_by: str | None = None` — column to stratify on.
- `train_name: str | None = None`, `test_name: str | None = None` — default
  `{name}_train` / `{name}_test`; same identifier regex as `materialize_query`
  (`^[A-Za-z_][A-Za-z0-9_]*$`).
- `overwrite: bool = False` — same collision scheme as `materialize_query`.

**Name preflight (atomic):** the three names (source, train, test) must be pairwise
distinct → `split_name_conflict` otherwise. Collisions with existing datasets are checked
for *both* output names before either table is created, so a half-applied split cannot
occur; with `overwrite=False`, either output name already registered →
`dataset_name_collision`.

**Determinism (the load-bearing decision):** row assignment uses
`np.random.RandomState(seed).permutation(n)` in Python — not DuckDB `hash()` or
`USING SAMPLE`. NumPy's NEP 19 policy froze the legacy `RandomState` (MT19937) stream:
the same seed yields the same permutation on any NumPy version (the modern `Generator`
API carries no such guarantee, which is why we do not use it). DuckDB's hash/sample
output is not contractually stable across DuckDB versions, and the replay environment may
run a different DuckDB. The permuted test indices are loaded into a temp index table and
the train/test tables are created by join/anti-join on `row_number() OVER ()`.

**Row-order tiers (mirrors the existing hash-guard tiers):**

- *File-backed sources:* scan order is replay-stable — DuckDB preserves insertion order
  for CSV/Parquet reads under the default `preserve_insertion_order=true` (the server
  never toggles it), and the file bytes are already SHA-256-guarded. Here the split is a
  pure function of (source rows, seed) and replays exactly.
- *Derived sources (`materialize_query` or a previous split):* at replay the derived
  table is recreated by re-running its SQL, and joins/group-bys carry no order guarantee.
  Recomputed row numbers could silently differ, so the split cell asserts a
  **membership checksum** — an order-independent digest (row count + XOR and sum of
  per-row hashes, so duplicate rows cannot cancel out) of the recreated test table
  against the session's value. Deterministic replays pass;
  order drift fails loudly at the assert, exactly like the existing SHA-256 file guards.
  The checksum is asserted for file-backed sources too (it is cheap and catches the
  unexpected).

**Sizing and stratification:** unstratified, `n_test = round(n * test_fraction)` clamped
to `[1, n - 1]`; source with `n < 2` → `dataset_too_small`. Stratified: the same sizing
rule is applied within each stratum, with a single `RandomState(seed)` instance consumed
across strata in sorted-stratum order (`NULL` stratum last) so the assignment stays
deterministic; a stratum with fewer than 2 rows goes entirely to train and adds a
`small_strata` warning. After assignment, both sides must be non-empty — if singleton
strata (or rounding) leave train or test with zero rows, the split is rejected with
`stratification_too_sparse` and nothing is registered.

**Registration:** both outputs register as first-class datasets with a dedicated
`format="split"` (path `"(split)"`) — a distinct format keeps the recorder's rehydration
branch and `materialize_query`'s overwrite logic from misreading a split as SQL-derived —
with `read_options` carrying `{source, seed, test_fraction, stratify_by, role}` plus the
membership checksum, so the recorder can rehydrate and guard them. Every existing tool can then target them by name.

**Output:**

```
{ok, source, train: {name, rows}, test: {name, rows}, seed, test_fraction,
 stratify_by, strata: [{value, train_rows, test_rows}] | null, warnings}
```

**Errors:** `dataset_not_found`, `dataset_name_collision`, `invalid_name`,
`split_name_conflict`, `test_fraction_out_of_range`, `stratify_column_missing`,
`dataset_too_small`, `stratification_too_sparse`, `internal`.

## `cross_validate` (SPEC §5.11d)

**Purpose:** k-fold cross-validated metrics for a model formula on a dataset — the
re-fitting complement to `evaluate_model`'s score-a-registered-fit. No `model_name`
parameter exists; fits are ephemeral and the registry is untouched.

**Input:**

- `name: str` — dataset.
- `formula: str`, `kind: Literal["ols", "logistic", "poisson", "negbin"] = "ols"`,
  `robust: bool = False` — identical contract and validation to `fit_model`.
- `k: int = 5` — range `[2, 20]`.
- `seed: int = 42` — fold assignment.
- `threshold: float = 0.5` — logistic confusion-derived metrics; open interval `(0, 1)`.

**Preflight fit:** before any folding, fit the formula once on the full dataset through
the same validation/coercion path `fit_model` uses (boolean-outcome coercion, negbin
non-negative-integer count checks, separation detection). This single fit surfaces
`fit_model`'s whole error taxonomy top-level — `formula_error`, `perfect_separation`,
`convergence_failed`, `outcome_dtype_mismatch` — before any fold work, and its patsy
design matrices are reused for the folds, so `cross_validate` cannot drift from
`fit_model`'s semantics.

**Fold construction:** slice the preflight `y, X` per fold (NaN-dropped rows reported as
`dropped_rows`, like `predict`). The global design matrix encodes categorical levels
consistently, so a level appearing in only one fold can never crash held-out scoring (the
classic naive-CV failure). Fold assignment is a `RandomState(seed)` permutation over the
post-drop rows; post-drop `n_obs < k` → `k_out_of_range` (the message states the
post-drop count). For `kind="logistic"` folds are stratified by outcome class
automatically and the response reports `stratified: true` — auto-and-report, no knob,
same philosophy as `compare_groups`' test selection. Stratification requires each
outcome class to have at least `k` members → `outcome_class_too_small` otherwise; this
makes single-class holdout folds structurally impossible, so ROC-AUC / log-loss are
always defined (the recorder cell still passes explicit `labels=[0, 1]` to sklearn's
`log_loss` as a defensive measure).

**Per-fold behavior:** fit on k−1 folds, score the held-out fold with the same metric
families as `evaluate_model` — logistic: ROC-AUC, PR-AUC, Brier, log-loss,
accuracy/precision/recall/F1 at `threshold`; OLS: RMSE, MAE, R²; Poisson/negbin: RMSE,
MAE, Pearson χ², deviance. Fold-local fit failures (separation or non-convergence that
the full-data preflight fit did not exhibit) are recorded in `fold_failures` — fold index
plus the same error-type string `fit_model` would have used — and excluded from
aggregates with a warning; all folds failing → `cv_fit_failed`. A fold whose training
slice has ≤ n_params rows → `fold_too_small`.

**Output:**

```
{ok, name, formula, kind, k, seed, stratified,
 metrics: {<metric>: {mean, std, per_fold}}, fold_sizes, n_obs,
 dropped_rows, fold_failures, warnings, interpretation}
```

**Errors:** `dataset_not_found`, `formula_error`, `perfect_separation`,
`convergence_failed`, `outcome_dtype_mismatch`, `outcome_class_too_small`,
`k_out_of_range`, `fold_too_small`, `cv_fit_failed`, `robust_not_supported`,
`threshold_out_of_range`, `internal`. (`perfect_separation` / `convergence_failed` refer
to the full-data preflight fit; the fold-local variants land in `fold_failures`.)

## Recorder / replay

- `split_dataset` emits a code cell running the identical
  `RandomState(seed).permutation` + index-table + join sequence against the notebook's
  DuckDB connection, followed by the membership-checksum assert (order-independent digest
  of the recreated test table vs. the session's value).
- The setup cell gains a third rehydration branch: split-derived entries are recreated
  **after** file-backed and `materialize_query`-derived lines (their source must exist
  first), in registration order — a split of a split works because chains register in
  order. File-backed provenance stays covered by the existing SHA-256 assert; the
  membership checksum covers row-order drift in derived sources.
- `cross_validate` emits a self-contained cell: same fold assignment, statsmodels fits in
  a loop, sklearn metrics, printing the per-fold and aggregate table.
- Overwrite interactions follow the existing `materialize_query` collision scheme,
  including the fit-then-overwrite guard behavior shipped in 1.2.x. A model fit on a
  split output (or any dataset without a carryable file loader) whose training table is
  later overwritten by `materialize_query` cannot be re-fit faithfully at replay — the
  setup cell emits a hard `raise AssertionError` for that model instead of silently
  re-fitting against the post-transform table. Overwriting a split output also drops
  that side's split recipe from the setup cell; replay then fails loudly (missing table
  or checksum) rather than recomputing silently.

## Testing

TDD (`red:` / `green:`) as enforced by `check_tdd_commits.py`.

- **Known-answer:** exact train/test row sets for a pinned seed on a small fixture;
  stratified per-stratum proportions; CV aggregates cross-checked against a hand-run
  sklearn `KFold` on the same fixture.
- **Error paths:** one test per error type listed above, including the preflight name
  rules (`train_name == test_name`, output name equal to source, one-of-two collision
  leaves nothing registered), all-singleton strata → `stratification_too_sparse`,
  post-drop `n_obs < k`, and logistic minority class < k → `outcome_class_too_small`.
- **Determinism:** same seed twice → identical membership; split of a derived
  (join/group-by) source and split-of-a-split both replay through the checksum assert;
  a multi-threaded DuckDB run (`threads > 1`) produces the same split as single-threaded
  for file-backed sources.
- **Integration evals:** emit → `jupyter nbconvert --execute` replay reproduces the
  identical split (membership checksum) and the identical CV metric table; drift case —
  edit the source file, replay must fail at the setup-cell assert before any split runs.

## Rollout

1. `docs/proposals/` draft for the two-tool cohort (this document seeds it).
2. SPEC fold-in as §5.6b and §5.11d.
3. Implement `split_dataset` first, then `cross_validate` (independent, but the README
   example reads best when the split exists).
4. README (tool count 22 → 24; worked example §4a switches from pre-split titanic files to
   `split_dataset`), ROADMAP, CHANGELOG, release 1.3.0.
