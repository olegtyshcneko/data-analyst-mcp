# `find_outliers`

## Purpose

`describe_column` flags per-column outliers via IQR / z-score, but joint
anomalies (a row that is unremarkable on any one axis yet extreme in the
K-dimensional manifold) need their own tool. `find_outliers` ships four
methods — `iqr`, `zscore`, `mahalanobis`, `isolation_forest` — over an
arbitrary numeric column set, returning row indices, per-row scores, and
the offending values for downstream investigation.

## Input

```python
class FindOutliersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    columns: list[str] = Field(min_length=1)
    method: Literal["iqr", "zscore", "mahalanobis", "isolation_forest"]
    threshold: float | None = None       # method-specific default applied when None
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)  # iforest only
    limit: int = Field(default=50, ge=1, le=10_000)
```

## Behavior

1. Resolve the dataset by `name`; missing → `not_found`.
2. Validate every entry of `columns` exists in the dataset; first miss
   → `column_not_found`.
3. Pull the column subset as a DataFrame.
4. Validate that every selected column is numeric; first non-numeric
   → `non_numeric_column`.
5. Dispatch by `method`:
   - **`iqr`**: per-column flag where `x < Q1 − k·IQR` or `x > Q3 + k·IQR`
     with `k = threshold` (default `1.5`). Reuses the shared helper
     extracted from `tools/datasets.py:339-366` (`_outliers`) into the
     new module `tools/_outlier_helpers.py`. A row is flagged if any
     column flags it; the score is the maximum per-column normalized
     excess.
   - **`zscore`**: per-column flag where `|z| > threshold` (default
     `3.0`). Reuses the same shared helper. Same row-aggregation rule
     as IQR; the score is `max(|z|)` across selected columns.
   - **`mahalanobis`**: joint detection. Drop rows with any NaN in the
     selected columns and record `dropped_n_na_rows` in `warnings`.
     Compute `μ` and covariance `Σ` of the N×K matrix using numpy.
     Compute `D² = (x − μ)ᵀ Σ⁻¹ (x − μ)` for each row. Flag where
     `D² > χ²(K, 1 − α)` with `α` default `0.025`; the echoed
     `threshold_used` is the chi² quantile (not `α`). If
     `np.linalg.inv(Σ)` raises `LinAlgError`, fall back to
     `np.linalg.pinv(Σ)` (Moore-Penrose pseudoinverse) and append
     `covariance_singular` to `warnings`. Only error
     (`singular_covariance`) if the pseudoinverse also fails.
   - **`isolation_forest`**: call
     `sklearn.ensemble.IsolationForest(contamination=contamination,
     random_state=42).fit_predict(X)`. Flag rows where the prediction
     is `−1`; the score is `−decision_function(X)` so larger = more
     anomalous.
6. Sort flagged rows by score descending, truncate to `limit`, and emit
   `row_index` (source-dataset 0-indexed), `score`, and `values` (the
   per-column raw values for that row).
7. Emit the recorder cell (success only).

## Output

```python
{
    "ok": True,
    "method": str,                # echoes payload.method
    "n_outliers": int,            # total flagged before truncation
    "n_rows_scored": int,         # rows actually scored (post-NA-drop for mahalanobis)
    "outliers": [                 # truncated to `limit`, sorted by score desc
        {"row_index": int, "score": float, "values": {col: val, ...}},
    ],
    "truncated": bool,            # True when n_outliers > limit
    "threshold_used": float,      # method default echoed back when payload.threshold is None
    "warnings": list[str],        # e.g. "covariance_singular", "dropped_n_na_rows"
}
```

## Errors

- `not_found` — `name` is not in the dataset registry.
- `column_not_found` — at least one entry of `columns` is not a column
  of the resolved dataset; names the first miss.
- `non_numeric_column` — at least one selected column has a
  non-numeric dtype; names the first offender. (Applies to all four
  methods — IQR / z-score with clearer messaging than the shared
  helper's existing raise.)
- `insufficient_rows` — `mahalanobis` only, when `n_rows_scored ≤ k`
  after the NA drop (covariance is rank-deficient by construction).
- `singular_covariance` — `mahalanobis` only, raised only when both
  `np.linalg.inv` and `np.linalg.pinv` fail (very rare). Hint: drop
  perfectly collinear columns.

## Recorder cells

```python
get_recorder().record(
    markdown=(
        f"### Outlier detection on `{payload.name}` ({payload.method})\n"
        f"- Columns: {', '.join(payload.columns)}\n"
        f"- Flagged: {n_outliers} / {n_rows_scored} rows"
    ),
    code=_outlier_code_snippet(payload),  # method-specific reproducer
    tool_name="find_outliers",
)
```

The code cell varies by method — for `mahalanobis` it inlines the numpy
compute; for `isolation_forest` it inlines the sklearn snippet; IQR /
z-score reuse the shared helper. Pattern parallels `compare_groups`
emitting different code per chosen test.

## TDD slices

~22 cycles, grouped:

**Shared (3 cycles):**
1. `not_found`.
2. `column_not_found`.
3. `non_numeric_column`.

**IQR (4 cycles):**
4. Default threshold `1.5`.
5. Custom threshold.
6. Known-answer fixture (hand-checked to ≥4 decimal places).
7. Per-column results aggregated correctly into row-level flags.

**Z-score (4 cycles):**
8. Default `3.0`.
9. Custom threshold.
10. Known-answer fixture.
11. NaN handling (NaN rows are excluded, not flagged).

**Mahalanobis (6 cycles):**
12. Known-answer 2D toy fixture.
13. `n ≤ k` → `insufficient_rows`.
14. Singular covariance → pseudoinverse fallback + `covariance_singular`
    warning (not an error).
15. `threshold_used` echoed as the chi² quantile (not `α`).
16. Drops NA rows and reports `dropped_n_na_rows` in `warnings`.
17. Custom `α` respected.

**Isolation Forest (5 cycles):**
18. Known-answer on a fixed-seed synthetic dataset with planted
    outliers.
19. `contamination` respected.
20. Deterministic across runs at `random_state=42`.
21. `score = −decision_function(X)`.
22. Insufficient `n` → error (smallest-viable n is sklearn-defined; use
    the typed error, not the bare sklearn exception).

## Acceptance criteria

- All ~22 TDD cycles green with `red:` / `green:` / `refactor:`
  commits.
- `tests/test_outliers.py` passes under `uv run pytest -q`.
- Characterization test in `tests/test_describe_column.py` confirms
  `describe_column` output is byte-identical after the helper
  extraction.
- `evals/eval_outliers.py` passes: load `fixtures/messy.csv`, run all
  four methods, verify each surfaces the planted `score=123456`
  outlier.
- `ruff check .`, `ruff format --check .`, `pyright src/`, and
  `scripts/check_tdd_commits.py` all green.

## ROADMAP impact

- Removes the `find_outliers` line from `ROADMAP.md` "Tooling".
- Adds `find_outliers` to SPEC §5; bumps the tool count.
- Documents the extracted helper module
  `src/data_analyst_mcp/tools/_outlier_helpers.py` as shared between
  `describe_column` and `find_outliers` (no public-API change to
  `describe_column`).
