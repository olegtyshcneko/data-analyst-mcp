# `data-analyst-mcp` — Implementation Spec

> You are an autonomous engineer. Implement this project end-to-end following the spec below. Work through the phases in order. Commit at every phase boundary. Run the acceptance gates before moving on. Ask the user **only** if you hit a true blocker — do not ask about architecture, library choice, or naming; those are decided here.

---

## 1. What you are building

A Python MCP server that turns Claude (or any MCP-capable agent) into a **reproducible data analyst**. It loads tabular data, runs proper exploratory analysis, performs statistical tests with assumption checking, fits regression models with diagnostics, and — the differentiator — emits the entire agent session as a runnable Jupyter notebook for audit and re-execution.

The product brief, in one sentence: **"Composes with `mcp-server-motherduck` to add the analytical reasoning layer on top of raw SQL: proper EDA, hypothesis tests, model fitting, and a reproducibility log."**

### Positioning (do not deviate)

- **In scope**: bounded tabular datasets (100 MB – 5 GB), human-in-the-loop ad-hoc analysis, local-stdio MCP transport.
- **Out of scope**: streaming data, dashboards, causal inference, BI replacement, datasets > 10 GB. Say so in the README.

### Target users (drives all UX trade-offs)

1. Analysts/PMs doing repetitive ad-hoc EDA — primary.
2. Consultants/FDEs walking into a fresh client dataset — primary.
3. Solo operators with a CSV export — secondary; nice side-effect, not the optimization target.

---

## 2. Non-negotiable decisions (do not relitigate)

- **Language**: Python 3.13+ (3.13 is the project floor; CI matrix tests 3.13 and 3.14; do not target 3.12)
- **Package manager**: `uv`. Not pip, not poetry, not pdm. All commands use `uv` / `uvx`.
- **MCP SDK**: `mcp[cli]` (official Anthropic Python SDK), `FastMCP` decorator style.
- **Query engine**: DuckDB, in-process. Reads CSV/Parquet/Excel/JSON natively.
- **DataFrame library**: pandas (not polars). Scipy/statsmodels expect pandas; polars adds interop friction with no benefit for our workload.
- **Stats**: `scipy.stats` for tests, `statsmodels` for regression. Nothing else.
- **Plots**: matplotlib only. Fixed style sheet. Return base64-encoded PNG. **Do not** add plotly, seaborn, altair, or bokeh.
- **Notebook**: `nbformat` for `.ipynb` generation.
- **Tool input validation**: pydantic v2 models, used as FastMCP tool input types.
- **Tests**: pytest + pytest-cov.
- **Lint/format**: ruff (configured in `pyproject.toml`).
- **Type checking**: enable `pyright` in `pyproject.toml`, strict mode for `src/`, off for `tests/` and `evals/`.
- **Transport**: stdio only for v1. No SSE, no HTTP.
- **License**: MIT.
- **Python project layout**: `src/data_analyst_mcp/`, importable as a package, exposes a `data-analyst-mcp` console script.

### Hard rules

- **Never `print()` or log to stdout** — it corrupts the MCP stdio stream. Use `logging` with stderr handler only.
- **Never raise exceptions out of a tool** — catch everything and return `{"error": {"type": "...", "message": "...", "hint": "..."}}`.
- **Never dump unbounded result sets to the LLM** — every tool that returns rows must support `limit` (default 50) and include `total_rows`, `truncated: bool`, and a `cursor` field for the next page.
- **Never embed `pd.read_*` directly in tool bodies** — go through DuckDB so the same code path works for CSV, Parquet, Excel, and remote files.
- **No new dependencies without recording the reason** in a `DEPS.md` file with one line per added dependency.

---

## 3. Development discipline: Red-Green-Refactor (mandatory)

Every line of production code in `src/` is written in response to a failing test. No exceptions for "trivial" code, no exceptions for "I'll add tests after." This is not aspiration — it is the workflow.

### The cycle (per tool, per behavior, per error path)

1. **Red.** Write a single failing test that captures one specific behavior or error case. Run it. Confirm it fails for the right reason (assertion failure or `NotImplementedError`, not `ImportError` or `SyntaxError`). Commit the failing test with message `red: <behavior>`.

2. **Green.** Write the **minimum** code in `src/` to make that test pass. Do not anticipate the next test. Do not generalize speculatively. Run the test, confirm it passes, run the full suite, confirm nothing else broke. Commit with message `green: <behavior>`.

3. **Refactor.** With tests green, improve the design — extract helpers, rename, dedupe, tighten types. Do not change behavior. Run the full suite after every refactor edit. Commit with message `refactor: <what>`.

Then take the next behavior and start at Red again.

### Granularity — one behavior per cycle

Not "implement `profile_dataset`" in one cycle. Slice it down:

| Cycle | Red test | Green implementation |
|---|---|---|
| 1 | "returns `ok: true` for a valid dataset name" | Stub returning `{"ok": True}` |
| 2 | "returns `ok: false, error.type='not_found'` for missing name" | Add the lookup + error path |
| 3 | "reports correct `rows` field" | Add row count via DuckDB |
| 4 | "reports correct `columns` field with name and dtype" | Add `DESCRIBE` query |
| 5 | "reports null_count per column" | Add per-column null aggregation |
| 6 | "flags column as `mostly_null` when >50% null" | Add the heuristic |

Each row above is a separate red → green → refactor cycle with its own commit triplet. Twenty cycles for `profile_dataset` is normal. Resist the urge to batch.

### Known-answer tests for statistics (mandatory for §5 stats tools)

The stats tools have a special obligation: every test for `compare_groups`, `test_hypothesis`, `correlate`, and `fit_model` must use a **known-answer fixture** — values where the expected result is computed independently (by hand, by scipy in a notebook with a fixed seed, or against published examples) and hard-coded into the test.

Example for `compare_groups`:

```python
def test_compare_groups_two_sample_known_answer(server, recorder):
    # Hand-computed: scipy.stats.ttest_ind([1,2,3,4,5], [3,4,5,6,7], equal_var=True)
    # → statistic = -2.8284271247461903, p = 0.0218...
    load_dataset_into_memory(server, "tiny", DataFrame({
        "group": ["A"]*5 + ["B"]*5,
        "value": [1,2,3,4,5,3,4,5,6,7],
    }))
    result = call_tool(server, "compare_groups", {
        "name": "tiny", "group_column": "group", "metric_column": "value"
    })
    assert result["ok"] is True
    assert result["test"] == "student_t"  # n small, equal var, normal-ish
    assert result["statistic"] == pytest.approx(-2.8284271, abs=1e-5)
    assert result["p_value"] == pytest.approx(0.02218, abs=1e-4)
    assert result["effect_size"]["name"] == "cohens_d"
```

A test that asserts only "`ok` is true" or "result has a `p_value` field" does not prove statistical correctness. Known-answer assertions to ≥4 decimal places do.

### What gets tested first vs. what doesn't

- **Strictly TDD'd**: every tool in `src/data_analyst_mcp/tools/`, the recorder, the session, the error builder, the formatters, the test-selection logic in `compare_groups`.
- **Test-after acceptable**: fixture *generators* (`fixtures/_build_messy.py`, `fixtures/generate_synthetic_crm.py`) — they're scripts, not library code. Verify by running them and inspecting output, then commit.
- **Characterization tests, not pure TDD**: `tools/plots.py` — write a test that calls the function and asserts PNG header + dimensions, then implement against it. Pixel-perfect TDD on matplotlib is wasted effort.
- **Eval suite (§8)**: written in Phase 7 after implementation is complete; evals validate the *integrated* product, not individual units.

### Pacing

- Aim for a red → green → refactor triplet every 5–15 minutes. If a cycle is taking 30+ minutes, the slice is too big — back out, commit nothing, slice smaller.
- Run `uv run pytest tests/ -q` after every code edit. Run `uv run pytest tests/ --cov` once per phase to check coverage trajectory.
- If you find a bug while writing other code, the next test you write is the one that would have caught it. Red → green that, then fix.

### Reporting in commit history

By the end of each phase, the git log should clearly show the cycle. Example:

```
red: load_dataset returns error for missing file
green: load_dataset returns error for missing file
red: load_dataset registers a CSV via DuckDB
green: load_dataset registers a CSV via DuckDB
refactor: extract _resolve_reader from load_dataset
red: load_dataset surfaces parse warnings
green: load_dataset surfaces parse warnings
```

---

## 4. Project layout

```
data-analyst-mcp/
├── pyproject.toml
├── README.md
├── LICENSE
├── DEPS.md
├── .gitignore
├── .python-version
├── claude_desktop_config.example.json
├── src/
│   └── data_analyst_mcp/
│       ├── __init__.py
│       ├── __main__.py              # console-script entry; calls server.run()
│       ├── server.py                # FastMCP instance, tool registration
│       ├── session.py               # singleton session: datasets + notebook log
│       ├── formatting.py            # truncation, table-to-dict, base64 helpers
│       ├── errors.py                # structured error builder
│       ├── recorder.py              # NotebookRecorder: records cells per tool call
│       └── tools/
│           ├── __init__.py
│           ├── datasets.py          # load_dataset, list_datasets, profile_dataset, describe_column
│           ├── query.py             # query
│           ├── stats.py             # correlate, compare_groups, test_hypothesis
│           ├── models.py            # fit_model
│           ├── plots.py             # plot
│           └── notebook.py          # emit_notebook
├── fixtures/
│   ├── README.md
│   ├── messy.csv
│   ├── generate_synthetic_crm.py
│   └── synthetic_crm/
│       ├── accounts.csv
│       ├── contacts.csv
│       └── opportunities.csv
├── tests/
│   ├── conftest.py
│   ├── test_datasets.py
│   ├── test_query.py
│   ├── test_stats.py
│   ├── test_models.py
│   ├── test_plots.py
│   ├── test_recorder.py
│   └── test_emit_notebook.py
└── evals/
    ├── conftest.py
    ├── README.md
    ├── eval_basic.py
    ├── eval_messy_csv.py
    ├── eval_stats.py
    └── eval_full_workflow.py
```

---

## 5. Tool specifications

Every tool follows this contract:

- Decorated with `@mcp.tool()` from FastMCP.
- Input is a pydantic v2 model with `Field(description=...)` on every field. **The descriptions are prompts to the LLM** — write them like product copy.
- Returns a `dict` with at least: `ok: bool`, plus tool-specific fields. On failure: `{"ok": false, "error": {...}}`.
- After successful execution, appends a markdown + code cell pair to the session notebook recorder (see §6).
- Logs to stderr only, never stdout.

### 5.1 `load_dataset`

**Purpose**: register a file as a named dataset queryable by all other tools.

**Input**:
- `path: str` — local filesystem path or `s3://` URL. Supported extensions: `.csv`, `.tsv`, `.parquet`, `.xlsx`, `.json`, `.jsonl`.
- `name: str | None` — name to register under. If omitted, derive from filename (slugified).
- `read_options: dict | None` — passed through to DuckDB's `read_*` function (e.g. `{"header": false, "delim": ";"}`).

**Behavior**:
- Uses DuckDB `CREATE OR REPLACE TABLE <name> AS SELECT * FROM read_csv(...)` / `read_parquet(...)` / etc.
- Auto-detects format from extension; falls back to `read_csv_auto` for ambiguous text.
- Stores file path and read_options in session for the recorder.

**Output**:
```json
{
  "ok": true,
  "name": "orders",
  "rows": 99441,
  "columns": [
    {"name": "order_id", "dtype": "VARCHAR"},
    {"name": "order_date", "dtype": "TIMESTAMP"}
  ],
  "file_size_bytes": 16823491,
  "warnings": ["3 rows had parse errors and were skipped"]
}
```

**Errors**: `file_not_found`, `unsupported_format`, `parse_failed` (with hint pointing at `read_options`).

### 5.2 `list_datasets`

**Input**: none.

**Output**:
```json
{
  "ok": true,
  "datasets": [
    {"name": "orders", "rows": 99441, "columns": 8, "registered_at": "..."}
  ]
}
```

### 5.3 `profile_dataset`

**Purpose**: full EDA profile in one call. This is the **headline tool**.

**Input**:
- `name: str` — dataset name.
- `sample_rows: int = 5` — number of head/tail rows.

**Behavior**:
- Total rows, total columns, memory estimate.
- Per column: dtype, null count, null %, distinct count, top-5 most-frequent values with counts.
- For numeric columns: min, max, mean, median, std, p25, p75, p99, count of zeros, count of negatives.
- For string columns: min length, max length, mean length, count empty strings, count whitespace-only.
- For temporal columns: min, max, range in days, count of nulls, modal weekday.
- Heuristic flags per column: `looks_like_id`, `looks_like_categorical`, `looks_like_timestamp`, `high_cardinality`, `mostly_null` (>50%), `constant`, `mixed_dtype_suspect`.
- Suggested next steps: top 3 actionable strings.

**Output**: a structured dict with `summary`, `columns: list[ColumnProfile]`, `head: list[dict]`, `suggestions: list[str]`.

### 5.4 `analyze_missingness`

**Purpose**: missingness diagnostics — per-column null stats plus the cross-column *structure* that `profile_dataset` does not surface (which columns are co-missing, whether nulls partition cleanly across a categorical, top missingness patterns, whether the planted nulls are MCAR).

**Input**:
- `name: str` — dataset name.
- `columns: list[str] | None = None` — restrict analysis to a subset; default is every column.
- `pattern_top_k: int = 10` — number of distinct (col → is_null) tuples returned, descending by count. Range `[1, 100]`.
- `pairwise_corr_threshold: float = 0.1` — hide pairwise φ entries with `|φ|` below this. Range `[0.0, 1.0]`.
- `run_mcar_test: bool = True` — run Little's MCAR test on the numeric columns; pass `False` to skip and return `mcar_test: null`.

**Behavior**: per-column `null_count` / `null_pct` / `null_grouping` (alignment with a categorical column; tiebreaker picks the fewest-groups categorical, high-cardinality candidates with `group_count > sqrt(n_rows)` are skipped) / `variance_zero`; top-K patterns as `{col: bool}` objects; pairwise φ-correlation between null indicators (Pearson on 0/1 cast); Little's (1988) MCAR test on the numeric subset via a vendored closed-form Gaussian computation (single-step EM); severity-sorted suggestions (structural > MCAR-violation > high-null > co-missing > imputation-OK > no-nulls), capped at 6.

**Output**: `{ok, summary, per_column, patterns, pairwise_correlation, mcar_test, suggestions}`. `mcar_test` is `{"name": "little", "statistic", "df", "p_value", "violated", "consequence"}` on success, `{"name": "little", "skipped": true, "reason": "insufficient_patterns"}` when fewer than 2 numeric patterns with `n_j ≥ 2` survive, and `null` when `run_mcar_test=False`.

**Errors**: `dataset_not_found`, `unknown_columns`, `pattern_top_k_out_of_range`, `pairwise_corr_threshold_out_of_range`.

### 5.5 `describe_column`

**Input**:
- `name: str` — dataset name.
- `column: str` — column name.
- `bins: int = 20`.

**Behavior**:
- Numeric: full quantile vector (1, 5, 10, 25, 50, 75, 90, 95, 99), skewness, kurtosis, IQR, histogram counts.
- Categorical: full value-counts up to 50, entropy.
- Temporal: counts by year/month/weekday/hour.
- Outliers: IQR rule, z-score > 3, count and 5 example rows.

### 5.5a `find_outliers`

**Purpose**: multi-column joint anomaly detection. Complements `describe_column`'s per-column flagging by surfacing rows that are unremarkable on any single axis yet extreme in the K-dimensional manifold.

**Input**:
- `name: str` — dataset name.
- `columns: list[str]` — one or more numeric columns (`min_length=1`).
- `method: Literal["iqr", "zscore", "mahalanobis", "isolation_forest"]`.
- `threshold: float | None = None` — method-specific default applied when omitted (IQR: `1.5`, z-score: `3.0`, Mahalanobis: `α = 0.025` upper-tail).
- `contamination: float = 0.05` — Isolation Forest only, open interval `(0, 0.5)`.
- `limit: int = 50` — range `[1, 10_000]`.

**Behavior**:
1. Resolve the dataset; validate every entry of `columns` exists and is numeric.
2. Dispatch by `method`:
   - `iqr` — per-column flag where `x < Q1 − k·IQR` or `x > Q3 + k·IQR`; row score = `max` per-column normalized excess.
   - `zscore` — per-column flag where `|z| > threshold`; row score = `max(|z|)` across selected columns.
   - `mahalanobis` — joint D² over the N×K matrix (drop NA rows, count them). Flag where `D² > χ²(K, 1 − α)`; echo the chi² quantile (not α) as `threshold_used`. Falls back to `np.linalg.pinv` with a `covariance_singular` warning if `np.linalg.inv` fails; errors only when the pseudoinverse also fails.
   - `isolation_forest` — `sklearn.ensemble.IsolationForest(contamination=..., random_state=42)`; row score = `−decision_function(X)`. Drops NA rows and counts them. Requires `n ≥ max(10, 2·k)`.
3. Sort flagged rows by score descending, truncate to `limit`, emit per-column raw `values`.

**Output**: `{ok, method, n_outliers, n_rows_scored, outliers: [{row_index, score, values}], truncated, threshold_used, warnings}`. `n_outliers` is the pre-truncation count; `warnings` covers `covariance_singular` and `dropped_N_na_rows`. The `iqr` and `zscore` branches additionally return a `per_column_flags: dict[str, list[int]]` mapping each column to the list of row indices flagged on that column; `mahalanobis` and `isolation_forest` omit this field because their score is intrinsically joint across columns.

**Errors**: `not_found`, `column_not_found`, `non_numeric_column`, `insufficient_rows` (Mahalanobis / Isolation Forest), `singular_covariance`.

### 5.6 `query`

**Input**:
- `sql: str`
- `limit: int = 50`

**Behavior**:
- Reject statements that aren't `SELECT`, `WITH`, `DESCRIBE`, `SHOW`, `EXPLAIN`, `PRAGMA show_tables`. (`SET`, `CREATE`, `DROP`, `INSERT`, `UPDATE`, `DELETE` → rejected with `error.type = "write_not_allowed"`.)
- Apply `LIMIT` if not already present.
- Return rows, columns, total_rows (separate `COUNT(*)` over a subquery), execution_time_ms.

### 5.6a `materialize_query`

**Purpose**: persist a `SELECT` / `WITH` result as a named DuckDB table and register it as a first-class derived dataset. Unblocks cohort / funnel / multi-step join workflows where `query` (read-only) would force the agent to re-run the same join inline on every call.

**Input**:
- `sql: str` — the `SELECT` or `WITH` statement; non-empty.
- `name: str` — identifier for the new derived dataset (`^[A-Za-z_][A-Za-z0-9_]*$`).
- `overwrite: bool = False` — whether to replace an existing dataset of the same name.

**Behavior**:
1. Validate the leading keyword against the narrower allowlist `("SELECT", "WITH")` — `DESCRIBE` / `SHOW` / `PRAGMA` (valid for `query`) are meaningless as a table source and rejected with `write_not_allowed`.
2. If `name` is already registered and `overwrite=False` → `dataset_name_collision`.
3. Execute `CREATE OR REPLACE TABLE "{name}" AS {sql}` on the live DuckDB connection.
4. Probe row count via `SELECT COUNT(*)` and columns via `DESCRIBE`; register a `DatasetEntry` with `path="(query)"`, `format="derived"`, `read_options={"sql": payload.sql}`.
5. The recorder's setup-cell rehydration (§6) emits derived `CREATE OR REPLACE TABLE` lines *after* every file-backed line so the upstream tables exist at replay.

**Output**: `{ok, name, rows, columns: [{name, dtype}], total_rows}` — mirrors `load_dataset`'s shape; `rows == total_rows`.

**Errors**: `write_not_allowed`, `dataset_name_collision`, `invalid_name`, `query_error`, `internal`.

### 5.7 `correlate`

**Input**:
- `name: str`
- `columns: list[str] | None = None`
- `method: Literal["pearson", "spearman", "kendall"] = "pearson"`
- `plot: bool = True`

**Output**: matrix as list-of-lists with row/col labels, plus optional `heatmap_png_base64`.

### 5.8 `adjust_pvalues`

**Purpose**: correct a family of raw p-values for multiple testing. Stateless — no dataset lookup, no session coupling. Designed to compose with `compare_groups` / `test_hypothesis` results that the agent has collected.

**Input**:
- `p_values: list[float]` — each in `[0, 1]`. NaN, inf, negative, and `>1` are rejected.
- `method: Literal["bonferroni", "sidak", "holm", "bh", "by"] = "bh"` — `bh` → statsmodels `fdr_bh`, `by` → `fdr_by`.
- `alpha: float = 0.05` — must satisfy `0 < alpha < 1`.
- `labels: list[str] | None = None` — optional row labels echoed unchanged; must equal `len(p_values)` when provided. Duplicates allowed.

**Behavior**:
- Delegate to `statsmodels.stats.multitest.multipletests`. Validate inputs *before* the statsmodels call so error types are deterministic.
- Output rows are in **input order**. `returnsorted=True` is never passed.
- Empty `p_values` → `ok: true`, empty `results`, `n_tests=0`, `n_rejected=0`. No statsmodels call.

**Output**: `{ok, method, alpha, results: [{label, p_raw, p_adj, rejected}, ...], n_tests, n_rejected}`. `label` is `null` when `labels` is omitted.

**Errors**: `invalid_p_value` (names the first offending index), `length_mismatch`, `invalid_alpha`, `unknown_method`.

### 5.9 `compare_groups`

**Input**:
- `name: str`
- `group_column: str`
- `metric_column: str`
- `groups: list[str] | None = None`

**Behavior**:
1. Check metric column type. If continuous → numeric path; if categorical → chi-square path.
2. Numeric path: check normality (Shapiro–Wilk if n < 5000, else D'Agostino) and equal variances (Levene). Pick t-test (equal var), Welch's t-test (unequal var), or Mann–Whitney U (non-normal). If >2 groups: one-way ANOVA or Kruskal–Wallis.
3. Categorical path: chi-square; if any expected cell < 5 and 2×2, use Fisher's exact.
4. Report: test name, statistic, p-value, effect size (Cohen's d, η², Cramér's V), 95% CI for the difference, sample sizes per group, **and a one-sentence plain-English interpretation**.

**Hard requirement**: explicitly state assumption-check results in the output. E.g. `"normality_test": {"name": "shapiro", "p": 0.003, "violated": true, "consequence": "Switched from t-test to Mann–Whitney U."}`.

### 5.10 `test_hypothesis`

**Input** (discriminated union via `kind` field):
- `kind: "t_test" | "welch" | "mann_whitney" | "chi_square" | "fisher" | "anova" | "kruskal" | "ks"`
- Plus test-specific fields.

**Output**: identical shape across tests where possible: `test`, `statistic`, `p_value`, `effect_size`, `df`, `n_a`, `n_b`, `interpretation`.

### 5.10a `power_analysis`

**Purpose**: stateless sample-size / MDE / achieved-power solver across the five test families that map onto `compare_groups` / `test_hypothesis`. Solves for whichever of `effect_size` / `n` / `power` is omitted, so agents can plan a design *before* running the test.

**Input**:
- `test: Literal["two_sample_t", "one_sample_t", "paired_t", "two_proportion_z", "anova_oneway"]`.
- `effect_size: float | None = None`, `n: int | float | None = None`, `power: float | None = None` — exactly one must be `None`; that is the unknown to solve for.
- `alpha: float = 0.05` — open interval `(0, 1)`.
- `p1: float | None = None`, `p2: float | None = None` — `two_proportion_z` only; if both are supplied, Cohen's h is derived via `statsmodels.stats.proportion.proportion_effectsize(p1, p2)` and used as `effect_size`.
- `k_groups: int | None = None` — `anova_oneway` only; required, must be ≥ 2.
- `ratio: float = 1.0` — `two_sample_t` only; `n2 / n1`.
- `alternative: Literal["two-sided", "larger", "smaller"] = "two-sided"`.

**Behavior**:
1. Validate that exactly one of `{effect_size, n, power}` is `None`. For `two_proportion_z`, the `(p1, p2)` pair counts as a provided `effect_size`.
2. Dispatch to the matching statsmodels solver: `TTestIndPower` (`two_sample_t`), `TTestPower` (`one_sample_t` / `paired_t`), `NormalIndPower` (`two_proportion_z`), `FTestAnovaPower` (`anova_oneway`). For ANOVA, `n` is the *total* sample size across groups; for two-sample tests it is per-group (group 1).
3. Wrap `ValueError` / `FloatingPointError` / `nan` solver returns in `infeasible_solution`.

**Output**: `{ok, test, solved_for, effect_size, effect_size_metric, n, n_total, power, alpha, alternative, interpretation}`. `effect_size_metric` is one of `cohens_d` / `cohens_h` / `cohens_f`; `n_total` is populated when meaningful (two-sample, paired, ANOVA).

**Errors**: `invalid_inputs` (wrong number of unknowns), `missing_proportions` (`two_proportion_z` without `p1`+`p2` or `effect_size`), `missing_k_groups`, `infeasible_solution`.

### 5.11 `fit_model`

**Input**:
- `name: str`
- `formula: str` — Wilkinson-style, e.g. `"price ~ sqft + bedrooms + C(neighborhood)"`.
- `kind: Literal["ols", "logistic", "poisson", "negbin"] = "ols"` — `negbin` is NB2 (variance = μ + α·μ²), the canonical remedy when a Poisson fit emits `overdispersion`.
- `robust: bool = False` — for OLS, use HC3 standard errors. Rejected for `negbin` (`robust_not_supported`).
- `model_name: str | None = None` — optional registry handle. Non-empty, no whitespace. When provided, the fitted result is stored in the session model registry for downstream `predict` / `evaluate_model` / `list_models` calls; duplicates → `model_name_collision`, invalid → `model_name_invalid`. When stored, the response echoes `"model_name": "..."`.

**Output**:
- `coefficients`: list of `{name, estimate, std_err, t, p_value, ci_low, ci_high}`.
- `fit`: `r_squared`, `adj_r_squared`, `aic`, `bic`, `n_obs`, `df_resid`. For `negbin` only: `dispersion_alpha`, `dispersion_alpha_se`, `pearson_chi2_over_df`.
- `diagnostics`: VIF per coefficient (if OLS), Breusch–Pagan p, Durbin–Watson, Jarque–Bera p, condition number.
- `warnings`: list of flagged issues — VIF > 10, BP p < 0.05, JB p < 0.05, Poisson `overdispersion`, NB `underdispersion_vs_negbin` (Pearson < 1.1) and `unstable_dispersion` (SE/α > 0.5).
- `interpretation`: 2–3 sentences; the `negbin` branch reports the IRR (incidence rate ratio).
- `model_name`: present only when stored in the registry.

### 5.11a `list_models`

**Input**: none.

**Output**: `{ok, models: [{name, kind, formula, fitted_on_dataset, n_obs, fitted_at}]}` — registration order. Read-only, no recorder cell (mirrors `list_datasets`).

### 5.11b `predict`

**Purpose**: score a registered model on a registered dataset.

**Input**:
- `model_name: str` — registry handle from a previous `fit_model` call.
- `dataset: str` — dataset to score; must contain every predictor referenced by the model's formula. Outcome column is optional.
- `output: Literal["link", "response", "class"] = "response"` — `link` = linear predictor η; `response` = inverse-link (probability / expected count / identity); `class` = thresholded class label (logistic only).
- `threshold: float = 0.5` — open interval `(0, 1)`; endpoints rejected.
- `limit: int = 50`, `cursor: str | None = None` — pagination via stringified source-dataset row index.
- `include_se: bool = False` — OLS only; adds per-row `se_mean` / `mean_ci_lower` / `mean_ci_upper` from `get_prediction(...).summary_frame()`.

**Behavior**:
- Reconstruct the design matrix via `patsy.dmatrix(rhs, df)`; report rows patsy drops (NaN predictors) in `dropped_rows` rather than skipping silently.
- `row_index` is 0-indexed in the **source dataset**; non-contiguous after drops so callers can SQL-join predictions back.
- In-sample prediction (`dataset == fitted_on_dataset`) is allowed; result matches the in-sample fit.

**Output**: `{ok, model_name, dataset, output_mode, predictions: [{row_index, y_pred, ...}], dropped_rows, total_rows, truncated, cursor}`. `y_class` appears when `output="class"`; SE block appears when `include_se=True`.

**Errors**: `model_not_found`, `dataset_not_found`, `missing_predictors`, `class_output_requires_logistic`, `threshold_out_of_range`, `include_se_requires_ols`, `formula_error`, `invalid_cursor`.

### 5.11c `evaluate_model`

**Purpose**: compute held-out (or in-sample) metrics for a registered model.

**Input**:
- `model_name: str`, `dataset: str` — registry / dataset handles.
- `threshold: float = 0.5` — classification cutoff for confusion-matrix-derived metrics.
- `n_calibration_bins: int = 10` — range `[2, 50]`. Auto-reduced to `min(10, n_obs // 20)` on small datasets; below 2 → `calibration: null` with a note.

**Behavior** (dispatch by model kind):
- **Logistic**: ROC-AUC, PR-AUC, Brier, log-loss (all sklearn); accuracy / precision / recall / F1; confusion matrix at `threshold`; calibration table (quantile bins, `{bin, mean_predicted, mean_observed, n}`).
- **OLS**: RMSE, MAE, R², adjusted R².
- **Poisson / NegBin**: RMSE on counts, MAE, Pearson χ², Poisson deviance.

Outcome dtype validation: logistic requires binary 0/1 or boolean; OLS requires numeric (boolean → `boolean_outcome_lpm` warning, proceed); poisson / negbin require non-negative integer.

**Output**: `{ok, model_name, dataset, metrics, n_obs, warnings}` plus `confusion_matrix` and `calibration` (logistic only). Recorder cell emits sklearn-based code; calibration DataFrame is included in code-cell output, summary stats in markdown.

**Errors**: `model_not_found`, `dataset_not_found`, `outcome_column_missing`, `outcome_dtype_mismatch`, `n_calibration_bins_out_of_range`.

### 5.12 `plot`

**Input**:
- `name: str`
- `kind: Literal["hist", "bar", "line", "scatter", "box", "violin", "heatmap"]`
- `x: str | None`, `y: str | None`, `hue: str | None`
- `title: str | None`
- `bins: int | None`

**Output**: `{"ok": true, "png_base64": "...", "width": 800, "height": 600}`.

### 5.12a `regression_line`

**Purpose**: scatter the response against a chosen predictor with the OLS fit line and 95 % mean-CI band overlaid. Pairs with `fit_model(kind="ols")` for visual diagnostics. **OLS-only** — logistic / Poisson / negbin have different canonical visuals and are out of scope.

**Input**:
- `model_name: str` — registry handle from a previous `fit_model` call.
- `predictor: str` — one of the model's exog names (Wilkinson RHS terms, `Intercept` excluded); must be numeric in the training DataFrame.
- `title: str | None = None`.

**Behavior**:
1. Look up the model; OLS guard otherwise.
2. Validate `predictor` is in `result.model.exog_names` (sans `Intercept`) and numeric in the training DataFrame.
3. Scatter `(predictor, endog)`; overlay the fit line by predicting across a dense grid of the predictor while holding every other predictor at its mean (numeric) or first value (factor).
4. Shade the 95 % mean-CI band from `get_prediction(...).summary_frame()`.

**Output**: `{ok, png_base64, width, height, model_name, plot_kind: "regression_line"}` — mirrors `plot`.

**Errors**: `model_not_found`, `regression_diagnostics_ols_only`, `column_not_found`, `non_numeric_predictor`, `internal`.

### 5.12b `residual_diagnostic`

**Purpose**: render the four canonical OLS residual diagnostics — residuals vs fitted, Q-Q, scale-location, residuals vs leverage with Cook's distance. Companion to `regression_line`; same OLS-only guard.

**Input**:
- `model_name: str` — registry handle from a previous `fit_model` call.
- `kind: Literal["resid_vs_fitted", "qq", "scale_location", "all"] = "all"` — `all` renders a 2×2 grid; everything else a single-axes figure.
- `title: str | None = None`.

**Behavior**:
1. Same lookup + OLS guard as §5.12a.
2. Compute `fitted`, `resid`, and `infl.resid_studentized_internal` once.
3. Render per `kind`:
   - `resid_vs_fitted` — scatter `(fitted, resid)` with `y=0` reference line and LOWESS overlay (`statsmodels.nonparametric.smoothers_lowess.lowess`).
   - `qq` — `scipy.stats.probplot(resid)` on the supplied axes.
   - `scale_location` — scatter `(fitted, sqrt(|standardized_resid|))` with LOWESS overlay.
   - `all` — 2×2 grid; fourth panel is residuals vs leverage with Cook's distance via `infl.cooks_distance`.
4. Figure size: `(12, 9)` for `all`, default `(8, 6)` otherwise.

**Output**: `{ok, png_base64, width, height, model_name, plot_kind}` where `plot_kind` echoes `payload.kind`.

**Errors**: `model_not_found`, `regression_diagnostics_ols_only`, `invalid_kind`, `internal`.

### 5.13 `emit_notebook`

**Input**:
- `path: str | None = None` — default: `./session_<timestamp>.ipynb`.
- `include_outputs: bool = False`.

**Behavior**:
- Take accumulated cells from recorder.
- Prepend a setup cell: imports, DuckDB connection, dataset reloads.
- Serialize via `nbformat.v4`.
- Write to disk.
- Return `{"ok": true, "path": "/abs/path/session_2026....ipynb", "n_cells": 42}`.

**Acceptance criterion**: a user can `jupyter nbconvert --execute` the emitted notebook and it runs to completion against the same input files.

---

## 6. The notebook recorder

### `recorder.py` interface

```python
class NotebookRecorder:
    def __init__(self) -> None: ...
    def record(self, *, markdown: str, code: str, tool_name: str) -> None: ...
    def reset(self) -> None: ...
    def to_notebook(self, include_setup: bool = True) -> "nbformat.NotebookNode": ...
```

### Each tool contributes one markdown + one code cell

**`load_dataset(path="fixtures/messy.csv", name="raw")`** markdown cell:
```markdown
### Loaded dataset `raw`
- Source: `fixtures/messy.csv`
- 5000 rows × 12 columns
- 3 parse warnings (see below)
```
Code cell:
```python
con.execute("""
    CREATE OR REPLACE TABLE raw AS
    SELECT * FROM read_csv_auto('fixtures/messy.csv', SAMPLE_SIZE=-1)
""")
raw_df = con.sql("SELECT * FROM raw").df()
raw_df.head()
```

**`compare_groups(name="orders", group_column="channel", metric_column="revenue")`** markdown cell:
```markdown
### Compared `revenue` across `channel` groups
- Test selected: **Welch's t-test** (normality OK, variances unequal at p=0.003)
- t = 4.31, p < 0.001, Cohen's d = 0.42 (small-to-medium)
- Mean revenue: organic = 142.10, paid = 118.40
- **Interpretation**: organic customers spend ~24 more per order on average.
```
Code cell:
```python
from scipy import stats
g_a = con.sql("SELECT revenue FROM orders WHERE channel = 'organic'").df()["revenue"]
g_b = con.sql("SELECT revenue FROM orders WHERE channel = 'paid'").df()["revenue"]
_, p_norm_a = stats.shapiro(g_a.sample(min(5000, len(g_a)), random_state=0))
_, p_norm_b = stats.shapiro(g_b.sample(min(5000, len(g_b)), random_state=0))
_, p_levene = stats.levene(g_a, g_b)
result = stats.ttest_ind(g_a, g_b, equal_var=False)
print(f"t = {result.statistic:.3f}, p = {result.pvalue:.4f}")
```

### Setup cell (prepended automatically by `to_notebook`)

```python
import duckdb
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

con = duckdb.connect()

# Reload datasets registered in the session
con.execute("""CREATE OR REPLACE TABLE raw AS SELECT * FROM read_csv_auto('fixtures/messy.csv', SAMPLE_SIZE=-1)""")
con.execute("""CREATE OR REPLACE TABLE orders AS SELECT * FROM read_parquet('orders.parquet')""")
```

The setup cell is rebuilt from the session's dataset registry at emit-time.

---

## 7. Test fixtures

### `fixtures/messy.csv` — the hostile CSV

5000-row CSV with:
- A UTF-8 BOM (`﻿`) at the start of the file.
- Trailing whitespace in 2 of the 12 column headers.
- A column `signup_date` with three formats mixed in: `2024-01-15`, `15/01/2024`, `Jan 15, 2024`.
- A column `revenue` containing the string `"N/A"` in 4% of rows and an empty string in another 2%.
- A column `notes` with one row containing an unquoted comma inside the value.
- Two exact duplicate rows.
- A column `email` that is 78% null.
- A column `score` with 20 values outside the IQR bounds, of which 5 are clearly data-entry errors (e.g. 99999).
- A column `country` with 95% values in `{"PL", "US", "DE", "UA"}` and 5% in `{"poland", "Poland", "POL", "pl "}`.

Generate with `fixtures/_build_messy.py`.

### `fixtures/synthetic_crm/`

Run `fixtures/generate_synthetic_crm.py` with a fixed seed:

- `accounts.csv` — 2000 rows: `account_id`, `account_name`, `industry`, `country`, `employees`, `arr`, `created_at`.
- `contacts.csv` — 20000 rows: `contact_id`, `account_id` (FK), `first_name`, `last_name`, `email`, `title`, `is_primary`.
- `opportunities.csv` — 8000 rows: `opp_id`, `account_id` (FK), `stage`, `amount`, `created_at`, `closed_at` (nullable), `won` (bool, null while open).

70% of opportunities closed, win rate ~28%, ARR log-normally distributed, opportunity amount correlated with account ARR, churn flag on ~10% of accounts. `faker` with fixed seed.

---

## 8. Evals

In `evals/`, pytest-style scripts that exercise the server through `mcp.client.stdio`. Programmatic pass/fail.

Minimum 24 evals:

**`eval_basic.py`** (6): load each fixture, list datasets, profile each, basic SQL, describe one column per fixture.

**`eval_messy_csv.py`** (8): one per planted issue.

**`eval_stats.py`** (6): on `synthetic_crm`:
- `compare_groups("opportunities", "stage", "amount")` returns ANOVA (>2 groups).
- Correlation between account `arr` and opportunity `amount` (joined) is positive and significant.
- `fit_model` on `won ~ amount + C(industry)` returns logistic with reasonable AIC.
- `test_hypothesis(kind="chi_square", ...)` for industry × won is well-formed.
- `compare_groups` switches to Mann–Whitney when normality is violated.
- Assumption-check warnings populated.

**`eval_full_workflow.py`** (4):
- Load → profile → compare_groups → fit_model → emit_notebook → nbconvert --execute exit 0.
- Notebook output contains expected variables (`con`, `raw_df`, model object).
- Session reset clears state.
- Determinism: same workflow twice → identical numerics.

Target: 100% pass.

---

## 9. Quality gates

| Gate | Command | Threshold |
|---|---|---|
| Format | `uv run ruff format --check .` | clean |
| Lint | `uv run ruff check .` | zero violations |
| Type check | `uv run pyright src/` | zero errors |
| Unit tests | `uv run pytest tests/ --cov=data_analyst_mcp` | >90% line |
| Branch coverage | `uv run pytest tests/ --cov=data_analyst_mcp --cov-branch` | >85% branch |
| Known-answer stats tests | `uv run pytest tests/test_stats.py tests/test_models.py -v` | all pass ≤1e-4 |
| TDD discipline check | `scripts/check_tdd_commits.py` | every `green:` preceded by matching `red:` |
| Evals | `uv run pytest evals/` | 100% pass |
| Server boot | `timeout 10 uvx --from . data-analyst-mcp` | exits 0 or runs >5s |
| Notebook re-execute | `jupyter nbconvert --execute evals/_artifacts/full_workflow.ipynb` | exit 0 |

---

## 10. Phased execution plan

### Phase 0 — Scaffold
Project metadata, deps, ruff/pyright/pytest config, `check_tdd_commits.py`, `conftest.py`, GH Actions, console script smoke test.

### Phase 1 — Fixtures
`fixtures/_build_messy.py` + `messy.csv`; `fixtures/generate_synthetic_crm.py` + 3 CSVs; `fixtures/README.md`.

### Phase 2 — Core dataset tools + recorder
`session.py`, `errors.py`, `formatting.py`, `recorder.py`, `tools/datasets.py`, `tools/query.py`. TDD per behavior.

### Phase 3 — Statistical tools
`tools/stats.py`. Known-answer TDD. Test-selection decision tree. `assumption_checks` always populated.

### Phase 4 — Modeling
`tools/models.py`. Coefficient match to published values ≤1e-3. Diagnostic & warning-rule cycles.

### Phase 5 — Plots
`tools/plots.py`. 7 kinds. Characterization tests.

### Phase 6 — Notebook emission
`tools/notebook.py`. 6-step round-trip exits 0 via `jupyter nbconvert --execute`.

### Phase 7 — Evals
≥24 evals through `mcp.client.stdio`. 100% pass.

### Phase 8 — Docs, packaging, release
README with worked example, `DEPS.md`, LICENSE, ROADMAP, tag `v0.1.0`, `uv build`, TestPyPI dry-run.

---

## 11. Anti-patterns — do not do these

- Production code without a failing test driving it.
- After-the-fact tests "covering" existing code.
- Batching multiple behaviors into one cycle.
- Skipping refactor.
- `green:` without prior matching `red:`.
- `print()` for debugging. Use stderr `logging`.
- Silently swallowing exceptions.
- Loading entire datasets into pandas at tool-call time.
- Adding `plotly`, `seaborn`, `altair`, `bokeh`, `dash`, `streamlit`, `polars`, `dask`, `ray`.
- Adding `Anthropic`/`OpenAI` clients to call an LLM from inside the server.
- Adding tools beyond the 21 in §5 (park in `ROADMAP.md`).
- Silently coercing data types in `load_dataset`.
- Making the recorder optional/feature-flagged.
- `emit_notebook` depending on internet, API key, or anything non-local.
- Spending time on logos, marketing sites, Twitter.

---

## 12. What "done" looks like

Fresh machine, under 5 minutes:

1. Adds 4 lines to `claude_desktop_config.json`.
2. Restarts Claude Desktop.
3. *"Profile the dataset at `~/Downloads/messy.csv` and tell me what's broken."* → structured profile naming the BOM, the 78%-null email, the mixed date formats, the case-inconsistent country values, and the 20 outliers in `score`.
4. *"Compare revenue between organic and paid channels and tell me if it's significant."* → Welch's t-test result + Cohen's d + plain-English explanation + **why Welch's and not Student's**.
5. *"Save this whole session as a notebook."* → `.ipynb` on disk.
6. `jupyter nbconvert --execute session_*.ipynb` reproduces every number from steps 3–5 exactly.
