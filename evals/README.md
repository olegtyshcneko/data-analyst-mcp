# Evals

End-to-end evals that exercise the `data-analyst-mcp` server through the
live MCP stdio protocol. Each test spawns a fresh `uv run
data-analyst-mcp` subprocess via `mcp.client.stdio.stdio_client` and
drives it through an initialized `mcp.ClientSession` — the protocol path
is what's under test, not the in-process function imports covered by
`tests/`.

## Run

```bash
# every eval
uv run pytest evals/

# one file
uv run pytest evals/eval_basic.py -v

# select by marker
uv run pytest evals/ -m eval -v
```

The default `pytest` (no path) only runs `tests/`; evals are slower
because each test pays a subprocess-spawn cost for isolation. CI runs
them as a separate step.

## What each file checks

| File | Count | Coverage |
|---|---|---|
| `eval_basic.py` | 6 | load → list → profile → query → describe smoke against both fixtures |
| `eval_messy_csv.py` | 8 | one eval per planted issue in `fixtures/messy.csv` (BOM, header whitespace, mixed date formats, "N/A" revenue, duplicates, 78%-null email, IQR outliers, case-inconsistent country) |
| `eval_stats.py` | 6 | `compare_groups` test selection (ANOVA/Kruskal/Mann-Whitney), `correlate`, `test_hypothesis` chi-square, `fit_model` logistic, assumption_checks always populated |
| `eval_titanic.py` | 8 | real-world reference dataset (`fixtures/titanic.csv`): survival rate by Sex / Pclass, χ² Sex × Survived, Kruskal Fare ~ Pclass, logit `Survived ~ Sex + Age + C(Pclass)`, slash-in-column-name SQL quoting |
| `eval_materialize.py` | 3 | `materialize_query` persists a derived dataset; downstream tools target it by name; notebook round-trip rehydrates it |
| `eval_outliers.py` | 7 | `find_outliers` all four methods surface the planted `messy.csv` extreme (≤2-D), plus high-`k` paths on `fixtures/breast_cancer.csv`: Mahalanobis over 10 features (χ²(df=10) cutoff + largest tumor), Isolation Forest over 10 features, and the singular-covariance pseudoinverse fallback |
| `eval_power.py` | 3 | `power_analysis` solves for `n`/MDE across the test families and echoes the effect-size metric |
| `eval_diagnostic_plots.py` | 2 | `regression_line` + `residual_diagnostic` return valid PNGs for an OLS model and reject non-OLS with `regression_diagnostics_ols_only` |
| `eval_pairwise.py` | 4 | `pairwise_comparisons` auto-picks Dunn on skewed CRM amounts, explicit Tukey recovers a planted mean shift, `method='tukey'`+`p_adjust` returns the `p_adjust_not_applicable` error envelope, and the recorded Dunn notebook round-trips through `nbconvert` |
| `eval_full_workflow.py` | 4 | six-step recorded session → emit → `jupyter nbconvert --execute` exit 0; emitted setup-cell contents; session-reset isolation; determinism across two runs |
| `eval_split_cv.py` | 3 | split → fit → evaluate → cross_validate emitted-notebook round-trip via nbconvert: clean session exits 0, mutated source CSV fails loudly |

Total: 54 evals.

## Conventions

- Every test is marked `@pytest.mark.eval` so it can be selected with
  `-m eval` or excluded with `-m "not eval"`.
- Test names start with `eval_` (configured via `python_functions` in
  `pyproject.toml`) — keep them descriptive of the planted condition.
- Sessions are opened via the `mcp_session()` async context manager
  (defined in `conftest.py`) inside each test body rather than via a
  yield-fixture, because anyio's cancel-scope discipline rejects fixture
  teardown that crosses task boundaries.
- Emitted notebooks land in `evals/_artifacts/` (gitignored).
- Assertions are programmatic — no "LLM thinks this looks good" checks.

## Exit codes

- `0`: every eval passed.
- `1`: at least one eval failed. `pytest` prints the failing assertion
  and the structured response from the server.
- Non-zero from `_nbconvert` (in `eval_full_workflow.py`) indicates a
  notebook re-execution failure — the differentiator is broken.
