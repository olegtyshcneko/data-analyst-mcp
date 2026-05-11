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
| `eval_full_workflow.py` | 4 | six-step recorded session → emit → `jupyter nbconvert --execute` exit 0; emitted setup-cell contents; session-reset isolation; determinism across two runs |

Total: 24 evals (the spec floor).

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
