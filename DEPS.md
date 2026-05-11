# Dependencies

Why each dependency is in `pyproject.toml`. One line per package; bumping or removing anything below should reference this file.

## Runtime

| Package | Why it's here |
|---|---|
| `mcp[cli]` | Official Anthropic MCP SDK. Provides `FastMCP` plus the stdio transport that Claude Desktop speaks. The `[cli]` extra pulls in the `mcp` console entry-point used by local debugging. |
| `duckdb` | In-process columnar query engine. Reads CSV / TSV / Parquet / Excel / JSON natively (no extra readers needed), runs analytical SQL fast on 100 MB – 5 GB datasets, and is the single code path behind every tool — no `pd.read_*` lurking inside tool bodies (spec §2). |
| `pandas` | Interop layer between DuckDB result sets and `scipy` / `statsmodels`, both of which expect pandas DataFrames / Series. We do not use pandas for I/O — DuckDB owns reads. |
| `scipy` | `scipy.stats` for hypothesis tests (t-test, Welch, Mann–Whitney, ANOVA, Kruskal–Wallis, chi-square, Fisher exact, KS) and the assumption checks (Shapiro, D'Agostino, Levene). |
| `statsmodels` | Regression engine for `fit_model` — OLS, logistic (Logit), Poisson — plus the diagnostic battery (VIF, Breusch–Pagan, Durbin–Watson, Jarque–Bera, condition number) that ships with each fit. |
| `matplotlib` | Plot rendering. Returns base64-encoded PNG so the MCP client can inline the image. Fixed style sheet; nothing else (no plotly / seaborn / altair / bokeh — spec §2). |
| `nbformat` | Jupyter notebook construction. `emit_notebook` uses `nbformat.v4` to serialize the recorder's cells into a `.ipynb` that `jupyter nbconvert --execute` can replay end-to-end. |
| `pydantic` | Tool input validation. FastMCP wires pydantic v2 models directly into JSON-schema for the MCP `inputSchema`, so the `Field(description=...)` text on every model is what the LLM actually reads. |

## Dev

| Package | Why it's here |
|---|---|
| `pytest` | Test runner. Drives the 146 unit tests under `tests/` and the 24 evals under `evals/`. |
| `pytest-cov` | Coverage reporter. Gates the build at >90% line and >85% branch on `src/data_analyst_mcp/`. |
| `pytest-asyncio` | Async test support, used by evals that drive the server through `mcp.client.stdio`. Note: `pytest-asyncio` and `anyio` cancel-scopes do not cooperate well in yield-style fixtures — eval modules use `async with` inside the test body instead of yield-fixtures (see `evals/conftest.py`). |
| `ruff` | Formatter and linter. Single tool replaces black + isort + flake8 + several pylint rules. |
| `pyright` | Strict-mode type checker on `src/`. Off for `tests/` and `evals/` so test-time monkey-patching stays ergonomic. |
| `faker` | Deterministic synthetic CRM fixtures. Seeded once per fixture file so `accounts.csv` / `contacts.csv` / `opportunities.csv` are reproducible. |
| `jupyter` | Pulls in `nbconvert` and friends — the eval suite shells out to `jupyter nbconvert --execute` to prove the emitted `.ipynb` actually round-trips. |
| `nbclient` | Programmatic notebook execution used by the round-trip evals to assert that the emitted notebook re-computes the same numbers as the live session. |
| `pillow` | Decodes the base64 PNG returned by `plot` so plot tests can assert image dimensions without eyeballing pixels. |
