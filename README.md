# data-analyst-mcp

A reproducible data-analyst MCP server. Point Claude (or any MCP-capable agent) at a CSV, Parquet, Excel, or JSON file and it gets a proper analytical reasoning layer: exploratory profiling, hypothesis tests with assumption checks, regression with diagnostics, and matplotlib plots — all via DuckDB so the same code path handles 100 MB or 5 GB. The differentiator is the **recorder**: every tool call appends a markdown + code cell to a session log, and `emit_notebook` writes the whole conversation out as a runnable `.ipynb` that `jupyter nbconvert --execute` will replay end-to-end. It composes naturally with [`mcp-server-motherduck`](https://github.com/motherduckdb/mcp-server-motherduck) — that one gives the agent raw SQL, this one gives it the statistical thinking on top.

## Not for

- Streaming data
- Dashboards / BI replacement
- Causal inference
- Datasets > 10 GB

## Install

```bash
uvx data-analyst-mcp
```

That's it — `uvx` resolves the package, builds an isolated environment, and runs the console-script. Claude Desktop will invoke this for you once you've added the server to its config.

## Configure Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "uvx",
      "args": ["data-analyst-mcp"]
    }
  }
}
```

The file lives at:

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

Restart Claude Desktop and the 11 tools (`load_dataset`, `list_datasets`, `profile_dataset`, `describe_column`, `query`, `correlate`, `compare_groups`, `test_hypothesis`, `fit_model`, `plot`, `emit_notebook`) become available in any new conversation.

## Worked example

A six-step session against `fixtures/messy.csv` — a deliberately hostile 5 000-row CSV with a UTF-8 BOM, mixed date formats, a 78%-null email column, case-inconsistent country values, and a handful of data-entry-error outliers in `score`. Every number below is what the server actually returns on that fixture.

### 1. Profile the file

> **You:** Profile the dataset at `fixtures/messy.csv` and tell me what's broken.

Claude calls `load_dataset(path="fixtures/messy.csv", name="messy")` followed by `profile_dataset(name="messy")` and comes back with:

> Loaded `messy` — 5 000 rows × 12 columns. A few things look off:
>
> - **`revenue` is stored as text.** 200 rows contain the literal string `"N/A"` and another 100 are empty. The column reads as `VARCHAR` even though it should be numeric — you'll want to coerce it before any aggregation.
> - **`signup_date` is a mixed-format string column.** Sampling shows `2024-04-18`, `18/08/2023`, and `Jul 21, 2024` all coexisting. Parsing this needs a multi-format dispatch.
> - **`country` has 8 distinct values but only 4 real countries.** The big buckets are `DE` (1 238), `UA` (1 194), `PL` (1 186), `US` (1 132); the long tail is `pl ` (72, trailing space), `POL` (66), `Poland` (56), `poland` (56) — all PL in disguise.
> - **`email` is 78% null** (3 900 / 5 000). The profiler flagged it `mostly_null`; treat it as essentially missing.
> - **`score` has 20 IQR outliers, of which 5 are >3σ.** Median 70.0, p99 87.3, but the max is 123 456.0 and the mean is dragged up to 167.9. Suggested follow-up: `describe_column(name="messy", column="score")`.
>
> Suggested next steps: drop or impute `email`; clean `country` casing; describe `score` to inspect the outliers.

### 2. Fix the date column

> **You:** Normalize `signup_date` into a real date.

Claude calls `query` with DuckDB's `try_strptime` over the three formats:

```sql
SELECT
  customer_id,
  COALESCE(
    TRY_STRPTIME(signup_date, '%Y-%m-%d'),
    TRY_STRPTIME(signup_date, '%d/%m/%Y'),
    TRY_STRPTIME(signup_date, '%b %d, %Y')
  ) AS signup_date_clean
FROM messy
LIMIT 5
```

The tool returns the parsed timestamps plus `total_rows`, `truncated`, and `execution_time_ms`. (Note: `query` is read-only — see "Known gotchas" below.)

### 3. Compare groups

> **You:** Is the opportunity `amount` different across `stage` in `fixtures/synthetic_crm/opportunities.csv`?

Claude calls `load_dataset` then `compare_groups(name="opps", group_column="stage", metric_column="amount")` and reports:

> 6 stages, n ranging from 561 (Proposal) to 4 063 (Closed Lost). Normality (Shapiro) is violated at p ≈ 3 × 10⁻⁷⁶, so the server **switched from ANOVA to Kruskal–Wallis**. Levene says variances are fine (p = 0.31). Kruskal–Wallis H = 8.97, p = 0.110, ε² = 0.001. **No evidence of group differences at α = 0.05.**

The raw response includes the `assumption_checks` block verbatim — both the Shapiro and Levene results, each with `violated` and `consequence` fields — so the test selection is auditable.

### 4. Fit a model

> **You:** Fit `amount ~ C(stage)` as OLS and tell me what the diagnostics say.

Claude calls `fit_model(name="opps", formula="amount ~ C(stage)", kind="ols")`:

> n = 8 000, R² = 0.0008, adj R² = 0.0002 — stage explains essentially none of the variance, which matches the Kruskal result above. Intercept = 6 859.7 (SE 200.8). Diagnostics: condition number 5.0 (clean), Breusch–Pagan p = 0.34 (homoskedastic), Durbin–Watson 2.02 (no autocorrelation), Jarque–Bera p ≈ 0 (residuals non-normal → `warnings: ["non_normal_residuals"]`). Inference on coefficients is reliable; predictive use of the model is not.

### 5. Plot

> **You:** Show me the distribution of `amount`.

Claude calls `plot(name="opps", kind="hist", x="amount", bins=30, title="Opportunity amounts")` and gets back `{"ok": true, "png_base64": "...", "width": 701, "height": 546}`. Claude Desktop renders the PNG inline.

### 6. Save the session

> **You:** Save this whole session as a notebook.

Claude calls `emit_notebook()`. The recorder has been accumulating one markdown + one code cell per tool call since step 1, plus a setup cell that imports DuckDB / pandas / scipy / statsmodels / matplotlib and re-registers each dataset from its original path. The tool returns `{"ok": true, "path": "/abs/path/session_2026-05-11_*.ipynb", "n_cells": 14}`.

Then, in a shell:

```bash
jupyter nbconvert --execute session_2026-05-11_*.ipynb
```

Exit code 0. Every number from steps 1 – 5 is recomputed from the source files. The notebook is the audit trail.

## Compose with MotherDuck MCP

This server gives the agent the **analytical reasoning** layer; [`mcp-server-motherduck`](https://github.com/motherduckdb/mcp-server-motherduck) gives it the **raw-SQL-against-a-warehouse** layer. They are complementary, not competing — run both in the same `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "data-analyst": { "command": "uvx", "args": ["data-analyst-mcp"] },
    "motherduck":   { "command": "uvx", "args": ["mcp-server-motherduck", "--token", "${MOTHERDUCK_TOKEN}"] }
  }
}
```

The agent will reach for MotherDuck when the task is "query the warehouse" and for `data-analyst` when the task is "do statistics on what came back." The split is intentional; do not look for one server that does both.

## Architecture

```
   Claude Desktop / any MCP client
            │  stdio (JSON-RPC)
            ▼
        FastMCP server
            │
   ┌────────┴────────────────────────────────┐
   ▼                                         ▼
 11 tools (datasets / query / stats /     NotebookRecorder
 models / plots / notebook)               (markdown + code cells)
            │                                         │
            ▼                                         ▼
       DuckDB connection ──────────────► nbformat → session_*.ipynb
            │
            ▼
  CSV · Parquet · Excel · JSON · s3://
```

Single process, single DuckDB connection, single recorder. No network calls. No LLM calls from inside the server — the server returns structured data, the agent does the reasoning.

## Known gotchas

- **`query` is read-only.** It accepts `SELECT` / `WITH` / `DESCRIBE` / `SHOW` / `EXPLAIN` / `PRAGMA show_tables` and rejects writes with `error.type = "write_not_allowed"`. There is no tool to register a query result as a new dataset in v1 — if you want to keep a join-derived table around, re-run the join inline each time, or use the MotherDuck server alongside. See `ROADMAP.md`.
- **Logistic regression with perfectly-separating predictors** raises a `statsmodels` warning rather than a structured error today. The fit still returns, but standard errors will be `NaN`. Tracked in `ROADMAP.md`.
- **Boolean response columns in `fit_model`.** Pandas' nullable `BooleanDtype` is coerced to `{0, 1}` numeric before being handed to statsmodels' `Logit`; plain Python booleans work too. Mixed `True` / `"yes"` strings are not.
- **Datasets are in-process state.** Restarting Claude Desktop drops the registry. Re-`load_dataset` after a restart, or run the emitted notebook to rehydrate.
- **Stdio only in v1.** No SSE, no HTTP. If you want to share one server across multiple clients, that's on the roadmap.

## Development

```bash
uv sync --dev                    # install runtime + dev deps
uv run pytest tests/             # 146 unit tests
uv run pytest evals/             # 24 integration evals via mcp.client.stdio (~30s)
uv run ruff format --check .     # formatter gate
uv run ruff check .              # linter gate
uv run pyright src/              # strict type-check on src/
uv run python scripts/check_tdd_commits.py   # every green: must have a matching red:
```

The implementation spec is `docs/SPEC.md`. It is the source of truth — when in doubt, the spec wins.

## Contributing

Issues and PRs are welcome. **Open an issue first for any new tool**: the 11-tool surface is intentionally closed in v1 (spec §5, §11), and tool ideas are parked in `ROADMAP.md` until they're either promoted to v2 or explicitly declined. Bug fixes, doc improvements, and test additions are easier to land — just open a PR. Every change in `src/` must come with a failing test first (`red:` then `green:`); the `check_tdd_commits.py` script enforces this on the commit log.

## License

MIT — see `LICENSE`.
