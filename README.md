# data-analyst-mcp

A reproducible data-analyst MCP server. Point Claude (or any MCP-capable agent) at a CSV, Parquet, Excel, or JSON file and it gets a proper analytical reasoning layer: exploratory profiling, hypothesis tests with assumption checks, regression with diagnostics, and matplotlib plots — all via DuckDB so the same code path handles 100 MB or 5 GB. The differentiator is the **recorder**: every state-changing or analytic tool call appends a markdown + code cell to a session log (only the read-only listings, `emit_notebook` itself, and `load_session_from_notebook` don't), and `emit_notebook` writes the whole conversation out as a runnable `.ipynb` that `jupyter nbconvert --execute` will replay end-to-end — and that `load_session_from_notebook` can resume after a restart. It composes naturally with [`mcp-server-motherduck`](https://github.com/motherduckdb/mcp-server-motherduck) — that one gives the agent raw SQL, this one gives it the statistical thinking on top.

## Not for

- Streaming data
- Dashboards / BI replacement
- Causal inference
- Datasets > 10 GB

## Install

Two ways to run the server. Pick one — the rest of the doc only cares about the *client* configuration.

**Installed mode (recommended for non-contributors).** `uvx` clones the tagged commit, builds an isolated environment, and runs the console-script:

```bash
uvx --from git+https://github.com/olegtyshcneko/data-analyst-mcp@v0.1.0 data-analyst-mcp
```

There's no PyPI release. Pin to a tag (recommended) or use `@main` to track the tip; bump the ref when you want to update.

**Local-checkout mode (for contributors).** Clone the repo and let `uv` resolve the project root:

```bash
git clone https://github.com/olegtyshcneko/data-analyst-mcp.git
cd data-analyst-mcp
uv sync
uv run data-analyst-mcp
```

Edits to `src/` are picked up on the next server restart (clients spawn the server as a subprocess per session).

## Wire it into your MCP client

The server speaks stdio JSON-RPC. Every client below needs the same three things in its config: a `command`, an `args` list, and a stable identifier. The shape of that config differs per client.

### Scripted setup

`scripts/install_mcp.py` writes the right config for any of the six clients, merging into existing settings without trampling other servers:

```bash
# wire the in-tree checkout (default — picks up your edits)
uv run python scripts/install_mcp.py claude-code
uv run python scripts/install_mcp.py claude-desktop
uv run python scripts/install_mcp.py codex
uv run python scripts/install_mcp.py cursor
uv run python scripts/install_mcp.py opencode
uv run python scripts/install_mcp.py antigravity        # prints a snippet to paste

# or install once for every client above
uv run python scripts/install_mcp.py all

# preview without writing
uv run python scripts/install_mcp.py codex --dry-run

# wire the published git tag instead of this checkout
uv run python scripts/install_mcp.py claude-desktop --installed
```

| Client | Config file written | Scope |
|---|---|---|
| `claude-code` | `<repo>/.mcp.json` | project |
| `claude-desktop` | OS-specific (see below) | user |
| `codex` | `~/.codex/config.toml` | user |
| `cursor` | `<repo>/.cursor/mcp.json` | project |
| `opencode` | `<repo>/opencode.json` | project |
| `antigravity` | none — prints a snippet to paste via the editor UI | — |

All four project-scoped outputs are gitignored — they embed an absolute path to the local checkout, so they're per-user, not shared.

### Manual snippets

If you'd rather paste the config yourself, here's what `install_mcp.py` would write. Replace `<REPO>` with the absolute path to your local clone, or use the `--installed` snippet at the bottom of this section if you're not running from a checkout.

<details>
<summary><b>Claude Code</b> — project: <code>.mcp.json</code></summary>

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "uv",
      "args": ["--directory", "<REPO>", "run", "data-analyst-mcp"]
    }
  }
}
```

Claude Code prompts once to approve the server when it starts in this directory. `/mcp` lists connected servers from inside the session. For user-global scope, use `claude mcp add data-analyst --scope user -- uv --directory <REPO> run data-analyst-mcp`.

</details>

<details>
<summary><b>Claude Desktop</b> — user-global</summary>

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "uv",
      "args": ["--directory", "<REPO>", "run", "data-analyst-mcp"]
    }
  }
}
```

Restart Claude Desktop after editing.

</details>

<details>
<summary><b>Codex CLI / IDE extension</b> — user: <code>~/.codex/config.toml</code></summary>

```toml
[mcp_servers.data-analyst]
command = "uv"
args = ["--directory", "<REPO>", "run", "data-analyst-mcp"]
```

The CLI and the IDE extension share this file. Codex also accepts a project-scoped `.codex/config.toml` if the project is marked trusted.

</details>

<details>
<summary><b>Cursor</b> — project: <code>.cursor/mcp.json</code></summary>

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "uv",
      "args": ["--directory", "<REPO>", "run", "data-analyst-mcp"]
    }
  }
}
```

For user-global Cursor scope, write the same JSON to `~/.cursor/mcp.json`.

</details>

<details>
<summary><b>OpenCode</b> — project: <code>opencode.json</code></summary>

OpenCode uses a different shape: a top-level `mcp` key, `type: "local"`, and a single `command` array (no separate `args`).

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "data-analyst": {
      "type": "local",
      "command": ["uv", "--directory", "<REPO>", "run", "data-analyst-mcp"],
      "enabled": true
    }
  }
}
```

For user-global OpenCode scope, write the same JSON to `~/.config/opencode/opencode.json`.

</details>

<details>
<summary><b>Antigravity</b> — editor UI only</summary>

Antigravity stores its MCP config in an editor-managed file. In the editor: **Agent panel → MCP Servers → Manage MCP Servers → "View raw config"**, then paste this entry into the `mcpServers` object:

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "uv",
      "args": ["--directory", "<REPO>", "run", "data-analyst-mcp"]
    }
  }
}
```

Click **Refresh** in Manage MCP Servers to reload.

</details>

<details>
<summary><b>Installed-mode variant</b> — same shape, no local checkout required</summary>

Replace the `command` / `args` with the published git-tag form. Identical across Claude Code, Claude Desktop, Cursor, and Antigravity:

```json
{
  "command": "uvx",
  "args": ["--from", "git+https://github.com/olegtyshcneko/data-analyst-mcp@v0.1.0", "data-analyst-mcp"]
}
```

For Codex (TOML):

```toml
command = "uvx"
args = ["--from", "git+https://github.com/olegtyshcneko/data-analyst-mcp@v0.1.0", "data-analyst-mcp"]
```

For OpenCode (single command array):

```json
"command": ["uvx", "--from", "git+https://github.com/olegtyshcneko/data-analyst-mcp@v0.1.0", "data-analyst-mcp"]
```

</details>

After restarting (or reloading MCP servers in) your client, all 25 tools become available in any new conversation: the original 11 (`load_dataset`, `list_datasets`, `profile_dataset`, `describe_column`, `query`, `correlate`, `compare_groups`, `test_hypothesis`, `fit_model`, `plot`, `emit_notebook`), the v1.x additions (`adjust_pvalues`, `analyze_missingness`, `list_models`, `predict`, `evaluate_model`), the tier-1 bundle (`materialize_query`, `find_outliers`, `power_analysis`, `regression_line`, `residual_diagnostic`), the post-tier-1 addition `pairwise_comparisons` (post-hoc pairs after `compare_groups`), the model-workflow bundle (`split_dataset`, `cross_validate`), and the session-resume tool `load_session_from_notebook` (1.6.0).

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

#### 4a. Score and evaluate (Phase 5 composition recipe)

When the agent wants to validate predictive performance on held-out data, the model-registry trio composes one-shot. The `titanic_train` / `titanic_test` datasets can now be produced in-session with `split_dataset(name="titanic")` (seeded, replayable) instead of pre-splitting the files outside the server:

```python
fit_model(name="titanic_train", formula="Survived ~ C(Sex) + C(Pclass) + Age",
          kind="logistic", model_name="titanic_logit")   # 1. fit + register
predict(model_name="titanic_logit", dataset="titanic_test",
        output="response", limit=1000)                   # 2. score
evaluate_model(model_name="titanic_logit", dataset="titanic_test",
               n_calibration_bins=10)                    # 3. metrics
list_models()                                            # 4. inspect registry
```

`evaluate_model` returns ROC-AUC, PR-AUC, Brier, log-loss, accuracy / precision / recall / F1, the confusion matrix at `threshold`, and a quantile calibration table for logistic models; RMSE / MAE / R² / adj-R² for OLS; RMSE / MAE / Pearson χ² / deviance for Poisson / negbin. The emitted notebook re-fits every registered model in its setup cell behind a hard SHA-256 assert on the training file — silent drift between session and replay becomes a loud `AssertionError`.

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

### 7. Resume the session later

> **You** (after restarting Claude Desktop): Pick up where we left off — load `/abs/path/session_2026-05-11_*.ipynb`.

Claude calls `load_session_from_notebook(path="/abs/path/session_2026-05-11_*.ipynb")`. The server verifies the notebook's embedded journal manifest (cell hashes, source-file hashes), replays every recorded operation inside one transaction — comparing table digests, split membership, and model coefficients/standard errors against the recorded evidence — and restores the full session: datasets, registered models, and the recorder history. The next `emit_notebook` produces one unified notebook covering both sittings. If anything drifted (a source CSV edited, a non-deterministic query), the tool aborts atomically with a structured error naming the diverging operation, and the session stays empty.

## Tier-1 bundle: cohorts, outliers, power, diagnostics

The four tools shipped in the tier-1 bundle close the most common gaps the agent runs into during exploratory and modeling work. Each one takes a single call.

**Persist a derived dataset** — `query` is read-only, so `materialize_query` is the way to keep a join around:

```python
materialize_query(
    sql="SELECT * FROM opps WHERE stage IN ('Closed Won', 'Closed Lost')",
    name="opps_closed",
)
# → {"ok": true, "name": "opps_closed", "rows": 5_528, "columns": [...]}
```

Every downstream tool (`describe_column`, `correlate`, `fit_model`, `find_outliers`, …) can now target `opps_closed` by name, and the emitted notebook rehydrates the derived table after the file-backed ones at replay.

**Joint-outlier detection** — `describe_column` flags per-column outliers; `find_outliers` catches rows that are extreme in the K-dimensional manifold:

```python
find_outliers(
    name="messy",
    columns=["score", "revenue"],
    method="mahalanobis",   # or "iqr" / "zscore" / "isolation_forest"
)
# → {"ok": true, "n_outliers": 23, "outliers": [{"row_index": 4_182,
#     "score": 234.7, "values": {"score": 123_456.0, "revenue": ...}}, ...]}
```

**Plan an A/B test before running it** — `power_analysis` solves for whichever of `effect_size` / `n` / `power` is omitted, across the five test families that map onto `compare_groups`:

```python
power_analysis(test="two_proportion_z", p1=0.10, p2=0.12, power=0.8, alpha=0.05)
# → {"ok": true, "solved_for": "n", "n": 3_835, "n_total": 7_670,
#    "effect_size_metric": "cohens_h", "interpretation": "Need 3835 per group ..."}
```

**Visual model diagnostics** — once an OLS model is in the registry, `regression_line` overlays the fit on a chosen predictor and `residual_diagnostic` renders the four canonical residual plots:

```python
fit_model(name="opps", formula="amount ~ employees + arr",
          kind="ols", model_name="opp_amount")
regression_line(model_name="opp_amount", predictor="arr")
# → PNG with scatter + fit line + 95% mean-CI band
residual_diagnostic(model_name="opp_amount", kind="all")
# → 2×2 grid: resid-vs-fitted, Q-Q, scale-location, leverage with Cook's distance
```

Both diagnostic tools are OLS-only — logistic / Poisson / negbin return `regression_diagnostics_ols_only`.

## Post-hoc pairwise comparisons

`compare_groups` runs the omnibus test and stops at "the groups differ." `pairwise_comparisons` is the follow-up — *which pairs differ* — over all `n·(n−1)/2` pairs, gated by the same Shapiro auto-selection: **Tukey HSD** after ANOVA, a vendored tie-corrected **Dunn's test** (Holm-adjusted by default) after Kruskal–Wallis.

```python
pairwise_comparisons(name="opps", group_column="stage", metric_column="amount")
# → {"ok": true, "method": "dunn", "p_adjust": "holm",
#    "comparisons": [{"group_a": "Closed Lost", "group_b": "Closed Won",
#                     "estimate": 812.4, "p_adj": 0.031, "reject": true}, ...],
#    "n_comparisons": 15, "n_rejected": 3, "omnibus": {"test": "kruskal_wallis", ...}}
```

It recomputes the omnibus inline and caveats the interpretation when the family is not significant. The emitted code cell is fully runnable — it rehydrates the groups from the notebook's DuckDB connection and reproduces the table.

## Train/test splits and cross-validation

`split_dataset` closes the loop the model registry opened: a seeded,
optionally stratified partition registered as two first-class datasets.
The same seed always produces the same split — membership comes from
`np.random.RandomState`, not DuckDB sampling — and the emitted notebook
recreates each side behind its own order-independent membership checksum, so
silent drift is impossible on either side.

```python
split_dataset(name="titanic", seed=7)          # → titanic_train / titanic_test
fit_model(name="titanic_train", formula="Survived ~ C(Sex) + C(Pclass)",
          kind="logistic", model_name="surv")
evaluate_model(model_name="surv", dataset="titanic_test")
```

`cross_validate` is the re-fitting complement to `evaluate_model` — k-fold
CV metrics for a formula, fits ephemeral, registry untouched. Logistic
folds are auto-stratified by outcome class; a full-data preflight fit
surfaces `fit_model`'s error taxonomy before any fold work.

```python
cross_validate(name="titanic_train", formula="Survived ~ C(Sex) + C(Pclass)",
               kind="logistic", k=5)
# → {"metrics": {"roc_auc": {"mean": ..., "std": ..., "per_fold": [...]}, ...}}
```

## Compose with MotherDuck MCP

This server gives the agent the **analytical reasoning** layer; [`mcp-server-motherduck`](https://github.com/motherduckdb/mcp-server-motherduck) gives it the **raw-SQL-against-a-warehouse** layer. They are complementary, not competing — run both in the same `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/olegtyshcneko/data-analyst-mcp@v0.1.0", "data-analyst-mcp"]
    },
    "motherduck": {
      "command": "uvx",
      "args": ["mcp-server-motherduck", "--token", "${MOTHERDUCK_TOKEN}"]
    }
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
 25 tools (datasets / query / stats /     NotebookRecorder
 models / registry / plots / notebook)    (markdown + code cells)
            │                                         │
            ▼                                         ▼
       DuckDB connection ──────────────► nbformat → session_*.ipynb
            │                              (setup cell hash-guards datasets,
            ▼                               re-fits models behind SHA-256)
  CSV · Parquet · Excel · JSON · s3://
```

Single process, single DuckDB connection, single recorder. No network calls. No LLM calls from inside the server — the server returns structured data, the agent does the reasoning.

## Known gotchas

- **`query` is read-only.** It accepts `SELECT` / `WITH` / `DESCRIBE` / `SHOW` / `EXPLAIN` / `PRAGMA show_tables` and rejects writes with `error.type = "write_not_allowed"`. To persist a join-derived table, use `materialize_query(sql=..., name=...)` — it registers the result as a first-class derived dataset that every other tool can target by name.
- **Logistic regression with (quasi-)perfectly-separating predictors** returns a structured `error.type = "perfect_separation"` instead of a fit: the maximum-likelihood estimates diverge, so no coefficients/diagnostics are produced and the model is not registered even if `model_name` was supplied. Perfect collinearity in the design is reported as `formula_error`; a non-converged logit without the divergence signature returns `convergence_failed`.
- **Boolean response columns in `fit_model`.** Pandas' nullable `BooleanDtype` is coerced to `{0, 1}` numeric before being handed to statsmodels' `Logit`; plain Python booleans work too. Mixed `True` / `"yes"` strings are not.
- **Datasets are in-process state.** Restarting Claude Desktop drops the registry. Emit a notebook before you stop, then `load_session_from_notebook(path=...)` in the fresh session — it verifies the notebook's embedded journal manifest, replays every recorded operation with drift guards, and restores datasets, models, and the recorder history atomically (any divergence from the recorded evidence aborts with the session untouched). Requires an empty fresh session; sessions holding in-memory dataframes are not resumable (`resume_supported: false` in the manifest).
- **Emitted notebooks are drift-guarded.** The setup cell asserts a SHA-256 provenance hash for every file-backed dataset (and for base files behind `materialize_query` overwrites) before reloading it, and since 1.5.0 every `load_dataset` cell in the notebook body re-asserts its own load-time hash — so a file mutated mid-session fails replay at the load that saw the old bytes, even when a later reload keeps the setup cell happy. `cross_validate` / unregistered `fit_model` cells recorded against in-memory datasets now open with an explanatory `raise AssertionError` instead of a bare `CatalogException`. If a source file changed since the session, replay fails with a loud `AssertionError` instead of silently recomputing different numbers. Files over 100 MB use a weaker `(path, mtime, size)` check; s3/http sources reload unguarded (a comment in the cell says so). Models re-fit from the original file only when they were fit *on* the file-backed state (identified by registration revision); a model fit on any table state that no longer exists at replay — re-materialized, re-loaded, or re-split, even with identical bytes — fails the setup cell with a loud `AssertionError` instead of silently re-fitting.
- **Stdio only in v1.** No SSE, no HTTP. If you want to share one server across multiple clients, that's on the roadmap.

## Development

```bash
uv sync --dev                    # install runtime + dev deps
uv run pytest tests/             # 644 unit tests
uv run pytest evals/             # 62 integration evals via mcp.client.stdio (~40s)
uv run ruff format --check .     # formatter gate
uv run ruff check .              # linter gate
uv run pyright src/              # strict type-check on src/
uv run python scripts/check_tdd_commits.py   # every green: must have a matching red:
```

The implementation spec is `docs/SPEC.md`. It is the source of truth — when in doubt, the spec wins.

## Contributing

Issues and PRs are welcome. **Open an issue first for any new tool**: the 24-tool surface is intentionally closed at the v2 boundary (spec §5, §11, ROADMAP), and tool ideas are parked in `ROADMAP.md` until they're either promoted to v3 or explicitly declined. `pairwise_comparisons` is the reference example of that flow working end to end — issue → `docs/proposals/` draft → design conversation → fold into SPEC §5.9a. Bug fixes, doc improvements, and test additions are easier to land — just open a PR. Every change in `src/` must come with a failing test first (`red:` then `green:`); the `check_tdd_commits.py` script enforces this on the commit log.

## License

MIT — see `LICENSE`.
