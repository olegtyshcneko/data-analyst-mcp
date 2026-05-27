# `materialize_query`

## Purpose

`query` is read-only, so every cohort / funnel / multi-step join is
re-executed inline by the agent on each call. `materialize_query` persists
a `SELECT` (or `WITH`) result as a named DuckDB table and registers it in
the session dataset registry, so downstream tools (`describe_column`,
`correlate`, `compare_groups`, `fit_model`, `find_outliers`, …) can target
the derived table by name. Unblocks the standard cohort → metric →
model workflow.

## Input

```python
class MaterializeQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str = Field(min_length=1)
    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    overwrite: bool = False
```

## Behavior

1. Validate the leading SQL keyword against the narrower allowlist
   `("SELECT", "WITH")` — `DESCRIBE` / `SHOW` / `PRAGMA` are valid for
   `query` but meaningless as a table source and are rejected with
   `write_not_allowed`.
2. If `name` already exists in the dataset registry and `overwrite=False`,
   return `dataset_name_collision`.
3. Execute `CREATE OR REPLACE TABLE "{name}" AS {sql}` on the live
   DuckDB connection.
4. Probe row count via `SELECT COUNT(*) FROM "{name}"` and columns via
   `DESCRIBE "{name}"`.
5. Register a `DatasetEntry` through `session.register(...)` with
   `path="(query)"`, `format="derived"`,
   `read_options={"sql": payload.sql}`, plus the probed row count and
   column list.
6. Emit the recorder cell (success path only).

**Setup-cell rehydration.** `recorder._build_setup_source`
(`src/data_analyst_mcp/recorder.py:43-140`) iterates
`session.get_datasets()` twice: the first pass emits file-backed
`CREATE OR REPLACE TABLE` lines, the second pass emits derived ones
using `entry.read_options["sql"]`. Derived-dataset lines **must** be
emitted after every file-backed line — the derived SQL references those
base tables and DuckDB would fail at replay otherwise. No hash assert is
written for the derived block: the recipe is the SQL plus the upstream
datasets, which already carry their own asserts via the model
rehydration block when models depend on them.

## Output

```python
{
    "ok": True,
    "name": str,            # echoed payload.name
    "rows": int,            # COUNT(*) of the materialized table
    "columns": list[dict],  # [{"name": ..., "type": ...}, ...] from DESCRIBE
    "total_rows": int,      # alias for rows; mirrors load_dataset's shape
}
```

## Errors

- `write_not_allowed` — leading keyword is not `SELECT` / `WITH`.
- `dataset_name_collision` — `name` already registered and
  `overwrite=False`. Hint: pass `overwrite=True` or pick a fresh name.
- `invalid_name` — `name` violates the identifier regex (raised by
  pydantic validation).
- `query_error` — DuckDB raised during execution (missing table, syntax
  error, type mismatch); the DuckDB message is surfaced in `details`.
- `internal` — fallback for unexpected exceptions.

## Recorder cells

```python
get_recorder().record(
    markdown=f"### Materialize query as dataset `{payload.name}`\n\n```sql\n{payload.sql}\n```",
    code=f'con.execute("""CREATE OR REPLACE TABLE \"{payload.name}\" AS {payload.sql}""")',
    tool_name="materialize_query",
)
```

## TDD slices

~14 cycles:

1. Tool exists, returns `ok: True` on a trivial `SELECT 1`.
2. Rejects `INSERT` / `UPDATE` / `DROP` with `write_not_allowed`.
3. Rejects `DESCRIBE` / `SHOW` / `PRAGMA` with `write_not_allowed`
   (narrower allowlist than `query`).
4. Creates a registered dataset with correct row count.
5. Reports columns with name + dtype matching `DESCRIBE` output.
6. `overwrite=False` + existing name → `dataset_name_collision`.
7. `overwrite=True` replaces an existing derived dataset.
8. Invalid name (special chars) → `invalid_name`.
9. Bad SQL (missing referenced table) → `query_error` with DuckDB
   message in `details`.
10. Recorder cell written on success only (not on error).
11. Setup cell emits derived `CREATE OR REPLACE TABLE` line **after**
    base tables.
12. Setup cell emits derived lines in dependency order across multiple
    derived datasets.
13. Notebook round-trip: `materialize_query` → `emit_notebook` →
    `jupyter nbconvert --execute` exits 0.
14. `session.reset()` clears derived entries alongside loaded ones
    (characterization test — should already be true).

## Acceptance criteria

- All 14 TDD cycles green with `red:` / `green:` / `refactor:` commits.
- `tests/test_materialize.py` passes under `uv run pytest -q`.
- `evals/eval_materialize.py` passes: load → materialize join → query
  the result → `emit_notebook` → `jupyter nbconvert --execute` exits 0.
- `evals/eval_full_workflow.py` still passes (existing notebook
  round-trip not regressed).
- `ruff check .`, `ruff format --check .`, `pyright src/`, and
  `scripts/check_tdd_commits.py` all green.

## ROADMAP impact

- Removes the `materialize_query` line from `ROADMAP.md` "Tooling".
- Adds `materialize_query` to the SPEC §5 tool list and bumps the tool
  count from 16 → 17 (further bumps land with the other tools in this
  bundle).
- Documents the new `DatasetEntry.format == "derived"` value in SPEC §4
  (session registry).
- Notes the second-pass derived-emission rule in SPEC §6 (recorder
  rehydration).
