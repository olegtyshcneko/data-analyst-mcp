# Fit-then-overwrite Replay Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A model whose training dataset is later overwritten by `materialize_query` replays correctly: the setup cell's hash guard targets the original source file (not the `"(query)"` placeholder, which crashes), and the re-fit trains on a dedicated frame loaded from that file (not the post-transform table).

**Architecture:** All source changes live in `src/data_analyst_mcp/recorder.py`'s `_build_setup_source` model loop, plus a small behavior-preserving helper extraction (`_file_select_expr`) shared with `_file_load_stmt`. Detection uses registry state only: the current entry is `derived` with a carried `base_loader` and the model's fit-time hash differs from the derived entry's own sentinel hash.

**Tech Stack:** Python 3.13, DuckDB, statsmodels, nbformat; pytest + `jupyter nbconvert --execute` for integration.

**Spec:** `docs/superpowers/specs/2026-07-05-fit-then-overwrite-replay-design.md` — the spec wins on any disagreement with this plan.

## Global Constraints

- TDD commit discipline (enforced by `scripts/check_tdd_commits.py`): every behavior change is a `red: <behavior>` commit (failing test only) immediately followed by `green: <behavior>` with a byte-identical message suffix. `refactor:` / `test:` / `docs:` / `chore:` commits are exempt. NEVER amend commits.
- Run `uv run ruff format <edited files>` BEFORE every commit.
- No tool-surface changes: the tool count stays at 22 and tool-response schemas are untouched.
- The only source file modified is `src/data_analyst_mcp/recorder.py`; tests go in `tests/test_recorder.py`; docs changes are Task 5.
- `<dataset>_df` frames stay post-transform (predict/evaluate scoring cells reference them); the pre-overwrite training frame is a *new* variable `<model_name>_train_df`.
- Gates that must pass at the end: `uv run pytest tests/`, `uv run pytest evals/`, `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright src/`, `uv run python scripts/check_tdd_commits.py`.

---

### Task 1: Extract `_file_select_expr` from `_file_load_stmt`

Behavior-preserving refactor. Later tasks emit `SELECT * FROM <reader>(...)` without the `CREATE OR REPLACE TABLE` wrapper; extracting the select expression keeps the reader/`read_options` rendering in exactly one place.

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py:46-63`

**Interfaces:**
- Produces: `_file_select_expr(fmt: str, path: str, read_options: dict[str, Any] | None = None) -> str` returning e.g. `SELECT * FROM read_csv_auto('/abs/x.csv', SAMPLE_SIZE=-1)`. Task 2 consumes it.

- [ ] **Step 1: Replace `_file_load_stmt` with the extracted pair**

Replace the existing `_file_load_stmt` function (recorder.py lines 46-63) with:

```python
def _file_select_expr(fmt: str, path: str, read_options: dict[str, Any] | None = None) -> str:
    """``SELECT * FROM <reader>(...)`` expression for a file-backed source.

    ``repr()`` quotes the path safely — embedded ``'`` / ``"`` / ``\"\"\"`` no
    longer break out of the host literal. ``read_options`` is rendered via the
    same fragment builder the live load used, so replay parses identically.
    """
    reader = _FORMAT_TO_READER.get(fmt, "read_csv_auto")
    path_lit = repr(path)
    extra = render_read_options_fragment(read_options or {})
    if reader == "read_csv_auto":
        return f"SELECT * FROM {reader}({path_lit}, SAMPLE_SIZE=-1{extra})"
    return f"SELECT * FROM {reader}({path_lit}{extra})"


def _file_load_stmt(
    name: str, fmt: str, path: str, read_options: dict[str, Any] | None = None
) -> str:
    """Build the ``CREATE OR REPLACE TABLE`` line that reloads a file-backed
    dataset from disk via the format-appropriate DuckDB reader.
    """
    return f"CREATE OR REPLACE TABLE {name} AS {_file_select_expr(fmt, path, read_options)}"
```

- [ ] **Step 2: Verify output is byte-identical via the existing suite**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: all tests PASS (existing tests pin `_file_load_stmt` output, including read-options rendering and quote-containing paths).

- [ ] **Step 3: Format and commit**

```bash
uv run ruff format src/data_analyst_mcp/recorder.py
git add src/data_analyst_mcp/recorder.py
git commit -m "refactor: extract _file_select_expr from _file_load_stmt"
```

---

### Task 2: Guard and re-fit overwritten training datasets from the carried base loader

The core fix. Detection + guard-path swap + `<model>_train_df` emission + re-fit target change.

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (the model loop inside `_build_setup_source`, currently lines ~204-254)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `_file_select_expr` from Task 1; `DatasetEntry.base_loader` (`{"path", "format", "read_options", "source_hash"}`) carried by `materialize_query`; `ModelEntry.training_dataset_hash`.
- Produces: setup-cell emission shape that Tasks 3-4 pin — comment line containing `was overwritten by materialize_query`, guard against the base path, `<model_name>_train_df = con.sql(<select>!r).df()`, re-fit `data=<model_name>_train_df`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recorder.py`:

```python
def test_setup_cell_refits_overwritten_training_dataset_from_base_loader(
    call_tool, tmp_path
) -> None:
    """A model whose training dataset is later overwritten by
    materialize_query must guard against the original file (not the
    '(query)' placeholder, which crashes the hash recompute) and re-fit on
    a dedicated train frame loaded from that file — not on the
    post-transform table."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    # Guard recomputes against the original file, not the placeholder.
    assert f"expected_hash_m = '{expected}'" in setup_src
    assert "open('(query)'" not in setup_src
    # Re-fit uses a dedicated frame loaded from the original file...
    assert "m_train_df = con.sql(" in setup_src
    assert "data=m_train_df" in setup_src
    assert "was overwritten by materialize_query" in setup_src
    # ...while train_df stays the (post-transform) current table for
    # predict/evaluate scoring cells.
    assert 'train_df = con.sql("SELECT * FROM train").df()' in setup_src
    compile(setup_src, "<setup>", "exec")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_recorder.py::test_setup_cell_refits_overwritten_training_dataset_from_base_loader -v`
Expected: FAIL — `open('(query)'` IS in the setup source today (assertion `"open('(query)'" not in setup_src` fails, or the earlier `expected_hash_m` assertion fails because the guard shape differs).

- [ ] **Step 3: Commit the red**

```bash
uv run ruff format tests/test_recorder.py
git add tests/test_recorder.py
git commit -m "red: overwritten training dataset guards and re-fits from base_loader"
```

- [ ] **Step 4: Implement**

In `_build_setup_source`, replace the model-loop body. The current code begins:

```python
        for model_name, model_entry in models.items():
            lines.append("")
            lines.append(f"# --- Re-fit model {model_name!r} (kind={model_entry.kind}) ---")
            ds_path = (
                _session.get_datasets()[model_entry.fitted_on_dataset].path
                if model_entry.fitted_on_dataset in _session.get_datasets()
                else None
            )
            hash_val = model_entry.training_dataset_hash
```

Replace that opening (through the `hash_val` line) with:

```python
        for model_name, model_entry in models.items():
            lines.append("")
            lines.append(f"# --- Re-fit model {model_name!r} (kind={model_entry.kind}) ---")
            ds_entry = _session.get_datasets().get(model_entry.fitted_on_dataset)
            hash_val = model_entry.training_dataset_hash
            # A model whose training dataset was later overwritten by
            # materialize_query must not guard or re-fit against the current
            # entry: its path is the "(query)" placeholder (the hash recompute
            # would crash at replay) and its table is post-transform (the
            # re-fit would train on the wrong data). The carried base_loader
            # is the truthful source. A model fit *after* the overwrite copied
            # the derived entry's own sentinel hash at fit time, so the hash
            # inequality keeps it on the normal path.
            overwritten_base: dict[str, Any] | None = None
            if (
                ds_entry is not None
                and ds_entry.format == "derived"
                and ds_entry.base_loader is not None
                and hash_val != ds_entry.source_hash
            ):
                overwritten_base = ds_entry.base_loader
            if overwritten_base is not None:
                ds_path = overwritten_base["path"]
                data_ref = f"{model_name}_train_df"
                lines.append(
                    f"# Dataset {model_entry.fitted_on_dataset!r} was overwritten by "
                    f"materialize_query after this model was fit; re-fitting from "
                    f"the original source file, not the current derived table."
                )
            else:
                ds_path = ds_entry.path if ds_entry is not None else None
                data_ref = f"{model_entry.fitted_on_dataset}_df"
```

The three guard branches that follow (`if ds_path is not None and not hash_val.startswith(...)` / `elif ... fallback ...` / `else` comment) stay **unchanged** — they already key off `ds_path` and `hash_val`.

Then, between the guard branches and the `smf_fn = ...` line, insert:

```python
            if overwritten_base is not None:
                select = _file_select_expr(
                    overwritten_base["format"],
                    overwritten_base["path"],
                    overwritten_base.get("read_options"),
                )
                lines.append(f"{model_name}_train_df = con.sql({select!r}).df()")
```

Finally, change the re-fit line from `data={model_entry.fitted_on_dataset}_df` to `data={data_ref}`:

```python
            lines.append(
                f'{model_name} = smf.{smf_fn}("{model_entry.formula}", '
                f"data={data_ref}).fit({fit_args})"
            )
```

Also update `_build_setup_source`'s docstring: after the numbered list item 2 (the "For each model..." paragraph), add one sentence: `If the model's training dataset was later overwritten by ``materialize_query``, the guard and the re-fit both target the carried base loader (original file) instead of the post-transform table.`

- [ ] **Step 5: Run the test and the full recorder suite**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: ALL PASS (new test plus no regressions — existing model/guard/overwrite tests must stay green).

- [ ] **Step 6: Commit the green**

```bash
uv run ruff format src/data_analyst_mcp/recorder.py
git add src/data_analyst_mcp/recorder.py
git commit -m "green: overwritten training dataset guards and re-fits from base_loader"
```

---

### Task 3: Pin the fallback, sentinel, and post-overwrite-fit shapes

These behaviors land as a side effect of Task 2's shared guard branches, so the tests pass immediately — they are coverage pins, committed with the exempt `test:` prefix.

**Files:**
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: Task 2's emission shape; `provenance.HASH_CONTENT_CEILING_BYTES` monkeypatch pattern (see existing `test_setup_cell_emits_fallback_guard_above_content_ceiling`).

- [ ] **Step 1: Add the three tests**

Append to `tests/test_recorder.py`:

```python
def test_setup_cell_fallback_guard_targets_base_path_after_overwrite(
    call_tool, tmp_path, monkeypatch
) -> None:
    """Above-ceiling training file + overwrite: the fallback stat/recompute
    must target the original path, not the '(query)' placeholder."""
    from data_analyst_mcp import provenance
    from data_analyst_mcp.recorder import get_recorder

    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 4)
    csv = tmp_path / "big.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "expected_hash_m = 'fallback:" in setup_src
    assert f"_st = _os.stat({str(csv)!r})" in setup_src
    assert "_os.stat('(query)')" not in setup_src
    assert "data=m_train_df" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_sentinel_base_after_overwrite_refits_unguarded(tmp_path) -> None:
    """An s3-backed training dataset overwritten after fit: no hash assert
    is possible (sentinel), but the re-fit still loads from the original
    s3 loader rather than the post-transform table."""
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder
    from data_analyst_mcp.tools import models as _models

    session.reset()
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    session.register(
        name="train",
        path="s3://bucket/train.csv",
        read_options={},
        format="csv",
        rows=5,
        columns=[{"name": "y", "dtype": "DOUBLE"}, {"name": "x", "dtype": "DOUBLE"}],
    )
    con = session.get_connection()
    con.register("__train_df", df)
    con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM __train_df")
    con.unregister("__train_df")
    assert session.get_datasets()["train"].source_hash.startswith("sentinel:")

    _models.fit_model(
        _models.FitModelInput(
            name="train", formula="y ~ x", kind="ols", robust=False, model_name="m"
        )
    )
    # Overwrite via a direct derived registration mirroring materialize_query
    # (an s3 read through the tool would need network; the registry state is
    # what the recorder consumes).
    entry = session.get_datasets()["train"]
    session.register(
        name="train",
        path="(query)",
        read_options={"sql": "SELECT y * 10 AS y, x FROM train"},
        format="derived",
        rows=5,
        columns=entry.columns,
        base_loader={
            "path": entry.path,
            "format": entry.format,
            "read_options": dict(entry.read_options),
            "source_hash": entry.source_hash,
        },
    )
    con.execute("CREATE OR REPLACE TABLE train AS SELECT y * 10 AS y, x FROM train")

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert "assert actual_hash_m" not in setup_src
    assert "m_train_df = con.sql(" in setup_src
    assert "s3://bucket/train.csv" in setup_src
    assert "data=m_train_df" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_model_fit_after_overwrite_keeps_refitting_current_table(
    call_tool, tmp_path
) -> None:
    """A model fit on the already-derived table re-fits on <dataset>_df —
    the post-transform table is exactly what it trained on. No train frame,
    no hash assert (the derived entry's hash is a sentinel)."""
    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")
    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "m_train_df" not in setup_src
    assert "data=train_df" in setup_src
    assert "assert actual_hash_m" not in setup_src
    assert "non-file dataset" in setup_src
    compile(setup_src, "<setup>", "exec")
```

- [ ] **Step 2: Run the three tests**

Run: `uv run pytest tests/test_recorder.py -k "fallback_guard_targets_base_path or sentinel_base_after_overwrite or fit_after_overwrite_keeps" -v`
Expected: 3 PASS. If any fails, STOP — Task 2's implementation has a shape bug; report it rather than adjusting the test to match.

- [ ] **Step 3: Commit**

```bash
uv run ruff format tests/test_recorder.py
git add tests/test_recorder.py
git commit -m "test: pin fallback/sentinel/post-overwrite-fit model guard shapes"
```

---

### Task 4: nbconvert integration — clean replay and drift direction

End-to-end proof in both directions, mirroring the existing subprocess pattern (`test_emitted_notebook_replays_overwrite_of_file_backed_dataset`). Both pass only with Task 2 in place, but they exercise behavior already red/green-committed there — `test:` prefix.

**Files:**
- Test: `tests/test_recorder.py`

- [ ] **Step 1: Add the two tests**

Append to `tests/test_recorder.py`:

```python
def test_emitted_notebook_replays_fit_then_overwrite_end_to_end(
    tmp_path, call_tool
) -> None:
    """fit_model → materialize_query overwrite → predict → emit → nbconvert
    executes cleanly: the model re-fits from the original file while the
    scoring cell sees the post-transform table, matching the live session."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r
    r = call_tool("predict", {"model_name": "m", "dataset": "train", "limit": 5})
    assert r["ok"], r

    nb_path = tmp_path / "fit_then_overwrite.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


def test_emitted_notebook_fit_then_overwrite_guard_fires_on_mutated_source(
    tmp_path, call_tool
) -> None:
    """Same session as the clean test, but the source CSV is edited between
    emit and replay: the setup cell must die with a loud AssertionError
    instead of silently re-fitting on different data."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r

    nb_path = tmp_path / "fit_then_overwrite_drift.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    csv.write_text("y,x\n9.0,0.0\n8.0,1.0\n7.0,2.0\n6.0,3.0\n5.0,4.0\n4.0,5.0\n")

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0, "replay should fail after source mutation"
    assert "AssertionError" in result.stderr
    assert "changed since the session was recorded" in result.stderr
```

(The drift test asserts the shared message suffix, not the model-specific one: the dataset-level base_loader guard sits earlier in the setup cell and legitimately fires first.)

- [ ] **Step 2: Run the two tests**

Run: `uv run pytest tests/test_recorder.py -k "fit_then_overwrite" -v`
Expected: 2 PASS (each spawns a jupyter kernel; allow ~30-60 s total). If the clean-replay test fails, STOP and report — that is a Task 2 defect.

- [ ] **Step 3: Commit**

```bash
uv run ruff format tests/test_recorder.py
git add tests/test_recorder.py
git commit -m "test: fit-then-overwrite nbconvert replay in both drift directions"
```

---

### Task 5: Documentation

**Files:**
- Modify: `ROADMAP.md:27` (delete one bullet)
- Modify: `README.md:427` (extend one bullet)
- Modify: `docs/SPEC.md:513` (extend one bullet)
- Modify: `CHANGELOG.md` (new Unreleased section)

- [ ] **Step 1: ROADMAP — delete the tracked-gap bullet**

Delete this entire bullet from the Reproducibility section of `ROADMAP.md` (it is now fixed):

```markdown
- **Fit-then-overwrite replay.** A model fitted on a dataset that is later overwritten by `materialize_query` emits a model block whose hash recompute targets `(query)` (crashes at replay even with zero drift) and whose re-fit would use the post-transform table. Fix: recompute against the carried `base_loader` path/hash, and decide the re-fit semantics (skip with a loud comment vs. materialize the base frame). Pre-existing gap, surfaced by the provenance-hashes review.
```

- [ ] **Step 2: README — extend the drift-guard gotcha**

In `README.md`, the "Known gotchas" bullet starting `**Emitted notebooks are drift-guarded.**` — after the sentence ending `s3/http sources reload unguarded (a comment in the cell says so).`, append this sentence to the same bullet:

```markdown
Models registered before a `materialize_query` overwrite of their training dataset re-fit from the original file (guarded by the fit-time hash), not the post-transform table.
```

- [ ] **Step 3: SPEC — extend the §5.11 provenance bullet**

In `docs/SPEC.md`, the §5.11 bullet starting `- When stored, the model's ` + "`training_dataset_hash`" + ` is copied from the dataset entry's load-time ` — append this sentence to the same bullet:

```markdown
If the training dataset is later overwritten by `materialize_query`, the emitted setup cell guards against and re-fits from the carried base loader (the original file) rather than the post-transform table; the re-fit uses a dedicated `<model_name>_train_df` frame so `<dataset>_df` stays current for scoring cells.
```

- [ ] **Step 4: CHANGELOG — add an Unreleased section**

In `CHANGELOG.md`, directly above the `## [1.2.0] - 2026-07-05` line, insert:

```markdown
## [Unreleased]

### Fixed
- A model fitted on a dataset that was later overwritten by
  `materialize_query` no longer crashes notebook replay: the setup cell's
  hash guard now targets the original source file carried in `base_loader`
  (not the `"(query)"` placeholder), and the model re-fits on a dedicated
  `<model_name>_train_df` frame loaded from that file instead of the
  post-transform table. Scoring cells keep seeing the current table.

```

- [ ] **Step 5: Commit**

```bash
git add ROADMAP.md README.md docs/SPEC.md CHANGELOG.md
git commit -m "docs: fit-then-overwrite replay fix — ROADMAP/README/SPEC/CHANGELOG"
```

---

### Task 6: Final gates

- [ ] **Step 1: Run all six gates**

```bash
uv run pytest tests/
uv run pytest evals/
uv run ruff format --check .
uv run ruff check .
uv run pyright src/
uv run python scripts/check_tdd_commits.py
```

Expected: tests all pass (469 unit — 463 + 6 new; 51 evals), both ruff gates clean, pyright `0 errors`, TDD audit OK (this plan adds exactly 1 red/green pair, 1 refactor, 2 test commits, 1 docs commit). The 33 pytest warnings in `tests/` are pre-existing scipy/statsmodels internals — not a failure.

- [ ] **Step 2: Fix anything that fails and re-run until green**

Formatting drift → `uv run ruff format .` and commit as `chore: ruff-format`. Any test failure is a real defect: report it, do not paper over it.
