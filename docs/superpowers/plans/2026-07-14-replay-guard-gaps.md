# Replay-Guard Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every replacement-surface replay gap (spec S4–S17): registration revisions + fit-time loader identity make silently-wrong model re-fits impossible; recorded split-overwrite provenance and symmetric per-side split replay make every remaining failure loud *and explained*.

**Architecture:** A monotonic per-session revision stamped by `session.register()` becomes the identity anchor for "is the current entry the same table state the model was fit on" — replacing the recorder's hash-inequality heuristics. `materialize_query` records split-overwrite provenance on the derived entry itself (no sibling inference). `split_dataset` checksums both sides; `split_replay_source` recreates and asserts each surviving side independently; the setup cell's second pass emits in revision order.

**Tech Stack:** Python 3.13, uv, FastMCP, DuckDB, pandas/numpy, statsmodels, nbformat + `jupyter nbconvert` for replay tests, pytest.

**Spec:** `docs/superpowers/specs/2026-07-14-replay-guard-gaps-design.md` (v3). Scenario IDs (S1–S17) below refer to its behavior matrix.

## Global Constraints

- **TDD discipline is enforced by `scripts/check_tdd_commits.py`:** every commit whose subject starts with `green:` must be *immediately preceded* by a `red:` commit with the **identical suffix text**. `refactor:`/`docs:`/`chore:` commits may appear anywhere. Never squash a red into its green.
- **Red commits must actually fail:** run the new test(s), confirm the failure mode stated in the step, then commit the test file(s) alone.
- **Gates (run all five before the final task is called done):**
  - `uv run pytest tests/` (currently 528 tests)
  - `uv run pytest evals/` (slow, ~30s+, spawns stdio subprocesses)
  - `uv run ruff format --check .` and `uv run ruff check .`
  - `uv run pyright src/` (strict)
  - `uv run python scripts/check_tdd_commits.py`
- **Pinned message contracts (copy verbatim, tests depend on these substrings):**
  - Split-overwrite wrap message MUST begin: `Dataset {name!r} was created by overwriting the {side} side of the split of {source!r}.` (pins `overwriting the train side` / `overwriting the test side`, `tests/test_split.py:561,591`).
  - The file-backed mismatch raise MUST contain: `Training data for {model_name!r} changed since the session was recorded.` (pins `tests/test_recorder.py:805`).
  - Split checksum assert messages MUST contain `drifted at replay` (pinned in `tests/test_split.py`).
  - The per-side checksum key stays `membership_checksum` in `read_options` for BOTH entries (renaming breaks `tests/test_split.py:75,375`).
- **Version:** 1.4.0 (final task). No API/tool-surface changes — all 24 tools unchanged; every change is registry-metadata + emit/replay-side.
- All emitted-notebook code lines are built with `{...!r}` interpolation so quotes in names/SQL can't break the generated Python — follow that idiom for every new emitted line.
- Deliberate conservatism is spec'd: any replacement bumps the revision, so byte-identical-recipe re-runs after replacement fail the pre-replacement model's replay loudly. Do NOT add "same recipe" escape hatches.

## File Structure

| File | Role in this plan |
|---|---|
| `src/data_analyst_mcp/session.py` | `DatasetEntry.revision` + counter, `DatasetEntry.split_overwrite`, `ModelEntry.training_dataset_revision`/`training_loader`, `register()`/`register_model()`/`reset()` |
| `src/data_analyst_mcp/tools/materialize.py` | base_loader records file entry's revision; records split-overwrite provenance |
| `src/data_analyst_mcp/tools/models.py` | fit-time revision + loader capture |
| `src/data_analyst_mcp/tools/split.py` | per-side membership checksums; per-call cell passes both |
| `src/data_analyst_mcp/recorder.py` | model-block rewrite (Part 2), provenance-keyed wrap (Part 3), symmetric `split_replay_source` + keying contract + revision-ordered second pass (Part 4) |
| `tests/test_session.py`, `tests/test_materialize.py`, `tests/test_model_registry.py`, `tests/test_recorder.py`, `tests/test_split.py` | new tests per task |
| `ROADMAP.md`, `CHANGELOG.md`, `README.md`, `docs/SPEC.md`, `evals/README.md`, `pyproject.toml`, `src/data_analyst_mcp/__init__.py`, `tests/test_smoke.py`, `uv.lock` | final docs/release task |

Fixtures available to tests: `call_tool` (invokes tools through FastMCP), `load_df_into_session(name, df)` (registers an in-memory DataFrame as `format="dataframe"`), autouse session+recorder reset (`tests/conftest.py`).

---

### Task 1: Registration revisions in `session`

**Files:**
- Modify: `src/data_analyst_mcp/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: `DatasetEntry.revision: int` (default `-1` for direct constructions; every `session.register()` call stamps a unique value from a monotonic counter starting at 0). `session.reset()` zeroes the counter. Later tasks compare `ModelEntry.training_dataset_revision` and `base_loader["revision"]` against this field.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_session.py`:

```python
def test_register_stamps_monotonic_revisions() -> None:
    from data_analyst_mcp import session

    session.reset()
    session.register(
        name="a", path="(dataframe)", read_options={}, format="dataframe", rows=1, columns=[]
    )
    session.register(
        name="b", path="(dataframe)", read_options={}, format="dataframe", rows=1, columns=[]
    )

    assert session.get_datasets()["a"].revision == 0
    assert session.get_datasets()["b"].revision == 1


def test_reregistering_same_name_gets_a_fresh_revision() -> None:
    """Replacement identity: overwriting a name must be distinguishable from
    the original registration even when every other field matches."""
    from data_analyst_mcp import session

    session.reset()
    session.register(
        name="a", path="(query)", read_options={"sql": "SELECT 1"}, format="derived", rows=1, columns=[]
    )
    session.register(
        name="b", path="(query)", read_options={"sql": "SELECT 2"}, format="derived", rows=1, columns=[]
    )
    session.register(
        name="a", path="(query)", read_options={"sql": "SELECT 1"}, format="derived", rows=1, columns=[]
    )

    assert session.get_datasets()["a"].revision == 2
    assert session.get_datasets()["b"].revision == 1


def test_reset_restarts_the_revision_counter() -> None:
    from data_analyst_mcp import session

    session.reset()
    session.register(
        name="a", path="(dataframe)", read_options={}, format="dataframe", rows=1, columns=[]
    )
    assert session.get_datasets()["a"].revision == 0

    session.reset()
    session.register(
        name="z", path="(dataframe)", read_options={}, format="dataframe", rows=1, columns=[]
    )
    assert session.get_datasets()["z"].revision == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_session.py -v -k revision`
Expected: 3 FAIL with `AttributeError: 'DatasetEntry' object has no attribute 'revision'`

- [ ] **Step 3: Commit red**

```bash
git add tests/test_session.py
git commit -m "red: every register() stamps a fresh session revision"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/session.py`, add the field to `DatasetEntry` (between `source_hash` and `registered_at`):

```python
    # Content hash of the source file at registration time (the recorder's
    # drift-guard anchor). ``sentinel:``-prefixed when there is no
    # verifiable file. Default covers direct constructions in tests.
    source_hash: str = "sentinel:unset"
    # Monotonic per-session registration revision stamped by register().
    # Identity of the *registration*, not the content: replacement through
    # any tool (materialize_query, load_dataset, split_dataset) gets a fresh
    # value even when source_hash stays constant (per-format sentinels,
    # byte-identical reloads). Default covers direct constructions in tests;
    # register() always stamps >= 0.
    revision: int = -1
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

Add the counter next to the registries (after `_connection`):

```python
_datasets: dict[str, DatasetEntry] = {}
_models: dict[str, ModelEntry] = {}
_connection: duckdb.DuckDBPyConnection | None = None
_revision_counter = 0
```

In `register()`, stamp and bump:

```python
def register(
    *,
    name: str,
    path: str,
    read_options: dict[str, Any],
    format: str,
    rows: int,
    columns: list[dict[str, str]],
    base_loader: dict[str, Any] | None = None,
) -> None:
    """Insert (or replace) a dataset entry under ``name``."""
    global _revision_counter
    _datasets[name] = DatasetEntry(
        path=path,
        read_options=dict(read_options),
        format=format,
        rows=rows,
        columns=list(columns),
        base_loader=dict(base_loader) if base_loader is not None else None,
        source_hash=compute_source_hash(path),
        revision=_revision_counter,
    )
    _revision_counter += 1
```

In `reset()`, zero the counter (at the end of the function body):

```python
    global _revision_counter
    _datasets.clear()
    _models.clear()
    _revision_counter = 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_session.py -v`
Expected: all PASS (including the pre-existing ones)

- [ ] **Step 6: Full unit suite + commit green**

Run: `uv run pytest tests/ -q` — expected: all pass (the field is additive with a default).

```bash
git add src/data_analyst_mcp/session.py
git commit -m "green: every register() stamps a fresh session revision"
```

---

### Task 2: `base_loader` pins the original file entry's revision

**Files:**
- Modify: `src/data_analyst_mcp/tools/materialize.py:114-125`
- Test: `tests/test_materialize.py`

**Interfaces:**
- Consumes: `DatasetEntry.revision` (Task 1).
- Produces: `base_loader["revision"]: int` — always the ORIGINAL file entry's revision (R0), carried unchanged across chained derived overwrites (`file@R0 → derived@R1 → derived@R2` keeps `base_loader["revision"] == R0`). Task 4's recorder branch compares the model's fit-time revision against it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_materialize.py`:

```python
def test_overwrite_base_loader_records_original_file_revision(call_tool, tmp_path) -> None:
    """base_loader must pin the replaced FILE entry's revision (R0) and keep
    it unchanged across chained derived overwrites — the model guard uses it
    to recognize a fit on the pre-overwrite file-backed state."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    csv = tmp_path / "base.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    r0 = _session.get_datasets()["data"].revision

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    base = _session.get_datasets()["data"].base_loader
    assert base is not None
    assert base["revision"] == r0

    # Second chained overwrite: the carried dict still says R0, never R1.
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    base = _session.get_datasets()["data"].base_loader
    assert base is not None
    assert base["revision"] == r0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_materialize.py::test_overwrite_base_loader_records_original_file_revision -v`
Expected: FAIL with `KeyError: 'revision'`

- [ ] **Step 3: Commit red**

```bash
git add tests/test_materialize.py
git commit -m "red: base_loader pins the original file entry's revision across overwrite chains"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/tools/materialize.py`, edit the base_loader capture:

```python
        if existing.format not in ("derived", "dataframe", "split"):
            base_loader = {
                "path": existing.path,
                "format": existing.format,
                "read_options": dict(existing.read_options),
                "source_hash": existing.source_hash,
                "revision": existing.revision,
            }
        elif existing.format == "derived":
            base_loader = existing.base_loader
```

(The `elif` carry-forward branch is already correct — carrying the dict unchanged is exactly what keeps R0.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_materialize.py -v`
Expected: all PASS

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/tools/materialize.py
git commit -m "green: base_loader pins the original file entry's revision across overwrite chains"
```

---

### Task 3: fit-time revision + loader identity on `ModelEntry`

**Files:**
- Modify: `src/data_analyst_mcp/session.py` (ModelEntry, register_model), `src/data_analyst_mcp/tools/models.py:152-160`
- Test: `tests/test_model_registry.py`

**Interfaces:**
- Consumes: `DatasetEntry.revision` (Task 1).
- Produces: `ModelEntry.training_dataset_revision: int = -1` and `ModelEntry.training_loader: dict[str, Any] | None = None` (fit-time `{"path", "format", "read_options"}`). `session.register_model()` gains both as optional keyword params (defaults keep existing direct-construction tests green). `fit_model` always populates both. Tasks 4–6 read them in the recorder.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_registry.py`:

```python
def test_fit_model_stamps_training_revision_and_loader(call_tool, tmp_path) -> None:
    """fit_model must capture the training dataset's registration revision
    and its fit-time loader identity {path, format, read_options} — the
    recorder's replacement and same-semantics-reload guards read both."""
    from data_analyst_mcp import session

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    entry = session.get_datasets()["train"]
    model = session.get_model("m")
    assert model is not None
    assert model.training_dataset_revision == entry.revision
    assert model.training_loader == {
        "path": str(csv),
        "format": "csv",
        "read_options": {},
    }


def test_register_model_defaults_for_direct_constructions() -> None:
    """Direct register_model calls (tests, hypothetical bypass paths) get
    sentinel defaults: revision -1 (matches no real registration) and no
    loader."""
    from data_analyst_mcp import session

    session.reset()
    session.register_model(
        name="m1",
        kind="ols",
        formula="y ~ x",
        fitted_on_dataset="ds",
        n_obs=10,
        training_dataset_hash="h",
        result=object(),
    )

    entry = session.get_model("m1")
    assert entry is not None
    assert entry.training_dataset_revision == -1
    assert entry.training_loader is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_registry.py -v -k "training_revision or direct_constructions"`
Expected: 2 FAIL with `AttributeError: 'ModelEntry' object has no attribute 'training_dataset_revision'`

- [ ] **Step 3: Commit red**

```bash
git add tests/test_model_registry.py
git commit -m "red: fit_model captures training revision and loader identity"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/session.py`, extend `ModelEntry` (append after `_result`):

```python
    training_dataset_hash: str
    _result: Any  # statsmodels Results object, in-process only
    # Fit-time registration revision of the training dataset — the model
    # guard's identity anchor: the recorder trusts the current entry only
    # when this matches entry.revision (or base_loader["revision"] for a
    # fit on the pre-overwrite file-backed state). -1 for direct
    # constructions in tests.
    training_dataset_revision: int = -1
    # Fit-time loader identity {"path", "format", "read_options"} — proves a
    # later same-name reload has the same loading semantics, which a content
    # hash alone cannot (identical bytes re-parse differently under changed
    # read_options).
    training_loader: dict[str, Any] | None = None
```

Extend `register_model`:

```python
def register_model(
    *,
    name: str,
    kind: str,
    formula: str,
    fitted_on_dataset: str,
    n_obs: int,
    training_dataset_hash: str,
    result: Any,
    training_dataset_revision: int = -1,
    training_loader: dict[str, Any] | None = None,
) -> None:
```

and in the `ModelEntry(...)` construction add:

```python
        training_dataset_hash=training_dataset_hash,
        _result=result,
        training_dataset_revision=training_dataset_revision,
        training_loader=dict(training_loader) if training_loader is not None else None,
    )
```

In `src/data_analyst_mcp/tools/models.py` (the `session.register_model(` call inside `fit_model`):

```python
        session.register_model(
            name=payload.model_name,
            kind=payload.kind,
            formula=payload.formula,
            fitted_on_dataset=payload.name,
            n_obs=n_obs_val,
            training_dataset_hash=ds_entry.source_hash,
            training_dataset_revision=ds_entry.revision,
            training_loader={
                "path": ds_entry.path,
                "format": ds_entry.format,
                "read_options": dict(ds_entry.read_options),
            },
            result=live_result,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_registry.py tests/test_session.py -v`
Expected: all PASS

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/session.py src/data_analyst_mcp/tools/models.py
git commit -m "green: fit_model captures training revision and loader identity"
```

---

### Task 4: recorder derived branch keyed on revisions (S4, S5, S10; S2 pin)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py:449-487` (model block dispatch), `tests/test_recorder.py:884` area (deliberate test-fixture update)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `ModelEntry.training_dataset_revision` (Task 3), `base_loader["revision"]` (Task 2).
- Produces: the derived-format dispatch of the model re-fit block: `rev == entry.revision` → normal re-fit; `base_loader is not None and rev == base_loader.get("revision")` → guarded base-file re-fit; otherwise → emitted `raise AssertionError` whose message contains `was later replaced` and `cannot be replayed faithfully`. Tasks 5–6 extend this same `if rev != ds_entry.revision:` dispatch to the other formats — keep the structure shown below.

- [ ] **Step 1: Update the manual base_loader fixture in the pinned s3 test**

In `tests/test_recorder.py`, inside `test_setup_cell_sentinel_base_after_overwrite_refits_unguarded` (~line 918), the manually-built `base_loader` dict mirrors what `materialize_query` records — it must now include the replaced entry's revision (the recorder ignores the key until this task's green, so the test stays green at red time):

```python
        base_loader={
            "path": entry.path,
            "format": entry.format,
            "read_options": dict(entry.read_options),
            "source_hash": entry.source_hash,
            "revision": entry.revision,
        },
```

Also: that test registers the model via `_models.fit_model(...)` BEFORE the derived re-registration, so the model's fit-time revision equals `entry.revision` — the new base-branch condition matches and the pinned unguarded-s3-re-fit behavior survives. No other change to that test.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_recorder.py`:

```python
def test_setup_cell_raises_for_model_fit_on_pure_query_overwritten_dataset(
    call_tool, tmp_path
) -> None:
    """S4 (ROADMAP gap): a model fit on a pure-query derived dataset that is
    later overwritten shares the constant '(query)' sentinel hash with the
    replacement, so hash comparison cannot see the overwrite. The revision
    can: the setup cell must emit a loud raise, never a silent re-fit on the
    post-transform table."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT y, x FROM base", "name": "d"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM base", "name": "d", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "'m'" in setup_src
    assert "'d'" in setup_src
    assert "was later replaced" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_emitted_notebook_pure_query_fit_overwrite_fails_loudly_end_to_end(
    tmp_path, call_tool
) -> None:
    """S4 end-to-end: emit the fit-then-overwrite pure-query session and run
    it via nbconvert — replay must fail with the explanatory AssertionError
    (today it silently succeeds on the wrong table)."""
    import os
    import subprocess

    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT y, x FROM base", "name": "d"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM base", "name": "d", "overwrite": True},
    )["ok"] is True

    nb_path = tmp_path / "s4.ipynb"
    assert call_tool("emit_notebook", {"path": str(nb_path)})["ok"] is True

    result = subprocess.run(
        [
            "uv", "run", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--inplace", str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0, "replay must not silently re-fit on the replaced table"
    combined = result.stderr + result.stdout
    assert "cannot be replayed faithfully" in combined


def test_setup_cell_raises_for_model_fit_on_base_carrying_derived_then_overwritten(
    call_tool, tmp_path
) -> None:
    """S5: model fit on the middle base-carrying derived state, then another
    overwrite. Neither the current revision nor the carried base revision
    matches the fit — silently re-fitting from the base FILE would be as
    wrong as the post-transform table. Loud raise, no m_train_df frame."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "data"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 2 AS y, x FROM data", "name": "data", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "data", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y + 1 AS y, x FROM data", "name": "data", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "was later replaced" in setup_src
    assert "m_train_df" not in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_raises_for_model_fit_on_middle_materialization(
    call_tool, tmp_path
) -> None:
    """S10: fresh derived name (base_loader stays None throughout),
    materialized twice with a fit in between — the middle state is gone;
    replay must raise instead of silently re-fitting on the latest table."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT y, x FROM base WHERE x < 5", "name": "d"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y, x FROM base WHERE x < 4", "name": "d", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "was later replaced" in setup_src


def test_setup_cell_still_refits_from_base_file_after_chained_overwrites(
    call_tool, tmp_path
) -> None:
    """S2 regression pin: a model fit ON the file-backed state, followed by
    two chained overwrites, still re-fits from the carried base file behind
    the fit-time hash guard — identified by base_loader['revision'] == R0."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    assert call_tool("load_dataset", {"path": str(csv), "name": "train"})["ok"] is True
    assert call_tool(
        "fit_model", {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y + 1 AS y, x FROM train", "name": "train", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_m = '{expected}'" in setup_src
    assert "m_train_df = con.sql(" in setup_src
    assert "data=m_train_df" in setup_src
    assert "raise AssertionError" not in setup_src
    compile(setup_src, "<setup>", "exec")
```

- [ ] **Step 3: Run tests to verify red state**

Run: `uv run pytest tests/test_recorder.py -v -k "pure_query or base_carrying or middle_materialization or chained_overwrites"`
Expected: the S4 unit, S4 end-to-end, S5, and S10 tests FAIL (no raise emitted / nbconvert exits 0); the S2 chained pin PASSES (it pins behavior that already works — committed with the red so the green can prove it survives).

- [ ] **Step 4: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: model replay guards derived replacements by revision, not hash"
```

- [ ] **Step 5: Implement**

In `src/data_analyst_mcp/recorder.py`, inside `_build_setup_source()`'s model loop, replace this block (currently at lines ~453-487 — the comment starting `# A model whose training dataset was later overwritten by` down through the `continue` of the loud-raise branch):

```python
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
            elif (
                ds_entry is not None
                and ds_entry.format == "derived"
                and ds_entry.base_loader is None
                and hash_val != ds_entry.source_hash
            ):
                # Fit-then-overwrite with nothing to re-fit from (e.g. the
                # training dataset was a split output): silently re-fitting
                # on the post-transform table would betray the drift
                # guarantee — fail the replay loudly instead.
                msg = (
                    f"Model {model_name!r} was fit on dataset "
                    f"{model_entry.fitted_on_dataset!r}, which was later "
                    f"overwritten by materialize_query and has no re-loadable "
                    f"source; the re-fit cannot be replayed faithfully."
                )
                lines.append(f"raise AssertionError({msg!r})")
                continue
```

with:

```python
            hash_val = model_entry.training_dataset_hash
            rev = model_entry.training_dataset_revision
            # The fit-time registration REVISION — not the content hash —
            # decides whether the current entry is the same table state the
            # model was fit on: derived/split/dataframe states share constant
            # per-format sentinel hashes, so hash comparison cannot see a
            # replacement. A fit on the pre-overwrite file-backed state is
            # recognized by the carried base_loader's pinned revision (always
            # the original file entry's, unchanged across chained overwrites)
            # and re-fits from the original file behind the fit-time hash
            # guard. Any other revision mismatch on a derived entry means the
            # fit-time table state no longer exists anywhere reachable.
            overwritten_base: dict[str, Any] | None = None
            if (
                ds_entry is not None
                and ds_entry.format == "derived"
                and rev != ds_entry.revision
            ):
                base = ds_entry.base_loader
                if base is not None and rev == base.get("revision"):
                    overwritten_base = base
                else:
                    msg = (
                        f"Model {model_name!r} was fit on dataset "
                        f"{model_entry.fitted_on_dataset!r}, which was later "
                        f"replaced; the table state it was fit on no longer "
                        f"exists anywhere reachable, so the re-fit cannot be "
                        f"replayed faithfully."
                    )
                    lines.append(f"raise AssertionError({msg!r})")
                    continue
```

Leave everything from `if overwritten_base is not None:` onward untouched.

- [ ] **Step 6: Run the full unit suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS. Pay attention to the pre-existing derived-branch tests — they must survive on revision logic alone:
- `test_setup_cell_refits_overwritten_training_dataset_from_base_loader` (fit rev == base revision → base path)
- `test_setup_cell_model_fit_after_overwrite_keeps_refitting_current_table` (fit rev == entry revision → normal path)
- `test_setup_cell_sentinel_base_after_overwrite_refits_unguarded` (updated fixture from Step 1)
- `tests/test_split.py::test_model_fit_on_split_then_overwritten_raises_at_replay` (derived, base None, rev mismatch → raise; asserts only `raise AssertionError` + `m_split`, so the generalized message is fine)
- `test_emitted_notebook_replays_fit_then_overwrite_end_to_end` and `..._guard_fires_on_mutated_source` (nbconvert, base path)

- [ ] **Step 7: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: model replay guards derived replacements by revision, not hash"
```

---

### Task 5: recorder split & dataframe branches (S11; S7a pin; dataframe re-registration)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (the dispatch from Task 4)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: Task 4's dispatch structure.
- Produces: the merged replacement dispatch — for `format in ("derived", "split", "dataframe")` a revision mismatch (that isn't the derived base-file case) emits the same `was later replaced` raise. File-backed formats still fall through to the normal path on mismatch (Task 6 tightens that).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recorder.py`:

```python
def test_setup_cell_raises_for_model_fit_on_split_then_resplit(
    call_tool, load_df_into_session
) -> None:
    """S11 (review r1): split-over-split under the same names. Old and new
    split entries share the constant '(split)' sentinel hash; only the
    revision distinguishes the seed-1 membership the model was fit on from
    the seed-2 membership at replay. Loud raise required."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session(
        "base",
        pd.DataFrame({"x": list(range(10)), "y": [float(i % 3 + i) for i in range(10)]}),
    )
    assert call_tool("split_dataset", {"name": "base", "seed": 1})["ok"] is True
    assert call_tool(
        "fit_model",
        {"name": "base_train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )["ok"] is True
    assert call_tool(
        "split_dataset", {"name": "base", "seed": 2, "overwrite": True}
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "was later replaced" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_raises_for_model_fit_on_replaced_dataframe_dataset(
    call_tool, load_df_into_session
) -> None:
    """Dataframe branch: re-registering an in-memory dataset under the same
    name after a fit leaves the model's fit-time state unreachable — the
    '(dataframe)' sentinel hash is constant, only the revision can tell."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session(
        "t", pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    )
    assert call_tool(
        "fit_model", {"name": "t", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    load_df_into_session(
        "t", pd.DataFrame({"y": [5.0, 4.0, 3.0, 2.0, 1.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    )

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "was later replaced" in setup_src


def test_setup_cell_raises_for_model_fit_on_dataframe_then_materialize_overwrite(
    call_tool, load_df_into_session
) -> None:
    """S7a pin (already loud today): dataframe dataset overwritten by
    materialize_query after a fit — stays a loud raise under the revision
    dispatch (derived entry, no base_loader, revision mismatch)."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session(
        "t", pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    )
    assert call_tool(
        "fit_model", {"name": "t", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM t", "name": "t", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
```

- [ ] **Step 2: Run tests to verify red state**

Run: `uv run pytest tests/test_recorder.py -v -k "resplit or replaced_dataframe or dataframe_then_materialize"`
Expected: split-over-split and replaced-dataframe FAIL (no raise — both take today's silent normal path); the S7a pin PASSES (Task 4's derived branch already raises).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: model replay guards split and dataframe replacements by revision"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py`, generalize Task 4's dispatch. Replace:

```python
            overwritten_base: dict[str, Any] | None = None
            if (
                ds_entry is not None
                and ds_entry.format == "derived"
                and rev != ds_entry.revision
            ):
                base = ds_entry.base_loader
                if base is not None and rev == base.get("revision"):
                    overwritten_base = base
                else:
```

with:

```python
            overwritten_base: dict[str, Any] | None = None
            if (
                ds_entry is not None
                and ds_entry.format in ("derived", "split", "dataframe")
                and rev != ds_entry.revision
            ):
                base = ds_entry.base_loader
                if ds_entry.format == "derived" and base is not None and rev == base.get("revision"):
                    overwritten_base = base
                else:
```

(The `else:` raise body and everything after it is unchanged — split and dataframe replacements now flow into the same `was later replaced` raise. A split/dataframe entry never carries a base_loader — `materialize_query` leaves it `None` for those formats — so the derived-only base condition is not narrowing behavior, just stating it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_recorder.py tests/test_split.py -q`
Expected: all PASS (rev-match split/dataframe fits — e.g. `test_setup_cell_skips_hash_assert_for_in_memory_model` — still take the normal path).

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: model replay guards split and dataframe replacements by revision"
```

---

### Task 6: recorder file-backed mismatch rule + entry-None raise (S12, S7b, S15, S15b, S17; S15c stays green)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (same dispatch)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `ModelEntry.training_loader` (Task 3).
- Produces: the COMPLETE Part-2 dispatch. File-backed revision mismatch is allowed onto the normal path only when `hash_val == entry.source_hash` AND `training_loader == {"path": entry.path, "format": entry.format, "read_options": entry.read_options}`; otherwise raise with the pinned `Training data for {model_name!r} changed since the session was recorded.` prefix. `ds_entry is None` → unconditional raise ("no longer registered").

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recorder.py`:

```python
def test_setup_cell_raises_for_model_fit_on_derived_replaced_by_load_dataset(
    call_tool, tmp_path
) -> None:
    """S12 (review r1): load_dataset over a derived name after a fit. The
    current entry is file-backed, no overwrite branch applies today, and the
    recorder silently re-fits on the new file behind a 'no hash assert
    possible' comment. Must raise instead."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT y, x FROM base", "name": "d"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool("load_dataset", {"path": str(csv), "name": "d"})["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "Training data for 'm' changed since the session was recorded." in setup_src


def test_setup_cell_raises_for_model_fit_on_split_replaced_by_load_dataset(
    call_tool, tmp_path
) -> None:
    """S12, split flavor: load_dataset over a split output name after a fit."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"x": list(range(10)), "y": [float(i % 3 + i) for i in range(10)]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    assert call_tool(
        "fit_model",
        {"name": "base_train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )["ok"] is True
    assert call_tool("load_dataset", {"path": str(csv), "name": "base_train"})["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "Training data for 'm' changed since the session was recorded." in setup_src


def test_setup_cell_raises_for_model_fit_on_dataframe_replaced_by_load_dataset(
    call_tool, load_df_into_session, tmp_path
) -> None:
    """S7b (review r1): dataframe dataset replaced by load_dataset after a
    fit — today a silent re-fit on the new file. Must raise."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    load_df_into_session("t", df)
    assert call_tool(
        "fit_model", {"name": "t", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True

    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "t"})["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "Training data for 'm' changed since the session was recorded." in setup_src


def test_setup_cell_same_loader_reload_after_fit_stays_guarded_pass(
    call_tool, tmp_path
) -> None:
    """S15: reloading the same file, same read_options, same content under
    the same name is the innocent case — the normal hash-guarded re-fit
    path must survive the revision mismatch (hash AND loader identity both
    match)."""
    import hashlib

    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    assert call_tool("load_dataset", {"path": str(csv), "name": "train"})["ok"] is True
    assert call_tool(
        "fit_model", {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool("load_dataset", {"path": str(csv), "name": "train"})["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" not in setup_src
    assert f"expected_hash_m = '{expected}'" in setup_src
    assert "data=train_df" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_raises_for_same_bytes_reload_with_changed_read_options(
    call_tool, tmp_path
) -> None:
    """S15b (review r2, reproduced): identical bytes reloaded with different
    read_options parse a DIFFERENT table while the SHA-256 stays equal. A
    content hash proves identical bytes, not identical loading semantics —
    the fit-time loader identity must catch this."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "train"})["ok"] is True
    assert call_tool(
        "fit_model", {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    # Same bytes, same name — but explicit read_options change the loader
    # identity (and would change parsing for options like nullstr/header).
    assert call_tool(
        "load_dataset",
        {"path": str(csv), "name": "train", "read_options": {"header": True}},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "Training data for 'm' changed since the session was recorded." in setup_src


def test_setup_cell_remote_reload_same_url_keeps_unguarded_refit(call_tool) -> None:
    """S17: a remote (s3) dataset re-registered under the same URL and
    options after a fit. Path-keyed sentinels and loader identity compare
    equal, so the unguarded-comment re-fit path is unchanged — the
    conservative rule must NOT turn remote reloads into false failures."""
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import get_recorder
    from data_analyst_mcp.tools import models as _models

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    con = session.get_connection()
    con.register("__t_df", df)
    con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM __t_df")
    con.unregister("__t_df")
    cols = [{"name": "y", "dtype": "DOUBLE"}, {"name": "x", "dtype": "DOUBLE"}]
    session.register(
        name="train", path="s3://bucket/train.csv", read_options={}, format="csv",
        rows=5, columns=cols,
    )
    _models.fit_model(
        _models.FitModelInput(
            name="train", formula="y ~ x", kind="ols", robust=False, model_name="m"
        )
    )
    # Simulated re-load of the same URL (a real s3 read needs network; the
    # registry state is what the recorder consumes).
    session.register(
        name="train", path="s3://bucket/train.csv", read_options={}, format="csv",
        rows=5, columns=cols,
    )

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" not in setup_src
    assert "non-file dataset" in setup_src
    assert "data=train_df" in setup_src


def test_setup_cell_raises_when_training_dataset_no_longer_registered() -> None:
    """entry=None (reachable only by direct registry mutation): emit an
    unconditional raise instead of a NameError-at-replay comment path."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    session.register_model(
        name="m",
        kind="ols",
        formula="y ~ x",
        fitted_on_dataset="ghost",
        n_obs=5,
        training_dataset_hash="sentinel:unset",
        result=object(),
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "no longer registered" in setup_src
```

- [ ] **Step 2: Run tests to verify red state**

Run: `uv run pytest tests/test_recorder.py -v -k "load_dataset or same_loader or changed_read_options or remote_reload or no_longer_registered"`
Expected: the three S12/S7b tests, S15b, and entry-None FAIL; S15 and S17 PASS (pins for the innocent paths — they discriminate against an over-eager "any revision mismatch raises" implementation).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: model replay guards file reloads by content and loader identity"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py`, three edits inside the model loop:

**(a)** Immediately after `hash_val = ...` / `rev = ...` and before the dispatch comment, add the entry-None raise:

```python
            if ds_entry is None:
                # Reachable only by direct registry mutation — no tool
                # deregisters a dataset while models still reference it.
                msg = (
                    f"Model {model_name!r} was fit on dataset "
                    f"{model_entry.fitted_on_dataset!r}, which is no longer "
                    f"registered; the re-fit cannot be replayed faithfully."
                )
                lines.append(f"raise AssertionError({msg!r})")
                continue
```

**(b)** Extend the dispatch with the file-backed rule. Replace the dispatch condition and add an `elif` branch after the non-file raise's `continue`, so the whole dispatch reads:

```python
            overwritten_base: dict[str, Any] | None = None
            if rev != ds_entry.revision:
                base = ds_entry.base_loader
                if ds_entry.format == "derived" and base is not None and rev == base.get("revision"):
                    overwritten_base = base
                elif ds_entry.format in ("derived", "split", "dataframe"):
                    msg = (
                        f"Model {model_name!r} was fit on dataset "
                        f"{model_entry.fitted_on_dataset!r}, which was later "
                        f"replaced; the table state it was fit on no longer "
                        f"exists anywhere reachable, so the re-fit cannot be "
                        f"replayed faithfully."
                    )
                    lines.append(f"raise AssertionError({msg!r})")
                    continue
                elif not (
                    hash_val == ds_entry.source_hash
                    and model_entry.training_loader
                    == {
                        "path": ds_entry.path,
                        "format": ds_entry.format,
                        "read_options": ds_entry.read_options,
                    }
                ):
                    # File-backed replacement that is not provably the same
                    # loading semantics: the innocent same-file reload needs
                    # content equality (hash) AND loader identity (path,
                    # format, read_options) — a hash alone cannot see
                    # re-parsing under changed read options. Remote URLs pass
                    # via equal path-keyed sentinels + equal loaders.
                    msg = (
                        f"Training data for {model_name!r} changed since the "
                        f"session was recorded. Dataset "
                        f"{model_entry.fitted_on_dataset!r} was reloaded after "
                        f"the fit with different content or loading options, "
                        f"so the re-fit cannot be replayed faithfully."
                    )
                    lines.append(f"raise AssertionError({msg!r})")
                    continue
```

(Note the outer condition loses its `ds_entry is not None and ds_entry.format in (...)` qualifiers: None already `continue`d, and file-backed formats are now handled by the final `elif`. A file-backed mismatch that passes both equalities falls through to the normal path.)

**(c)** In the normal-path code below, simplify the now-impossible None handling:

```python
            else:
                ds_path = ds_entry.path if ds_entry is not None else None
```

becomes

```python
            else:
                ds_path = ds_entry.path
```

(Keep the `ds_path is not None` conditions below unchanged — pyright-wise `ds_path` is now always `str`, so if pyright flags dead `is not None` checks, simplify `if ds_path is not None and not hash_val.startswith(...)` to `if not hash_val.startswith(...)` and `elif ds_path is not None and hash_val.startswith("fallback:")` to `elif hash_val.startswith("fallback:")`; behavior is identical because `ds_entry.path` is always a string.)

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS. Specifically confirm the pinned S15c contract still holds: `uv run pytest tests/test_recorder.py::test_emitted_notebook_model_guard_fires_when_dataset_reloaded_after_fit -v` — the changed-content reload now takes the new raise, whose message begins with the same pinned `Training data for 'm' changed since the session was recorded.` substring that test asserts on.

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: model replay guards file reloads by content and loader identity"
```

---

### Task 7: `materialize_query` records split-overwrite provenance on the entry

**Files:**
- Modify: `src/data_analyst_mcp/session.py` (DatasetEntry + register), `src/data_analyst_mcp/tools/materialize.py`
- Test: `tests/test_materialize.py`, `tests/test_session.py`

**Interfaces:**
- Produces: `DatasetEntry.split_overwrite: dict[str, Any] | None = None` with keys `{"side": "train"|"test", "source": <split source name>}`; `session.register(..., split_overwrite=...)` optional param (copied defensively). `materialize_query` records it when overwriting a `format == "split"` entry (side/source from `existing.read_options`), carries it forward from a `format == "derived"` entry, leaves `None` otherwise. Task 8 keys the recorder wrap off it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_materialize.py`:

```python
def test_overwrite_of_split_entry_records_split_provenance(
    call_tool, load_df_into_session
) -> None:
    """Overwriting a split side must record {side, source} on the derived
    entry itself — the recorder's wrap must not depend on a surviving
    sibling (the double-overwrite case has none)."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True

    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 5', "name": "base_train", "overwrite": True},
    )["ok"] is True
    assert _session.get_datasets()["base_train"].split_overwrite == {
        "side": "train",
        "source": "base",
    }

    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x <= 2', "name": "base_test", "overwrite": True},
    )["ok"] is True
    assert _session.get_datasets()["base_test"].split_overwrite == {
        "side": "test",
        "source": "base",
    }


def test_chained_overwrite_carries_split_provenance_forward(
    call_tool, load_df_into_session
) -> None:
    import pandas as pd

    from data_analyst_mcp import session as _session

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 5', "name": "base_train", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 6', "name": "base_train", "overwrite": True},
    )["ok"] is True

    assert _session.get_datasets()["base_train"].split_overwrite == {
        "side": "train",
        "source": "base",
    }


def test_plain_overwrites_leave_split_provenance_none(call_tool, tmp_path) -> None:
    import pandas as pd

    from data_analyst_mcp import session as _session

    csv = tmp_path / "base.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "data"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )["ok"] is True
    assert _session.get_datasets()["data"].split_overwrite is None
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
    )["ok"] is True
    assert _session.get_datasets()["data"].split_overwrite is None
```

Append to `tests/test_session.py`:

```python
def test_register_copies_split_overwrite_defensively() -> None:
    from data_analyst_mcp import session

    session.reset()
    provenance = {"side": "train", "source": "base"}
    session.register(
        name="d",
        path="(query)",
        read_options={"sql": "SELECT 1"},
        format="derived",
        rows=1,
        columns=[],
        split_overwrite=provenance,
    )
    provenance["side"] = "mutated"

    assert session.get_datasets()["d"].split_overwrite == {"side": "train", "source": "base"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_materialize.py tests/test_session.py -v -k "provenance or split_overwrite"`
Expected: 4 FAIL — either `AttributeError: ... no attribute 'split_overwrite'` or `TypeError: register() got an unexpected keyword argument`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_materialize.py tests/test_session.py
git commit -m "red: materialize overwrite records split-side provenance on the entry"
```

- [ ] **Step 4: Implement**

`src/data_analyst_mcp/session.py` — add to `DatasetEntry` (after `base_loader`):

```python
    # When a derived (materialize_query) entry overwrote one side of a split,
    # {"side": "train"|"test", "source": <split source name>} recorded at
    # overwrite time and carried across chained derived overwrites. The
    # recorder wraps this entry's CREATE so an unreplayable self-referential
    # recipe fails with an explanation instead of a raw CatalogException —
    # detection must not depend on a surviving sibling (a double overwrite
    # has none).
    split_overwrite: dict[str, Any] | None = None
```

Add the param to `register()` and thread it through:

```python
def register(
    *,
    name: str,
    path: str,
    read_options: dict[str, Any],
    format: str,
    rows: int,
    columns: list[dict[str, str]],
    base_loader: dict[str, Any] | None = None,
    split_overwrite: dict[str, Any] | None = None,
) -> None:
    """Insert (or replace) a dataset entry under ``name``."""
    global _revision_counter
    _datasets[name] = DatasetEntry(
        path=path,
        read_options=dict(read_options),
        format=format,
        rows=rows,
        columns=list(columns),
        base_loader=dict(base_loader) if base_loader is not None else None,
        split_overwrite=dict(split_overwrite) if split_overwrite is not None else None,
        source_hash=compute_source_hash(path),
        revision=_revision_counter,
    )
    _revision_counter += 1
```

`src/data_analyst_mcp/tools/materialize.py` — after the base_loader capture block, add:

```python
    # Record split-overwrite provenance on the entry itself: the recorder's
    # replay wrap must work even when the sibling split entry is gone too
    # (double overwrite), so sibling inference is not enough. Chained derived
    # overwrites carry the original provenance forward.
    split_overwrite: dict[str, Any] | None = None
    if existing is not None:
        if existing.format == "split":
            split_overwrite = {
                "side": str(existing.read_options["role"]),
                "source": str(existing.read_options["source"]),
            }
        elif existing.format == "derived":
            split_overwrite = existing.split_overwrite
```

and pass it in the `session.register(` call:

```python
    session.register(
        name=payload.name,
        path="(query)",
        read_options={"sql": payload.sql},
        format="derived",
        rows=rows,
        columns=columns,
        base_loader=base_loader,
        split_overwrite=split_overwrite,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_materialize.py tests/test_session.py tests/test_split.py -q`
Expected: all PASS

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/session.py src/data_analyst_mcp/tools/materialize.py
git commit -m "green: materialize overwrite records split-side provenance on the entry"
```

---

### Task 8: recorder wrap keys off recorded provenance; delete the sibling scan (S9 wrapped, S14 pin, S8 unchanged)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py:255-282` (delete `_split_side_overwritten`), `:377-406` (derived emission)
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `DatasetEntry.split_overwrite` (Task 7).
- Produces: the derived-CREATE wrap keyed off `entry.split_overwrite`; `_split_side_overwritten()` deleted. Wrap message prefix unchanged (pinned): `Dataset {name!r} was created by overwriting the {side} side of the split of {source!r}.`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split.py`:

```python
def test_double_split_side_overwrite_raises_explained_at_replay(
    call_tool, tmp_path: Any
) -> None:
    """S9 (ROADMAP gap): BOTH sides self-referentially overwritten — no split
    entry survives, so sibling inference finds nothing and replay dies with a
    raw CatalogException today. Recorded provenance must wrap both CREATEs;
    the first failing wrapper (train, earlier revision) halts the cell with
    the pinned friendly message and the catalog error chained."""
    import duckdb
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_train" WHERE x > 10', "name": "base_train", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_test" WHERE x > 1', "name": "base_test", "overwrite": True},
    )["ok"] is True

    src = _setup_source(call_tool)
    # Both derived CREATEs carry the provenance wrap.
    assert src.count("except duckdb.CatalogException") == 2
    ns: dict[str, Any] = {}
    with pytest.raises(AssertionError, match="overwriting the train side") as excinfo:
        exec(src, ns)  # first unreplayable CREATE halts the cell, explained
    assert isinstance(excinfo.value.__cause__, duckdb.CatalogException)


def test_both_sides_replayable_overwrite_replays_transparently(
    call_tool, tmp_path: Any
) -> None:
    """S14 pin (success case): both sides overwritten with SQL that reads
    only from the source table. The wraps must be transparent — replay
    succeeds and both tables hold the overwrite results."""
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 10', "name": "base_train", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x <= 3', "name": "base_test", "overwrite": True},
    )["ok"] is True

    src = _setup_source(call_tool)
    ns: dict[str, Any] = {}
    exec(src, ns)  # wrapped but replayable: must run clean
    con = ns["con"]
    train_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_train"').fetchall())
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    assert train_x == list(range(11, 20))
    assert test_x == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests to verify red state**

Run: `uv run pytest tests/test_split.py -v -k "double_split_side or both_sides_replayable"`
Expected: the S9 test FAILS (exec raises raw `duckdb.CatalogException`, not `AssertionError` — pytest reports the wrong exception type; also the `count(...) == 2` assert fails first with 0). The S14 pin PASSES today (bare CREATEs read only from `base`).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_split.py
git commit -m "red: split-side overwrite wrap keys off recorded provenance"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py`:

**(a)** Delete the whole `_split_side_overwritten` function (lines ~255-282, including its docstring).

**(b)** In the second-pass derived branch, replace:

```python
            overwrite = _split_side_overwritten(name, datasets)
            if overwrite is None:
                lines.append(f"con.execute({stmt!r})")
            else:
```
and the wrap body's opening lines
```python
                side, split_source = overwrite
```

with:

```python
            if entry.split_overwrite is None:
                lines.append(f"con.execute({stmt!r})")
            else:
```
and
```python
                side = str(entry.split_overwrite["side"])
                split_source = str(entry.split_overwrite["source"])
```

**(c)** Update the wrap's trailing explanation (the message MUST keep its pinned first sentence; the trailing sentence generalizes because with a double overwrite the missing table may be the *other* side, not `name` itself). Replace the `msg = (...)` construction with:

```python
                msg = (
                    f"Dataset {name!r} was created by overwriting the {side} side "
                    f"of the split of {split_source!r}. The split's pre-overwrite "
                    f"tables are not recreated at replay (that would clobber the "
                    f"overwriting datasets), so this SQL references a table that "
                    f"does not exist at replay; rematerialize it from a table "
                    f"that exists at replay."
                )
```

**(d)** Update the stale comment above the wrap (it says "overwrote one side of a still-live split" / explains sibling detection) to describe recorded provenance:

```python
                # This derived entry overwrote one side of a split (recorded
                # at materialize_query time — detection must not depend on a
                # surviving sibling; a double overwrite has none). The
                # pre-overwrite split tables are deliberately NOT recreated at
                # replay, so a recipe reading them hits a DuckDB catalog error
                # at its own CREATE. Wrap it so replay explains the missing
                # table instead of surfacing a bare CatalogException; the
                # CREATE is otherwise unchanged, so a replayable overwrite
                # recipe still succeeds transparently.
```

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS. The pinned S8 tests must be green UNMODIFIED: `test_split_setup_cell_train_overwrite_self_ref_raises_at_replay`, `test_split_setup_cell_test_overwrite_self_ref_raises_at_replay` (message prefix contract), `test_plain_derived_create_not_wrapped_beside_split_overwrite` (exactly one handler in that scenario), `test_split_setup_cell_train_overwrite_drops_train_recreation`, `test_split_setup_cell_train_overwrite_replays_overwrite_not_split`.

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: split-side overwrite wrap keys off recorded provenance"
```

---

### Task 9: `split_dataset` stores a membership checksum for both sides

**Files:**
- Modify: `src/data_analyst_mcp/tools/split.py:245-266`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `membership_checksum(df)` (existing, `split.py:98`).
- Produces: BOTH split entries carry their own side's digest under the SAME key `read_options["membership_checksum"]` (test entry's value unchanged from today; train entry's is new). Tasks 10–11 read the train value.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split.py`:

```python
def test_split_stores_membership_checksum_on_both_sides(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Per-side checksums (spec Part 4): each entry stores its OWN side's
    digest under the same 'membership_checksum' key. A split-source drift
    that only changes train rows is invisible to the test checksum (S16) —
    the train side needs its own."""
    import re

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.split import membership_checksum

    _load_ten_rows(load_df_into_session)
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True

    datasets = _session.get_datasets()
    con = _session.get_connection()
    for side in ("base_train", "base_test"):
        stored = datasets[side].read_options["membership_checksum"]
        assert re.fullmatch(r"[0-9a-f]+:[0-9a-f]{32}:[0-9a-f]{32}", stored)
        actual = membership_checksum(con.execute(f'SELECT * FROM "{side}"').df())
        assert stored == actual
    # Different rows on each side -> different digests.
    assert (
        datasets["base_train"].read_options["membership_checksum"]
        != datasets["base_test"].read_options["membership_checksum"]
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split.py::test_split_stores_membership_checksum_on_both_sides -v`
Expected: FAIL with `KeyError: 'membership_checksum'` (on the train entry)

- [ ] **Step 3: Commit red**

```bash
git add tests/test_split.py
git commit -m "red: split_dataset stores a membership checksum for both sides"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/tools/split.py`, replace:

```python
    test_df = con.execute(f"SELECT * FROM {_quote(test_name)}").df()
    checksum = membership_checksum(test_df)
```

with:

```python
    # One digest per side (one extra hashing pass over the train frame):
    # test-side drift and train-side drift are independent failure modes —
    # a split-source change that only moves train rows passes the test
    # checksum (spec S16), so replay asserts each side against its own.
    test_df = con.execute(f"SELECT * FROM {_quote(test_name)}").df()
    test_checksum = membership_checksum(test_df)
    train_df = con.execute(f"SELECT * FROM {_quote(train_name)}").df()
    train_checksum = membership_checksum(train_df)
```

and in the registration loop replace:

```python
        opts = {**common_opts, "role": role}
        if role == "test":
            opts["membership_checksum"] = checksum
```

with:

```python
        opts = {**common_opts, "role": role}
        opts["membership_checksum"] = test_checksum if role == "test" else train_checksum
```

and update the `_record_split` call's checksum argument from `checksum` to `test_checksum`:

```python
    _record_split(payload, train_name, test_name, n - n_test, n_test, test_checksum, rid)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_split.py -q`
Expected: all PASS (existing tests never forbid extra keys; the test-side value is computed exactly as before).

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/tools/split.py
git commit -m "green: split_dataset stores a membership checksum for both sides"
```

---

### Task 10: symmetric `split_replay_source` with per-side asserts (S16 at snippet level)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py:143-206` (`split_replay_source`), `src/data_analyst_mcp/tools/split.py` (`_record_split`), `tests/test_split.py:375` (deliberate direct-call update)
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: per-side checksums (Task 9).
- Produces: `split_replay_source(*, source, train_name, test_name, seed, test_fraction, stratify_by, rid_column, membership_checksum: str | None, train_membership_checksum: str | None = None, include_train: bool = True, include_test: bool = True) -> str`. `membership_checksum` keeps meaning the TEST-side value and is asserted iff `include_test`; the train assert is emitted iff `include_train and train_membership_checksum is not None`. Callers never pass `include_train=False, include_test=False`. Task 11's recorder keying calls all three shapes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split.py`:

```python
def test_split_replay_source_emits_per_side_checksum_asserts() -> None:
    from data_analyst_mcp.recorder import split_replay_source

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column="__split_rid",
        membership_checksum="aa",
        train_membership_checksum="bb",
    )
    assert "'aa'" in snippet
    assert "'bb'" in snippet
    assert snippet.count("drifted at replay") == 2
    assert "SELECT * FROM \"base_train\"" in snippet


def test_split_replay_source_train_only_block() -> None:
    """Train-only shape (surviving train side, test overwritten): recreates
    and asserts ONLY the train table."""
    import duckdb
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import split_replay_source
    from data_analyst_mcp.tools.split import _assign_is_test, membership_checksum

    df = pd.DataFrame({"x": list(range(10))})
    is_test, _ = _assign_is_test(10, 0.25, 42, None)
    train_checksum = membership_checksum(df[~is_test])

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column="__split_rid",
        membership_checksum=None,
        train_membership_checksum=train_checksum,
        include_train=True,
        include_test=False,
    )
    con = duckdb.connect()
    con.register("__base_src", df)
    con.execute('CREATE TABLE "base" AS SELECT * FROM __base_src')
    exec(snippet, {"con": con, "np": np, "pd": pd})  # train-only replay under test
    train_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_train"').fetchall())
    assert train_x == [0, 2, 3, 4, 5, 6, 7, 9]
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "base_test" not in tables


def test_split_replay_snippet_train_only_drift_fails_train_checksum(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """S16 core (review r2, reproduced): replay-time source rows differ at a
    TRAIN-side position only. Test rows are byte-identical, so the test
    checksum passes; only the train-side assert can catch the drift."""
    import duckdb
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.recorder import split_replay_source

    load_df_into_session("base", pd.DataFrame({"x": list(range(20))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    datasets = _session.get_datasets()
    test_opts = datasets["base_test"].read_options

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column=str(test_opts["rid_column"]),
        membership_checksum=str(test_opts["membership_checksum"]),
        train_membership_checksum=str(
            datasets["base_train"].read_options["membership_checksum"]
        ),
    )

    # Membership is positional: find a position the seed sends to TRAIN and
    # perturb only that value in the replayed source.
    perm = np.random.RandomState(42).permutation(20)
    test_positions = set(perm[:5].tolist())
    train_pos = min(set(range(20)) - test_positions)
    drifted = pd.DataFrame({"x": list(range(20))})
    drifted.loc[train_pos, "x"] = 999

    con = duckdb.connect()
    con.register("__base_src", drifted)
    con.execute('CREATE TABLE "base" AS SELECT * FROM __base_src')
    with pytest.raises(AssertionError, match="base_train"):
        exec(snippet, {"con": con, "np": np, "pd": pd})  # train drift must be loud
```

- [ ] **Step 2: Run tests to verify red state**

Run: `uv run pytest tests/test_split.py -v -k "per_side_checksum_asserts or train_only"`
Expected: all 3 FAIL with `TypeError: split_replay_source() got an unexpected keyword argument 'train_membership_checksum'`

- [ ] **Step 3: Commit red**

```bash
git add tests/test_split.py
git commit -m "red: split replay asserts each recreated side against its own checksum"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py`, replace `split_replay_source` in full:

```python
def split_replay_source(
    *,
    source: str,
    train_name: str,
    test_name: str,
    seed: int,
    test_fraction: float,
    stratify_by: str | None,
    rid_column: str,
    membership_checksum: str | None,
    train_membership_checksum: str | None = None,
    include_train: bool = True,
    include_test: bool = True,
) -> str:
    """Self-contained notebook snippet that recreates a train/test split.

    Rebuilds membership with the same ``RandomState`` algorithm the live
    tool used, recreates each included side, then asserts every recreated
    table against its OWN order-independent membership checksum
    (``membership_checksum`` is the test-side value,
    ``train_membership_checksum`` the train-side one). Per-side asserts are
    the point: a source drift that only changes train rows passes the test
    checksum, so the test-side digest alone cannot guard the train table.
    For file-backed sources the upstream source-hash assert makes replay
    deterministic; for derived sources whose SQL is not order-preserving,
    row-order drift fails loudly here instead of silently changing the
    split (spec §5.6b row-order tiers).

    ``include_train`` / ``include_test`` (setup-cell only): a side that was
    later overwritten by ``materialize_query`` lost its split recipe, so its
    ``CREATE`` (and assert) is skipped — re-creating it here would clobber
    the derived table the second pass builds. The per-call cell keeps both
    defaults (both sides are fresh at call time). Callers never pass both
    flags false.
    """
    src_q = '"' + source.replace('"', '""') + '"'
    train_q = '"' + train_name.replace('"', '""') + '"'
    test_q = '"' + test_name.replace('"', '""') + '"'
    rid_q = '"' + rid_column.replace('"', '""') + '"'
    lines = [
        f"# --- split_dataset: {source} -> {train_name} / {test_name} "
        f"(seed={seed}, test_fraction={test_fraction}) ---",
    ]
    lines.extend(_split_assignment_lines(source, seed, test_fraction, stratify_by))
    base = (
        f"SELECT s.* EXCLUDE ({rid_q}) FROM "
        f"(SELECT *, row_number() OVER () - 1 AS {rid_q} FROM {src_q}) s "
        f"JOIN __data_analyst_split_assign a ON s.{rid_q} = a.rid"
    )
    train_stmt = f"CREATE OR REPLACE TABLE {train_q} AS {base} WHERE NOT a.is_test"
    test_stmt = f"CREATE OR REPLACE TABLE {test_q} AS {base} WHERE a.is_test"
    exec_lines = [
        "_split_assign = pd.DataFrame({'rid': np.arange(len(_split_is_test), "
        "dtype=np.int64), 'is_test': _split_is_test})",
        "con.register('__data_analyst_split_assign', _split_assign)",
    ]
    if include_train:
        exec_lines.append(f"con.execute({train_stmt!r})")
    if include_test:
        exec_lines.append(f"con.execute({test_stmt!r})")
    exec_lines.append("con.unregister('__data_analyst_split_assign')")
    exec_lines.append(_SPLIT_CHECKSUM_DEF)
    if include_test:
        message = (
            f"Split membership for {test_name!r} drifted at replay (source row order changed)."
        )
        exec_lines.append(
            f"assert _split_checksum(con.sql('SELECT * FROM {test_q}').df()) == "
            f"{membership_checksum!r}, {message!r}"
        )
    if include_train and train_membership_checksum is not None:
        train_message = (
            f"Split membership for {train_name!r} drifted at replay (source row order changed)."
        )
        exec_lines.append(
            f"assert _split_checksum(con.sql('SELECT * FROM {train_q}').df()) == "
            f"{train_membership_checksum!r}, {train_message!r}"
        )
    lines.extend(exec_lines)
    return "\n".join(lines)
```

In `src/data_analyst_mcp/tools/split.py`, thread the train checksum through the per-call cell. Change the `_record_split` call to:

```python
    _record_split(payload, train_name, test_name, n - n_test, n_test, test_checksum, train_checksum, rid)
```

and `_record_split` to:

```python
def _record_split(
    payload: SplitDatasetInput,
    train_name: str,
    test_name: str,
    n_train: int,
    n_test: int,
    checksum: str,
    train_checksum: str,
    rid: str,
) -> None:
```

with the `split_replay_source(` call inside gaining:

```python
        membership_checksum=checksum,
        train_membership_checksum=train_checksum,
```

Update the direct-call test at `tests/test_split.py` (`test_split_replay_snippet_executes_and_reproduces_membership`, ~line 375) deliberately — add the train-side value so the strengthened snippet is what's exercised:

```python
    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column=entry.read_options["rid_column"],
        membership_checksum=entry.read_options["membership_checksum"],
        train_membership_checksum=_session.get_datasets()["base_train"].read_options[
            "membership_checksum"
        ],
    )
```

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS. Note: the setup cell's split block does NOT yet pass a train checksum (the recorder call site is Task 11) — only the per-call cell and direct calls gained the train assert.

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py src/data_analyst_mcp/tools/split.py tests/test_split.py
git commit -m "green: split replay asserts each recreated side against its own checksum"
```

---

### Task 11: emission keying contract — train-only block, reciprocal siblings, setup-cell train assert (S13; S16 at setup level)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (second-pass split branch + new helper)
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: symmetric `split_replay_source` (Task 10), per-side checksums (Task 9).
- Produces: the emission keying contract — a "matching" sibling means reciprocal pair metadata (`format == "split"`, expected role, identical `train_name`/`test_name` pair):

  | Surviving matching sides | Block owner | include_train, include_test |
  |---|---|---|
  | train + test | test entry | True, True |
  | test only | test entry | False, True |
  | train only | train entry | True, False |
  | neither | — | no block |

  Helper: `_matching_split_sibling(datasets, *, opts, sibling_role) -> DatasetEntry | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split.py`:

```python
def test_split_setup_cell_asserts_both_side_checksums(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """S16 at setup-cell level: with both sides alive, the split block must
    assert train AND test digests."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True

    src = _setup_source(call_tool)
    datasets = _session.get_datasets()
    assert datasets["base_test"].read_options["membership_checksum"] in src
    assert datasets["base_train"].read_options["membership_checksum"] in src


def test_split_setup_cell_test_overwrite_emits_train_only_block(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """S13 (review r1): the test side is overwritten with replayable SQL and
    the train side survives. Today no split block is emitted at all (it was
    keyed solely off the test entry) and the train table is never recreated.
    The surviving train entry must own a train-only block."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 5', "name": "base_test", "overwrite": True},
    )
    assert result["ok"] is True

    src = _setup_source(call_tool)
    # Train side recreated via the split JOIN, test side NOT (the derived
    # CREATE owns that name now).
    assert '"base_train" AS SELECT s.* EXCLUDE' in src
    assert '"base_test" AS SELECT s.* EXCLUDE' not in src
    assert 'CREATE OR REPLACE TABLE "base_test" AS SELECT * FROM "base" WHERE x > 5' in src
    # The train table is guarded by its own digest; the test digest is gone
    # with its recipe.
    assert _session.get_datasets()["base_train"].read_options["membership_checksum"] in src


def test_split_setup_cell_test_overwrite_replays(call_tool: Any, tmp_path: Any) -> None:
    """S13 end-to-end: exec the self-contained setup cell — the train table
    holds the original split rows, the test table holds the overwrite."""
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base" WHERE x > 10', "name": "base_test", "overwrite": True},
    )
    assert result["ok"] is True

    src = _setup_source(call_tool)
    ns: dict[str, Any] = {}
    exec(src, ns)  # train-only split block + derived test CREATE under test
    con = ns["con"]
    train_n = con.execute('SELECT COUNT(*) FROM "base_train"').fetchone()[0]
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    assert train_n == 15  # original split train (20 rows, 5 test)
    assert test_x == list(range(11, 20))  # the overwrite result


def test_split_setup_cell_stratified_replays_in_both_asymmetric_directions(
    call_tool: Any, tmp_path: Any
) -> None:
    """Stratified splits through both one-sided shapes: a train-side
    overwrite (test-owned block, include_train=False) and a test-side
    overwrite (train-owned block, include_test=False) must both replay."""
    import pandas as pd

    csv = tmp_path / "strat.csv"
    pd.DataFrame(
        {"g": ["a", "b"] * 10, "x": list(range(20))}
    ).to_csv(csv, index=False)

    # Direction 1: overwrite the TRAIN side.
    assert call_tool("load_dataset", {"path": str(csv), "name": "s1"})["ok"] is True
    assert call_tool("split_dataset", {"name": "s1", "stratify_by": "g"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "s1" WHERE x > 15', "name": "s1_train", "overwrite": True},
    )["ok"] is True
    # Direction 2: overwrite the TEST side.
    assert call_tool("load_dataset", {"path": str(csv), "name": "s2"})["ok"] is True
    assert call_tool("split_dataset", {"name": "s2", "stratify_by": "g"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "s2" WHERE x <= 3', "name": "s2_test", "overwrite": True},
    )["ok"] is True

    src = _setup_source(call_tool)
    ns: dict[str, Any] = {}
    exec(src, ns)  # both asymmetric stratified blocks under test
    con = ns["con"]
    assert con.execute('SELECT COUNT(*) FROM "s1_test"').fetchone()[0] > 0
    assert sorted(
        r[0] for r in con.execute('SELECT x FROM "s1_train"').fetchall()
    ) == list(range(16, 20))
    assert con.execute('SELECT COUNT(*) FROM "s2_train"').fetchone()[0] > 0
    assert sorted(
        r[0] for r in con.execute('SELECT x FROM "s2_test"').fetchall()
    ) == [0, 1, 2, 3]


def test_split_block_not_claimed_by_unrelated_split_reusing_a_name(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """'Matching' means reciprocal pair metadata: after the test side is
    re-split under names that reuse the old test name, the ORIGINAL train
    entry has no matching sibling and must own a train-only block."""
    import pandas as pd

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))
    load_df_into_session("other", pd.DataFrame({"x": list(range(10, 30))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True
    # Re-split ANOTHER source, reusing the original test name for its test
    # side. base_train's recorded sibling name now belongs to a different
    # pair (train_name='o_tr', not 'base_train').
    assert call_tool(
        "split_dataset",
        {"name": "other", "train_name": "o_tr", "test_name": "base_test", "overwrite": True},
    )["ok"] is True

    src = _setup_source(call_tool)
    # The original train side still gets recreated (train-only block).
    assert '"base_train" AS SELECT s.* EXCLUDE' in src
```

- [ ] **Step 2: Run tests to verify red state**

Run: `uv run pytest tests/test_split.py -v -k "both_side_checksums or train_only_block or test_overwrite_replays or asymmetric_directions or unrelated_split"`
Expected: all 5 FAIL — no train checksum in setup src; no train-only block; `base_train`/`s2_train` missing at exec (the exec tests die inside the emitted frames/queries or on the missing-table assertion).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_split.py
git commit -m "red: surviving train side emits a train-only split block"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py`:

**(a)** Add the helper above `_build_setup_source` (where `_split_side_overwritten` used to live):

```python
def _matching_split_sibling(
    datasets: dict[str, DatasetEntry], *, opts: dict[str, Any], sibling_role: str
) -> DatasetEntry | None:
    """The reciprocal sibling entry of a split output, or ``None``.

    "Matching" means reciprocal pair metadata: the entry at the recorded
    sibling name is ``format == "split"``, has the expected role, and
    records the same ``train_name`` / ``test_name`` pair. A name merely
    reused by a *different* split (re-split of another source under the
    same output name) does not match — the survivor then owns a one-sided
    block of its own.
    """
    key = "train_name" if sibling_role == "train" else "test_name"
    sibling = datasets.get(str(opts.get(key)))
    if sibling is None or sibling.format != "split":
        return None
    sopts = sibling.read_options
    if sopts.get("role") != sibling_role:
        return None
    if sopts.get("train_name") != opts.get("train_name"):
        return None
    if sopts.get("test_name") != opts.get("test_name"):
        return None
    return sibling
```

**(b)** Replace the second-pass split branch (the whole `elif entry.format == "split" and entry.read_options.get("role") == "test":` block, including its inline `train_entry`/`include_train` computation) with:

```python
        elif entry.format == "split":
            opts = entry.read_options
            role = opts.get("role")
            # Block-owner contract: with both matching sides alive the
            # test-role entry owns the (two-sided) block; a lone surviving
            # test side owns a test-only block; a lone surviving train side
            # owns a train-only block; with neither alive no block is
            # emitted. One-sided shapes skip the overwritten side's CREATE —
            # the derived entry emitted above owns that name, and replay
            # must not clobber it with stale split rows.
            if role == "test":
                train_entry = _matching_split_sibling(datasets, opts=opts, sibling_role="train")
                lines.append(
                    split_replay_source(
                        source=str(opts["source"]),
                        train_name=str(opts["train_name"]),
                        test_name=str(opts["test_name"]),
                        seed=int(opts["seed"]),
                        test_fraction=float(opts["test_fraction"]),
                        stratify_by=opts.get("stratify_by"),
                        rid_column=str(opts["rid_column"]),
                        membership_checksum=str(opts["membership_checksum"]),
                        train_membership_checksum=(
                            str(train_entry.read_options["membership_checksum"])
                            if train_entry is not None
                            else None
                        ),
                        include_train=train_entry is not None,
                    )
                )
            elif role == "train" and _matching_split_sibling(
                datasets, opts=opts, sibling_role="test"
            ) is None:
                lines.append(
                    split_replay_source(
                        source=str(opts["source"]),
                        train_name=str(opts["train_name"]),
                        test_name=str(opts["test_name"]),
                        seed=int(opts["seed"]),
                        test_fraction=float(opts["test_fraction"]),
                        stratify_by=opts.get("stratify_by"),
                        rid_column=str(opts["rid_column"]),
                        membership_checksum=None,
                        train_membership_checksum=str(opts["membership_checksum"]),
                        include_train=True,
                        include_test=False,
                    )
                )
```

**(c)** Update the stale second-pass comment block (recorder.py ~:360-374, the part describing "keyed off the test-role entry" and "for a test-side overwrite the test-keyed branch below never fires, so no split block is emitted at all") to describe the block-owner contract above (both-sides → test-owned two-sided block with per-side asserts; lone test → test-only; lone train → train-only; neither → none).

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS — including the pre-existing one-sided tests (`test_split_setup_cell_train_overwrite_drops_train_recreation`, `..._replays_overwrite_not_split`, both self-ref raise tests) which now go through `_matching_split_sibling`.

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: surviving train side emits a train-only split block"
```

---

### Task 12: revision-ordered second pass (problem 8)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py:354-376` (second-pass loop + comment)
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `DatasetEntry.revision` (Task 1); the train-only block (Task 11).
- Produces: the second pass iterates `sorted(datasets.items(), key=lambda kv: kv[1].revision)` — true temporal order, which dict insertion order is not for overwrites (re-assigning an existing key keeps its old position). First pass unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split.py`:

```python
def test_setup_cell_second_pass_orders_by_revision_for_reused_names(
    call_tool: Any, tmp_path: Any
) -> None:
    """Problem 8 (review r2, reproduced): dict insertion order is NOT
    registration order for overwrites — a pre-registered name keeps its old
    dict position when split_dataset(overwrite=True) re-assigns it. The
    derived test-side overwrite here reads from the surviving train table,
    so its CREATE must emit AFTER the train-only split block; in dict order
    it emits first and replay dies in the provenance wrapper even though
    the recipe is replayable."""
    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame({"x": list(range(20))}).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    # Pre-register the future test name at an EARLY dict position.
    assert call_tool(
        "materialize_query", {"sql": 'SELECT * FROM "base"', "name": "t_test"}
    )["ok"] is True
    assert call_tool(
        "split_dataset",
        {"name": "base", "train_name": "t_train", "test_name": "t_test", "overwrite": True},
    )["ok"] is True
    # Overwrite the test side with SQL that reads the surviving train side.
    assert call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "t_train" WHERE x > 10', "name": "t_test", "overwrite": True},
    )["ok"] is True

    src = _setup_source(call_tool)
    # The train-only block must precede the derived CREATE that reads it.
    assert src.index('"t_train" AS SELECT s.* EXCLUDE') < src.index(
        'CREATE OR REPLACE TABLE "t_test" AS SELECT * FROM "t_train" WHERE x > 10'
    )
    ns: dict[str, Any] = {}
    exec(src, ns)  # replayable recipe — must succeed once ordered correctly
    con = ns["con"]
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "t_test"').fetchall())
    train_x = {r[0] for r in con.execute('SELECT x FROM "t_train"').fetchall()}
    assert test_x == sorted(x for x in train_x if x > 10)
    assert len(test_x) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split.py::test_setup_cell_second_pass_orders_by_revision_for_reused_names -v`
Expected: FAIL — the `src.index(...)` ordering assert fails (derived CREATE appears first), or the `exec` raises the Task-8 provenance `AssertionError`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_split.py
git commit -m "red: setup second pass emits in revision order"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py`, change the second-pass loop from:

```python
    datasets = _session.get_datasets()
    for name, entry in datasets.items():
```

to:

```python
    datasets = _session.get_datasets()
    for name, entry in sorted(datasets.items(), key=lambda kv: kv[1].revision):
```

and correct the second-pass comment (recorder.py ~:354): replace the sentences claiming "Registration order IS topological order — ... ``_session.get_datasets()`` is a regular dict whose insertion order Python preserves" with:

```python
    # Second pass: derived and split datasets, in REVISION order. Revisions
    # are true temporal order; dict insertion order is not — re-assigning an
    # existing key keeps its old position, so a pre-registered name that is
    # later overwritten (split_dataset/materialize_query with overwrite=True)
    # would emit too early. Registration (revision) order is topological
    # order: a derived/split entry can only reference tables that existed —
    # i.e. were registered — before it.
```

(Keep the rest of that comment block — the derived-no-hash-assert rationale and the block-owner contract from Task 11 — intact.)

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/ -q`
Expected: all PASS (entries never overwritten sort exactly as before — `sorted` is stable and their revisions are already in insertion order).

- [ ] **Step 6: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: setup second pass emits in revision order"
```

---

### Task 13: docs, ROADMAP, and release 1.4.0

**Files:**
- Modify: `ROADMAP.md`, `CHANGELOG.md`, `README.md`, `docs/SPEC.md`, `evals/README.md`, `src/data_analyst_mcp/recorder.py` (docstrings only), `pyproject.toml`, `src/data_analyst_mcp/__init__.py`, `tests/test_smoke.py`, `uv.lock`

**Interfaces:**
- Consumes: everything above, finished and green.

- [ ] **Step 1: Run every gate before touching docs**

```bash
uv run pytest tests/
uv run pytest evals/
uv run ruff format --check . && uv run ruff check .
uv run pyright src/
uv run python scripts/check_tdd_commits.py
```
Expected: all clean. Fix anything that isn't before proceeding.

- [ ] **Step 2: ROADMAP**

In `ROADMAP.md` "Reproducibility": delete the two closed bullets (`**Pure-query fit-then-overwrite guard gap.**`, `**Double split-side overwrite replay message.**`, currently lines 27–28) and add two parked notes:

```markdown
- **Plain-derived self-referential overwrite replay message (S4b).** A
  `materialize_query` overwrite of a *plain* derived dataset whose final
  recipe self-/cross-references tables nothing recreates still fails replay
  with a raw `duckdb.CatalogException` — loud, never silent, just
  unexplained. Split-side overwrites got recorded provenance + a wrapped
  explanation in 1.4.0; wrapping *every* derived CREATE would churn emitted
  notebook shape and pinned tests for marginal benefit, so the plain case is
  parked.
- **Ephemeral-fit replay provenance.** `cross_validate` and
  `fit_model(model_name=None)` re-fit inside their *per-call* cells with no
  fit-time revision/hash guard (the setup-cell guards only cover registered
  models). A training source mutated *and reloaded* between the call and
  replay re-runs those cells on the new data and silently reports different
  CV/fit numbers. Separate failure class from the 1.4.0 setup-cell work:
  closing it means stamping fit-time provenance into per-call cells.
```

Also update the Phase-5 "Reproducibility caveat" paragraph (line 5): after "re-fitting every registered model in its setup cell, guarded by a hard SHA-256 assert on the training file", add a sentence: "Since 1.4.0 the guard also carries the training dataset's registration revision and fit-time loader identity, so *any* replacement of the training dataset (re-materialize, re-load, re-split — even with an identical content hash) fails replay loudly instead of silently re-fitting."

- [ ] **Step 3: CHANGELOG**

Add at the top of `CHANGELOG.md` (below the header paragraph, above `## [1.3.1]`):

```markdown
## [1.4.0] - 2026-07-14

(Date the heading with the actual day of the release commit if it differs.)

Replay-guard hardening: registration revisions + fit-time loader identity +
recorded split-overwrite provenance + symmetric per-side split replay. Every
known way to make an emitted notebook silently re-fit a model on the wrong
table now fails loudly, and the known loud-but-unexplained split failure now
explains itself. Tool surface unchanged (24).

### Fixed
- A model fit on a pure-query derived dataset later overwritten by
  `materialize_query` silently re-fit on the post-transform table at replay
  (constant `(query)` sentinel hashes were indistinguishable). Replay now
  raises a purpose-written `AssertionError`. Same fix covers: fit on the
  middle materialization of an overwrite chain, fit on a base-carrying
  derived state then overwritten again, split-over-split under the same
  names, `load_dataset` over a derived/split/dataframe name after a fit,
  and same-name reloads with changed content or changed `read_options`
  (identical bytes re-parsed differently are now caught by fit-time loader
  identity — a content hash alone cannot see loading semantics).
- Overwriting **both** sides of a split with recipes that read the missing
  pre-overwrite split tables failed replay with a raw
  `duckdb.CatalogException`. Split-overwrite provenance is now recorded on
  the derived entry at `materialize_query` time (no sibling inference), so
  both CREATEs carry the 1.3.1-style explained `AssertionError`.
- Overwriting the **test** side of a split left the surviving train table
  without any recreation at replay (raw `CatalogException` downstream). The
  surviving train entry now emits a train-only split block guarded by its
  own membership checksum.
- A split-source drift that changed only train-side rows passed the
  test-side membership checksum and replayed silently drifted numbers. Both
  sides now store and assert their own checksum.
- The setup cell's second pass emitted in dict insertion order, which is
  wrong for overwrites (a re-assigned name keeps its old position): with
  pre-registered output names + `split_dataset(overwrite=True)` a derived
  CREATE could emit before the split block it reads from. The second pass
  now emits in registration-revision order.

### Changed
- `DatasetEntry` gains `revision` (monotonic per-session registration
  counter) and `split_overwrite` (recorded overwrite provenance);
  `ModelEntry` gains `training_dataset_revision` and `training_loader`
  (fit-time `{path, format, read_options}`); `base_loader` records the
  replaced file entry's revision. Registry metadata only — no tool-response
  changes.
- Emitted notebooks: split blocks now assert a per-side membership checksum
  (train and test), and one-sided blocks exist for a surviving train side.
- Deliberate conservatism: re-running a byte-identical recipe (or
  re-splitting with the same seed) over an unchanged source still *replaces*
  the dataset, so a model fit before the replacement now fails replay loudly
  even though a faithful re-fit might have been possible. Determinism of the
  recipe is not verifiable; loud beats silently-maybe-right.
```

Add the release link at the bottom, above the 1.3.1 line:

```markdown
[1.4.0]: https://github.com/olegtyshcneko/data-analyst-mcp/releases/tag/v1.4.0
```

- [ ] **Step 4: README**

- "Known gotchas" (`README.md` ~line 454), the "Emitted notebooks are drift-guarded." bullet: replace its last sentence ("Models registered before a `materialize_query` overwrite ... not the post-transform table.") with: "Models re-fit from the original file only when they were fit *on* the file-backed state (identified by registration revision); a model fit on any table state that no longer exists at replay — re-materialized, re-loaded, or re-split, even with identical bytes — fails the setup cell with a loud `AssertionError` instead of silently re-fitting."
- Split section (~line 378): after "recreates it behind an order-independent membership checksum, so silent drift is impossible", change to "...recreates each side behind its own order-independent membership checksum, so silent drift is impossible on either side."
- Test counts (~line 461): refresh both numbers from reality:
  `uv run pytest tests/ --collect-only -q | tail -1` and `uv run pytest evals/ --collect-only -q | tail -1`.

- [ ] **Step 5: docs/SPEC.md**

- §5.6b Registration (~line 372): change "plus `membership_checksum` on the test entry" to "plus a per-side `membership_checksum` on each entry (each side's own digest under the same key)". In the same behavior list, point 3 (row-order tiers): change "membership checksum ... of the recreated test table" to "of each recreated table (per side)". Update point 6's final sentence about emission order: "The recorder's setup cell recreates split-derived entries after file-backed lines, in registration-revision order (true temporal order even across overwrites; a split of a split works because chains register in order)."
- §5.11 `fit_model` (~line 537), the `training_dataset_hash` paragraph: append "The model also captures the dataset entry's registration `revision` and its fit-time loader identity `{path, format, read_options}`; at replay the setup cell re-fits only when the current entry is provably the same table state (matching revision, or the carried base loader's revision for pre-overwrite file-backed fits, or an identical-content *and* identical-loader reload) — anything else fails the setup cell with a loud `AssertionError`."
- §5.13 `emit_notebook` behavior (~line 674): in the setup-cell bullet, mention per-side split checksums and the revision-ordered second pass.

- [ ] **Step 6: evals/README.md**

The table (~line 27) is stale (omits `eval_split_cv.py`). Add its row:

```markdown
| `eval_split_cv.py` | 3 | split → fit → evaluate → cross_validate emitted-notebook round-trip via nbconvert: clean session exits 0, mutated source CSV fails loudly |
```

(Verify the count with `uv run pytest evals/eval_split_cv.py --collect-only -q` and use the real number.) Refresh the `Total: N evals.` line from `uv run pytest evals/ --collect-only -q | tail -1`.

- [ ] **Step 7: recorder docstrings**

In `src/data_analyst_mcp/recorder.py`, update `_build_setup_source`'s docstring (point 2 of the numbered list, ~line 301): replace "emit a SHA-256 assert against the training file and a `smf.<kind>(...).fit(disp=False)` rehydration line. ... If the model's training dataset was later overwritten by `materialize_query`, the guard and the re-fit both target the carried base loader (original file) instead of the post-transform table." with a version that states the revision dispatch: re-fit happens only when the current entry matches the fit-time revision (normal path), or the carried base loader's revision (guarded original-file re-fit), or is a provably-identical file reload (equal content hash and loader identity); every other replacement emits a loud `raise AssertionError` line.

- [ ] **Step 8: Commit docs**

```bash
git add ROADMAP.md CHANGELOG.md README.md docs/SPEC.md evals/README.md src/data_analyst_mcp/recorder.py
git commit -m "docs: replay-guard hardening — revisions, loader identity, per-side split replay"
```

- [ ] **Step 9: Version bump 1.4.0**

- `pyproject.toml:3`: `version = "1.4.0"`
- `src/data_analyst_mcp/__init__.py:3`: `__version__ = "1.4.0"`
- `tests/test_smoke.py`: `assert data_analyst_mcp.__version__ == "1.4.0"`
- Refresh the lockfile's own version record: `uv lock` (then `git diff uv.lock` — expect only the project's version line).

- [ ] **Step 10: Final gates + release commit**

```bash
uv run pytest tests/ && uv run pytest evals/
uv run ruff format --check . && uv run ruff check . && uv run pyright src/
uv run python scripts/check_tdd_commits.py
git add pyproject.toml src/data_analyst_mcp/__init__.py tests/test_smoke.py uv.lock
git commit -m "chore: release 1.4.0"
```

Expected: everything green; commit lands.

---

## Verification (whole plan)

- Behavior matrix spot-check against the spec: S4/S5/S7b/S9/S10/S11/S12/S13/S15b/S16 each have a dedicated test above; S1/S2/S3/S6/S7a/S8/S14/S15/S15c/S17 are pinned by existing or new pin tests. If any matrix row lacks a passing test at the end, that's a plan execution failure — add the missing test before release.
- `git log --oneline` should show 12 red/green pairs with matching suffixes, then docs + release commits; `scripts/check_tdd_commits.py` enforces this mechanically.
