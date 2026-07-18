# Prefix Replay Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every `load_dataset` recorder cell asserts its own load-time source hash before its `CREATE`, and `cross_validate` / ephemeral `fit_model` cells recorded against in-memory datasets get an explanatory `raise AssertionError` prefix — closing the ROADMAP "Ephemeral-fit replay provenance" failure class per `docs/proposals/2026-07-18-ephemeral-fit-replay-provenance.md`.

**Architecture:** Emitted notebooks replay the setup cell first, then every recorded cell in call order; historical `load_dataset` cells re-read files unguarded, which is the drift vector. Fix: a new public `load_guard_lines()` helper in `recorder.py` (wrapping the existing private `_hash_guard_lines` + `_sanitized_guard_var`) is prepended to each load cell at record time, anchored to that load's own `source_hash`. The in-memory raise prefixes are stamped at record time in `_record_cross_validate` / `_record_fit_model` — no recorder API change, no emit-time step, setup cell untouched.

**Tech Stack:** Python 3.13, uv, FastMCP, DuckDB, pandas, pytest, jupyter nbconvert (evals), ruff 0.15.12 (pinned), pyright strict.

## Global Constraints

- TDD commit discipline: every commit touching `src/` needs a preceding failing-test commit. Subjects: `red: <behavior>` then `green: <behavior>` — **identical `<behavior>` text**, enforced by `uv run python scripts/check_tdd_commits.py`. Test-only / docs-only commits use `test:` / `docs:` prefixes (ignored by the checker).
- Gates that must pass at the end of every task: `uv run pytest tests/ -q`, `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright src/`. Evals (`uv run pytest evals/ -q`) run in Tasks 5–7.
- The setup cell (`_build_setup_source`) must not change: `tests/test_recorder.py` must pass untouched in Tasks 1–4.
- Do not add cell metadata, recorder parameters, or emit-time resolution anywhere — the spec explicitly forbids them.
- Spec is the source of truth: `docs/proposals/2026-07-18-ephemeral-fit-replay-provenance.md`. When in doubt, the spec wins.
- Commit trailer on every commit: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Per-load drift guards in `load_dataset` cells

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (add `load_guard_lines` right after `_hash_guard_lines`, ~line 269)
- Modify: `src/data_analyst_mcp/tools/datasets.py:624-631` (the `code = (...)` block in `load_dataset`)
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: existing private `_hash_guard_lines(var, display_name, path, hash_val) -> list[str]` and `_sanitized_guard_var(name, idx) -> str` in `recorder.py`; `session.get_datasets()`; `get_recorder().cells`.
- Produces: `load_guard_lines(*, name: str, path: str, source_hash: str, ordinal: int) -> list[str]` — public, imported by `datasets.py`. Later tasks rely on the emitted guard-variable stems being `<sanitized_name>_<ordinal>` where `ordinal` is `len(get_recorder().cells)` at record time (0 for the first load, 2 for the second, …).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_datasets.py`:

```python
def test_load_dataset_cell_asserts_load_time_content_hash(call_tool: Any) -> None:
    """Spec: prefix replay guards, mechanism 1. The load cell must assert the
    file's SHA-256 as captured at THIS load, before its CREATE, so a file
    mutated after the load fails replay at this cell."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    src = get_recorder().cells[1]["source"]
    expected = hashlib.sha256(open(MESSY_CSV, "rb").read()).hexdigest()
    assert f"expected_hash_ds_messy_0 = '{expected}'" in src
    assert "actual_hash_ds_messy_0 = hashlib.sha256(open(" in src
    assert (
        "assert actual_hash_ds_messy_0 == expected_hash_ds_messy_0" in src
    )
    # The assert must precede the CREATE so drift fails before recreation.
    assert src.index("assert actual_hash_ds_messy_0") < src.index(
        "CREATE OR REPLACE TABLE"
    )


def test_load_dataset_cell_uses_fallback_recompute_above_ceiling(
    call_tool: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """Above HASH_CONTENT_CEILING_BYTES the entry stores a fallback:(path,
    mtime,size) digest; the load cell must recompute and assert that same
    fallback instead of a content hash."""
    from data_analyst_mcp import provenance
    from data_analyst_mcp.recorder import get_recorder

    csv_path = tmp_path / "big.csv"
    csv_path.write_text("x,y\n1,2\n3,4\n")
    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 1)

    call_tool("load_dataset", {"path": str(csv_path), "name": "big"})

    src = get_recorder().cells[1]["source"]
    assert "expected_hash_ds_big_0 = 'fallback:" in src
    assert "_st = _os.stat(" in src
    assert "'fallback:' + hashlib.sha256(" in src
    assert "assert actual_hash_ds_big_0 == expected_hash_ds_big_0" in src


def test_load_dataset_cell_emits_comment_for_sentinel_sources(
    call_tool: Any, monkeypatch: Any
) -> None:
    """Remote (s3/http) sources hash to a sentinel; the load cell must emit
    the unguarded-reload comment and no assert, mirroring the setup cell."""
    from data_analyst_mcp import provenance
    from data_analyst_mcp.recorder import get_recorder

    # Force the sentinel shape without touching the network: the loader
    # still reads the real local file; only the registration hash is faked.
    # session.py binds the function at import (`from ... import
    # compute_source_hash`), so patch the session-module binding.
    def _sentinel(path: str) -> str:
        return f"sentinel:no-file:{path}"

    monkeypatch.setattr(
        "data_analyst_mcp.session.compute_source_hash", _sentinel
    )
    del provenance  # imported only to document where the real hasher lives

    call_tool("load_dataset", {"path": MESSY_CSV, "name": "remote"})

    src = get_recorder().cells[1]["source"]
    assert "no verifiable source hash; reload is unguarded" in src
    assert "expected_hash_ds_remote_0" not in src
    assert "assert actual_hash" not in src
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_datasets.py -q -k "load_time_content_hash or fallback_recompute or sentinel_sources"`
Expected: 3 FAILED (assertions on missing guard lines; current cells contain only the `CREATE`).

- [ ] **Step 3: Commit the red**

```bash
git add tests/test_datasets.py
git commit -m "red: load_dataset cells carry per-load drift guards

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Add `load_guard_lines` to `recorder.py`** — insert directly after `_hash_guard_lines` (after its closing `return`, ~line 269):

```python
def load_guard_lines(*, name: str, path: str, source_hash: str, ordinal: int) -> list[str]:
    """Drift-guard lines for one ``load_dataset`` per-call cell.

    Anchored to THIS load's ``source_hash`` — not the registry's latest —
    so an earlier load of since-mutated bytes fails replay at its own cell
    even when the setup cell's latest-registration assert passes. Shapes
    (content / fallback / sentinel comment) are ``_hash_guard_lines``'s;
    ``ordinal`` is the recorder cell index at record time, keeping stems
    unique across repeated loads of one name.
    """
    return _hash_guard_lines(_sanitized_guard_var(name, ordinal), name, path, source_hash)
```

- [ ] **Step 5: Wire it into `load_dataset`** — in `src/data_analyst_mcp/tools/datasets.py`, change the import line and the `code = (...)` block:

Import (line 14): `from data_analyst_mcp.recorder import get_recorder` → `from data_analyst_mcp.recorder import get_recorder, load_guard_lines`

Replace the `code = (...)` block (currently lines 624–631):

```python
    entry = session.get_datasets()[name]
    guard_lines = load_guard_lines(
        name=name,
        path=entry.path,
        source_hash=entry.source_hash,
        ordinal=len(get_recorder().cells),
    )
    create_block = (
        f'con.execute("""\n'
        f"    CREATE OR REPLACE TABLE {name} AS\n"
        f"    SELECT * FROM {read_call}\n"
        f'""")\n'
        f'{name}_df = con.sql("SELECT * FROM {name}").df()\n'
        f"{name}_df.head()"
    )
    code = "\n".join([*guard_lines, create_block])
```

- [ ] **Step 6: Run the new tests and the full unit suite**

Run: `uv run pytest tests/test_datasets.py -q && uv run pytest tests/ -q`
Expected: all PASS — including `tests/test_recorder.py` untouched (setup cell unchanged) and `test_load_dataset_records_markdown_and_code_cell_pair` (it only asserts `CREATE OR REPLACE TABLE` is present, which still holds).

- [ ] **Step 7: Gates**

Run: `uv run ruff format --check . && uv run ruff check . && uv run pyright src/`
Expected: clean. If `ruff format` complains about the edited blocks, run `uv run ruff format src/data_analyst_mcp/` and re-check.

- [ ] **Step 8: Commit the green**

```bash
git add src/data_analyst_mcp/recorder.py src/data_analyst_mcp/tools/datasets.py
git commit -m "green: load_dataset cells carry per-load drift guards

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Characterization tests — reloads and stem uniqueness

No `src/` change: Task 1's ordinal-based stems already give each load cell its own hash and a unique stem. These tests pin the two properties the spec's slices 4–5 name, so a future refactor cannot silently lose them.

**Files:**
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: guard-line shapes and `<name>_<ordinal>` stems from Task 1.
- Produces: nothing new — regression pins only.

- [ ] **Step 1: Write the passing characterization tests** — append to `tests/test_datasets.py`:

```python
def test_reloaded_name_keeps_each_loads_own_hash(call_tool: Any, tmp_path: Any) -> None:
    """Each load cell asserts the bytes THAT load saw. After a mutate-and-
    reload, the first cell must still carry the pre-mutation hash — this is
    the mechanism that fails the ROADMAP scenario at the first load cell."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n3,4\n")
    hash_v1 = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    call_tool("load_dataset", {"path": str(csv_path), "name": "data"})
    csv_path.write_text("x,y\n9,9\n3,4\n")
    hash_v2 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    call_tool("load_dataset", {"path": str(csv_path), "name": "data"})

    cells = get_recorder().cells
    assert f"expected_hash_ds_data_0 = '{hash_v1}'" in cells[1]["source"]
    assert f"expected_hash_ds_data_2 = '{hash_v2}'" in cells[3]["source"]
    assert hash_v1 != hash_v2


def test_guard_stems_unique_across_loads_of_same_name(call_tool: Any, tmp_path: Any) -> None:
    """Two loads of one name must not share a guard-variable stem — the
    ordinal (recorder cell index at record time) disambiguates."""
    from data_analyst_mcp.recorder import get_recorder

    csv_path = tmp_path / "d.csv"
    csv_path.write_text("a\n1\n")
    call_tool("load_dataset", {"path": str(csv_path), "name": "d"})
    call_tool("load_dataset", {"path": str(csv_path), "name": "d"})

    first = get_recorder().cells[1]["source"]
    second = get_recorder().cells[3]["source"]
    assert "expected_hash_ds_d_0" in first
    assert "expected_hash_ds_d_2" in second
    assert "expected_hash_ds_d_0" not in second


def test_fallback_guard_lines_abort_on_drift_when_executed(
    call_tool: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """Execute the emitted fallback guard block against a since-drifted
    file: the assert must raise. Pins that the fallback shape is a working
    guard, not just right-looking text (spec slice 6)."""
    import hashlib

    import pytest as _pytest

    from data_analyst_mcp import provenance
    from data_analyst_mcp.recorder import get_recorder

    csv_path = tmp_path / "big.csv"
    csv_path.write_text("x,y\n1,2\n3,4\n")
    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 1)
    call_tool("load_dataset", {"path": str(csv_path), "name": "big"})

    src = get_recorder().cells[1]["source"]
    guard_block = src.split('con.execute("""')[0]
    assert "assert actual_hash_ds_big_0" in guard_block
    # Drift: append a row — size changes, so the fallback digest changes.
    csv_path.write_text("x,y\n1,2\n3,4\n5,6\n")

    with _pytest.raises(AssertionError):
        exec(guard_block, {"hashlib": hashlib})  # noqa: S102 — our own emitted guard
```

Note: no dedicated "setup cell unchanged" characterization test — none can compare across versions; the spec pins that property via the untouched `tests/test_recorder.py` suite instead. Guard-stem overlap with the setup cell's own `guard_idx` stems (both can emit `messy_0`) is harmless: every emitted shape assigns immediately before asserting.

- [ ] **Step 2: Run them — they must already pass**

Run: `uv run pytest tests/test_datasets.py -q -k "reloaded_name_keeps or stems_unique or abort_on_drift"`
Expected: 3 PASS. (If any fails, Task 1 was implemented wrong — fix Task 1, do not adjust these tests.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_datasets.py
git commit -m "test: pin per-load hashes and stem uniqueness across reloads

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `cross_validate` cells on in-memory datasets raise at replay

**Files:**
- Modify: `src/data_analyst_mcp/tools/crossval.py:448-459` (`_record_cross_validate`)
- Test: `tests/test_crossval.py`

**Interfaces:**
- Consumes: `session.get_datasets()` (already imported in `crossval.py`); `DatasetEntry.format == "dataframe"` marks in-memory registrations; `load_df_into_session` fixture from `tests/conftest.py`.
- Produces: the raise-prefix contract Task 4 mirrors — first line `raise AssertionError('The <tool> call in this cell ran on in-memory dataset ...')`, original computation retained below.

- [ ] **Step 1: Write the failing test** — append to `tests/test_crossval.py`:

```python
def test_cv_cell_on_dataframe_dataset_gets_raise_prefix(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Spec: prefix replay guards, mechanism 2. In-memory datasets are never
    recreated at replay (setup emits only a comment), so the CV cell must
    open with an explanatory raise; the computation stays below as the
    audit trail."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    df = pd.DataFrame({"y": [float(i % 7) for i in range(30)], "x": [float(i) for i in range(30)]})
    load_df_into_session("mem", df)

    result = call_tool(
        "cross_validate", {"name": "mem", "formula": "y ~ x", "kind": "ols", "k": 3}
    )
    assert result["ok"] is True

    src = get_recorder().cells[-1]["source"]
    first_line = src.splitlines()[0]
    assert first_line.startswith("raise AssertionError(")
    assert "cross_validate" in first_line
    assert "'mem'" in first_line or '"mem"' in first_line
    assert "in-memory" in first_line
    # Original computation retained below the raise.
    assert "_cv_df = con.sql(" in src


def test_cv_cell_on_file_dataset_has_no_raise_prefix(
    call_tool: Any, tmp_path: Any
) -> None:
    """File-backed sources replay via guarded load cells — no prefix."""
    from data_analyst_mcp.recorder import get_recorder

    csv_path = tmp_path / "file_backed.csv"
    csv_path.write_text(
        "y,x\n" + "\n".join(f"{(i * 7) % 13}.0,{i}.0" for i in range(30)) + "\n"
    )
    call_tool("load_dataset", {"path": str(csv_path), "name": "fb"})
    result = call_tool(
        "cross_validate", {"name": "fb", "formula": "y ~ x", "kind": "ols", "k": 3}
    )
    assert result["ok"] is True
    assert not get_recorder().cells[-1]["source"].startswith("raise AssertionError(")
```

- [ ] **Step 2: Run to verify the first fails**

Run: `uv run pytest tests/test_crossval.py -q -k "raise_prefix"`
Expected: `test_cv_cell_on_dataframe_dataset_gets_raise_prefix` FAILED (no raise prefix today); the file-backed test PASSES.

- [ ] **Step 3: Commit the red**

```bash
git add tests/test_crossval.py
git commit -m "red: cross_validate cells on in-memory datasets raise at replay

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement** — in `_record_cross_validate` (crossval.py), replace `code = _cv_cell_source(payload)`:

```python
    code = _cv_cell_source(payload)
    entry = session.get_datasets().get(payload.name)
    if entry is not None and entry.format == "dataframe":
        msg = (
            f"The cross_validate call in this cell ran on in-memory dataset "
            f"{payload.name!r}; in-memory datasets are not recreated at "
            f"replay, so this cell cannot replay faithfully."
        )
        code = f"raise AssertionError({msg!r})\n{code}"
```

- [ ] **Step 5: Run tests and gates**

Run: `uv run pytest tests/test_crossval.py -q && uv run pytest tests/ -q && uv run ruff format --check . && uv run ruff check . && uv run pyright src/`
Expected: all PASS / clean.

- [ ] **Step 6: Commit the green**

```bash
git add src/data_analyst_mcp/tools/crossval.py
git commit -m "green: cross_validate cells on in-memory datasets raise at replay

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Ephemeral `fit_model` cells on in-memory datasets raise at replay

**Files:**
- Modify: `src/data_analyst_mcp/tools/models.py:281-300` (`_record_fit_model`)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: the raise-prefix contract from Task 3 (same message shape, `fit_model` in place of `cross_validate`); `payload.model_name` distinguishes ephemeral (None) from registered fits.
- Produces: nothing further — registered fits stay untouched (transitively guarded by the setup-cell model block).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_models.py`:

```python
def test_ephemeral_fit_cell_on_dataframe_dataset_gets_raise_prefix(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Spec: prefix replay guards, mechanism 2 — fit_model WITHOUT model_name
    on an in-memory dataset opens with an explanatory raise."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    load_df_into_session("mem", df)

    result = call_tool("fit_model", {"name": "mem", "formula": "y ~ x", "kind": "ols"})
    assert result["ok"] is True

    src = get_recorder().cells[-1]["source"]
    first_line = src.splitlines()[0]
    assert first_line.startswith("raise AssertionError(")
    assert "fit_model" in first_line
    assert "in-memory" in first_line
    assert "df = con.sql(" in src  # original computation retained


def test_registered_fit_cell_on_dataframe_dataset_unchanged(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Registered fits are guarded by the setup-cell model block; their
    per-call cell must NOT gain the prefix (scope decision in the spec)."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    load_df_into_session("mem", df)

    result = call_tool(
        "fit_model",
        {"name": "mem", "formula": "y ~ x", "kind": "ols", "model_name": "m1"},
    )
    assert result["ok"] is True
    assert not get_recorder().cells[-1]["source"].startswith("raise AssertionError(")
```

- [ ] **Step 2: Run to verify the first fails**

Run: `uv run pytest tests/test_models.py -q -k "raise_prefix or dataframe_dataset_unchanged"`
Expected: ephemeral test FAILED, registered test PASSES.

- [ ] **Step 3: Commit the red**

```bash
git add tests/test_models.py
git commit -m "red: ephemeral fit cells on in-memory datasets raise at replay

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement** — in `_record_fit_model` (models.py), replace `code = _code_for_fit(payload)`:

```python
    code = _code_for_fit(payload)
    if payload.model_name is None:
        entry = session.get_datasets().get(payload.name)
        if entry is not None and entry.format == "dataframe":
            msg = (
                f"The fit_model call in this cell ran on in-memory dataset "
                f"{payload.name!r}; in-memory datasets are not recreated at "
                f"replay, so this cell cannot replay faithfully."
            )
            code = f"raise AssertionError({msg!r})\n{code}"
```

- [ ] **Step 5: Run tests and gates**

Run: `uv run pytest tests/ -q && uv run ruff format --check . && uv run ruff check . && uv run pyright src/`
Expected: all PASS / clean. Watch `tests/test_models.py::test_fit_model_emitted_code_cell_matches_runtime_template` — it pins the ephemeral fit cell's template; it asserts the reload + smf lines are present, which they still are (below the prefix only for in-memory sources; that test uses `load_df_into_session`, so if it asserts the cell *starts* with the template, adjust the implementation is NOT the fix — the test loads via `load_df_into_session("duncan", ...)`, making it in-memory, so if it breaks, update that test to assert the template lines are present rather than positional, and fold the edit into this green commit with a note in the message body).

- [ ] **Step 6: Commit the green**

```bash
git add src/data_analyst_mcp/tools/models.py tests/test_models.py
git commit -m "green: ephemeral fit cells on in-memory datasets raise at replay

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Eval — mutate-and-reload fails replay at the first load cell

**Files:**
- Create: `evals/eval_replay_guards.py`

**Interfaces:**
- Consumes: `conftest.PROJECT_ROOT`, `call`, `mcp_session` from `evals/conftest.py`; the `_nbconvert` pattern from `evals/eval_split_cv.py` (copy the helper — evals are standalone files, no shared helper module).
- Produces: the end-to-end proof of the ROADMAP scenario. No later task consumes it.

- [ ] **Step 1: Write the eval**

```python
"""End-to-end evals for prefix replay guards (spec: prefix-guard realization).

The ROADMAP failure class: load → cross_validate/fit → mutate the source
CSV → reload → emit. The setup cell's latest-registration assert passes,
but the FIRST load cell's own hash assert must fail replay loudly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import PROJECT_ROOT, call, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"


def _nbconvert(nb_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv", "run", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--inplace", str(nb_path),
            "--ExecutePreprocessor.timeout=180",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("y,x\n" + "\n".join(rows) + "\n")


@pytest.mark.eval
async def eval_mutated_and_reloaded_source_fails_replay_at_first_load_cell():
    """load → cross_validate → edit CSV → reload → emit → nbconvert must
    fail with the load-cell drift message, not silently recompute."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_source.csv"
    nb_path = ARTIFACTS / "eval_replay_guards.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(
            s, "cross_validate", {"name": "g", "formula": "y ~ x", "kind": "ols", "k": 4}
        )
        assert r["ok"] is True
        # Mutate the source, then reload — setup will assert the NEW hash.
        _write_csv(csv_path, [f"{(i * 5) % 11}.0,{i}.0" for i in range(40)])
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode != 0, "replay must fail loudly, not recompute silently"
    combined = proc.stdout + proc.stderr
    assert "changed since the session was recorded" in combined
    assert "'g'" in combined or '"g"' in combined


@pytest.mark.eval
async def eval_mutated_and_reloaded_source_fails_replay_for_ephemeral_fit():
    """Sibling case: ephemeral fit_model instead of cross_validate."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_source_fit.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_fit.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "fit_model", {"name": "g", "formula": "y ~ x", "kind": "ols"})
        assert r["ok"] is True
        _write_csv(csv_path, [f"{(i * 5) % 11}.0,{i}.0" for i in range(40)])
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode != 0
    assert "changed since the session was recorded" in (proc.stdout + proc.stderr)


@pytest.mark.eval
async def eval_stable_source_ephemeral_fit_replays_cleanly():
    """Success sibling (spec slice 9): an untouched source with an ephemeral
    fit must round-trip through nbconvert with exit 0 — the guards must not
    fire on a faithful session."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_stable.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_stable.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "fit_model", {"name": "g", "formula": "y ~ x", "kind": "ols"})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

`emit_notebook` accepts an optional `path` (`src/data_analyst_mcp/tools/notebook.py:30`), so the calls above are valid as written.

- [ ] **Step 2: Run the eval**

Run: `uv run pytest evals/eval_replay_guards.py -q`
Expected: 3 PASS (~30-60 s; nbconvert spawns kernels).

- [ ] **Step 3: Commit**

```bash
git add evals/eval_replay_guards.py
git commit -m "test: eval — mutated-and-reloaded source fails replay at its load cell

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Eval — faithful replacement histories replay cleanly

Pins the design-review false-positive counterexamples as regressions: replacement-after-fit histories are *faithful* under prefix replay and must exit 0.

**Files:**
- Modify: `evals/eval_replay_guards.py` (append)

**Interfaces:**
- Consumes: `_nbconvert`, `_write_csv`, `ARTIFACTS` from Task 5's file.
- Produces: nothing — regression pins only.

- [ ] **Step 1: Append the evals**

```python
@pytest.mark.eval
async def eval_rematerialize_after_cv_replays_cleanly():
    """materialize S1 → CV(d) → materialize S2 (overwrite) is faithful at
    replay (the prefix recreates S1 before the CV cell) and must exit 0 —
    the first-draft dispatch wrongly raised here; pin the fix.

    Constraint (spec slice 10): BOTH recipes read the stable surviving
    file-backed `base` — a self-referential S2 (SELECT ... FROM d) would
    hit the parked S4b setup failure instead and invalidate the pin."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_base.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_remat.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "base"})
        assert r["ok"] is True
        r = await call(
            s, "materialize_query",
            {"sql": "SELECT y, x FROM base WHERE x < 30", "name": "d"},
        )
        assert r["ok"] is True
        r = await call(
            s, "cross_validate", {"name": "d", "formula": "y ~ x", "kind": "ols", "k": 3}
        )
        assert r["ok"] is True
        r = await call(
            s, "materialize_query",
            {"sql": "SELECT y, x FROM base", "name": "d", "overwrite": True},
        )
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.eval
async def eval_resplit_after_cv_replays_cleanly():
    """split(seed=1) → CV(train) → split(seed=2, overwrite) under the same
    names: the prefix recreates the seed-1 sides (own checksums) before the
    CV cell, then the seed-2 sides after. Must exit 0."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_split.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_resplit.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(60)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "src"})
        assert r["ok"] is True
        r = await call(s, "split_dataset", {"name": "src", "seed": 1})
        assert r["ok"] is True
        r = await call(
            s, "cross_validate",
            {"name": "src_train", "formula": "y ~ x", "kind": "ols", "k": 3},
        )
        assert r["ok"] is True
        r = await call(s, "split_dataset", {"name": "src", "seed": 2, "overwrite": True})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

Both tools take `overwrite: true` for same-name replacement and `split_dataset` outputs `<name>_train` / `<name>_test` (confirmed in `src/data_analyst_mcp/recorder.py`'s overwrite comment and the README's `split_dataset` example) — the calls above are valid as written.

- [ ] **Step 2: Run the full eval file and unit suite**

Run: `uv run pytest evals/eval_replay_guards.py -q && uv run pytest tests/ -q`
Expected: 5 PASS in the eval file; unit suite green.

- [ ] **Step 3: Commit**

```bash
git add evals/eval_replay_guards.py
git commit -m "test: eval — faithful replacement histories replay cleanly

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Docs — SPEC, README, ROADMAP, proposal fold-in

**Files:**
- Modify: `docs/SPEC.md` (§5.1 Behavior, §5.11 recorder note, §5.11d recorder note, §6)
- Modify: `README.md` ("Known gotchas" drift bullet)
- Modify: `ROADMAP.md` (remove shipped item; add parked item; clear active-proposal line)
- Modify: `docs/proposals/README.md` (back to "No active proposals" + history)
- Delete: `docs/proposals/2026-07-18-ephemeral-fit-replay-provenance.md` (convention: fold and delete)

**Interfaces:**
- Consumes: shipped behavior from Tasks 1–6.
- Produces: docs consistent with the code; the proposal is folded into SPEC and deleted.

- [ ] **Step 1: SPEC §5.1** — append to the Behavior bullet list (after the provenance-hash bullet, currently SPEC line 217):

```markdown
- The recorder cell for each load asserts **that load's own** `source_hash` before its `CREATE` (content assert ≤ 100 MB; `(path, mtime, size)` fallback recompute above the ceiling; explanatory comment for `sentinel:` sources). The setup cell keeps asserting the *latest* registration only — the per-cell guards are what make an earlier load of a since-mutated file fail at replay instead of silently recomputing downstream cells.
```

- [ ] **Step 2: SPEC §5.11 and §5.11d** — append one sentence to each section's recorder-cell paragraph:

```markdown
When the source dataset is in-memory (`format == "dataframe"`), the recorded cell is prefixed with a `raise AssertionError(...)` explaining that the table is not recreated at replay; the computation is retained below the raise as the audit trail. (For `fit_model` this applies only when `model_name` is omitted — registered fits are guarded by the setup cell's model block.)
```

- [ ] **Step 3: SPEC §6** — after the "Setup cell (prepended automatically by `to_notebook`)" subsection, add:

```markdown
### Replay order and per-call load guards

`to_notebook` emits the setup cell first, then every recorded cell in call order. A per-call cell therefore reads the table state recreated by its *historical prefix* (the last state-recreating cell for that name before it), not the final session state — `load_dataset`, `materialize_query`, and `split_dataset` all record cells that re-execute their state change. Content fidelity along that prefix comes from each `load_dataset` cell asserting its own load-time hash (see §5.1); derived recipes re-execute over those guarded roots and split recreation asserts per-side membership checksums. Known residual gaps (parked in ROADMAP): row-order drift through order-independent checksums vs positional CV folds, and remote / above-ceiling sources whose hashes cannot prove content.
```

- [ ] **Step 4: README** — in "Known gotchas", extend the "**Emitted notebooks are drift-guarded.**" bullet's opening (keep the rest of the bullet as-is):

Replace: `The setup cell asserts a SHA-256 provenance hash for every file-backed dataset (and for base files behind `materialize_query` overwrites) before reloading it.`
With: `The setup cell asserts a SHA-256 provenance hash for every file-backed dataset (and for base files behind `materialize_query` overwrites) before reloading it, and since 1.5.0 every `load_dataset` cell in the notebook body re-asserts its own load-time hash — so a file mutated mid-session fails replay at the load that saw the old bytes, even when a later reload keeps the setup cell happy. `cross_validate` / unregistered `fit_model` cells recorded against in-memory datasets now open with an explanatory `raise AssertionError` instead of a bare `CatalogException`.`

- [ ] **Step 5: ROADMAP** — three edits:

1. Change the active-proposal line back to: `No active proposals (the prefix-replay-guards proposal shipped in 1.5.0 and was folded into SPEC §5.1 / §5.11 / §5.11d / §6; see docs/proposals/README.md).`
2. Delete the entire "**Ephemeral-fit replay provenance.**" item under § Reproducibility.
3. Add in its place:

```markdown
- **Row-order drift under order-independent checksums.** Split membership
  checksums and derived recipes tolerate row-order changes that preserve
  multisets, but CV fold assignment is positional: an order-permuting
  drift in a *derived* source (file roots are hash-guarded since 1.5.0)
  can change CV numbers while every existing assert passes. Closing it
  needs an order-sensitive digest. An emit-time re-hash of fit-time
  lineage was considered and dropped during the 1.5.0 design: strictly
  dominated by the per-load-cell asserts, and a file edited then reverted
  before replay would bake in a false-positive raise.
- **Nondeterministic derived recipes.** `materialize_query` SQL containing
  `random()`, `current_timestamp`, or sampling re-evaluates at replay
  behind passing load guards and silently changes downstream numbers.
  Closing it needs a content digest captured at materialize time.
```

4. In the § Reproducibility caveat paragraph near the top (the 1.4.0 text reading "every file-backed dataset reload now carries its own assert too"), clarify the pre-existing sentence to "(in the setup cell)" so it cannot be misread as the 1.5.0 per-load-cell guards, and add a sentence: "Since 1.5.0 each `load_dataset` cell in the notebook body additionally asserts its own load-time hash."

- [ ] **Step 6: proposals README + delete the proposal**

In `docs/proposals/README.md`, replace the current-proposals bullet with `No active proposals.` and append to the history paragraph: `The prefix-replay-guards proposal (2026-07-18) shipped in 1.5.0 and was folded into SPEC §5.1 / §5.11 / §5.11d / §6.`

```bash
git rm docs/proposals/2026-07-18-ephemeral-fit-replay-provenance.md
```

- [ ] **Step 7: Full gates, then commit**

Run: `uv run pytest tests/ -q && uv run pytest evals/ -q && uv run ruff format --check . && uv run ruff check . && uv run pyright src/ && uv run python scripts/check_tdd_commits.py`
Expected: everything green.

```bash
git add docs/SPEC.md README.md ROADMAP.md docs/proposals/README.md
git commit -m "docs: fold prefix replay guards into SPEC; retire the proposal

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Release 1.5.0

**Files:**
- Modify: `pyproject.toml:3` (`version = "1.4.0"` → `version = "1.5.0"`)
- Modify: `CHANGELOG.md` (new entry above `## [1.4.0]`)

**Interfaces:**
- Consumes: everything shipped in Tasks 1–7.
- Produces: the released version. **Ask the user before executing this task** — tagging/releasing is their call; stop here if unconfirmed.

- [ ] **Step 1: CHANGELOG entry** — insert above the `## [1.4.0]` heading:

```markdown
## [1.5.0] - 2026-07-18

Prefix replay guards: emitted notebooks replay setup **then** the full
recorded history, and historical `load_dataset` cells re-read files
unguarded — the last way to make a notebook silently recompute on drifted
data. Every load cell now asserts its own load-time hash, and
`cross_validate` / unregistered `fit_model` cells on in-memory datasets
open with an explanatory raise. Tool surface unchanged (24).

### Fixed
- A source file mutated and reloaded mid-session replayed the pre-mutation
  cells (`cross_validate`, ephemeral and registered `fit_model` inputs,
  every analytic cell) against the new bytes with exit code 0 — the setup
  cell only asserts each dataset's *latest* registration. Each
  `load_dataset` cell now carries its own content assert (fallback digest
  above 100 MB; explanatory comment for remote sources), so replay fails
  loudly at the first load that saw the old bytes.
- `cross_validate` and unregistered `fit_model` cells recorded against
  in-memory (dataframe-registered) datasets failed replay with a bare
  `duckdb.CatalogException`; they now open with a purpose-written
  `AssertionError` naming the tool and dataset.
```

- [ ] **Step 2: Bump version, verify, commit**

Run: `uv run pytest tests/ -q && uv run pytest evals/ -q`
Expected: green.

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release 1.5.0

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
