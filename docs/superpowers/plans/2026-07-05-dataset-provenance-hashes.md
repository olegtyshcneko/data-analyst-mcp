# Dataset Provenance Hashes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every dataset gets a content hash at load time; the emitted notebook's setup cell asserts on it (and renders `read_options`) so any source-file drift or parse divergence fails replay loudly; `fit_model` copies the load-time hash instead of re-hashing.

**Architecture:** A new leaf module `provenance.py` owns the hash function (moved from `tools/models.py`); `session.register()` computes `DatasetEntry.source_hash` for every entry; the recorder emits per-dataset guard lines before each file-backed `CREATE` (content / fallback / sentinel shapes, mirroring the existing model guard); a second leaf module `read_options.py` shares the option-fragment renderer between the live loader and the recorder.

**Tech Stack:** Python 3.13, uv, DuckDB, nbformat, pytest, ruff, pyright (strict on `src/`).

**Spec:** `docs/superpowers/specs/2026-07-05-dataset-provenance-hashes-design.md` — the source of truth. When this plan and the spec disagree, the spec wins.

## Global Constraints

- **TDD commit discipline (enforced by `scripts/check_tdd_commits.py`):** every behavior lands as two commits — `red: <behavior>` (failing test only) immediately followed by `green: <behavior>` (implementation) with the **identical** behavior suffix. `refactor:`, `test:`, `docs:`, `chore:` prefixes are exempt. Never commit a `green:` whose test wasn't seen failing first.
- **Gates that must pass at the end (and should be run after each task):**
  - `uv run pytest tests/` (fast; run per task)
  - `uv run pytest evals/` (slow ~30s; run at Task 13)
  - `uv run ruff format --check .` and `uv run ruff check .`
  - `uv run pyright src/` (strict — annotate everything in `src/`)
  - `uv run python scripts/check_tdd_commits.py`
- **No new MCP tools; no tool-response schema changes.** Tool count stays 22.
- **Hash semantics (copied from spec):** content SHA-256 streamed in 1 MB chunks up to the 100 MB ceiling; `fallback:<sha256 of "path|mtime|size">` above it; `sentinel:no-file:<path>` / `sentinel:stat-failed:<path>` / `sentinel:read-failed:<path>` when the path isn't a readable local file. `sentinel:`-prefixed values never produce an assert — they produce a comment.
- **Guard variable naming (from spec):** `expected_hash_ds_<var>` / `actual_hash_ds_<var>` where `<var>` = dataset name with non-identifier chars replaced by `_`, suffixed with the emission index (e.g. `expected_hash_ds_my_data_0`).
- **Assert message (exact):** `"Source file for dataset '<name>' changed since the session was recorded."` (name rendered with `{name!r}`, matching the model guard's style).
- **Path quoting in emitted SQL:** keep the recorder's existing `repr(path)` quoting (`_file_load_stmt`). Do NOT switch the recorder to `datasets._build_read_call` — it interpolates the path naively and would regress the repr-quoting behavior that has its own red/green history. Share only the options fragment.
- **Derived entries' `read_options` hold `{"sql": ...}` — never pass them to `_file_load_stmt`.** Only file-backed entries' and `base_loader`'s `read_options` are render targets.
- All file paths below are relative to the repo root `/home/oleg/projects/personal_projects/dataanalysis_mcp`.

---

### Task 1: Extract `provenance.py` (refactor, no behavior change)

**Files:**
- Create: `src/data_analyst_mcp/provenance.py`
- Create: `tests/test_provenance.py`
- Modify: `src/data_analyst_mcp/tools/models.py` (remove function + ceiling, lines 22–27 and 108–144; repoint call at line 218)
- Modify: `tests/test_model_registry.py` (remove the two hash tests at lines 117–133)

**Interfaces:**
- Consumes: nothing new.
- Produces: `provenance.compute_source_hash(path: str) -> str` and `provenance.HASH_CONTENT_CEILING_BYTES: int` — every later task uses these exact names. The ceiling constant is **public** (tests monkeypatch it; pyright-strict flags cross-module private access).

- [ ] **Step 1: Create the new module**

Create `src/data_analyst_mcp/provenance.py`:

```python
"""Source-file provenance hashing — the drift guard shared by datasets and models."""

from __future__ import annotations

import hashlib
import os

# Above this file size (bytes) we skip content-hashing in favour of a
# cheap ``(path, mtime, size)`` tuple — content-hash on a 5 GB CSV is
# slow enough that the pause is user-visible. Documented as a weaker
# drift guarantee in the provenance-hashes design spec.
HASH_CONTENT_CEILING_BYTES = 100 * 1024 * 1024


def compute_source_hash(path: str) -> str:
    """Hash a source file for the recorder's drift guard.

    Files up to ``HASH_CONTENT_CEILING_BYTES`` are content-hashed
    (SHA-256 of bytes). Larger files fall back to a cheap
    ``(path, mtime, size)`` tuple — a weaker guarantee. In-memory
    datasets (``path == "(dataframe)"``), derived datasets
    (``path == "(query)"``), and any other non-file path are tagged with
    a stable sentinel so the recorder can detect them and skip the hash
    assert without silently mismatching.
    """
    if not os.path.isfile(path):
        return f"sentinel:no-file:{path}"
    try:
        size = os.path.getsize(path)
    except OSError:
        return f"sentinel:stat-failed:{path}"
    if size <= HASH_CONTENT_CEILING_BYTES:
        h = hashlib.sha256()
        # Stream in 1 MB chunks; SHA-256 of a 100 MB file at ~500 MB/s is
        # under a quarter second on commodity hardware.
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    # Above the ceiling: fall back to (path, mtime, size). Weaker guarantee
    # — a careful edit that preserves mtime + size will not trigger the
    # drift assert — but content-hashing 5 GB is too slow for an
    # interactive session.
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    fallback_hash = hashlib.sha256(f"{path}|{mtime}|{size}".encode()).hexdigest()
    return f"fallback:{fallback_hash}"
```

(This is `tools/models.py::compute_training_dataset_hash` moved with the constant renamed public and the docstring updated. The `sentinel:read-failed` branch is **Task 2** — do not add it here.)

- [ ] **Step 2: Repoint `tools/models.py`**

In `src/data_analyst_mcp/tools/models.py`:
- Delete the `_HASH_CONTENT_CEILING_BYTES` constant (lines 22–27) and the whole `compute_training_dataset_hash` function (lines 108–144).
- Add to the imports: `from data_analyst_mcp.provenance import compute_source_hash`
- Change the `register_model` call site (was line 218):

```python
            training_dataset_hash=compute_source_hash(ds_path),
```

- Delete `import hashlib` (line 9). Check whether `os` is still used before deleting it: run `grep -n "os\." src/data_analyst_mcp/tools/models.py` — if the only hits were inside the deleted function, delete `import os` too; otherwise keep it.

- [ ] **Step 3: Move the two hash tests**

Create `tests/test_provenance.py`:

```python
"""Tests for provenance.compute_source_hash — the shared drift-guard hash."""

from __future__ import annotations

import hashlib
from pathlib import Path


def test_compute_source_hash_matches_sha256(tmp_path: Path) -> None:
    from data_analyst_mcp.provenance import compute_source_hash

    csv = tmp_path / "tiny.csv"
    csv.write_bytes(b"a,b\n1,2\n3,4\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    assert compute_source_hash(str(csv)) == expected


def test_compute_source_hash_handles_in_memory_dataset() -> None:
    """In-memory datasets (no file path) get a deterministic sentinel
    rather than throwing — the recorder cell uses this to skip the
    hash assert without silently mismatching."""
    from data_analyst_mcp.provenance import compute_source_hash

    h = compute_source_hash("(dataframe)")
    assert h.startswith("sentinel:")
```

Delete `test_compute_training_dataset_hash_matches_sha256` and `test_compute_training_dataset_hash_handles_in_memory_dataset` from `tests/test_model_registry.py` (lines 117–133). Run `uv run ruff check tests/test_model_registry.py` — if `hashlib` or `Path` imports became unused there, remove them.

- [ ] **Step 4: Verify no other references and run the suite**

Run: `grep -rn "compute_training_dataset_hash\|_HASH_CONTENT_CEILING_BYTES" src/ tests/ evals/`
Expected: no hits.

Run: `uv run pytest tests/ -x -q` — Expected: all pass.
Run: `uv run ruff format --check . && uv run ruff check . && uv run pyright src/` — Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/data_analyst_mcp/provenance.py src/data_analyst_mcp/tools/models.py tests/test_provenance.py tests/test_model_registry.py
git commit -m "refactor: extract provenance.compute_source_hash from tools.models"
```

---

### Task 2: `sentinel:read-failed` on mid-hash `OSError`

**Files:**
- Modify: `src/data_analyst_mcp/provenance.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Consumes: `provenance.compute_source_hash` (Task 1).
- Produces: the guarantee `compute_source_hash` never raises — later tasks call it inside `session.register()` with no try/except.

- [ ] **Step 1: Write the failing test** (append to `tests/test_provenance.py`)

```python
def test_compute_source_hash_returns_read_failed_sentinel_on_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    """A file that stats fine but fails to open/read (vanished mid-hash,
    permissions) must yield a sentinel, not an exception — hashing runs
    inside load_dataset's success path and must never add a failure mode."""
    import builtins

    from data_analyst_mcp.provenance import compute_source_hash

    csv = tmp_path / "vanish.csv"
    csv.write_bytes(b"a\n1\n")
    real_open = builtins.open

    def _raising_open(file: object, *args: object, **kwargs: object) -> object:
        if str(file) == str(csv):
            raise OSError("disappeared mid-hash")
        return real_open(file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", _raising_open)

    assert compute_source_hash(str(csv)) == f"sentinel:read-failed:{csv}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance.py::test_compute_source_hash_returns_read_failed_sentinel_on_oserror -q`
Expected: FAIL with `OSError: disappeared mid-hash`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_provenance.py
git commit -m "red: compute_source_hash returns read-failed sentinel when the content read raises"
```

- [ ] **Step 4: Implement**

In `provenance.py`, wrap the content-hash read:

```python
    if size <= HASH_CONTENT_CEILING_BYTES:
        h = hashlib.sha256()
        # Stream in 1 MB chunks; SHA-256 of a 100 MB file at ~500 MB/s is
        # under a quarter second on commodity hardware.
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
        except OSError:
            # Readable at load, unreadable now (vanished, permissions):
            # collapse to a sentinel — hashing must never raise.
            return f"sentinel:read-failed:{path}"
        return h.hexdigest()
```

- [ ] **Step 5: Run test to verify it passes, then commit green**

Run: `uv run pytest tests/test_provenance.py -q` — Expected: all pass.

```bash
git add src/data_analyst_mcp/provenance.py
git commit -m "green: compute_source_hash returns read-failed sentinel when the content read raises"
```

---

### Task 3: `DatasetEntry.source_hash` computed in `session.register()`

**Files:**
- Modify: `src/data_analyst_mcp/session.py:17-33` (dataclass) and `:113-131` (register)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `provenance.compute_source_hash` (Tasks 1–2).
- Produces: `DatasetEntry.source_hash: str` (default `"sentinel:unset"`), populated by `register()` from `path` for **every** caller — `load_dataset` (content hash), `materialize_query` (`"(query)"` → sentinel), dataframe fixtures (`"(dataframe)"` → sentinel). Tasks 5–10 read this field.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_session.py`; both fail together — the field doesn't exist — so they form one cycle)

```python
def test_register_records_content_hash_for_file_backed_dataset(tmp_path) -> None:
    import hashlib

    from data_analyst_mcp import session

    csv = tmp_path / "tiny.csv"
    csv.write_bytes(b"a,b\n1,2\n3,4\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    session.reset()
    session.register(
        name="tiny", path=str(csv), read_options={}, format="csv", rows=2, columns=[]
    )

    assert session.get_datasets()["tiny"].source_hash == expected


def test_register_records_sentinel_hash_for_non_file_paths() -> None:
    from data_analyst_mcp import session

    session.reset()
    session.register(
        name="derived", path="(query)", read_options={"sql": "SELECT 1"},
        format="derived", rows=1, columns=[],
    )
    session.register(
        name="mem", path="(dataframe)", read_options={}, format="dataframe",
        rows=1, columns=[],
    )

    assert session.get_datasets()["derived"].source_hash.startswith("sentinel:")
    assert session.get_datasets()["mem"].source_hash.startswith("sentinel:")
```

- [ ] **Step 2: Run to verify both fail**

Run: `uv run pytest tests/test_session.py -q`
Expected: the two new tests FAIL with `AttributeError: 'DatasetEntry' object has no attribute 'source_hash'`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_session.py
git commit -m "red: dataset registry records a source_hash for every entry"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/session.py` — add the import at top (after the existing imports):

```python
from data_analyst_mcp.provenance import compute_source_hash
```

Extend `DatasetEntry` (field order matters — defaults must trail non-defaults; keep it next to the other defaulted fields):

```python
    base_loader: dict[str, Any] | None = None
    # Content hash of the source file at registration time (the recorder's
    # drift-guard anchor). ``sentinel:``-prefixed when there is no
    # verifiable file. Default covers direct constructions in tests.
    source_hash: str = "sentinel:unset"
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

In `register()`, pass the computed hash:

```python
    _datasets[name] = DatasetEntry(
        path=path,
        read_options=dict(read_options),
        format=format,
        rows=rows,
        columns=list(columns),
        base_loader=dict(base_loader) if base_loader is not None else None,
        source_hash=compute_source_hash(path),
    )
```

- [ ] **Step 5: Run the full test file + suite, commit green**

Run: `uv run pytest tests/ -x -q` — Expected: all pass.

```bash
git add src/data_analyst_mcp/session.py
git commit -m "green: dataset registry records a source_hash for every entry"
```

---

### Task 4: `materialize_query` carries `source_hash` into `base_loader`

**Files:**
- Modify: `src/data_analyst_mcp/tools/materialize.py:113-123`
- Test: `tests/test_materialize.py`

**Interfaces:**
- Consumes: `DatasetEntry.source_hash` (Task 3).
- Produces: `base_loader["source_hash"]: str` — Task 9 reads it via `base.get("source_hash", "sentinel:unset")`. Derived-over-derived overwrites carry the whole dict forward unchanged (existing `base_loader = existing.base_loader` line), so chains need no extra code.

- [ ] **Step 1: Write the failing test** (append to `tests/test_materialize.py`)

```python
def test_overwrite_carries_source_hash_into_base_loader(call_tool, tmp_path) -> None:
    """Overwriting a file-backed dataset must retain the original file's
    load-time hash in base_loader so the emitted setup cell can guard the
    base reload; a second (derived-over-derived) overwrite carries it on."""
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "base.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    original_hash = session.get_datasets()["data"].source_hash
    assert not original_hash.startswith("sentinel:")

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 2 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    entry = session.get_datasets()["data"]
    assert entry.base_loader is not None
    assert entry.base_loader["source_hash"] == original_hash

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    entry = session.get_datasets()["data"]
    assert entry.base_loader is not None
    assert entry.base_loader["source_hash"] == original_hash
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_materialize.py::test_overwrite_carries_source_hash_into_base_loader -q`
Expected: FAIL with `KeyError: 'source_hash'`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_materialize.py
git commit -m "red: materialize_query carries the base file source_hash through overwrites"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/tools/materialize.py`, extend the `base_loader` dict (was lines 117–121):

```python
            base_loader = {
                "path": existing.path,
                "format": existing.format,
                "read_options": dict(existing.read_options),
                "source_hash": existing.source_hash,
            }
```

- [ ] **Step 5: Run tests, commit green**

Run: `uv run pytest tests/test_materialize.py -q` — Expected: all pass.

```bash
git add src/data_analyst_mcp/tools/materialize.py
git commit -m "green: materialize_query carries the base file source_hash through overwrites"
```

---

### Task 5: `fit_model` copies the load-time hash (the behavioral fix)

**Files:**
- Modify: `src/data_analyst_mcp/tools/models.py:209-220`
- Test: `tests/test_model_registry.py`

**Interfaces:**
- Consumes: `DatasetEntry.source_hash` (Task 3).
- Produces: `ModelEntry.training_dataset_hash` now equals the training dataset's **load-time** hash. The recorder's model-block emission (`recorder.py:146-181`) is untouched — it keeps reading `model_entry.training_dataset_hash` and emitting its assert.

- [ ] **Step 1: Write the failing test** (append to `tests/test_model_registry.py`)

```python
def test_fit_model_records_load_time_hash_not_fit_time(call_tool, tmp_path) -> None:
    """fit_model trains on the DuckDB table populated at load time, so its
    provenance hash must be the load-time file hash — not a re-hash of
    whatever the file contains at fit time."""
    import pandas as pd

    from data_analyst_mcp import session

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [1, 2, 3, 4, 5, 6]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    load_time_hash = session.get_datasets()["train"].source_hash

    # Edit the file after load, before fit. The in-session table is unchanged.
    with open(csv, "a") as fh:
        fh.write("7.0,7\n")

    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    assert session.get_models()["m"].training_dataset_hash == load_time_hash
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_model_registry.py::test_fit_model_records_load_time_hash_not_fit_time -q`
Expected: FAIL — the stored hash is the fit-time re-hash of the appended file, which differs from `load_time_hash`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_model_registry.py
git commit -m "red: fit_model records the load-time dataset hash without re-reading the file"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/tools/models.py` (was lines 209–220):

```python
    if result.get("ok") and payload.model_name is not None and live_result is not None:
        ds_entry = session.get_datasets()[payload.name]
        n_obs_val = int(result["fit"]["n_obs"])
        session.register_model(
            name=payload.model_name,
            kind=payload.kind,
            formula=payload.formula,
            fitted_on_dataset=payload.name,
            n_obs=n_obs_val,
            training_dataset_hash=ds_entry.source_hash,
            result=live_result,
        )
        result["model_name"] = payload.model_name
```

Remove the now-unused `from data_analyst_mcp.provenance import compute_source_hash` import (ruff will flag it).

- [ ] **Step 5: Run tests, commit green**

Run: `uv run pytest tests/ -x -q && uv run ruff check .` — Expected: clean. (The existing drift test `test_emitted_notebook_hash_assert_fires_when_training_csv_is_mutated` still passes: the file is mutated after fit, so load-time and fit-time hashes coincide there.)

```bash
git add src/data_analyst_mcp/tools/models.py
git commit -m "green: fit_model records the load-time dataset hash without re-reading the file"
```

---

### Task 6: Extract `read_options.py` (refactor, no behavior change)

**Files:**
- Create: `src/data_analyst_mcp/read_options.py`
- Modify: `src/data_analyst_mcp/tools/datasets.py:66-118`

**Interfaces:**
- Consumes: nothing new.
- Produces: `read_options.render_read_options_fragment(options: dict[str, Any]) -> str` — returns `""` for empty dict, else a leading `", key=value, ..."` fragment; raises `ValueError` on non-identifier keys and `TypeError` on unsupported value types. Task 7 imports it from `recorder.py`. This module imports only stdlib — it is a leaf; importing it from `recorder.py` at module level creates no cycle.

- [ ] **Step 1: Create the module**

Create `src/data_analyst_mcp/read_options.py`:

```python
"""Rendering of DuckDB ``read_*`` option fragments.

Shared by the live loader (``tools.datasets``) and the notebook recorder,
so the emitted setup cell reproduces exactly the options the live load
used. Keys are identifier-validated and values rendered as DuckDB
literals — SQL injection via the option dict stays impossible.
"""

from __future__ import annotations

import re
from typing import Any, cast

_READ_OPTION_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _render_read_option(value: Any) -> str:
    """Render a Python value as a DuckDB literal for the reader option list.

    Bools → ``TRUE``/``FALSE``; ints/floats unchanged; strings single-quoted
    with embedded quotes doubled; lists rendered as ``[a, b, c]`` (used for
    ``names``, ``columns``, etc.). Unknown shapes raise — the caller surfaces
    them as a ``bad_read_option`` error rather than producing broken SQL.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, list):
        items = cast(list[Any], value)
        return "[" + ", ".join(_render_read_option(v) for v in items) + "]"
    raise TypeError(f"unsupported read_option value type: {type(value).__name__}")


def render_read_options_fragment(options: dict[str, Any]) -> str:
    """Render ``read_options`` as a leading ``, key=value, ...`` fragment.

    Keys must be identifier-like (``[A-Za-z_][A-Za-z0-9_]*``) to keep SQL
    injection impossible via the option dict. Returns an empty string when
    ``options`` is empty.
    """
    if not options:
        return ""
    parts: list[str] = []
    for key, val in options.items():
        if not _READ_OPTION_KEY_RE.match(key):
            raise ValueError(f"read_options key {key!r} is not a valid identifier")
        parts.append(f"{key}={_render_read_option(val)}")
    return ", " + ", ".join(parts)
```

- [ ] **Step 2: Repoint `tools/datasets.py`**

Delete `_render_read_option`, `_READ_OPTION_KEY_RE`, and `_format_read_options` (lines 66–104). Add the import:

```python
from data_analyst_mcp.read_options import render_read_options_fragment
```

In `_build_read_call` (was line 113), change:

```python
    extra = render_read_options_fragment(read_options)
```

Check for test references: `grep -rn "_format_read_options\|_render_read_option" tests/ evals/` — if any test imports these privately from `tools.datasets`, repoint it to `data_analyst_mcp.read_options` in this commit.

- [ ] **Step 3: Run the suite and gates**

Run: `uv run pytest tests/ -x -q && uv run ruff format --check . && uv run ruff check . && uv run pyright src/`
Expected: all clean — behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add src/data_analyst_mcp/read_options.py src/data_analyst_mcp/tools/datasets.py
git commit -m "refactor: extract read_options rendering into a shared leaf module"
```

---

### Task 7: Setup cell renders `read_options`

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py:43-56` (`_file_load_stmt`) and call sites at `:100,103`
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `read_options.render_read_options_fragment` (Task 6).
- Produces: `_file_load_stmt(name: str, fmt: str, path: str, read_options: dict[str, Any] | None = None) -> str` — Task 9's base-loader emission passes `base.get("read_options")` through this same signature.

- [ ] **Step 1: Write the failing test** (append to `tests/test_recorder.py`)

```python
def test_setup_cell_renders_read_options_in_reload_statements() -> None:
    """The live load honored read_options; the replay reload must too,
    otherwise a passing hash can still parse differently (e.g. header or
    delimiter divergence). Covers both first-pass file-backed entries and
    base_loader re-creates."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    session.register(
        name="raw",
        path="fixtures/messy.csv",
        read_options={"header": False, "delim": ";"},
        format="csv",
        rows=10,
        columns=[],
    )
    session.register(
        name="data",
        path="(query)",
        read_options={"sql": "SELECT 1 AS a"},
        format="derived",
        rows=1,
        columns=[],
        base_loader={
            "path": "fixtures/messy.csv",
            "format": "csv",
            "read_options": {"delim": "|"},
            "source_hash": "sentinel:unset",
        },
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source

    assert "header=FALSE" in setup_src
    assert "delim=';'" in setup_src
    assert "delim='|'" in setup_src
    # Derived read_options ({"sql": ...}) must NOT leak into a reader call.
    assert "sql=" not in setup_src
    compile(setup_src, "<setup>", "exec")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_recorder.py::test_setup_cell_renders_read_options_in_reload_statements -q`
Expected: FAIL on `assert "header=FALSE" in setup_src`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: setup cell reload statements render the live load's read_options"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py` — add near the top (real import, not TYPE_CHECKING):

```python
from data_analyst_mcp.read_options import render_read_options_fragment
```

Replace `_file_load_stmt`:

```python
def _file_load_stmt(
    name: str, fmt: str, path: str, read_options: dict[str, Any] | None = None
) -> str:
    """Build the ``CREATE OR REPLACE TABLE`` line that reloads a file-backed
    dataset from disk via the format-appropriate DuckDB reader.

    ``repr()`` quotes the path safely — embedded ``'`` / ``"`` / ``\"\"\"`` no
    longer break out of the host literal. ``read_options`` is rendered via the
    same fragment builder the live load used, so replay parses identically.
    """
    reader = _FORMAT_TO_READER.get(fmt, "read_csv_auto")
    path_lit = repr(path)
    extra = render_read_options_fragment(read_options or {})
    if reader == "read_csv_auto":
        call = f"{reader}({path_lit}, SAMPLE_SIZE=-1{extra})"
    else:
        call = f"{reader}({path_lit}{extra})"
    return f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {call}"
```

Update both call sites in `_build_setup_source`:

```python
                stmt = _file_load_stmt(name, base["format"], base["path"], base.get("read_options"))
```

```python
        stmt = _file_load_stmt(name, entry.format, entry.path, entry.read_options)
```

(Do NOT touch the derived second-pass emission — its `read_options` hold `{"sql": ...}` and never reach `_file_load_stmt`.)

- [ ] **Step 5: Run tests, commit green**

Run: `uv run pytest tests/test_recorder.py tests/test_emit_notebook.py -q` — Expected: all pass.

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: setup cell reload statements render the live load's read_options"
```

---

### Task 8: Content-hash guard emission for file-backed datasets

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (new helper + first-pass emission)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `DatasetEntry.source_hash` (Task 3); `_file_load_stmt` (Task 7).
- Produces: `_hash_guard_lines(var: str, display_name: str, path: str, hash_val: str) -> list[str]` and `_sanitized_guard_var(name: str, idx: int) -> str` — Task 9 (fallback/sentinel shapes) and Task 10 (base-loader guards) extend/reuse these exact names. Guard variables: `expected_hash_ds_<var>` / `actual_hash_ds_<var>`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_recorder.py`; one cycle — both fail on the missing guard lines)

```python
def test_setup_cell_emits_source_hash_assert_for_file_backed_dataset(tmp_path) -> None:
    """Every file-backed dataset reload is preceded by a content-hash assert
    so an edited source file fails replay loudly (dataset-level counterpart
    of the model-block guard)."""
    import hashlib

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    csv = tmp_path / "tiny.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    session.reset()
    session.register(
        name="tiny", path=str(csv), read_options={}, format="csv", rows=1, columns=[]
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source

    assert f"expected_hash_ds_tiny_0 = '{expected}'" in setup_src
    assert "actual_hash_ds_tiny_0 = hashlib.sha256(" in setup_src
    assert (
        "assert actual_hash_ds_tiny_0 == expected_hash_ds_tiny_0, "
        "\"Source file for dataset 'tiny' changed since the session was recorded.\""
        in setup_src
    )
    # Guard precedes the reload it protects.
    assert setup_src.index("expected_hash_ds_tiny_0") < setup_src.index(
        "CREATE OR REPLACE TABLE tiny"
    )
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_guard_variable_sanitizes_non_identifier_names(tmp_path) -> None:
    """Dataset names are not validated as Python identifiers (load_dataset
    defaults the name from the file basename, so ``my-data.csv`` → ``my-data``).
    Guard variables sanitize the name and add an emission index so the setup
    cell stays valid Python and collision-free."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    csv = tmp_path / "my-data.csv"
    csv.write_bytes(b"a\n1\n")

    session.reset()
    session.register(
        name="my-data", path=str(csv), read_options={}, format="csv", rows=1, columns=[]
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert "expected_hash_ds_my_data_0" in setup_src
    compile(setup_src, "<setup>", "exec")
```

- [ ] **Step 2: Run to verify both fail**

Run: `uv run pytest tests/test_recorder.py -q -k "source_hash_assert or sanitizes_non_identifier"`
Expected: 2 FAIL — no guard lines are emitted yet.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: setup cell emits a content-hash assert before each file-backed dataset reload"
```

- [ ] **Step 4: Implement**

In `src/data_analyst_mcp/recorder.py` — add `import re` at the top, then add above `_build_setup_source`:

```python
_IDENT_SANITIZE_RE = re.compile(r"\W")


def _sanitized_guard_var(name: str, idx: int) -> str:
    """Guard-variable stem for a dataset: sanitized name + emission index.

    Dataset names are not validated as Python identifiers, and two names may
    sanitize identically — the index keeps the variables collision-free.
    """
    return f"{_IDENT_SANITIZE_RE.sub('_', name)}_{idx}"


def _hash_guard_lines(var: str, display_name: str, path: str, hash_val: str) -> list[str]:
    """Drift-guard lines emitted before one file-backed dataset reload.

    Content hashes get a hard assert. (Fallback and sentinel shapes are added
    in later cycles.)
    """
    message = (
        f'"Source file for dataset {display_name!r} changed since the session was recorded."'
    )
    return [
        f"expected_hash_ds_{var} = {hash_val!r}",
        f"actual_hash_ds_{var} = hashlib.sha256(open({path!r}, 'rb').read()).hexdigest()",
        f"assert actual_hash_ds_{var} == expected_hash_ds_{var}, {message}",
    ]
```

In `_build_setup_source`, initialize a counter before the first pass and emit guards for plain file-backed entries only (the `base_loader` branch is Task 10; leave it as-is here):

```python
    lines = [_SETUP_IMPORTS]
    guard_idx = 0
```

and change the tail of the first-pass loop (after the `dataframe` and `derived` branches, which keep their `continue`):

```python
        var = _sanitized_guard_var(name, guard_idx)
        guard_idx += 1
        lines.extend(_hash_guard_lines(var, name, entry.path, entry.source_hash))
        stmt = _file_load_stmt(name, entry.format, entry.path, entry.read_options)
        lines.append(f"con.execute({stmt!r})")
```

- [ ] **Step 5: Run the whole recorder + emit suites, commit green**

Run: `uv run pytest tests/test_recorder.py tests/test_emit_notebook.py -q`
Expected: all pass — including the pre-existing nbconvert round-trips (guards on unmodified files pass) and the quote-path compile test (paths are `repr()`-quoted inside the guard too).

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: setup cell emits a content-hash assert before each file-backed dataset reload"
```

---

### Task 9: Fallback and sentinel guard shapes

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (`_hash_guard_lines`)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `_hash_guard_lines`, `provenance.HASH_CONTENT_CEILING_BYTES`.
- Produces: the completed three-shape guard — Task 10 reuses it verbatim for base-loader entries.

Two red/green cycles in this task.

**Cycle A — fallback shape.**

- [ ] **Step A1: Write the failing test** (append to `tests/test_recorder.py`)

```python
def test_setup_cell_emits_fallback_guard_above_content_ceiling(tmp_path, monkeypatch) -> None:
    """Files above the content-hash ceiling carry a (path, mtime, size)
    fallback hash; the emitted guard recomputes the same tuple at replay.
    Weaker guarantee, still a hard assert."""
    from data_analyst_mcp import provenance, session
    from data_analyst_mcp.recorder import NotebookRecorder

    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 4)
    csv = tmp_path / "big.csv"
    csv.write_bytes(b"a,b\n1,2\n3,4\n")

    session.reset()
    session.register(
        name="big", path=str(csv), read_options={}, format="csv", rows=2, columns=[]
    )
    entry = session.get_datasets()["big"]
    assert entry.source_hash.startswith("fallback:")

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_ds_big_0 = '{entry.source_hash}'" in setup_src
    assert "_os.stat(" in setup_src
    assert "'fallback:' + hashlib.sha256(" in setup_src
    assert "assert actual_hash_ds_big_0 == expected_hash_ds_big_0" in setup_src
    compile(setup_src, "<setup>", "exec")
```

- [ ] **Step A2: Run to verify it fails**

Run: `uv run pytest tests/test_recorder.py::test_setup_cell_emits_fallback_guard_above_content_ceiling -q`
Expected: FAIL — the content shape is emitted (`hashlib.sha256(open(...)`), not the stat-based fallback recompute, so `"_os.stat("` is missing.

- [ ] **Step A3: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: setup cell recomputes the stat fallback for above-ceiling dataset guards"
```

- [ ] **Step A4: Implement** — extend `_hash_guard_lines` (mirrors the model-block fallback at `recorder.py:161-176`):

```python
def _hash_guard_lines(var: str, display_name: str, path: str, hash_val: str) -> list[str]:
    """Drift-guard lines emitted before one file-backed dataset reload.

    Three shapes keyed off the stored hash: content assert, ``(path, mtime,
    size)`` fallback assert, or (next cycle) a comment when only a sentinel
    is available.
    """
    message = (
        f'"Source file for dataset {display_name!r} changed since the session was recorded."'
    )
    if hash_val.startswith("fallback:"):
        # Above-ceiling files use a (path, mtime, size) fallback. Recompute
        # the same fallback at replay time; the assert remains hard, but the
        # weaker guarantee is documented.
        return [
            "import os as _os",
            f"_st = _os.stat({path!r})",
            f"expected_hash_ds_{var} = {hash_val!r}",
            f"actual_hash_ds_{var} = 'fallback:' + hashlib.sha256("
            f"f'{{{path!r}}}|{{_st.st_mtime}}|{{_st.st_size}}'.encode('utf-8')"
            f").hexdigest()",
            f"assert actual_hash_ds_{var} == expected_hash_ds_{var}, {message}",
        ]
    return [
        f"expected_hash_ds_{var} = {hash_val!r}",
        f"actual_hash_ds_{var} = hashlib.sha256(open({path!r}, 'rb').read()).hexdigest()",
        f"assert actual_hash_ds_{var} == expected_hash_ds_{var}, {message}",
    ]
```

- [ ] **Step A5: Run, commit green**

Run: `uv run pytest tests/test_recorder.py -q` — Expected: all pass.

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: setup cell recomputes the stat fallback for above-ceiling dataset guards"
```

**Cycle B — sentinel shape (comment, no assert).**

- [ ] **Step B1: Write the failing test** (append to `tests/test_recorder.py`)

```python
def test_setup_cell_emits_comment_not_assert_for_sentinel_hash() -> None:
    """Datasets without a verifiable local file (s3://, http) reload
    unguarded — the setup cell says so in a comment instead of asserting."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    session.register(
        name="remote",
        path="s3://bucket/file.csv",
        read_options={},
        format="csv",
        rows=1,
        columns=[],
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert "# Note: dataset 'remote' has no verifiable source hash" in setup_src
    assert "expected_hash_ds_remote_0" not in setup_src
    assert "CREATE OR REPLACE TABLE remote" in setup_src
    compile(setup_src, "<setup>", "exec")
```

- [ ] **Step B2: Run to verify it fails**

Run: `uv run pytest tests/test_recorder.py::test_setup_cell_emits_comment_not_assert_for_sentinel_hash -q`
Expected: FAIL — the sentinel value is emitted as a content assert (`expected_hash_ds_remote_0` present), which would raise at replay.

- [ ] **Step B3: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: setup cell emits a comment instead of an assert for sentinel dataset hashes"
```

- [ ] **Step B4: Implement** — add the sentinel branch at the top of `_hash_guard_lines`:

```python
    if not hash_val or hash_val.startswith("sentinel:"):
        return [
            f"# Note: dataset {display_name!r} has no verifiable source hash; "
            f"reload is unguarded."
        ]
```

- [ ] **Step B5: Run, commit green**

Run: `uv run pytest tests/test_recorder.py -q` — Expected: all pass.

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: setup cell emits a comment instead of an assert for sentinel dataset hashes"
```

---

### Task 10: Guard the `base_loader` reload after overwrites

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py:93-102` (derived-with-base_loader branch in `_build_setup_source`)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: `_hash_guard_lines`, `_sanitized_guard_var` (Tasks 8–9); `base_loader["source_hash"]` (Task 4).
- Produces: complete guard coverage — every file that the setup cell reloads is preceded by a guard (or a sentinel comment).

- [ ] **Step 1: Write the failing test** (append to `tests/test_recorder.py`)

```python
def test_setup_cell_guards_base_loader_reload_after_overwrite(call_tool, tmp_path) -> None:
    """Overwriting a file-backed dataset moves its reload into the
    base_loader branch — the carried load-time hash must guard that reload,
    and a second overwrite must keep the guard."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    csv.write_bytes(b"a\n1\n2\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_ds_data_0 = '{expected}'" in setup_src
    assert setup_src.index("expected_hash_ds_data_0") < setup_src.index(
        "CREATE OR REPLACE TABLE data"
    )
    compile(setup_src, "<setup>", "exec")

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_ds_data_0 = '{expected}'" in setup_src
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_recorder.py::test_setup_cell_guards_base_loader_reload_after_overwrite -q`
Expected: FAIL — the base_loader branch emits its CREATE with no guard lines.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_recorder.py
git commit -m "red: setup cell guards the base_loader reload with the carried source hash"
```

- [ ] **Step 4: Implement**

In `_build_setup_source`, extend the derived-with-base_loader branch (was lines 93–102):

```python
        if entry.format == "derived":
            # A derived entry that overwrote a file-backed dataset retains the
            # original loader in base_loader; emit it here (first pass) so the
            # second-pass derived CREATE — which may self-reference this same
            # name (transform-in-place) — has its base table at replay. The
            # carried load-time hash guards the base file exactly like a
            # first-class file-backed entry.
            base = entry.base_loader
            if base is not None:
                var = _sanitized_guard_var(name, guard_idx)
                guard_idx += 1
                lines.extend(
                    _hash_guard_lines(
                        var, name, base["path"], base.get("source_hash", "sentinel:unset")
                    )
                )
                stmt = _file_load_stmt(name, base["format"], base["path"], base.get("read_options"))
                lines.append(f"con.execute({stmt!r})")
            continue
```

(`base.get("source_hash", "sentinel:unset")` tolerates dicts written before Task 4 — in-process registries can't actually contain them, but the recorder must not KeyError on a hand-built entry in tests.)

- [ ] **Step 5: Run tests, commit green**

Run: `uv run pytest tests/test_recorder.py tests/test_materialize.py -q` — Expected: all pass, including the pre-existing `test_emitted_notebook_replays_overwrite_of_file_backed_dataset` nbconvert round-trip (unmodified file → guard passes).

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: setup cell guards the base_loader reload with the carried source hash"
```

---

### Task 11: Replay drift integration tests (`test:` commit)

**Files:**
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: the completed emission (Tasks 8–10).
- Produces: end-to-end evidence for the spec's acceptance criteria. These tests pass immediately against the finished implementation, so they land as a single `test:` commit (the auditor exempts that prefix — precedent: `test: add pairwise_comparisons integration evals`).

- [ ] **Step 1: Write both tests** (append to `tests/test_recorder.py`; they mirror the structure of the existing `test_emitted_notebook_hash_assert_fires_when_training_csv_is_mutated` at line 361, including the nbconvert invocation)

```python
def test_emitted_notebook_dataset_guard_fires_when_source_csv_is_mutated(
    tmp_path, call_tool
) -> None:
    """No model involved: a plain loaded dataset whose CSV is edited between
    emit and replay must fail the setup cell loudly (spec acceptance
    criterion, plain file-backed case)."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]})
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "plain"})
    assert r["ok"], r
    r = call_tool("describe_column", {"name": "plain", "column": "a"})
    assert r["ok"], r

    nb_path = tmp_path / "plain_drift.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    with open(csv, "a") as fh:
        fh.write("99\n")

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
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "AssertionError" in combined or "changed since the session was recorded" in combined


def test_emitted_notebook_overwrite_chain_guard_fires_when_base_csv_is_mutated(
    tmp_path, call_tool
) -> None:
    """Drift on a file reachable only via base_loader (after a
    transform-in-place overwrite) must also fail replay loudly (spec
    acceptance criterion, overwrite-chain case)."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]})
    csv = tmp_path / "base.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r

    nb_path = tmp_path / "chain_drift.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    with open(csv, "a") as fh:
        fh.write("99\n")

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
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "AssertionError" in combined or "changed since the session was recorded" in combined
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_recorder.py -q -k "dataset_guard_fires or overwrite_chain_guard_fires"`
Expected: 2 PASS (each takes several seconds — nbconvert spawns a kernel). If either fails, STOP: that is a real bug in Tasks 8–10; fix it there (with a red/green cycle if it's a new behavior) before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_recorder.py
git commit -m "test: add dataset provenance replay drift integration tests"
```

---

### Task 12: Docs sync + version bump

**Files:**
- Modify: `docs/SPEC.md` (§5.1, §5.11, §5.13, §6 setup-cell example)
- Modify: `README.md` (Known gotchas, architecture diagram line)
- Modify: `ROADMAP.md` (Reproducibility bucket, Phase 5 caveat)
- Modify: `CHANGELOG.md`, `pyproject.toml`

**Interfaces:** none — documentation only.

- [ ] **Step 1: SPEC edits**

In `docs/SPEC.md` §5.1 `load_dataset` **Behavior**, append a bullet:

```markdown
- Records a provenance hash on the dataset entry at registration (`DatasetEntry.source_hash`): SHA-256 of the file bytes up to a 100 MB ceiling, a `(path, mtime, size)` fallback digest above it, and a `sentinel:` marker for non-file sources (s3/http, in-memory, derived). The emitted notebook's setup cell asserts on it at replay.
```

In §5.11 `fit_model` **Input** section, after the `model_name` bullet, add:

```markdown
- When stored, the model's `training_dataset_hash` is copied from the dataset entry's load-time `source_hash` — the model trains on the table populated at `load_dataset` time, so the load-time hash is the truthful provenance anchor (no re-hash at fit time).
```

In §5.13 `emit_notebook` **Behavior**, change the setup-cell bullet to:

```markdown
- Prepend a setup cell: imports, DuckDB connection, then per-dataset drift guards (hash asserts) followed by dataset reloads that render the live load's `read_options`.
```

In §6 "Setup cell (prepended automatically by `to_notebook`)", replace the example code block with:

```python
import duckdb
import hashlib
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

con = duckdb.connect()

# Reload datasets registered in the session, each guarded by its
# load-time provenance hash (sentinel-hashed sources get a comment
# instead of an assert; >100 MB files use a (path, mtime, size) fallback).
expected_hash_ds_raw_0 = '3f5a...'
actual_hash_ds_raw_0 = hashlib.sha256(open('fixtures/messy.csv', 'rb').read()).hexdigest()
assert actual_hash_ds_raw_0 == expected_hash_ds_raw_0, "Source file for dataset 'raw' changed since the session was recorded."
con.execute("""CREATE OR REPLACE TABLE raw AS SELECT * FROM read_csv_auto('fixtures/messy.csv', SAMPLE_SIZE=-1)""")
```

- [ ] **Step 2: README edits**

In `README.md` "Known gotchas", add a bullet after the "Datasets are in-process state" one:

```markdown
- **Emitted notebooks are drift-guarded.** The setup cell asserts a SHA-256 provenance hash for every file-backed dataset (and for base files behind `materialize_query` overwrites) before reloading it. If a source file changed since the session, replay fails with a loud `AssertionError` instead of silently recomputing different numbers. Files over 100 MB use a weaker `(path, mtime, size)` check; s3/http sources reload unguarded (a comment in the cell says so).
```

In the architecture diagram, change the parenthetical `(setup cell re-fits models behind SHA-256 assert)` to `(setup cell hash-guards every dataset and re-fits models behind SHA-256 asserts)`.

- [ ] **Step 3: ROADMAP edits**

- Delete the "**Provenance hashes.**" bullet from the Reproducibility section (line 27).
- In the Phase 5 reproducibility caveat paragraph (line 5), change "guarded by a hard SHA-256 assert on the training file" to "guarded by a hard SHA-256 assert on the training file (captured at `load_dataset` time — every file-backed dataset reload now carries its own assert too)".

- [ ] **Step 4: CHANGELOG + version**

`pyproject.toml`: `version = "1.2.0"`.

Prepend to `CHANGELOG.md` after the header block:

```markdown
## [1.2.0] - 2026-07-05

### Added
- **Dataset provenance hashes.** Every dataset records a SHA-256 content hash
  of its source file at `load_dataset` time (`(path, mtime, size)` fallback
  above 100 MB; sentinels for s3/http/in-memory/derived sources). The emitted
  notebook's setup cell asserts on it before each reload — including base
  files behind `materialize_query` overwrites — so editing a source file
  between session and replay fails loudly instead of silently recomputing
  different numbers.
- Setup-cell reloads now render the live load's `read_options`, so a passing
  hash also implies the same parse at replay.

### Changed
- `fit_model` no longer re-hashes the training file at fit time; the model's
  provenance hash is copied from the dataset entry's load-time hash (the
  model trains on the table loaded then, not the file as it exists at fit
  time). A same-name dataset reload after a fit still fails replay loudly
  via the model-block assert.

### Internal
- New `provenance.py` (shared hash) and `read_options.py` (shared DuckDB
  reader-option rendering) leaf modules.
```

- [ ] **Step 5: Commit**

```bash
git add docs/SPEC.md README.md ROADMAP.md CHANGELOG.md pyproject.toml
git commit -m "docs: dataset provenance hashes — SPEC/README/ROADMAP/CHANGELOG sync, bump to 1.2.0"
```

---

### Task 13: Full gates

**Files:** none (verification only).

- [ ] **Step 1: Run every gate**

```bash
uv run pytest tests/ -q
uv run pytest evals/ -q
uv run ruff format --check .
uv run ruff check .
uv run pyright src/
uv run python scripts/check_tdd_commits.py
```

Expected: all clean. `check_tdd_commits.py` must report the new red/green pairs matched (9 red, 9 green added by this plan: Tasks 2, 3, 4, 5, 7, 8, 9A, 9B, 10) with zero mismatches.

- [ ] **Step 2: If any gate fails**

Fix forward: formatting failures → `uv run ruff format .` then amend the offending commit is NOT allowed (history is the TDD audit trail) — land a `refactor:` or `chore:` commit instead. Test failures → treat as a bug in the corresponding task; add a red/green cycle if it's a behavior change.
