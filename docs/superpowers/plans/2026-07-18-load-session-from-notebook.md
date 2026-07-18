# `load_session_from_notebook` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tool 25 — `load_session_from_notebook` — which resumes a previously-emitted notebook into a fresh session (datasets, derived tables, splits, registered models, recorder history) by replaying an operation journal embedded in the notebook's metadata, per the approved spec `docs/superpowers/specs/2026-07-18-load-session-from-notebook-design.md` (v4, commit `8b83c46`).

**Architecture:** Every state-changing tool (`load_dataset`, `materialize_query`, `split_dataset`, registered `fit_model`) appends a structured journal entry at call time, capturing an operation-output digest (`damcp-digest-v1`) and evidence (source hashes, membership checksums, named params/SEs). `emit_notebook` serializes journal + per-cell SHA-256 descriptors + final-state digests + a final-registry descriptor into `nb.metadata["data_analyst_mcp"]`. The new tool validates strictly (Pydantic `extra="forbid"`), replays the journal inside a DuckDB transaction on the live connection comparing recorded evidence at every step, and on success publishes staged Python state (registries, recorder cells, journal, revision counter) in one swap. Any divergence → `ROLLBACK`, live state untouched.

**Tech Stack:** Python 3.13, DuckDB (min raised to 1.5.2), pandas, numpy, statsmodels, pydantic, FastMCP, nbformat. **No new dependencies** (digest extraction uses DuckDB's native cursor + SQL epoch projections — no pyarrow).

## Global Constraints

- Every behavior change lands as a `red:` commit (failing test) followed by a `green:` commit. `scripts/check_tdd_commits.py` enforces this. Behavior-preserving refactors use `refactor:`; docs use `docs:`; release uses `chore:`.
- Gates before **every** commit: `uv run ruff format .` && `uv run ruff check .`. Before every `green:` / `test:` / `refactor:` commit additionally: `uv run pyright src/` and green `uv run pytest tests/ -q`.
- Tests call tools through the FastMCP layer via the `call_tool` fixture (`tests/conftest.py`) — never by direct function import — except narrowly-scoped unit tests of private helpers.
- House typing style: lazy `Any`-returning module accessors for pandas/statsmodels (see `tools/models.py:26-31`); pyright strict runs on `src/` only.
- Error envelopes always via `build_error(type=..., message=..., hint=...)` from `data_analyst_mcp/errors.py`.
- Tool count goes 24 → 25 (explicit waiver — governance task). Target release **1.6.0**.
- Spec-fixed constants (copy verbatim): digest algorithm id `"damcp-digest-v1"`; comparison tolerances `rtol=1e-7, atol=1e-12`; caps: notebook ≤ 32 MB, manifest ≤ 8 MB, ≤ 2000 cells, ≤ 500 journal ops, ≤ 100 KB per SQL/formula/path string; replay budget 300 s (cooperative); manifest version `1`.
- Locking: the spec's readers-writer contract is implemented as a single re-entrant mutex (`session.state_lock()`) used by writers and by resume end-to-end. Mutual exclusion is strictly stronger than RW consistency; FastMCP serializes calls anyway. Do not build an RW lock.
- Provenance hash formats (already shipped, `provenance.py`): bare hex (content), `fallback:<hex>` (>100 MB), `sentinel:<reason>:<path>` (non-file). The manifest carries them verbatim.
- New `error.type` values (spec taxonomy): `session_not_empty`, `catalog_not_empty`, `notebook_not_found`, `notebook_invalid`, `manifest_missing`, `manifest_version_unsupported`, `manifest_invalid`, `notebook_modified`, `unreplayable_dataset`, `source_drift`, `split_drift`, `model_drift`, `state_digest_mismatch`, `registry_mismatch`, `resume_budget_exceeded`, `resume_failed`.

---

### Task 1: Phase 0a — `ModelEntry.fit_options` recorded at registration

**Files:**
- Modify: `src/data_analyst_mcp/session.py` (ModelEntry dataclass ~line 56-85; `register_model` ~line 178-209)
- Modify: `src/data_analyst_mcp/tools/models.py` (`fit_model` registration call ~line 152-166)
- Test: `tests/test_model_registry.py` (append)

**Interfaces:**
- Consumes: existing `session.register_model(...)`, `FitModelInput.robust`.
- Produces: `ModelEntry.fit_options: dict[str, Any]` (default `{}` via `field(default_factory=dict)`), `register_model(..., fit_options: dict[str, Any] | None = None)`. Tasks 2, 8, 9, 13 rely on `entry.fit_options["robust"]` being present for OLS fits.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_registry.py`:

```python
def test_registered_model_carries_fit_options(call_tool: Any, load_df_into_session: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session as _session

    rng = np.random.RandomState(0)
    df = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)})
    load_df_into_session("d", df)

    result = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x", "kind": "ols", "robust": True, "model_name": "m_rob"},
    )
    assert result["ok"] is True
    entry = _session.get_models()["m_rob"]
    assert entry.fit_options == {"robust": True}


def test_fit_options_defaults_to_robust_false(call_tool: Any, load_df_into_session: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session as _session

    rng = np.random.RandomState(0)
    df = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)})
    load_df_into_session("d", df)

    result = call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m_plain"}
    )
    assert result["ok"] is True
    assert _session.get_models()["m_plain"].fit_options == {"robust": False}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_model_registry.py -k fit_options -v`
Expected: FAIL — `AttributeError: 'ModelEntry' object has no attribute 'fit_options'` (or TypeError on unexpected kwarg).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_model_registry.py
git commit -m "red: ModelEntry carries fit_options at registration"
```

- [ ] **Step 4: Implement**

In `session.py`, add to `ModelEntry` (after `training_loader`):

```python
    # Fit-time options that change the fitted result's inference state —
    # currently only {"robust": bool} for OLS HC3. Recorded so setup-cell
    # re-fit and journal replay reproduce the same covariance, not just the
    # same coefficients (robust and plain OLS share coefficients exactly).
    fit_options: dict[str, Any] = field(default_factory=dict)
```

In `register_model`, add parameter `fit_options: dict[str, Any] | None = None` and pass `fit_options=dict(fit_options) if fit_options is not None else {}` into the `ModelEntry(...)` construction.

In `tools/models.py::fit_model`, in the `session.register_model(...)` call add:

```python
            fit_options={"robust": payload.robust} if payload.kind == "ols" else {},
```

- [ ] **Step 5: Run tests, gates, commit green**

Run: `uv run pytest tests/test_model_registry.py -q` then full gates.
Expected: PASS.

```bash
git add src/data_analyst_mcp/session.py src/data_analyst_mcp/tools/models.py
git commit -m "green: ModelEntry carries fit_options at registration"
```

---

### Task 2: Phase 0b — setup-cell re-fit honors `fit_options` (HC3 bug fix)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py` (`_build_setup_source` model re-fit tail, ~line 637-643)
- Test: `tests/test_recorder.py` (append)

**Interfaces:**
- Consumes: `ModelEntry.fit_options` (Task 1).
- Produces: setup-cell re-fit line emits `.fit(cov_type="HC3")` for robust OLS models. Task 15's fidelity eval relies on this.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recorder.py`:

```python
def test_setup_refit_emits_hc3_for_robust_ols(call_tool: Any, tmp_path: Any) -> None:
    """Pre-existing replay drift (spec v4 phase 0): robust=True models
    re-fit as plain OLS in the setup cell, silently changing SEs."""
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "rob.csv"
    pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "rob"})["ok"] is True
    assert (
        call_tool(
            "fit_model",
            {"name": "rob", "formula": "y ~ x", "kind": "ols", "robust": True, "model_name": "mr"},
        )["ok"]
        is True
    )

    nb = get_recorder().to_notebook(include_setup=True)
    setup = nb.cells[0].source
    assert 'mr = smf.ols("y ~ x", data=rob_df).fit(cov_type="HC3")' in setup


def test_setup_refit_plain_ols_keeps_bare_fit(call_tool: Any, tmp_path: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "pl.csv"
    pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "pl"})["ok"] is True
    assert (
        call_tool(
            "fit_model", {"name": "pl", "formula": "y ~ x", "kind": "ols", "model_name": "mp"}
        )["ok"]
        is True
    )

    setup = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert 'mp = smf.ols("y ~ x", data=pl_df).fit()' in setup
```

- [ ] **Step 2: Run to verify failure, commit red**

Run: `uv run pytest tests/test_recorder.py -k hc3 -v` — expected FAIL (bare `.fit()` emitted).

```bash
git add tests/test_recorder.py
git commit -m "red: setup-cell re-fit honors robust fit_options (HC3)"
```

- [ ] **Step 3: Implement**

In `recorder.py::_build_setup_source`, replace the fit-args line (~638):

```python
            smf_fn = _KIND_TO_SMF.get(model_entry.kind, "ols")
            if model_entry.kind in ("logistic", "poisson", "negbin"):
                fit_args = "disp=False"
            elif model_entry.fit_options.get("robust"):
                fit_args = 'cov_type="HC3"'
            else:
                fit_args = ""
```

- [ ] **Step 4: Run tests + gates, commit green**

```bash
git add src/data_analyst_mcp/recorder.py
git commit -m "green: setup-cell re-fit honors robust fit_options (HC3)"
```

---

### Task 3: Session journal, state lock, and `source_hash` pass-through

**Files:**
- Modify: `src/data_analyst_mcp/session.py`
- Test: `tests/test_session.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces (exact names later tasks use):
  - `session.state_lock() -> threading.RLock` — module-level re-entrant lock.
  - `session.append_journal_entry(entry: dict[str, Any]) -> None` (stores a `dict(entry)` copy).
  - `session.get_journal() -> list[dict[str, Any]]` (live list).
  - `session.register(..., source_hash: str | None = None)` — when provided, stored verbatim instead of calling `compute_source_hash(path)`.
  - `session.install_state(datasets, models, journal, next_revision)` — atomic publish used by resume phase 3: replaces `_datasets`, `_models`, `_journal`, `_revision_counter` contents under the lock. Signature: `install_state(*, datasets: dict[str, DatasetEntry], models: dict[str, ModelEntry], journal: list[dict[str, Any]], next_revision: int) -> None`.
  - `session.reset()` additionally clears the journal.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_session.py`:

```python
def test_journal_append_and_reset() -> None:
    from data_analyst_mcp import session

    session.append_journal_entry({"op": "load", "op_id": "x"})
    assert session.get_journal() == [{"op": "load", "op_id": "x"}]
    session.reset()
    assert session.get_journal() == []


def test_journal_append_stores_a_copy() -> None:
    from data_analyst_mcp import session

    entry = {"op": "load", "op_id": "x"}
    session.append_journal_entry(entry)
    entry["op"] = "mutated"
    assert session.get_journal()[0]["op"] == "load"


def test_register_accepts_explicit_source_hash(tmp_path: Any) -> None:
    from data_analyst_mcp import session

    p = tmp_path / "f.csv"
    p.write_text("a\n1\n")
    session.register(
        name="t",
        path=str(p),
        read_options={},
        format="csv",
        rows=1,
        columns=[{"name": "a", "dtype": "BIGINT"}],
        source_hash="deadbeef",
    )
    assert session.get_datasets()["t"].source_hash == "deadbeef"


def test_state_lock_is_reentrant() -> None:
    from data_analyst_mcp import session

    with session.state_lock():
        with session.state_lock():
            pass  # RLock: nested acquisition must not deadlock


def test_install_state_replaces_everything() -> None:
    from data_analyst_mcp import session
    from data_analyst_mcp.session import DatasetEntry

    entry = DatasetEntry(
        path="(query)",
        read_options={"sql": "SELECT 1"},
        format="derived",
        rows=1,
        columns=[{"name": "a", "dtype": "BIGINT"}],
        revision=3,
    )
    session.install_state(
        datasets={"d": entry}, models={}, journal=[{"op": "materialize"}], next_revision=4
    )
    assert session.get_datasets() == {"d": entry}
    assert session.get_journal() == [{"op": "materialize"}]
    # Next registration must mint revision 4, not 0.
    session.register(
        name="e", path="(query)", read_options={}, format="derived", rows=0, columns=[]
    )
    assert session.get_datasets()["e"].revision == 4
```

- [ ] **Step 2: Run to verify failure, commit red**

Run: `uv run pytest tests/test_session.py -k "journal or source_hash or state_lock or install_state" -v` — expected FAIL (`AttributeError`).

```bash
git add tests/test_session.py
git commit -m "red: session journal, state lock, install_state, source_hash pass-through"
```

- [ ] **Step 3: Implement**

In `session.py` add near the module state (after `_revision_counter = 0`):

```python
_journal: list[dict[str, Any]] = []
_state_lock = threading.RLock()


def state_lock() -> threading.RLock:
    """Session-wide mutex guarding all state mutation and resume.

    The spec's readers-writer contract is implemented as a single mutex:
    mutual exclusion is strictly stronger, and FastMCP serializes tool
    calls anyway — the lock is the invariant, not the framework.
    """
    return _state_lock


def append_journal_entry(entry: dict[str, Any]) -> None:
    """Append one operation-journal entry (stored as a defensive copy)."""
    _journal.append(dict(entry))


def get_journal() -> list[dict[str, Any]]:
    """Return the live journal list (mutating it mutates the session)."""
    return _journal


def install_state(
    *,
    datasets: dict[str, DatasetEntry],
    models: dict[str, ModelEntry],
    journal: list[dict[str, Any]],
    next_revision: int,
) -> None:
    """Atomically publish a fully-prepared session state (resume phase 3).

    Replaces the contents of every registry in place so existing handles
    returned by get_datasets()/get_models()/get_journal() stay valid.
    """
    global _revision_counter
    with _state_lock:
        _datasets.clear()
        _datasets.update(datasets)
        _models.clear()
        _models.update(models)
        _journal.clear()
        _journal.extend(journal)
        _revision_counter = next_revision
```

Add `import threading` to the module imports. In `register(...)` add keyword param `source_hash: str | None = None` and change the construction line to:

```python
        source_hash=source_hash if source_hash is not None else compute_source_hash(path),
```

In `reset()` add `_journal.clear()` alongside `_models.clear()`.

- [ ] **Step 4: Run tests + gates, commit green**

```bash
git add src/data_analyst_mcp/session.py
git commit -m "green: session journal, state lock, install_state, source_hash pass-through"
```

---

### Task 4: `damcp-digest-v1` — the digest engine

**Files:**
- Create: `src/data_analyst_mcp/digest.py`
- Test: `tests/test_digest.py` (new)

**Interfaces:**
- Consumes: a DuckDB connection + table name.
- Produces (later tasks use these exact names):
  - `digest.DIGEST_ALGORITHM = "damcp-digest-v1"`
  - `digest.digest_table(con: Any, table: str) -> str | None` — hex SHA-256, or `None` when the table contains an undigestable type (never raises for type reasons).
  - `digest.single_thread_scan(con)` — context manager pinning `threads=1` and restoring the prior value in `finally`.
  - `digest.CHUNK_ROWS = 8192` (module constant; tests monkeypatch it for chunk invariance).

**Byte layout (fixed here per spec §3; stable forever under this algorithm id):**
domain separators `0x01` (schema part) / `0x02` (value part); all length prefixes unsigned 64-bit little-endian; type tags:
`NULL=0x00, BOOL=0x10, INT=0x11` (all signed/unsigned widths + HUGEINT/UHUGEINT/BIGNUM; value = length-prefixed two's-complement little-endian), `FLOAT32=0x12` (4-byte raw IEEE LE), `FLOAT64=0x13` (8-byte raw IEEE LE), `DECIMAL=0x14` (length-prefixed two's-complement LE unscaled integer; width/scale live in the schema type string), `VARCHAR=0x20` (UTF-8 bytes), `BLOB=0x21`, `BIT=0x22` (canonical text), `DATE=0x30` (i64 days since epoch), `TIME=0x31` (canonical text), `TIME_TZ=0x37` (canonical text), `TIMESTAMP_S=0x32` (i64 seconds), `TIMESTAMP_MS=0x33` (i64 millis), `TIMESTAMP=0x34` (i64 micros), `TIMESTAMP_NS=0x35` (i64 nanos), `TIMESTAMPTZ=0x36` (i64 UTC micros), `INTERVAL=0x38` (canonical text), `UUID=0x40` (canonical text), `ENUM=0x41` (string value), `LIST=0x50, STRUCT=0x51, MAP=0x52` (recursive; element count u64-LE then per-element tagged values; STRUCT fields in declaration order name-prefixed). Undigestable: `UNION`, `VARIANT`, any unrecognized/extension type → `digest_table` returns `None`.

**Extraction strategy (why not plain fetch):** DuckDB's Python fetch truncates `TIMESTAMP_NS` to microseconds via `datetime`. Temporal and date columns are therefore projected to integers **in SQL** (`epoch_ns(col)`, `epoch_us(col)`, `epoch_ms(col)`, `epoch(col)::BIGINT`, `date_diff('day', DATE '1970-01-01', col)`); FLOAT/DOUBLE fetch as Python floats and are bit-packed with `struct.pack('<f'/'<d', v)` (float32→float64 widening is exact); DECIMAL fetches as `decimal.Decimal`; HUGEINT/BIGNUM fetch as exact Python ints; INTERVAL/TIME/TIME_TZ/BIT/UUID are projected `CAST(col AS VARCHAR)` (canonical text). Nested LIST/STRUCT/MAP fetch as Python lists/dicts and encode recursively; nested temporal values encode at fetched (micro) resolution — documented limitation, comment required in code.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_digest.py`:

```python
"""Conformance tests for damcp-digest-v1 (spec v4 §3)."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import pytest


@pytest.fixture
def con() -> Any:
    from data_analyst_mcp import session

    return session.get_connection()


def _make(con: Any, name: str, select_sql: str) -> None:
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS {select_sql}')


def test_golden_vector_single_int_column(con: Any) -> None:
    """Pins the exact byte layout: one BIGINT column 'a', rows [1, NULL]."""
    from data_analyst_mcp.digest import digest_table

    _make(con, "g", "SELECT * FROM (VALUES (CAST(1 AS BIGINT)), (NULL)) t(a)")

    h = hashlib.sha256()
    h.update(b"\x01")  # schema part
    h.update((0).to_bytes(8, "little"))  # column position
    h.update(len(b"a").to_bytes(8, "little") + b"a")
    h.update(len(b"BIGINT").to_bytes(8, "little") + b"BIGINT")
    h.update(b"\x02")  # value part
    one = (1).to_bytes(1, "little", signed=True)
    h.update(b"\x11" + len(one).to_bytes(8, "little") + one)  # INT tag
    h.update(b"\x00" + (0).to_bytes(8, "little"))  # NULL tag, empty payload
    assert digest_table(con, "g") == h.hexdigest()


def test_same_table_twice_is_stable(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "s", "SELECT range AS a, range * 1.5 AS b FROM range(1000)")
    assert digest_table(con, "s") == digest_table(con, "s")


def test_row_order_changes_digest(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "o1", "SELECT * FROM (VALUES (1), (2)) t(a)")
    d1 = digest_table(con, "o1")
    _make(con, "o1", "SELECT * FROM (VALUES (2), (1)) t(a)")
    assert digest_table(con, "o1") != d1


def test_chunk_size_invariance(con: Any, monkeypatch: Any) -> None:
    from data_analyst_mcp import digest as digest_mod

    _make(con, "c", "SELECT range AS a FROM range(50)")
    big = digest_mod.digest_table(con, "c")
    monkeypatch.setattr(digest_mod, "CHUNK_ROWS", 3)
    assert digest_mod.digest_table(con, "c") == big


def test_signed_zero_and_nan_are_distinct_and_stable(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "f1", "SELECT CAST(0.0 AS DOUBLE) AS a")
    _make(con, "f2", "SELECT CAST(-0.0 AS DOUBLE) AS a")
    assert digest_table(con, "f1") != digest_table(con, "f2")
    _make(con, "f3", "SELECT CAST('nan' AS DOUBLE) AS a")
    assert digest_table(con, "f3") == digest_table(con, "f3")


def test_null_differs_from_null_string(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "n1", "SELECT CAST(NULL AS VARCHAR) AS a")
    _make(con, "n2", "SELECT '<null>' AS a")
    assert digest_table(con, "n1") != digest_table(con, "n2")


def test_timestamp_ns_nanoseconds_are_not_truncated(con: Any) -> None:
    """The exact collision the review demonstrated: two TIMESTAMP_NS values
    equal at microsecond resolution must digest differently."""
    from data_analyst_mcp.digest import digest_table

    _make(con, "t1", "SELECT TIMESTAMP_NS '2024-01-01 00:00:00.123456789' AS a")
    _make(con, "t2", "SELECT TIMESTAMP_NS '2024-01-01 00:00:00.123456780' AS a")
    assert digest_table(con, "t1") != digest_table(con, "t2")


def test_timestamp_variants_have_distinct_digests(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "v1", "SELECT TIMESTAMP '2024-01-01 00:00:01' AS a")
    _make(con, "v2", "SELECT TIMESTAMP_S '2024-01-01 00:00:01' AS a")
    assert digest_table(con, "v1") != digest_table(con, "v2")  # schema type string differs


def test_decimal_scale_matters(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "d1", "SELECT CAST(1.50 AS DECIMAL(9,2)) AS a")
    _make(con, "d2", "SELECT CAST(1.500 AS DECIMAL(9,3)) AS a")
    assert digest_table(con, "d1") != digest_table(con, "d2")


def test_nested_list_and_struct_supported(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "ls", "SELECT [1, 2, 3] AS a, {'k': 'v'} AS b")
    assert digest_table(con, "ls") is not None
    assert digest_table(con, "ls") == digest_table(con, "ls")


def test_union_type_is_undigestable(con: Any) -> None:
    from data_analyst_mcp.digest import digest_table

    _make(con, "u", "SELECT union_value(num := 2) AS a")
    assert digest_table(con, "u") is None


def test_single_thread_scan_restores_threads_setting(con: Any) -> None:
    from data_analyst_mcp.digest import single_thread_scan

    con.execute("SET threads=4")
    with single_thread_scan(con):
        row = con.execute("SELECT current_setting('threads')").fetchone()
        assert int(row[0]) == 1
    row = con.execute("SELECT current_setting('threads')").fetchone()
    assert int(row[0]) == 4


def test_single_thread_scan_restores_on_exception(con: Any) -> None:
    from data_analyst_mcp.digest import single_thread_scan

    con.execute("SET threads=4")
    with pytest.raises(RuntimeError):
        with single_thread_scan(con):
            raise RuntimeError("boom")
    row = con.execute("SELECT current_setting('threads')").fetchone()
    assert int(row[0]) == 4


def test_float32_bit_packed(con: Any) -> None:
    """FLOAT column encodes as 4-byte pattern — golden vector."""
    from data_analyst_mcp.digest import digest_table

    _make(con, "f32", "SELECT CAST(1.5 AS FLOAT) AS a")
    h = hashlib.sha256()
    h.update(b"\x01")
    h.update((0).to_bytes(8, "little"))
    h.update(len(b"a").to_bytes(8, "little") + b"a")
    h.update(len(b"FLOAT").to_bytes(8, "little") + b"FLOAT")
    h.update(b"\x02")
    payload = struct.pack("<f", 1.5)
    h.update(b"\x12" + len(payload).to_bytes(8, "little") + payload)
    assert digest_table(con, "f32") == h.hexdigest()
```

- [ ] **Step 2: Run to verify failure, commit red**

Run: `uv run pytest tests/test_digest.py -v` — expected FAIL (`ModuleNotFoundError: data_analyst_mcp.digest`).

```bash
git add tests/test_digest.py
git commit -m "red: damcp-digest-v1 conformance vectors"
```

- [ ] **Step 3: Implement `src/data_analyst_mcp/digest.py`**

```python
"""damcp-digest-v1 — order-sensitive table digest (spec v4 §3).

The digest is the resume feature's primary state-equality algorithm. The
byte layout is FROZEN under this algorithm id — any change requires a new
id. Temporal columns are projected to integer epochs in SQL because
DuckDB's Python fetch truncates TIMESTAMP_NS to microseconds via datetime.
Nested temporal values (inside LIST/STRUCT/MAP) encode at fetched
resolution — a documented limitation of v1.
"""

from __future__ import annotations

import hashlib
import struct
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

DIGEST_ALGORITHM = "damcp-digest-v1"
CHUNK_ROWS = 8192

_SCHEMA_PART = b"\x01"
_VALUE_PART = b"\x02"

_TAG_NULL = 0x00
_TAG_BOOL = 0x10
_TAG_INT = 0x11
_TAG_FLOAT32 = 0x12
_TAG_FLOAT64 = 0x13
_TAG_DECIMAL = 0x14
_TAG_VARCHAR = 0x20
_TAG_BLOB = 0x21
_TAG_TEXTUAL = 0x22  # BIT / TIME / TIME_TZ / INTERVAL / UUID canonical text share
_TAG_DATE = 0x30
_TAG_TS_S = 0x32
_TAG_TS_MS = 0x33
_TAG_TS_US = 0x34
_TAG_TS_NS = 0x35
_TAG_TS_TZ = 0x36
_TAG_ENUM = 0x41
_TAG_LIST = 0x50
_TAG_STRUCT = 0x51
_TAG_MAP = 0x52

_INT_BASES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "BIGNUM",
}
_UNDIGESTABLE_BASES = {"UNION", "VARIANT"}


def _u64(n: int) -> bytes:
    return n.to_bytes(8, "little")


def _lp(payload: bytes) -> bytes:
    return _u64(len(payload)) + payload


class _Column:
    """One column's SQL projection + value encoder."""

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype
        base = dtype.split("(")[0].strip().upper()
        q = '"' + name.replace('"', '""') + '"'
        self.select = q
        if base in _UNDIGESTABLE_BASES:
            raise _Undigestable(dtype)
        if base in _INT_BASES:
            self.tag, self.enc = _TAG_INT, _enc_int
        elif base == "BOOLEAN":
            self.tag, self.enc = _TAG_BOOL, _enc_bool
        elif base == "FLOAT" or base == "REAL":
            self.tag, self.enc = _TAG_FLOAT32, _enc_f32
        elif base == "DOUBLE":
            self.tag, self.enc = _TAG_FLOAT64, _enc_f64
        elif base == "DECIMAL":
            self.tag, self.enc = _TAG_DECIMAL, _enc_decimal
        elif base in {"VARCHAR", "CHAR", "TEXT", "STRING"}:
            self.tag, self.enc = _TAG_VARCHAR, _enc_text
        elif base == "BLOB":
            self.tag, self.enc = _TAG_BLOB, _enc_blob
        elif base == "DATE":
            self.tag, self.enc = _TAG_DATE, _enc_int
            self.select = f"date_diff('day', DATE '1970-01-01', {q})"
        elif base == "TIMESTAMP_S":
            self.tag, self.enc = _TAG_TS_S, _enc_int
            self.select = f"CAST(epoch({q}) AS BIGINT)"
        elif base == "TIMESTAMP_MS":
            self.tag, self.enc = _TAG_TS_MS, _enc_int
            self.select = f"epoch_ms({q})"
        elif base in {"TIMESTAMP", "DATETIME"}:
            self.tag, self.enc = _TAG_TS_US, _enc_int
            self.select = f"epoch_us({q})"
        elif base == "TIMESTAMP_NS":
            self.tag, self.enc = _TAG_TS_NS, _enc_int
            self.select = f"epoch_ns({q})"
        elif base in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
            self.tag, self.enc = _TAG_TS_TZ, _enc_int
            self.select = f"epoch_us({q})"
        elif base in {"BIT", "TIME", "TIMETZ", "TIME_TZ", "INTERVAL", "UUID"}:
            self.tag, self.enc = _TAG_TEXTUAL, _enc_text
            self.select = f"CAST({q} AS VARCHAR)"
        elif base == "ENUM":
            self.tag, self.enc = _TAG_ENUM, _enc_text
            self.select = f"CAST({q} AS VARCHAR)"
        elif base in {"LIST", "STRUCT", "MAP"} or dtype.endswith("[]") or base.startswith(
            ("STRUCT", "MAP")
        ):
            self.tag, self.enc = _TAG_LIST, _enc_nested
        else:
            raise _Undigestable(dtype)


class _Undigestable(Exception):
    """Internal: table contains a type outside the v1 encoding table."""


def _enc_int(v: Any) -> bytes:
    n = int(v)
    length = max(1, (n.bit_length() + 8) // 8)
    return n.to_bytes(length, "little", signed=True)


def _enc_bool(v: Any) -> bytes:
    return b"\x01" if v else b"\x00"


def _enc_f32(v: Any) -> bytes:
    return struct.pack("<f", float(v))


def _enc_f64(v: Any) -> bytes:
    return struct.pack("<d", float(v))


def _enc_decimal(v: Any) -> bytes:
    d = v if isinstance(v, Decimal) else Decimal(str(v))
    sign, digits, exponent = d.as_tuple()
    unscaled = int("".join(str(x) for x in digits)) * (-1 if sign else 1)
    return _enc_int(unscaled)


def _enc_text(v: Any) -> bytes:
    return str(v).encode("utf-8")


def _enc_blob(v: Any) -> bytes:
    return bytes(v)


def _enc_nested(v: Any) -> bytes:
    """Recursive encoding for fetched LIST/STRUCT/MAP Python values."""
    out = bytearray()
    if isinstance(v, dict):
        out += bytes([_TAG_STRUCT]) + _u64(len(v))
        for key, item in v.items():
            out += _lp(str(key).encode("utf-8"))
            out += _tagged_nested(item)
    elif isinstance(v, (list, tuple)):
        out += bytes([_TAG_LIST]) + _u64(len(v))
        for item in v:
            out += _tagged_nested(item)
    else:
        out += _tagged_nested(v)
    return bytes(out)


def _tagged_nested(v: Any) -> bytes:
    if v is None:
        return bytes([_TAG_NULL]) + _u64(0)
    if isinstance(v, bool):
        return bytes([_TAG_BOOL]) + _lp(_enc_bool(v))
    if isinstance(v, int):
        return bytes([_TAG_INT]) + _lp(_enc_int(v))
    if isinstance(v, float):
        return bytes([_TAG_FLOAT64]) + _lp(_enc_f64(v))
    if isinstance(v, Decimal):
        return bytes([_TAG_DECIMAL]) + _lp(_enc_decimal(v))
    if isinstance(v, (bytes, bytearray)):
        return bytes([_TAG_BLOB]) + _lp(bytes(v))
    if isinstance(v, (dict, list, tuple)):
        return _enc_nested(v)
    # datetimes, UUIDs, everything else: canonical text at fetched resolution.
    return bytes([_TAG_TEXTUAL]) + _lp(_enc_text(v))


@contextmanager
def single_thread_scan(con: Any) -> Any:
    """Pin threads=1 for a deterministic scan; ALWAYS restore.

    PRAGMAs are not transactional (survive ROLLBACK), so restoration in
    ``finally`` is the only thing standing between a failed digest/replay
    and a permanently single-threaded live connection.
    """
    row = con.execute("SELECT current_setting('threads')").fetchone()
    old = int(row[0])
    con.execute("SET threads=1")
    try:
        yield
    finally:
        con.execute(f"SET threads={old}")


def digest_table(con: Any, table: str) -> str | None:
    """Order-sensitive damcp-digest-v1 of a live table, or None if undigestable."""
    q = '"' + table.replace('"', '""') + '"'
    describe = con.execute(f"DESCRIBE {q}").fetchall()
    try:
        columns = [_Column(str(r[0]), str(r[1])) for r in describe]
    except _Undigestable:
        return None
    h = hashlib.sha256()
    h.update(_SCHEMA_PART)
    for pos, col in enumerate(columns):
        h.update(_u64(pos))
        h.update(_lp(col.name.encode("utf-8")))
        h.update(_lp(col.dtype.encode("utf-8")))
    h.update(_VALUE_PART)
    select = ", ".join(c.select for c in columns)
    with single_thread_scan(con):
        cur = con.execute(f"SELECT {select} FROM {q}")
        while True:
            rows = cur.fetchmany(CHUNK_ROWS)
            if not rows:
                break
            for row in rows:
                for col, value in zip(columns, row, strict=True):
                    if value is None:
                        h.update(bytes([_TAG_NULL]) + _u64(0))
                    else:
                        h.update(bytes([col.tag]) + _lp(col.enc(value)))
    return h.hexdigest()
```

Note: DuckDB reports LIST types as e.g. `BIGINT[]` and STRUCT as `STRUCT(k VARCHAR)` — the `_Column` nested branch matches on the `[]` suffix and `STRUCT`/`MAP` prefixes; keep that branch ordered after the scalar branches exactly as shown.

- [ ] **Step 4: Run tests + gates, commit green**

Run: `uv run pytest tests/test_digest.py -v` — all PASS (fix encoding until golden vectors pass; the golden tests are the layout authority).

```bash
git add src/data_analyst_mcp/digest.py
git commit -m "green: damcp-digest-v1 conformance vectors"
```

---

### Task 5: Journal + op-transaction in `load_dataset`

**Files:**
- Modify: `src/data_analyst_mcp/tools/datasets.py` (`load_dataset`, ~line 556-648)
- Modify: `src/data_analyst_mcp/recorder.py` (`record()` gains `op_id`)
- Test: `tests/test_datasets.py` (append), `tests/test_recorder.py` (append)

**Interfaces:**
- Consumes: Task 3 (`state_lock`, `append_journal_entry`, `register(source_hash=...)`), Task 4 (`digest_table`).
- Produces:
  - `recorder.record(*, markdown: str, code: str, tool_name: str, op_id: str | None = None)` — `op_id` stored in both cell dicts.
  - Journal entry shape for loads (spec §1): `{"op": "load", "op_id": <uuid4 str>, "name", "path", "format", "read_options", "source_hash", "rows", "revision", "output_digest"}`.
  - The op-transaction pattern all state-changing tools use from here on:

```python
with session.state_lock():
    con.execute("BEGIN TRANSACTION")
    try:
        # ... all fallible work: table mutation, DESCRIBE/COUNT, digest ...
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    # infallible publications only: register / append_journal_entry / record
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
def test_load_dataset_appends_journal_entry(call_tool: Any, tmp_path: Any) -> None:
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "j.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
    result = call_tool("load_dataset", {"path": str(csv), "name": "j"})
    assert result["ok"] is True

    journal = session.get_journal()
    assert len(journal) == 1
    op = journal[0]
    assert op["op"] == "load"
    assert op["name"] == "j"
    assert op["path"] == str(csv)
    assert op["format"] == "csv"
    assert op["rows"] == 2
    assert op["revision"] == session.get_datasets()["j"].revision
    assert op["source_hash"] == session.get_datasets()["j"].source_hash
    assert isinstance(op["output_digest"], str) and len(op["output_digest"]) == 64
    import uuid

    uuid.UUID(op["op_id"])  # raises if not a valid UUID


def test_load_dataset_op_id_binds_journal_to_cells(call_tool: Any, tmp_path: Any) -> None:
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "b.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "b"})["ok"] is True

    op_id = session.get_journal()[0]["op_id"]
    cells = get_recorder().cells
    assert cells[-2]["op_id"] == op_id  # markdown cell
    assert cells[-1]["op_id"] == op_id  # code cell


def test_failed_load_leaves_no_journal_and_no_table(call_tool: Any, tmp_path: Any) -> None:
    from data_analyst_mcp import session

    result = call_tool("load_dataset", {"path": str(tmp_path / "missing.csv"), "name": "x"})
    assert result["ok"] is False
    assert session.get_journal() == []


def test_mid_transaction_failure_rolls_back_table(
    call_tool: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """Fault-injection at the digest boundary: the CREATE must roll back."""
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.tools import datasets as datasets_mod

    csv = tmp_path / "r.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)

    def _boom(con: Any, table: str) -> str:
        raise RuntimeError("injected digest failure")

    monkeypatch.setattr(datasets_mod, "digest_table", _boom)
    result = call_tool("load_dataset", {"path": str(csv), "name": "r"})
    assert result["ok"] is False
    con = session.get_connection()
    names = {r[0] for r in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    assert "r" not in names
    assert session.get_journal() == []
    assert "r" not in session.get_datasets()
```

- [ ] **Step 2: Run to verify failure, commit red**

Run: `uv run pytest tests/test_datasets.py -k "journal or op_id or rolls_back" -v` — expected FAIL.

```bash
git add tests/test_datasets.py
git commit -m "red: load_dataset journals with op-transaction semantics"
```

- [ ] **Step 3: Implement**

In `recorder.py::NotebookRecorder.record`, change signature to `record(self, *, markdown: str, code: str, tool_name: str, op_id: str | None = None)` and add `"op_id": op_id` to both appended cell dicts.

In `tools/datasets.py`, add imports `import uuid`, `from data_analyst_mcp.digest import digest_table`. Rework the tail of `load_dataset` (everything from the `con.register("__dam_load_view", ...)` line, ~line 600, to the end) to:

```python
    source_hash = compute_source_hash(payload.path)
    op_id = str(uuid.uuid4())
    with session.state_lock():
        con.execute("BEGIN TRANSACTION")
        try:
            con.register("__dam_load_view", loaded_df)
            try:
                con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM __dam_load_view')
            finally:
                con.unregister("__dam_load_view")
            describe_rows = con.execute(f'DESCRIBE "{name}"').fetchall()
            columns = [{"name": str(row[0]), "dtype": str(row[1])} for row in describe_rows]
            rows = int(con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])  # type: ignore[index]
            output_digest = digest_table(con, name)
            con.execute("COMMIT")
        except Exception as exc:
            con.execute("ROLLBACK")
            return build_error(
                type="query_error",
                message=str(exc),
                hint="The load was rolled back; no table, journal entry, or cell was created.",
            )
        session.register(
            name=name,
            path=payload.path,
            read_options=payload.read_options or {},
            format=fmt,
            rows=rows,
            columns=columns,
            source_hash=source_hash,
        )
        entry = session.get_datasets()[name]
        session.append_journal_entry(
            {
                "op": "load",
                "op_id": op_id,
                "name": name,
                "path": payload.path,
                "format": fmt,
                "read_options": payload.read_options or {},
                "source_hash": entry.source_hash,
                "rows": rows,
                "revision": entry.revision,
                "output_digest": output_digest,
            }
        )
        md = (
            f"### Loaded dataset `{name}`\n"
            f"- Source: `{payload.path}`\n"
            f"- {rows} rows x {len(columns)} columns"
        )
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
        get_recorder().record(markdown=md, code=code, tool_name="load_dataset", op_id=op_id)

    return {"ok": True, "name": name, "rows": rows, "columns": columns, "warnings": []}
```

Add `from data_analyst_mcp.provenance import compute_source_hash` to imports. The hash is computed *before* the transaction and passed through `register(source_hash=...)` so the journal, registry, and guard lines all see identical bytes-evidence for this load.

- [ ] **Step 4: Run full tests + gates, commit green**

Run: `uv run pytest tests/ -q` — the pre-existing `test_datasets.py` suite pins guard-line shapes; they must still pass unchanged.

```bash
git add src/data_analyst_mcp/tools/datasets.py src/data_analyst_mcp/recorder.py
git commit -m "green: load_dataset journals with op-transaction semantics"
```

---

### Task 6: Journal + op-transaction in `materialize_query`

**Files:**
- Modify: `src/data_analyst_mcp/tools/materialize.py` (~line 89-164)
- Test: `tests/test_materialize.py` (append)

**Interfaces:**
- Consumes: Tasks 3-5 pattern.
- Produces journal entry: `{"op": "materialize", "op_id", "name", "sql", "overwrote": bool, "base_loader": <dict|None>, "split_overwrite": <dict|None>, "rows", "revision", "output_digest"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_materialize.py`:

```python
def test_materialize_appends_journal_entry(call_tool: Any, tmp_path: Any) -> None:
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "m.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "m"})["ok"] is True
    result = call_tool(
        "materialize_query", {"sql": "SELECT a * 2 AS a2 FROM m", "name": "m2"}
    )
    assert result["ok"] is True

    op = session.get_journal()[-1]
    assert op["op"] == "materialize"
    assert op["name"] == "m2"
    assert op["sql"] == "SELECT a * 2 AS a2 FROM m"
    assert op["overwrote"] is False
    assert op["base_loader"] is None
    assert op["rows"] == 3
    assert op["revision"] == session.get_datasets()["m2"].revision
    assert isinstance(op["output_digest"], str)


def test_materialize_overwrite_chain_journals_every_step(
    call_tool: Any, tmp_path: Any
) -> None:
    """The 13-vs-7 spec case: y*2 then +1 as self-overwrites — two entries,
    each with its own output digest, in order."""
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "y.csv"
    pd.DataFrame({"y": [6]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "y"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 2 AS y FROM y", "name": "y", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y + 1 AS y FROM y", "name": "y", "overwrite": True},
    )["ok"] is True

    con = session.get_connection()
    assert con.execute('SELECT y FROM "y"').fetchone()[0] == 13
    ops = [e for e in session.get_journal() if e["op"] == "materialize"]
    assert [e["sql"] for e in ops] == ["SELECT y * 2 AS y FROM y", "SELECT y + 1 AS y FROM y"]
    assert all(e["overwrote"] for e in ops)
    assert ops[0]["output_digest"] != ops[1]["output_digest"]
    assert ops[1]["base_loader"]["path"] == str(csv)


def test_failed_materialize_journals_nothing(call_tool: Any) -> None:
    from data_analyst_mcp import session

    result = call_tool("materialize_query", {"sql": "SELECT * FROM nope", "name": "n"})
    assert result["ok"] is False
    assert session.get_journal() == []
```

- [ ] **Step 2: Run to verify failure, commit red**

```bash
git add tests/test_materialize.py
git commit -m "red: materialize_query journals with op-transaction semantics"
```

- [ ] **Step 3: Implement**

In `materialize.py`, compute `base_loader` / `split_overwrite` (existing code, ~line 114-140) **before** the transaction (they read the pre-overwrite registry), then wrap:

```python
    op_id = str(uuid.uuid4())
    overwrote = existing is not None
    with session.state_lock():
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f'CREATE OR REPLACE TABLE "{payload.name}" AS {payload.sql}')
            rows = int(con.execute(f'SELECT COUNT(*) FROM "{payload.name}"').fetchone()[0])  # type: ignore[index]
            describe_rows = con.execute(f'DESCRIBE "{payload.name}"').fetchall()
            columns = [{"name": str(row[0]), "dtype": str(row[1])} for row in describe_rows]
            output_digest = digest_table(con, payload.name)
            con.execute("COMMIT")
        except Exception as exc:
            con.execute("ROLLBACK")
            return build_error(
                type="query_error",
                message=str(exc),
                hint="Check the SQL — verify referenced tables/columns exist and the syntax is valid.",
            )
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
        entry = session.get_datasets()[payload.name]
        session.append_journal_entry(
            {
                "op": "materialize",
                "op_id": op_id,
                "name": payload.name,
                "sql": payload.sql,
                "overwrote": overwrote,
                "base_loader": base_loader,
                "split_overwrite": split_overwrite,
                "rows": rows,
                "revision": entry.revision,
                "output_digest": output_digest,
            }
        )
        md = f"### Materialize query as dataset `{payload.name}`\n\n```sql\n{payload.sql}\n```"
        stmt = f'CREATE OR REPLACE TABLE "{payload.name}" AS {payload.sql}'
        code = f"con.execute({stmt!r})"
        get_recorder().record(markdown=md, code=code, tool_name="materialize_query", op_id=op_id)
```

The `existing = session.get_datasets().get(payload.name)` lookup (~line 114) stays where it is; add `import uuid` and `from data_analyst_mcp.digest import digest_table` to imports. Note `existing` is captured before the collision check returns, so `overwrote` reads it directly.

- [ ] **Step 4: Run full tests + gates, commit green**

```bash
git add src/data_analyst_mcp/tools/materialize.py
git commit -m "green: materialize_query journals with op-transaction semantics"
```

---

### Task 7: Journal + op-transaction in `split_dataset`

**Files:**
- Modify: `src/data_analyst_mcp/tools/split.py` (~line 226-295 and `_record_split`)
- Test: `tests/test_split.py` (append)

**Interfaces:**
- Consumes: Tasks 3-5 pattern; existing `membership_checksum`, `_assign_is_test`.
- Produces journal entry: `{"op": "split", "op_id", "source", "names": {"train": ..., "test": ...}, "params": {"test_fraction": float, "stratify_by": str|None, "rid_column": str}, "seed": int, "membership_checksums": {"train": ..., "test": ...}, "rows": {"train": int, "test": int}, "revisions": {"train": int, "test": int}, "output_digests": {"train": ..., "test": ...}}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_split.py`:

```python
def test_split_appends_one_journal_entry_with_both_sides(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    from data_analyst_mcp import session

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))
    assert call_tool("split_dataset", {"name": "base"})["ok"] is True

    ops = [e for e in session.get_journal() if e["op"] == "split"]
    assert len(ops) == 1
    op = ops[0]
    assert op["source"] == "base"
    assert op["names"] == {"train": "base_train", "test": "base_test"}
    assert op["seed"] == 42
    assert op["params"]["test_fraction"] == 0.25
    assert op["params"]["stratify_by"] is None
    assert op["rows"] == {"train": 8, "test": 2}
    ds = session.get_datasets()
    assert op["revisions"] == {
        "train": ds["base_train"].revision,
        "test": ds["base_test"].revision,
    }
    assert op["membership_checksums"]["test"] == ds["base_test"].read_options[
        "membership_checksum"
    ]
    assert set(op["output_digests"]) == {"train", "test"}
    assert all(isinstance(v, str) for v in op["output_digests"].values())
```

- [ ] **Step 2: Run to verify failure, commit red**

```bash
git add tests/test_split.py
git commit -m "red: split_dataset journals both sides in one entry"
```

- [ ] **Step 3: Implement**

In `split.py::split_dataset`, wrap the two `CREATE OR REPLACE TABLE` statements plus checksum computation, DESCRIBEs, and digests in the op-transaction pattern (BEGIN before `con.register(view, assign_df)`, COMMIT after both digests). After COMMIT, keep the two `session.register(...)` calls as-is, then append the journal entry and call `_record_split(...)` passing a new `op_id` parameter through to `get_recorder().record(..., op_id=op_id)`:

```python
            output_digests = {
                "train": digest_table(con, train_name),
                "test": digest_table(con, test_name),
            }
            con.execute("COMMIT")
        except Exception as exc:
            con.execute("ROLLBACK")
            return build_error(
                type="query_error",
                message=str(exc),
                hint="The split was rolled back; no tables, journal entry, or cell were created.",
            )
        # ... the existing two session.register(...) calls ...
        session.append_journal_entry(
            {
                "op": "split",
                "op_id": op_id,
                "source": payload.name,
                "names": {"train": train_name, "test": test_name},
                "params": {
                    "test_fraction": payload.test_fraction,
                    "stratify_by": payload.stratify_by,
                    "rid_column": rid,
                },
                "seed": payload.seed,
                "membership_checksums": {"train": train_checksum, "test": test_checksum},
                "rows": {"train": n - n_test, "test": n_test},
                "revisions": {
                    "train": session.get_datasets()[train_name].revision,
                    "test": session.get_datasets()[test_name].revision,
                },
                "output_digests": output_digests,
            }
        )
```

Add `import uuid`, `from data_analyst_mcp.digest import digest_table`; `_record_split` gains `op_id: str` parameter forwarded to `record(...)`.

- [ ] **Step 4: Run full tests + gates, commit green**

```bash
git add src/data_analyst_mcp/tools/split.py
git commit -m "green: split_dataset journals both sides in one entry"
```

---

### Task 8: Journal for registered `fit_model` (named evidence)

**Files:**
- Modify: `src/data_analyst_mcp/tools/models.py`
- Create: `src/data_analyst_mcp/journal_evidence.py` (nonfinite tagging + comparison helpers)
- Test: `tests/test_models.py` (append), `tests/test_journal_evidence.py` (new)

**Interfaces:**
- Consumes: Task 3; `fit_prepared`'s result dict, the live statsmodels result (`live_result`) with `.params` / `.bse` pandas Series.
- Produces:
  - `journal_evidence.tag_nonfinite(x: float) -> float | str` — returns `"NaN"` / `"Infinity"` / `"-Infinity"` for nonfinite, else the float.
  - `journal_evidence.untag(v: float | str) -> float` — inverse (`float("nan")` etc.).
  - `journal_evidence.evidence_equal(expected: dict[str, Any], actual: dict[str, Any], *, rtol: float = 1e-7, atol: float = 1e-12) -> bool` — exact key-set equality, NaN==NaN true, infinities sign-exact, finite values via `math.isclose(a, b, rel_tol=rtol, abs_tol=atol)`.
  - Journal entry for registered fits: `{"op": "fit", "op_id", "model_name", "dataset", "formula", "kind", "fit_options", "n_obs", "design_columns": [...], "params": {...}, "bse": {...}, "dispersion": <float|str|None>, "training_dataset_hash", "training_dataset_revision", "training_loader"}`. Unregistered fits do **not** journal.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_journal_evidence.py`:

```python
"""Unit tests for nonfinite tagging and evidence comparison."""

from __future__ import annotations

import math


def test_tag_and_untag_round_trip() -> None:
    from data_analyst_mcp.journal_evidence import tag_nonfinite, untag

    assert tag_nonfinite(1.5) == 1.5
    assert tag_nonfinite(float("nan")) == "NaN"
    assert tag_nonfinite(float("inf")) == "Infinity"
    assert tag_nonfinite(float("-inf")) == "-Infinity"
    assert math.isnan(untag("NaN"))
    assert untag("Infinity") == float("inf")
    assert untag(2.0) == 2.0


def test_evidence_equal_exact_key_set() -> None:
    from data_analyst_mcp.journal_evidence import evidence_equal

    assert evidence_equal({"a": 1.0}, {"a": 1.0 + 1e-12})
    assert not evidence_equal({"a": 1.0}, {"a": 1.0, "b": 2.0})  # extra key
    assert not evidence_equal({"a": 1.0, "b": 2.0}, {"a": 1.0})  # missing key
    assert not evidence_equal({"a": 1.0}, {"a": 1.001})  # outside rtol=1e-7


def test_evidence_equal_nonfinite_semantics() -> None:
    from data_analyst_mcp.journal_evidence import evidence_equal

    assert evidence_equal({"a": "NaN"}, {"a": "NaN"})
    assert not evidence_equal({"a": "Infinity"}, {"a": "-Infinity"})
    assert not evidence_equal({"a": "NaN"}, {"a": 0.0})
```

Append to `tests/test_models.py`:

```python
def test_registered_fit_journals_named_evidence(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session

    rng = np.random.RandomState(1)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(size=60)})
    load_df_into_session("d", df)

    result = call_tool(
        "fit_model",
        {"name": "d", "formula": "y ~ x", "kind": "ols", "robust": True, "model_name": "m"},
    )
    assert result["ok"] is True

    ops = [e for e in session.get_journal() if e["op"] == "fit"]
    assert len(ops) == 1
    op = ops[0]
    assert op["model_name"] == "m"
    assert op["kind"] == "ols"
    assert op["fit_options"] == {"robust": True}
    assert op["design_columns"] == ["Intercept", "x"]
    assert set(op["params"]) == {"Intercept", "x"}
    assert set(op["bse"]) == {"Intercept", "x"}
    assert op["n_obs"] == 60
    assert op["dispersion"] is None
    assert op["training_dataset_revision"] == session.get_datasets()["d"].revision


def test_unregistered_fit_does_not_journal(call_tool: Any, load_df_into_session: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session

    rng = np.random.RandomState(1)
    df = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
    load_df_into_session("d", df)
    assert call_tool("fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols"})["ok"] is True
    assert [e for e in session.get_journal() if e["op"] == "fit"] == []
```

- [ ] **Step 2: Run to verify failure, commit red**

```bash
git add tests/test_journal_evidence.py tests/test_models.py
git commit -m "red: registered fits journal named params/bse evidence"
```

- [ ] **Step 3: Implement**

Create `src/data_analyst_mcp/journal_evidence.py`:

```python
"""Nonfinite-safe model-evidence serialization + comparison (spec v4 §1)."""

from __future__ import annotations

import math
from typing import Any


def tag_nonfinite(x: float) -> float | str:
    """JSON-safe encoding: nonfinite floats become tagged strings."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return float(x)


def untag(v: float | str) -> float:
    """Inverse of tag_nonfinite."""
    if v == "NaN":
        return float("nan")
    if v == "Infinity":
        return float("inf")
    if v == "-Infinity":
        return float("-inf")
    return float(v)  # type: ignore[arg-type]


def evidence_equal(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-12,
) -> bool:
    """Exact key sets; NaN==NaN; infinities sign-exact; finites via isclose."""
    if set(expected) != set(actual):
        return False
    for key, exp in expected.items():
        act = actual[key]
        e, a = untag(exp), untag(act)
        if math.isnan(e) or math.isnan(a):
            if not (math.isnan(e) and math.isnan(a)):
                return False
        elif math.isinf(e) or math.isinf(a):
            if e != a:
                return False
        elif not math.isclose(e, a, rel_tol=rtol, abs_tol=atol):
            return False
    return True
```

In `tools/models.py::fit_model`, inside the `if result.get("ok") and payload.model_name is not None ...` block, after `session.register_model(...)`, append the journal entry (add `import uuid`, `from data_analyst_mcp.journal_evidence import tag_nonfinite`):

```python
        op_id = str(uuid.uuid4())
        params = {str(k): tag_nonfinite(float(v)) for k, v in live_result.params.items()}
        bse = {str(k): tag_nonfinite(float(v)) for k, v in live_result.bse.items()}
        dispersion = params.get("alpha") if payload.kind == "negbin" else None
        session.append_journal_entry(
            {
                "op": "fit",
                "op_id": op_id,
                "model_name": payload.model_name,
                "dataset": payload.name,
                "formula": payload.formula,
                "kind": payload.kind,
                "fit_options": {"robust": payload.robust} if payload.kind == "ols" else {},
                "n_obs": n_obs_val,
                "design_columns": [str(k) for k in live_result.params.index],
                "params": params,
                "bse": bse,
                "dispersion": dispersion,
                "training_dataset_hash": ds_entry.source_hash,
                "training_dataset_revision": ds_entry.revision,
                "training_loader": {
                    "path": ds_entry.path,
                    "format": ds_entry.format,
                    "read_options": dict(ds_entry.read_options),
                },
            }
        )
```

Then thread `op_id` into `_record_fit_model(payload, result, op_id=op_id)` (new keyword, default `None`) → `get_recorder().record(..., op_id=op_id)`. Registered-fit calls pass the id; unregistered fits pass `None`.

- [ ] **Step 4: Run full tests + gates, commit green**

```bash
git add src/data_analyst_mcp/journal_evidence.py src/data_analyst_mcp/tools/models.py
git commit -m "green: registered fits journal named params/bse evidence"
```

---

### Task 9: Manifest schema + emit-side embedding

**Files:**
- Create: `src/data_analyst_mcp/manifest.py`
- Modify: `src/data_analyst_mcp/recorder.py` (`to_notebook`)
- Test: `tests/test_manifest.py` (new), `tests/test_emit_notebook.py` (append)

**Interfaces:**
- Consumes: Tasks 3-8 (journal in session, `op_id` on cells, `digest_table`, `fit_options`).
- Produces (exact names Tasks 10-14 use):
  - `manifest.MANIFEST_VERSION = 1`, `manifest.COMPARISON = {"rtol": 1e-7, "atol": 1e-12}`
  - Caps: `manifest.MAX_NOTEBOOK_BYTES = 32 * 1024 * 1024`, `MAX_MANIFEST_BYTES = 8 * 1024 * 1024`, `MAX_CELLS = 2000`, `MAX_JOURNAL_OPS = 500`, `MAX_STRING_BYTES = 100 * 1024`
  - `manifest.build_manifest(nb: Any) -> dict[str, Any]` — reads live session + recorder + the already-built nbformat cells; returns the plain-dict manifest (spec §2 shape, keys: `manifest_version, digest_algorithm, comparison, producer, resume_supported, resume_unsupported_reasons, notebook_replayable, journal, cells, setup_cell_sha256, state_digests, final_registry`).
  - `manifest.validate_manifest(meta: dict[str, Any]) -> Manifest` — strict Pydantic parse; raises `ManifestInvalid(reasons: list[str])` on structural/semantic violation. `Manifest` is a pydantic `BaseModel` with `model_config = ConfigDict(extra="forbid")` throughout; `journal` items are a discriminated union on `op` (`JournalLoad | JournalMaterialize | JournalSplit | JournalFit`); `cells` items are `CellDescriptor(index: int, cell_type: Literal["markdown","code"], tool_name: str, op_id: str | None, source_sha256: str)`; `final_registry` is `FinalRegistry(datasets: list[FinalDataset], models: list[FinalModel], next_revision: int)` where `FinalDataset` has `name, format, read_options, path, columns, rows, source_hash, revision, base_loader, split_overwrite` and `FinalModel` has `name, kind, formula, fitted_on_dataset, n_obs, fit_options, training_dataset_hash, training_dataset_revision, training_loader`.
  - Semantic checks inside `validate_manifest`: unique final dataset names; journal revisions strictly increasing across entries (split uses its two revisions in order); every `op_id` unique and, for ops with cells, present on exactly one markdown+code cell pair in order; every final non-`dataframe` dataset has a `state_digests` entry; `next_revision` > every journal revision.
  - `nb.metadata["data_analyst_mcp"]` set by `to_notebook`; body cells carry `{"tool_name", "op_id"}` metadata, setup cell `{"role": "setup"}`.
  - `resume_supported` false (with reasons) when any registered dataset has `format == "dataframe"` or any journal `output_digest` is `None`. `notebook_replayable` false when any top-level (column-0) setup line starts with `raise AssertionError(`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manifest.py`:

```python
"""Manifest build + strict validation (spec v4 §2)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest


def _emit_nb(call_tool: Any, tmp_path: Any) -> Any:
    """One load + one materialize + one registered fit, then to_notebook."""
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "m.csv"
    pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "m"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT x, y FROM m", "name": "m2"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "m2", "formula": "y ~ x", "kind": "ols", "model_name": "mm"}
    )["ok"] is True
    return get_recorder().to_notebook(include_setup=True)


def test_notebook_carries_manifest_and_cell_metadata(call_tool: Any, tmp_path: Any) -> None:
    nb = _emit_nb(call_tool, tmp_path)

    meta = nb.metadata["data_analyst_mcp"]
    assert meta["manifest_version"] == 1
    assert meta["digest_algorithm"] == "damcp-digest-v1"
    assert meta["comparison"] == {"rtol": 1e-7, "atol": 1e-12}
    assert meta["resume_supported"] is True
    assert meta["notebook_replayable"] is True
    assert [e["op"] for e in meta["journal"]] == ["load", "materialize", "fit"]
    assert set(meta["producer"]) == {"duckdb", "pandas", "numpy", "statsmodels", "python"}

    assert nb.cells[0].metadata["role"] == "setup"
    body = nb.cells[1:]
    assert len(meta["cells"]) == len(body)
    for desc, cell in zip(meta["cells"], body, strict=True):
        assert desc["cell_type"] == cell.cell_type
        assert desc["source_sha256"] == hashlib.sha256(cell.source.encode("utf-8")).hexdigest()
    assert meta["setup_cell_sha256"] == hashlib.sha256(
        nb.cells[0].source.encode("utf-8")
    ).hexdigest()
    # Manifest is JSON-serializable (nbformat write requirement).
    json.dumps(meta)


def test_manifest_final_registry_matches_session(call_tool: Any, tmp_path: Any) -> None:
    from data_analyst_mcp import session

    nb = _emit_nb(call_tool, tmp_path)
    meta = nb.metadata["data_analyst_mcp"]
    fr = meta["final_registry"]
    assert {d["name"] for d in fr["datasets"]} == {"m", "m2"}
    m2 = next(d for d in fr["datasets"] if d["name"] == "m2")
    assert m2["rows"] == 30
    assert m2["revision"] == session.get_datasets()["m2"].revision
    assert [m["name"] for m in fr["models"]] == ["mm"]
    assert fr["models"][0]["fit_options"] == {"robust": False}
    assert fr["next_revision"] == max(d["revision"] for d in fr["datasets"]) + 1
    assert set(meta["state_digests"]) == {"m", "m2"}


def test_dataframe_dataset_marks_resume_unsupported(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("mem", pd.DataFrame({"a": [1]}))
    meta = get_recorder().to_notebook(include_setup=True).metadata["data_analyst_mcp"]
    assert meta["resume_supported"] is False
    assert any("mem" in r for r in meta["resume_unsupported_reasons"])


def test_ephemeral_model_marks_notebook_unreplayable(call_tool: Any, tmp_path: Any) -> None:
    """Fit on a table then overwrite it: setup cell raises → replayable false,
    but resume stays supported (journal replay recreates the fit)."""
    import numpy as np
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "e.csv"
    pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "e"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT x, y FROM e", "name": "d", "overwrite": False}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "em"}
    )["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT x, y FROM e", "name": "d", "overwrite": True}
    )["ok"] is True

    meta = get_recorder().to_notebook(include_setup=True).metadata["data_analyst_mcp"]
    assert meta["notebook_replayable"] is False
    assert meta["resume_supported"] is True


def test_validate_manifest_round_trips_and_forbids_extras(call_tool: Any, tmp_path: Any) -> None:
    from data_analyst_mcp.manifest import ManifestInvalid, validate_manifest

    meta = _emit_nb(call_tool, tmp_path).metadata["data_analyst_mcp"]
    validate_manifest(json.loads(json.dumps(meta)))  # round-trip clean

    bad = json.loads(json.dumps(meta))
    bad["surprise"] = 1
    with pytest.raises(ManifestInvalid):
        validate_manifest(bad)

    dup = json.loads(json.dumps(meta))
    dup["journal"][1]["op_id"] = dup["journal"][0]["op_id"]
    with pytest.raises(ManifestInvalid):
        validate_manifest(dup)
```

- [ ] **Step 2: Run to verify failure, commit red**

```bash
git add tests/test_manifest.py
git commit -m "red: emit embeds journal manifest + cell descriptors"
```

- [ ] **Step 3: Implement `src/data_analyst_mcp/manifest.py`**

Pydantic models exactly as the Interfaces block specifies (`ConfigDict(extra="forbid")` on every model; journal union discriminated on `op` via `Field(discriminator="op")`). `ManifestInvalid(Exception)` carries `self.reasons: list[str]`. `build_manifest(nb)`:

```python
def build_manifest(nb: Any) -> dict[str, Any]:
    """Build the resume manifest from live session + recorder + built cells."""
    import platform

    import duckdb
    import numpy
    import pandas
    import statsmodels

    from data_analyst_mcp import session
    from data_analyst_mcp.digest import DIGEST_ALGORITHM, digest_table

    con = session.get_connection()
    datasets = session.get_datasets()
    journal = [dict(e) for e in session.get_journal()]

    reasons: list[str] = []
    for name, entry in datasets.items():
        if entry.format == "dataframe":
            reasons.append(f"dataset {name!r} is in-memory (dataframe) — journal cannot recreate it")
    for e in journal:
        digests = e.get("output_digests", {}) if e["op"] == "split" else {"": e.get("output_digest")}
        if any(v is None for v in digests.values()):
            reasons.append(f"journal op {e['op_id']} produced an undigestable table")

    state_digests = {
        name: digest_table(con, name)
        for name, entry in datasets.items()
        if entry.format != "dataframe"
    }
    setup_src = nb.cells[0].source
    replayable = not any(
        line.startswith("raise AssertionError(") for line in setup_src.splitlines()
    )
    cells = [
        {
            "index": i,
            "cell_type": c.cell_type,
            "tool_name": c.metadata.get("tool_name", ""),
            "op_id": c.metadata.get("op_id"),
            "source_sha256": hashlib.sha256(c.source.encode("utf-8")).hexdigest(),
        }
        for i, c in enumerate(nb.cells[1:])
    ]
    final_datasets = [
        {
            "name": name,
            "format": e.format,
            "read_options": dict(e.read_options),
            "path": e.path,
            "columns": list(e.columns),
            "rows": e.rows,
            "source_hash": e.source_hash,
            "revision": e.revision,
            "base_loader": e.base_loader,
            "split_overwrite": e.split_overwrite,
        }
        for name, e in datasets.items()
    ]
    final_models = [
        {
            "name": m.name,
            "kind": m.kind,
            "formula": m.formula,
            "fitted_on_dataset": m.fitted_on_dataset,
            "n_obs": m.n_obs,
            "fit_options": dict(m.fit_options),
            "training_dataset_hash": m.training_dataset_hash,
            "training_dataset_revision": m.training_dataset_revision,
            "training_loader": m.training_loader,
        }
        for m in session.get_models().values()
    ]
    next_revision = max((e.revision for e in datasets.values()), default=-1) + 1
    return {
        "manifest_version": MANIFEST_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "comparison": dict(COMPARISON),
        "producer": {
            "duckdb": duckdb.__version__,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
        },
        "resume_supported": not reasons,
        "resume_unsupported_reasons": reasons,
        "notebook_replayable": replayable,
        "journal": journal,
        "cells": cells,
        "setup_cell_sha256": hashlib.sha256(setup_src.encode("utf-8")).hexdigest(),
        "state_digests": state_digests,
        "final_registry": {
            "datasets": final_datasets,
            "models": final_models,
            "next_revision": next_revision,
        },
    }
```

In `recorder.py::to_notebook`: pass metadata into the created cells and attach the manifest:

```python
        if include_setup:
            setup = _nbformat.v4.new_code_cell(_build_setup_source())  # type: ignore[reportUnknownMemberType]
            setup.metadata["role"] = "setup"
            nb.cells.append(setup)
        for cell in self.cells:
            maker = (
                _nbformat.v4.new_markdown_cell
                if cell["cell_type"] == "markdown"
                else _nbformat.v4.new_code_cell
            )
            new_cell = maker(cell["source"])  # type: ignore[reportUnknownMemberType]
            new_cell.metadata["tool_name"] = cell["metadata"]["tool_name"]
            if cell.get("op_id") is not None:
                new_cell.metadata["op_id"] = cell["op_id"]
            nb.cells.append(new_cell)
        if include_setup:
            from data_analyst_mcp.manifest import build_manifest

            nb.metadata["data_analyst_mcp"] = build_manifest(nb)
        return nb
```

- [ ] **Step 4: Run full tests + gates, commit green**

Existing `test_emit_notebook.py` pins may assert on cell sources only — if any pins exact `metadata == {}`, update them in this commit and say so in the commit body.

```bash
git add src/data_analyst_mcp/manifest.py src/data_analyst_mcp/recorder.py tests/test_emit_notebook.py
git commit -m "green: emit embeds journal manifest + cell descriptors"
```

---

### Task 10: `load_session_from_notebook` — happy path (loads + materialize + split)

**Files:**
- Create: `src/data_analyst_mcp/tools/resume.py`
- Modify: `src/data_analyst_mcp/server.py` (tool 25 wrapper, after the `emit_notebook` wrapper)
- Modify: `src/data_analyst_mcp/recorder.py` (add `install_cells`)
- Test: `tests/test_resume.py` (new)

**Interfaces:**
- Consumes: everything above; `datasets._build_read_call`, `session.read_file_as_df`, `split._assign_is_test`, `split.membership_checksum`, `compute_source_hash`, `digest_table`, `single_thread_scan`, `validate_manifest`, `evidence_equal`.
- Produces:
  - `resume.LoadSessionInput(BaseModel)` with `path: str` (`extra="forbid"`).
  - `resume.load_session_from_notebook(payload: LoadSessionInput) -> dict[str, Any]`.
  - `resume.RESUME_BUDGET_SECONDS = 300.0` (module constant; tests monkeypatch).
  - `recorder.NotebookRecorder.install_cells(cells: list[dict[str, Any]]) -> None` — replaces `self.cells` contents in place.
  - Success response: `{"ok": True, "path", "datasets": [{"name","rows","format"}...], "models": [...], "n_cells_imported": int, "n_journal_ops": int, "warnings": [...]}`.
  - Internal `_Divergence(Exception)` with `error_type: str`, `message: str`, `hint: str | None`, `op_index: int | None`, `op_id: str | None` — phase-2 fail-fast carrier; converts to `build_error` with `op_index`/`op_id` appended to the message.

**Phase structure (all under `session.state_lock()`):**

1. *Validate:* empty session (`get_datasets()`, `get_models()`, `get_journal()`, `get_recorder().cells` all empty → else `session_not_empty`); empty live catalog (`SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'` + `SELECT view_name FROM duckdb_views() WHERE schema_name = 'main' AND NOT internal` → any row → `catalog_not_empty` listing names); file exists (`notebook_not_found`), `nbformat.read` succeeds (`notebook_invalid`), caps (file size vs `MAX_NOTEBOOK_BYTES`, `json.dumps(meta)` size vs `MAX_MANIFEST_BYTES`, cell/op counts, per-string caps → `manifest_invalid`); manifest present (`manifest_missing`), version supported (`manifest_version_unsupported`), `validate_manifest` (`manifest_invalid` with reasons); `resume_supported` true (`unreplayable_dataset` with recorded reasons); cell integrity — body cells match descriptors index-by-index (count, `cell_type`, `tool_name`, sha256) and setup sha matches (`notebook_modified`); source preflight — for every `load` op and every `materialize` op's `base_loader`, `compute_source_hash(path)` equals the recorded hash (accumulate ALL mismatches → single `source_drift` listing each).
2. *Replay (fail-fast):* `BEGIN TRANSACTION`; iterate journal ops with `deadline = time.monotonic() + RESUME_BUDGET_SECONDS` checked before each op (`resume_budget_exceeded`); apply per-op (below); after the last op compare `state_digests` for every staged non-dataframe dataset and the staged registry against `final_registry` (field-equal excluding `registered_at`; name-keyed) plus `next_revision` (`registry_mismatch`); on any `_Divergence` or exception → `ROLLBACK` → error return.
3. *Commit:* `COMMIT`; `session.install_state(datasets=staged_datasets, models=staged_models, journal=manifest journal, next_revision=...)`; `get_recorder().install_cells(imported_cells)`; return the response.

**Per-op apply (phase 2):**

- `load`: `pre = compute_source_hash(op["path"])`; mismatch vs `op["source_hash"]` → `_Divergence("source_drift", ...)` (TOCTOU: preflight passed but the file changed since). Remote paths (`s3://`/`http`) skip hashing, append warning `f"{op['path']} reloaded unguarded"`. `read_call = _build_read_call(...)`; `df = session.read_file_as_df(read_call)`; `post = compute_source_hash(op["path"])`; `post != pre` → `source_drift`. Register view + `CREATE OR REPLACE TABLE`; `digest_table` vs `op["output_digest"]` → `state_digest_mismatch`. Stage `DatasetEntry(path=op["path"], read_options=dict(op["read_options"]), format=op["format"], rows=op["rows"], columns=<DESCRIBE>, source_hash=op["source_hash"], revision=op["revision"])`.
- `materialize`: re-check `leading_keyword` + `contains_unsafe_semicolon` (same `_sql_safety` gate as live; violation → `manifest_invalid` — a journal carrying rejected SQL is malformed, not drifted); `CREATE OR REPLACE TABLE`; digest vs `output_digest` → `state_digest_mismatch`; stage entry (`format="derived"`, `path="(query)"`, `read_options={"sql": op["sql"]}`, `base_loader`/`split_overwrite` from op, `revision=op["revision"]`).
- `split`: recompute membership via `_assign_is_test(n, params["test_fraction"], op["seed"], strata)` (strata fetched when `params["stratify_by"]`), same join SQL as `split_dataset` (reuse the literal statements — build them with the identical `base` expression); per-side `membership_checksum(df)` vs `op["membership_checksums"]` → `_Divergence("split_drift", ...)`; per-side digest vs `output_digests` → `state_digest_mismatch`; stage both entries with the recorded `read_options` shape the live tool writes (source/seed/test_fraction/stratify_by/train_name/test_name/rid_column/role/membership_checksum).
- `fit`: `df = con.execute(f'SELECT * FROM "{op["dataset"]}"').df()`; `payload = FitModelInput(name=op["dataset"], formula=op["formula"], kind=op["kind"], robust=bool(op["fit_options"].get("robust", False)), model_name=op["model_name"])`; `result = fit_prepared(payload, df)`; not ok → `_Divergence("model_drift", message=result["error"]["message"])`; `live = result.pop("_result")`; compare `int(result["fit"]["n_obs"]) == op["n_obs"]` exactly and `evidence_equal(op["params"], <recomputed tagged params>)` / `evidence_equal(op["bse"], ...)` with the manifest's `comparison` tolerances → else `model_drift`; stage `ModelEntry(..., fit_options=dict(op["fit_options"]), training_dataset_hash=op["training_dataset_hash"], training_dataset_revision=op["training_dataset_revision"], training_loader=op["training_loader"], _result=live, fitted_at=datetime.now(UTC), n_obs=op["n_obs"], name=op["model_name"], kind=op["kind"], formula=op["formula"], fitted_on_dataset=op["dataset"])`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resume.py`:

```python
"""load_session_from_notebook — happy-path round trips (spec v4 §4)."""

from __future__ import annotations

import os
from typing import Any


def _fresh(call_tool: Any) -> None:
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import get_recorder

    session.reset()
    get_recorder().reset()


def _emit(call_tool: Any, tmp_path: Any) -> str:
    target = str(tmp_path / "session.ipynb")
    result = call_tool("emit_notebook", {"path": target})
    assert result["ok"] is True
    return result["path"]


def test_round_trip_load_materialize_split(call_tool: Any, tmp_path: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "rt.csv"
    pd.DataFrame({"x": rng.normal(size=40), "g": ["a", "b"] * 20}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "rt"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT x FROM rt WHERE x > 0", "name": "pos"}
    )["ok"] is True
    assert call_tool("split_dataset", {"name": "rt", "seed": 7})["ok"] is True
    n_cells_before = len(get_recorder().cells)
    journal_before = [dict(e) for e in session.get_journal()]
    datasets_before = {n: (e.rows, e.format, e.revision) for n, e in session.get_datasets().items()}
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is True, result
    assert result["n_cells_imported"] == n_cells_before
    assert result["n_journal_ops"] == len(journal_before)
    assert result["warnings"] == []
    assert {d["name"] for d in result["datasets"]} == {"rt", "pos", "rt_train", "rt_test"}

    after = {n: (e.rows, e.format, e.revision) for n, e in session.get_datasets().items()}
    assert after == datasets_before
    assert [e["op_id"] for e in session.get_journal()] == [e["op_id"] for e in journal_before]
    assert len(get_recorder().cells) == n_cells_before

    con = session.get_connection()
    assert con.execute('SELECT COUNT(*) FROM "rt_train"').fetchone()[0] == 30


def test_round_trip_overwrite_chain_restores_live_values(call_tool: Any, tmp_path: Any) -> None:
    """The 13-vs-7 case: snapshot reconstruction gives 7; journal replay must give 13."""
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "y.csv"
    pd.DataFrame({"y": [6]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "y"})["ok"] is True
    for sql in ("SELECT y * 2 AS y FROM y", "SELECT y + 1 AS y FROM y"):
        assert call_tool(
            "materialize_query", {"sql": sql, "name": "y", "overwrite": True}
        )["ok"] is True
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    assert call_tool("load_session_from_notebook", {"path": path})["ok"] is True
    con = session.get_connection()
    assert con.execute('SELECT y FROM "y"').fetchone()[0] == 13


def test_resume_then_continue_then_emit_again(call_tool: Any, tmp_path: Any) -> None:
    """Post-resume the session continues: new ops append, next emit validates."""
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.manifest import validate_manifest

    csv = tmp_path / "c.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "c"})["ok"] is True
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    assert call_tool("load_session_from_notebook", {"path": path})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT a + 1 AS a FROM c", "name": "c2"}
    )["ok"] is True
    # next_revision continued — no revision collision with the imported entry.
    assert session.get_datasets()["c2"].revision > session.get_datasets()["c"].revision

    import nbformat

    path2 = str(tmp_path / "session2.ipynb")
    assert call_tool("emit_notebook", {"path": path2})["ok"] is True
    nb2 = nbformat.read(path2, as_version=4)
    validate_manifest(dict(nb2.metadata["data_analyst_mcp"]))
    assert len(nb2.metadata["data_analyst_mcp"]["journal"]) == 2


def test_resume_restores_registered_model(call_tool: Any, tmp_path: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session

    rng = np.random.RandomState(2)
    x = rng.normal(size=50)
    csv = tmp_path / "mod.csv"
    pd.DataFrame({"x": x, "y": 3.0 * x + rng.normal(size=50)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "mod"})["ok"] is True
    assert call_tool(
        "fit_model",
        {"name": "mod", "formula": "y ~ x", "kind": "ols", "robust": True, "model_name": "rm"},
    )["ok"] is True
    params_before = dict(session.get_models()["rm"]._result.params.items())
    bse_before = dict(session.get_models()["rm"]._result.bse.items())
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is True, result
    assert result["models"] == ["rm"]
    entry = session.get_models()["rm"]
    assert entry.fit_options == {"robust": True}
    for k, v in params_before.items():
        assert abs(float(entry._result.params[k]) - float(v)) < 1e-9
    for k, v in bse_before.items():
        assert abs(float(entry._result.bse[k]) - float(v)) < 1e-9
    # predict works against the restored registry (live Results object).
    assert call_tool(
        "predict", {"model_name": "rm", "dataset": "mod", "limit": 5}
    )["ok"] is True
```

- [ ] **Step 2: Run to verify failure, commit red**

Run: `uv run pytest tests/test_resume.py -v` — expected FAIL (`Unknown tool: load_session_from_notebook`).

```bash
git add tests/test_resume.py
git commit -m "red: load_session_from_notebook happy-path round trips"
```

- [ ] **Step 3: Implement**

Create `src/data_analyst_mcp/tools/resume.py` implementing the phase structure and per-op appliers exactly as the Interfaces block above specifies. Key skeleton (fill the appliers per the block):

```python
"""load_session_from_notebook — resume an emitted session (spec v4)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from data_analyst_mcp import session
from data_analyst_mcp.digest import digest_table
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.journal_evidence import evidence_equal, tag_nonfinite
from data_analyst_mcp.manifest import (
    MANIFEST_VERSION,
    MAX_MANIFEST_BYTES,
    MAX_NOTEBOOK_BYTES,
    ManifestInvalid,
    validate_manifest,
)
from data_analyst_mcp.provenance import compute_source_hash
from data_analyst_mcp.recorder import get_recorder
from data_analyst_mcp.session import DatasetEntry, ModelEntry

logger = logging.getLogger(__name__)

RESUME_BUDGET_SECONDS = 300.0


class LoadSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class _Divergence(Exception):
    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        hint: str | None = None,
        op_index: int | None = None,
        op_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.hint = hint
        self.op_index = op_index
        self.op_id = op_id
```

`load_session_from_notebook(payload)` runs phases 1-3 under `with session.state_lock():`; phase-2 exceptions convert via:

```python
        except _Divergence as div:
            con.execute("ROLLBACK")
            loc = (
                f" (journal op {div.op_index}, op_id {div.op_id}; downstream ops unverified)"
                if div.op_index is not None
                else ""
            )
            return build_error(type=div.error_type, message=div.message + loc, hint=div.hint)
        except Exception as exc:
            con.execute("ROLLBACK")
            return build_error(
                type="resume_failed",
                message=f"Journal replay failed: {exc}",
                hint="The live session was left untouched.",
            )
```

Add to `recorder.py`:

```python
    def install_cells(self, cells: list[dict[str, Any]]) -> None:
        """Replace the recorded cells wholesale (resume phase 3 publish)."""
        self.cells.clear()
        self.cells.extend([dict(c) for c in cells])
```

Imported cell dicts are rebuilt from the notebook body: `{"cell_type": c.cell_type, "source": c.source, "metadata": {"tool_name": desc["tool_name"]}, "op_id": desc["op_id"]}`.

In `server.py`, after the `emit_notebook` wrapper add:

```python
@mcp.tool()
def load_session_from_notebook(path: str) -> dict[str, Any]:
    """Resume a previously-emitted session notebook: verify its journal
    manifest, replay every recorded operation transactionally with drift
    guards, and restore datasets, models, and the recorder history."""
    from data_analyst_mcp.tools.resume import LoadSessionInput
    from data_analyst_mcp.tools.resume import load_session_from_notebook as _impl

    return _impl(LoadSessionInput(path=path))
```

(match the existing wrapper style at the top of `server.py` — flat args, import-inside-function is NOT the existing style; place imports at module top like the other tools.)

- [ ] **Step 4: Run full tests + gates, commit green**

```bash
git add src/data_analyst_mcp/tools/resume.py src/data_analyst_mcp/server.py src/data_analyst_mcp/recorder.py
git commit -m "green: load_session_from_notebook happy-path round trips"
```

---

### Task 11: Phase-1 validation battery

**Files:**
- Modify: `src/data_analyst_mcp/tools/resume.py`
- Test: `tests/test_resume.py` (append)

**Interfaces:** consumes Task 10; produces the complete phase-1 error surface (spec §4 phase 1).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_resume.py`:

```python
def _tiny_emitted(call_tool: Any, tmp_path: Any) -> str:
    import pandas as pd

    csv = tmp_path / "t.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "t"})["ok"] is True
    return _emit(call_tool, tmp_path)


def test_non_empty_session_rejected(call_tool: Any, tmp_path: Any) -> None:
    import pandas as pd

    path = _tiny_emitted(call_tool, tmp_path)
    # Session still holds dataset "t" — not empty.
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "session_not_empty"


def test_ambient_catalog_table_rejected_and_never_dropped(
    call_tool: Any, tmp_path: Any
) -> None:
    from data_analyst_mcp import session

    path = _tiny_emitted(call_tool, tmp_path)
    _fresh(call_tool)
    con = session.get_connection()
    con.execute("CREATE OR REPLACE TABLE ambient_aux AS SELECT 1 AS x")
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "catalog_not_empty"
    assert "ambient_aux" in result["error"]["message"]
    assert con.execute("SELECT x FROM ambient_aux").fetchone()[0] == 1
    con.execute("DROP TABLE ambient_aux")


def test_missing_file_and_invalid_notebook(call_tool: Any, tmp_path: Any) -> None:
    result = call_tool("load_session_from_notebook", {"path": str(tmp_path / "no.ipynb")})
    assert result["error"]["type"] == "notebook_not_found"

    bad = tmp_path / "bad.ipynb"
    bad.write_text("not json at all")
    result = call_tool("load_session_from_notebook", {"path": str(bad)})
    assert result["error"]["type"] == "notebook_invalid"


def test_manifest_missing_on_pre_feature_notebook(call_tool: Any, tmp_path: Any) -> None:
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("pass"))
    plain = tmp_path / "plain.ipynb"
    nbformat.write(nb, str(plain))
    result = call_tool("load_session_from_notebook", {"path": str(plain)})
    assert result["error"]["type"] == "manifest_missing"
    assert "re-emit" in result["error"]["hint"]


def test_unsupported_manifest_version(call_tool: Any, tmp_path: Any) -> None:
    import nbformat

    path = _tiny_emitted(call_tool, tmp_path)
    _fresh(call_tool)
    nb = nbformat.read(path, as_version=4)
    nb.metadata["data_analyst_mcp"]["manifest_version"] = 99
    nbformat.write(nb, path)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["error"]["type"] == "manifest_version_unsupported"


def test_edited_body_cell_rejected(call_tool: Any, tmp_path: Any) -> None:
    import nbformat

    path = _tiny_emitted(call_tool, tmp_path)
    _fresh(call_tool)
    nb = nbformat.read(path, as_version=4)
    nb.cells[-1].source = nb.cells[-1].source + "\n# tampered"
    nbformat.write(nb, path)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["error"]["type"] == "notebook_modified"


def test_edited_setup_cell_rejected(call_tool: Any, tmp_path: Any) -> None:
    import nbformat

    path = _tiny_emitted(call_tool, tmp_path)
    _fresh(call_tool)
    nb = nbformat.read(path, as_version=4)
    nb.cells[0].source = nb.cells[0].source + "\n# tampered"
    nbformat.write(nb, path)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["error"]["type"] == "notebook_modified"


def test_dataframe_session_rejected(
    call_tool: Any, load_df_into_session: Any, tmp_path: Any
) -> None:
    import pandas as pd

    load_df_into_session("mem", pd.DataFrame({"a": [1]}))
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["error"]["type"] == "unreplayable_dataset"
    assert "mem" in result["error"]["message"]


def test_preflight_source_drift_lists_every_file(call_tool: Any, tmp_path: Any) -> None:
    import pandas as pd

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame({"x": [1]}).to_csv(a, index=False)
    pd.DataFrame({"x": [2]}).to_csv(b, index=False)
    assert call_tool("load_dataset", {"path": str(a), "name": "a"})["ok"] is True
    assert call_tool("load_dataset", {"path": str(b), "name": "b"})["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    a.write_text("x\n99\n")
    b.write_text("x\n98\n")
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["error"]["type"] == "source_drift"
    assert str(a) in result["error"]["message"] and str(b) in result["error"]["message"]

    from data_analyst_mcp import session

    assert session.get_datasets() == {}
    con = session.get_connection()
    assert con.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE schema_name = 'main'"
    ).fetchone()[0] == 0


def test_oversized_journal_rejected(call_tool: Any, tmp_path: Any, monkeypatch: Any) -> None:
    from data_analyst_mcp.tools import resume as resume_mod

    path = _tiny_emitted(call_tool, tmp_path)
    _fresh(call_tool)
    monkeypatch.setattr(resume_mod, "MAX_JOURNAL_OPS_EFFECTIVE", 0)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["error"]["type"] == "manifest_invalid"
```

- [ ] **Step 2: Run to verify failure, commit red**

Some tests may already pass from Task 10's phase 1 — that is fine as long as at least the new error paths fail. Verify: `uv run pytest tests/test_resume.py -v`. Deselect passing ones when confirming reds.

```bash
git add tests/test_resume.py
git commit -m "red: resume phase-1 validation battery"
```

- [ ] **Step 3: Implement**

Complete phase 1 in `resume.py` per the Interfaces block of Task 10; expose `MAX_JOURNAL_OPS_EFFECTIVE = MAX_JOURNAL_OPS` as a module-level name (imported cap, monkeypatchable). Accumulate independent phase-1 failures: preflight source mismatches collect into one `source_drift` message (one line per file: `f"{path}: expected {expected[:12]}…, found {actual[:12]}…"`).

- [ ] **Step 4: Run full tests + gates, commit green**

```bash
git add src/data_analyst_mcp/tools/resume.py
git commit -m "green: resume phase-1 validation battery"
```

---

### Task 12: Phase-2 divergence detection (digests, splits, fail-fast, settings)

**Files:**
- Modify: `src/data_analyst_mcp/tools/resume.py`
- Test: `tests/test_resume.py` (append)

**Interfaces:** consumes Tasks 10-11. Produces complete phase-2 evidence comparison per spec §4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_resume.py`:

```python
def test_toctou_mutation_between_preflight_and_load(call_tool: Any, tmp_path: Any, monkeypatch: Any) -> None:
    """File changes after phase-1 preflight, before the load op: the pre/post
    agreement check inside _apply_load must catch it."""
    import pandas as pd

    from data_analyst_mcp.tools import resume as resume_mod

    csv = tmp_path / "tc.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "tc"})["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)

    original_read = resume_mod.session.read_file_as_df
    state = {"mutated": False}

    def _mutating_read(read_call: str) -> Any:
        if not state["mutated"]:
            state["mutated"] = True
            csv.write_text("a\n42\n")  # mutate AFTER preflight, DURING the op
        return original_read(read_call)

    monkeypatch.setattr(resume_mod.session, "read_file_as_df", _mutating_read)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "source_drift"


def test_nondeterministic_recipe_fails_at_its_op(call_tool: Any, tmp_path: Any) -> None:
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "nd.csv"
    pd.DataFrame({"a": list(range(100))}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "nd"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT a, random() AS r FROM nd", "name": "randomized"},
    )["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "state_digest_mismatch"
    assert "op_id" in result["error"]["message"]
    # Rollback left nothing behind.
    con = session.get_connection()
    assert con.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE schema_name = 'main'"
    ).fetchone()[0] == 0
    assert session.get_datasets() == {}


def test_split_drift_when_derived_source_order_changes(call_tool: Any, tmp_path: Any) -> None:
    """Emit a split of a derived table; tamper the journal's recorded
    membership checksum to simulate order drift → split_drift."""
    import nbformat
    import pandas as pd

    csv = tmp_path / "sd.csv"
    pd.DataFrame({"a": list(range(10))}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "sd"})["ok"] is True
    assert call_tool("split_dataset", {"name": "sd"})["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)

    nb = nbformat.read(path, as_version=4)
    meta = nb.metadata["data_analyst_mcp"]
    split_op = next(e for e in meta["journal"] if e["op"] == "split")
    split_op["membership_checksums"]["test"] = "0:" + "0" * 32 + ":" + "0" * 32
    # Keep cell descriptors valid: cells untouched. Re-write the notebook.
    nbformat.write(nb, path)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "split_drift"


def test_registry_mismatch_on_tampered_rows(call_tool: Any, tmp_path: Any) -> None:
    import nbformat
    import pandas as pd

    csv = tmp_path / "rm.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "rm"})["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    nb = nbformat.read(path, as_version=4)
    nb.metadata["data_analyst_mcp"]["final_registry"]["datasets"][0]["rows"] = 999
    nbformat.write(nb, path)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "registry_mismatch"


def test_threads_setting_restored_after_failed_resume(call_tool: Any, tmp_path: Any) -> None:
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "th.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "th"})["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    con = session.get_connection()
    con.execute("SET threads=4")
    csv.write_text("a\n5\n")  # force source_drift mid-phase-2? preflight catches it — fine
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert int(con.execute("SELECT current_setting('threads')").fetchone()[0]) == 4


def test_budget_exceeded_is_cooperative(call_tool: Any, tmp_path: Any, monkeypatch: Any) -> None:
    import pandas as pd

    from data_analyst_mcp.tools import resume as resume_mod

    csv = tmp_path / "bu.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "bu"})["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    monkeypatch.setattr(resume_mod, "RESUME_BUDGET_SECONDS", -1.0)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "resume_budget_exceeded"

    from data_analyst_mcp import session

    assert session.get_datasets() == {}
```

- [ ] **Step 2: Run to verify failure, commit red**

```bash
git add tests/test_resume.py
git commit -m "red: resume phase-2 divergence detection and hygiene"
```

- [ ] **Step 3: Implement**

Complete phase 2 in `resume.py`: the deadline check `if time.monotonic() > deadline: raise _Divergence("resume_budget_exceeded", ...)` before each op; every `_Divergence` raised from an op carries `op_index`/`op_id`; final `state_digests` + registry-descriptor comparison (build the staged descriptor dicts with exactly the `FinalDataset`/`FinalModel` fields and compare as name-keyed maps; mismatch names go in the `registry_mismatch` message); `single_thread_scan` already restores threads around digest scans — ensure no other setting is mutated anywhere in resume.

- [ ] **Step 4: Run full tests + gates, commit green**

```bash
git add src/data_analyst_mcp/tools/resume.py
git commit -m "green: resume phase-2 divergence detection and hygiene"
```

---

### Task 13: Model-drift detection — the guard on the guard

**Files:**
- Modify: `src/data_analyst_mcp/tools/resume.py` (only if Step 1 fails — the comparison logic shipped in Task 10)
- Test: `tests/test_resume.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resume.py`:

```python
def test_stripped_hc3_is_caught_by_bse_not_params(call_tool: Any, tmp_path: Any) -> None:
    """Robust and plain OLS share coefficients exactly; only SEs differ.
    Tampering fit_options to robust=False must fail on bse evidence."""
    import nbformat
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(3)
    x = rng.normal(size=80)
    noise = rng.normal(size=80) * (1.0 + np.abs(x))  # heteroskedastic on purpose
    csv = tmp_path / "hc.csv"
    pd.DataFrame({"x": x, "y": 2.0 * x + noise}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "hc"})["ok"] is True
    assert call_tool(
        "fit_model",
        {"name": "hc", "formula": "y ~ x", "kind": "ols", "robust": True, "model_name": "hm"},
    )["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)

    nb = nbformat.read(path, as_version=4)
    meta = nb.metadata["data_analyst_mcp"]
    fit_op = next(e for e in meta["journal"] if e["op"] == "fit")
    fit_op["fit_options"] = {"robust": False}
    fr_model = next(m for m in meta["final_registry"]["models"] if m["name"] == "hm")
    fr_model["fit_options"] = {"robust": False}
    nbformat.write(nb, path)

    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is False
    assert result["error"]["type"] == "model_drift"
    assert "bse" in result["error"]["message"]


def test_tampered_coefficient_evidence_is_model_drift(call_tool: Any, tmp_path: Any) -> None:
    import nbformat
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(4)
    x = rng.normal(size=50)
    csv = tmp_path / "tc2.csv"
    pd.DataFrame({"x": x, "y": x + rng.normal(size=50)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "tc2"})["ok"] is True
    assert call_tool(
        "fit_model", {"name": "tc2", "formula": "y ~ x", "kind": "ols", "model_name": "tm"}
    )["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    nb = nbformat.read(path, as_version=4)
    fit_op = next(e for e in nb.metadata["data_analyst_mcp"]["journal"] if e["op"] == "fit")
    fit_op["params"]["x"] = float(fit_op["params"]["x"]) + 0.5
    nbformat.write(nb, path)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["error"]["type"] == "model_drift"
```

- [ ] **Step 2: Run — expect FAIL only if the comparison is incomplete; commit red, then fix**

If Task 10's `_apply_fit` already compares `params` **and** `bse` with the manifest tolerances and names the failing evidence map in the message, these pass immediately — then commit them as `test:` instead of `red:`/`green:`:

```bash
git add tests/test_resume.py
git commit -m "test: model_drift catches stripped HC3 via bse and tampered params"
```

Otherwise: commit `red:`, fix `_apply_fit` to include `f"bse mismatch for {op['model_name']!r}"` / `f"params mismatch ..."` messages, commit `green:`.

---

### Task 14: Warnings for degraded evidence (remote / fallback sources)

**Files:**
- Modify: `src/data_analyst_mcp/tools/resume.py`
- Test: `tests/test_resume.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_fallback_hash_source_resumes_with_warning(
    call_tool: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """Files above the hash ceiling carry fallback: hashes — resume succeeds
    (same (path, mtime, size) check as replay) but reports degraded continuity."""
    import pandas as pd

    from data_analyst_mcp import provenance

    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 0)  # force fallback
    csv = tmp_path / "big.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "big"})["ok"] is True
    path = _emit(call_tool, tmp_path)
    _fresh(call_tool)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is True, result
    assert any("fallback" in w for w in result["warnings"])
```

- [ ] **Step 2: red commit; implement; green commit**

Implementation: in `_apply_load`, when `op["source_hash"].startswith("fallback:")` recompute the same fallback (reuse `compute_source_hash` — with the ceiling monkeypatched to 0 in the test it recomputes fallback shape) and compare; on success append `f"{op['path']}: fallback (path, mtime, size) evidence only — continuity is degraded"` to warnings. Remote (`s3://`/`http`) paths append `f"{op['path']} reloaded unguarded"` (already specified in Task 10; assert both paths are covered by grep before closing the task).

```bash
git add tests/test_resume.py
git commit -m "red: degraded-evidence sources resume with warnings"
# ... implement ...
git add src/data_analyst_mcp/tools/resume.py
git commit -m "green: degraded-evidence sources resume with warnings"
```

---### Task 15: Black-box evals (fresh process, nbconvert)

**Files:**
- Create: `evals/eval_resume.py`

**Interfaces:** none new. **Before writing, read `evals/eval_replay_guards.py` and `evals/conftest.py`** and reuse their stdio-client harness and nbconvert invocation verbatim — the eval below names scenario bodies; the session-spawning fixture comes from the existing harness.

- [ ] **Step 1: Write the eval scenarios** (these run against a *fresh server process per scenario* — the point is that resume works across process death, which `tests/` cannot show):

1. **`test_eval_resume_round_trip_replayable`** — process A: load csv fixture (`fixtures/messy.csv` is available) → materialize an overwrite chain on a copy (`y*2` then `y+1` shape) → `split_dataset` → robust OLS `fit_model(model_name=...)` → `emit_notebook`. Kill process A. Process B: `load_session_from_notebook` → assert ok, datasets/models restored, chain value correct → continue with one more `materialize_query` → `emit_notebook` to a second path. Then `jupyter nbconvert --to notebook --execute` the second notebook via `subprocess.run` → exit code 0.
2. **`test_eval_resume_only_ephemeral_model`** — session that fits on a table then overwrites it: emitted manifest has `notebook_replayable: false`; nbconvert on it FAILS (assert non-zero exit); fresh process resume SUCCEEDS and restored model params equal the recorded journal evidence.
3. **`test_eval_resume_drift_atomicity`** — emit, mutate the csv, fresh process resume → `source_drift`; assert the process's `list_datasets` returns empty and a `query` for `duckdb_tables()` count is 0.

- [ ] **Step 2: Run, commit**

Run: `uv run pytest evals/eval_resume.py -v` (~needs jupyter; the existing eval suite already shells out to nbconvert).

```bash
git add evals/eval_resume.py
git commit -m "test: eval — resume round trips, resume-only sessions, drift atomicity"
```

---

### Task 16: Governance folds + version bump

**Files:**
- Modify: `docs/SPEC.md`, `README.md`, `ROADMAP.md`, `pyproject.toml`, `CHANGELOG.md`

**Steps (single `docs:` commit, then `chore:` release when the user says ship):**

- [ ] **Step 1: SPEC.md** — (a) §5: add the `load_session_from_notebook` tool section (inputs, response, error taxonomy, phase semantics — condense spec v4 §4); (b) §5 recorder contract (~line 194-200): replace "every successful tool records a cell" with the five-name no-record enumeration (`list_datasets`, `list_models`, `describe_column`, `emit_notebook`, `load_session_from_notebook`); (c) §5.11: manifest + journal subsection (shape, caps, digest algorithm id, trust model incl. Patsy formulas as process-level code execution); (d) §5.13 (~line 673) and §12 (~line 897): condition "emitted notebooks execute successfully" on `notebook_replayable: true`; (e) §11 (~line 877): explicit waiver text for tool 25 citing ROADMAP ¶1 precedent.
- [ ] **Step 2: README.md** — tool count 24 → 25 (line ~219 and the architecture diagram); "every tool call appends a markdown + code cell" → qualified with the enumeration; "Known gotchas": rewrite the "Datasets are in-process state" bullet to point at `load_session_from_notebook`; add a short worked-example subsection ("Resume a session") showing emit → restart → `load_session_from_notebook(path=...)`.
- [ ] **Step 3: ROADMAP.md** — replace the parked `load_session_from_notebook` entry (line ~25, snapshot-flavored description) with "shipped in 1.6.0 — journal-based, see the design spec"; update the header count 24 → 25 with the waiver note.
- [ ] **Step 4: pyproject.toml** — `"duckdb>=1.5.2"` (spec §5 version envelope; transactional-DDL and type-surface verified there).
- [ ] **Step 5: CHANGELOG.md** — 1.6.0 entry: phase-0 robust-OLS setup-cell fix (bug fix), journal + manifest emission (additive), `load_session_from_notebook`, DuckDB minimum raise.
- [ ] **Step 6: Commit**

```bash
git add docs/SPEC.md README.md ROADMAP.md pyproject.toml CHANGELOG.md
git commit -m "docs: fold load_session_from_notebook into SPEC/README/ROADMAP; raise duckdb floor"
```

Release (`version = "1.6.0"` + `chore: release 1.6.0`) happens only when the user asks.

---

## Self-Review (completed at plan time)

- **Spec coverage:** phase 0 → Tasks 1-2; §0 SessionState/lock → Task 3 (`install_state` + `state_lock`; composite-value semantics implemented as clear-and-update under the lock — publication is atomic w.r.t. every lock-holding observer); §1 journal + op-transaction → Tasks 5-8; §2 manifest/flags → Task 9; §3 digest → Task 4 (tag table fixed here as the spec delegated); §4 phases → Tasks 10-12; model evidence/HC3 guard → Task 13; degraded evidence warnings → Task 14; §6 caps/budget → Tasks 11-12 (caps, cooperative budget; trust-model text lands in SPEC via Task 16); §7 response/no-record → Tasks 10, 16; taxonomy → Tasks 10-14; governance → Task 16; testing section → mapped across Tasks 4-15 (fault-injection: Tasks 5, 12; conformance vectors: Task 4; idempotence: Task 10; evals: Task 15).
- **Known deliberate narrowings** (documented in code comments where they land): nested temporal values digest at fetched resolution (Task 4); `catalog_not_empty` checks `main`-schema tables + non-internal views (Task 10-11); settings hygiene tracks `threads` only — the sole setting resume mutates (Task 12).
- **Type consistency check:** `record(op_id=)` (Task 5) matches all later `record` calls; `install_state` signature (Task 3) matches Task 10's call; `evidence_equal` (Task 8) matches Task 10's `_apply_fit`; `MAX_JOURNAL_OPS_EFFECTIVE` (Task 11) is defined before its monkeypatch use; journal field names in Tasks 5-8 match the manifest models in Task 9 and the appliers in Task 10.
