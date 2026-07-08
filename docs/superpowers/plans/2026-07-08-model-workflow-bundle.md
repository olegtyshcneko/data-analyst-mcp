# Model Workflow Bundle (`split_dataset` + `cross_validate`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two MCP tools — `split_dataset` (seeded, optionally stratified train/test partition registered as derived datasets) and `cross_validate` (k-fold CV metrics for a formula, ephemeral fits) — with full notebook-replay support, per the approved spec `docs/superpowers/specs/2026-07-08-model-workflow-bundle-design.md`.

**Architecture:** Each tool is a new module in `src/data_analyst_mcp/tools/` following the existing pattern (pydantic Input model with `extra="forbid"`, pure function returning a dict, `build_error` envelopes, recorder cell on success) plus a flat-arg wrapper in `server.py`. Split membership is computed with `np.random.RandomState(seed)` (NEP-19-frozen) and materialized via a DuckDB join on `row_number() OVER ()`; replay drift is caught by an order-independent membership checksum asserted in the emitted notebook. `cross_validate` reuses `fit_model`'s validation/fit path via a new public `fit_prepared` helper (full-data preflight fit), then slices the preflight design matrices per fold.

**Tech Stack:** Python 3.13, DuckDB, pandas, numpy, statsmodels, patsy (via statsmodels formula API), scikit-learn (metrics), pydantic, FastMCP, nbformat. **No new dependencies.**

## Global Constraints

- Every behavior change lands as a `red:` commit (failing test) followed by a `green:` commit (implementation). `scripts/check_tdd_commits.py` enforces this on the log. Behavior-preserving refactors use `refactor:`; docs use `docs:`; release uses `chore:`.
- Before **every** commit run all four gates and fix anything they flag:
  `uv run ruff format .` && `uv run ruff check .` && `uv run pyright src/` && `uv run pytest tests/ -q`
- Tests call tools through the FastMCP layer via the `call_tool` fixture (`tests/conftest.py`) — never by direct function import — except for narrowly-scoped unit tests of private helpers.
- House typing style: lazy `Any`-returning module accessors for pandas/sklearn/statsmodels (see `tools/evaluate.py:28-39`); pyright strict runs on `src/` only.
- Error envelopes always via `build_error(type=..., message=..., hint=...)` from `data_analyst_mcp/errors.py`.
- Tool count goes 22 → 24. README line 219 lists all tools and must be updated in the docs task.
- Identifier regex for dataset names: `^[A-Za-z_][A-Za-z0-9_]*$` (same as `materialize_query`).
- `int(round(x))` uses Python banker's rounding (round-half-even); tests below pin this behavior — do not "fix" it to half-up.
- Pinned known answer used by tests: `np.random.RandomState(42).permutation(10)` → `[8, 1, 5, 0, 7, 2, 9, 4, 3, 6]`.

---

### Task 1: `split_dataset` happy path (unstratified)

**Files:**
- Create: `src/data_analyst_mcp/tools/split.py`
- Modify: `src/data_analyst_mcp/server.py` (add import + wrapper after the `materialize_query` wrapper, i.e. after line 98)
- Create: `tests/test_split.py`

**Interfaces:**
- Consumes: `session.get_datasets()`, `session.get_connection()`, `session.register(...)`, `build_error`, `get_recorder()` — all existing.
- Produces: `SplitDatasetInput` (pydantic model: `name: str`, `test_fraction: float = 0.25`, `seed: int = 42`, `stratify_by: str | None = None`, `train_name: str | None = None`, `test_name: str | None = None`, `overwrite: bool = False`) and `split_dataset(payload: SplitDatasetInput) -> dict[str, Any]`. Also `_assign_is_test(n: int, test_fraction: float, seed: int, strata: Any | None) -> tuple[Any, list[str]]` (numpy bool array + warnings) and `membership_checksum(df: Any) -> str` — Task 4's recorder work relies on these exact names. Datasets register with `format="split"`, `path="(split)"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_split.py`:

```python
"""Tests for the ``split_dataset`` tool."""

from __future__ import annotations

from typing import Any


def _load_ten_rows(load_df_into_session: Any) -> None:
    import pandas as pd

    load_df_into_session("base", pd.DataFrame({"x": list(range(10))}))


def test_split_dataset_returns_ok_and_row_counts(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_ten_rows(load_df_into_session)

    result = call_tool("split_dataset", {"name": "base"})

    assert result["ok"] is True
    assert result["source"] == "base"
    assert result["train"] == {"name": "base_train", "rows": 8}
    assert result["test"] == {"name": "base_test", "rows": 2}
    assert result["seed"] == 42
    assert result["test_fraction"] == 0.25
    assert result["stratify_by"] is None
    assert result["strata"] is None
    assert result["warnings"] == []


def test_split_dataset_membership_is_the_pinned_permutation(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """RandomState(42).permutation(10) = [8,1,5,0,7,2,9,4,3,6]; with
    test_fraction=0.25, n_test = int(round(2.5)) = 2 (banker's rounding),
    so rows 8 and 1 land in test."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)

    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True

    con = _session.get_connection()
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    train_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_train"').fetchall())
    assert test_x == [1, 8]
    assert train_x == [0, 2, 3, 4, 5, 6, 7, 9]


def test_split_dataset_registers_both_as_split_format(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True

    datasets = _session.get_datasets()
    for out_name, role in (("base_train", "train"), ("base_test", "test")):
        entry = datasets[out_name]
        assert entry.format == "split"
        assert entry.path == "(split)"
        assert entry.read_options["source"] == "base"
        assert entry.read_options["seed"] == 42
        assert entry.read_options["test_fraction"] == 0.25
        assert entry.read_options["role"] == role
        assert entry.read_options["train_name"] == "base_train"
        assert entry.read_options["test_name"] == "base_test"
    # The test-side entry carries the membership checksum (32 hex chars).
    checksum = datasets["base_test"].read_options["membership_checksum"]
    assert isinstance(checksum, str) and len(checksum) == 32


def test_split_dataset_same_seed_is_deterministic(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    call_tool("split_dataset", {"name": "base"})
    call_tool(
        "split_dataset",
        {"name": "base", "train_name": "tr2", "test_name": "te2"},
    )

    con = _session.get_connection()
    first = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    second = sorted(r[0] for r in con.execute('SELECT x FROM "te2"').fetchall())
    assert first == second


def test_split_dataset_custom_names_and_fraction_clamps(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """n=10, test_fraction=0.05 → round(0.5)=0 → clamped to 1 test row."""
    _load_ten_rows(load_df_into_session)

    result = call_tool(
        "split_dataset",
        {"name": "base", "test_fraction": 0.05, "train_name": "tr", "test_name": "te"},
    )

    assert result["ok"] is True
    assert result["train"] == {"name": "tr", "rows": 9}
    assert result["test"] == {"name": "te", "rows": 1}


def test_split_dataset_source_column_named_split_rid_survives(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """The internal row-number column must dodge a source column of the
    same name instead of colliding with it."""
    import pandas as pd

    load_df_into_session(
        "tricky", pd.DataFrame({"__split_rid": list(range(10)), "y": list(range(10))})
    )

    result = call_tool("split_dataset", {"name": "tricky"})

    assert result["ok"] is True
    from data_analyst_mcp import session as _session

    cols = [c["name"] for c in _session.get_datasets()["tricky_train"].columns]
    assert cols == ["__split_rid", "y"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_split.py -q`
Expected: FAIL — every test errors with a tool-not-found/unknown-tool result (`split_dataset` is not registered).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_split.py
git commit -m "red: split_dataset partitions a dataset into seeded train/test derived datasets"
```

- [ ] **Step 4: Implement `tools/split.py`**

Create `src/data_analyst_mcp/tools/split.py`:

```python
"""Seeded train/test split — registers two split-format datasets + recorder cell.

Determinism contract (spec §5.6b): membership is a pure function of
(source rows, seed) computed with ``np.random.RandomState`` — frozen by
NumPy's NEP 19 legacy guarantee — never with DuckDB ``hash()`` or
``USING SAMPLE``, whose output is not stable across DuckDB versions.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pd() -> Any:
    """Return ``pandas`` as untyped to keep strict pyright clean."""
    import pandas as _pd_mod  # type: ignore[reportMissingTypeStubs]

    return _pd_mod


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


class SplitDatasetInput(BaseModel):
    """Inputs for ``split_dataset``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    test_fraction: float = 0.25
    seed: int = 42
    stratify_by: str | None = None
    train_name: str | None = None
    test_name: str | None = None
    overwrite: bool = False


def _assign_is_test(
    n: int, test_fraction: float, seed: int, strata: Any | None
) -> tuple[Any, list[str]]:
    """Boolean test-membership array over row ids ``0..n-1`` plus warnings.

    Must stay algorithm-identical to the notebook snippet emitted by
    ``recorder.split_replay_source`` — the replay membership-checksum
    assert is the drift guard between the two (covered by
    ``evals/eval_split_cv.py``).

    Stratified mode consumes a single ``RandomState(seed)`` across strata
    in sorted-stratum order (``NULL`` stratum last) so assignment stays
    deterministic. A stratum with fewer than 2 rows goes entirely to
    train and adds a ``small_strata`` warning.
    """
    rng = np.random.RandomState(seed)
    is_test = np.zeros(n, dtype=bool)
    warnings: list[str] = []
    if strata is None:
        n_test = min(max(int(round(n * test_fraction)), 1), n - 1)
        is_test[rng.permutation(n)[:n_test]] = True
        return is_test, warnings
    null_mask: Any = strata.isna().to_numpy()
    values: list[Any] = sorted(strata[~strata.isna()].unique().tolist(), key=str)
    groups: list[Any] = [
        np.where((strata == v).to_numpy() & ~null_mask)[0] for v in values
    ]
    if bool(null_mask.any()):
        groups.append(np.where(null_mask)[0])
    saw_small = False
    for rids in groups:
        if len(rids) < 2:
            saw_small = True
            continue
        n_t = min(max(int(round(len(rids) * test_fraction)), 1), len(rids) - 1)
        is_test[rids[rng.permutation(len(rids))[:n_t]]] = True
    if saw_small:
        warnings.append("small_strata")
    return is_test, warnings


def membership_checksum(df: Any) -> str:
    """Order-independent digest of a DataFrame's row contents.

    XOR of truncated SHA-256 per-row digests over a canonical value
    serialization — stable across DuckDB/numpy versions because every
    value is converted to a builtin before ``repr``. Must stay
    algorithm-identical to the ``_split_checksum`` snippet emitted by
    ``recorder.split_replay_source``.
    """
    pd_mod = _pd()
    acc = 0
    for row in df.itertuples(index=False, name=None):
        parts: list[str] = []
        for v in row:
            is_na = False
            try:
                is_na = bool(pd_mod.isna(v))
            except (TypeError, ValueError):
                is_na = False
            if v is None or is_na:
                parts.append("<null>")
            elif isinstance(v, (bool, np.bool_)):
                parts.append("true" if bool(v) else "false")
            elif isinstance(v, (float, np.floating)):
                f = float(v)
                parts.append("<null>" if math.isnan(f) else repr(f))
            elif isinstance(v, (int, np.integer)):
                parts.append(repr(int(v)))
            elif isinstance(v, str):
                parts.append(v)
            else:
                parts.append(str(v))
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        acc ^= int.from_bytes(digest[:16], "big")
    return format(acc, "032x")


def split_dataset(payload: SplitDatasetInput) -> dict[str, Any]:
    """Partition a registered dataset into seeded train/test datasets."""
    entry = session.get_datasets().get(payload.name)
    if entry is None:
        return build_error(
            type="dataset_not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    if not (0.0 < payload.test_fraction < 1.0):
        return build_error(
            type="test_fraction_out_of_range",
            message=f"test_fraction must be in the open interval (0, 1); got {payload.test_fraction}.",
            hint="Pick a fraction strictly between 0 and 1, e.g. 0.25.",
        )
    train_name = payload.train_name or f"{payload.name}_train"
    test_name = payload.test_name or f"{payload.name}_test"
    for candidate in (train_name, test_name):
        if not _NAME_RE.fullmatch(candidate):
            return build_error(
                type="invalid_name",
                message=f"Invalid dataset name {candidate!r}.",
                hint=(
                    "Names must match ^[A-Za-z_][A-Za-z0-9_]*$ — letters, "
                    "digits, and underscores only; cannot start with a digit."
                ),
            )
    if len({payload.name, train_name, test_name}) < 3:
        return build_error(
            type="split_name_conflict",
            message=(
                f"Source, train, and test names must be pairwise distinct; got "
                f"source={payload.name!r}, train={train_name!r}, test={test_name!r}."
            ),
            hint="Pick distinct train_name / test_name that also differ from the source.",
        )
    if not payload.overwrite:
        collisions = [c for c in (train_name, test_name) if c in session.get_datasets()]
        if collisions:
            return build_error(
                type="dataset_name_collision",
                message=f"Dataset name(s) already registered: {collisions}.",
                hint="Pass overwrite=True to replace, or choose different names.",
            )
    con = session.get_connection()
    src_q = _quote(payload.name)
    describe_rows = con.execute(f"DESCRIBE {src_q}").fetchall()
    src_cols = {str(r[0]) for r in describe_rows}
    if payload.stratify_by is not None and payload.stratify_by not in src_cols:
        return build_error(
            type="stratify_column_missing",
            message=f"Column {payload.stratify_by!r} is not in dataset {payload.name!r}.",
            hint=f"Available columns: {sorted(src_cols)}.",
        )
    n = int(con.execute(f"SELECT COUNT(*) FROM {src_q}").fetchone()[0])  # type: ignore[index]
    if n < 2:
        return build_error(
            type="dataset_too_small",
            message=f"Dataset {payload.name!r} has {n} row(s); a split needs at least 2.",
            hint="Both sides of a split must be non-empty.",
        )

    strata: Any | None = None
    if payload.stratify_by is not None:
        strata = con.execute(
            f"SELECT {_quote(payload.stratify_by)} FROM {src_q}"
        ).df().iloc[:, 0]

    is_test, warnings = _assign_is_test(n, payload.test_fraction, payload.seed, strata)
    n_test = int(is_test.sum())
    if n_test == 0 or n_test == n:
        return build_error(
            type="stratification_too_sparse",
            message=(
                "Stratified assignment left one side of the split empty "
                f"(test rows: {n_test} of {n})."
            ),
            hint=(
                "Every stratum has fewer than 2 rows, so all rows went to "
                "train. Use a coarser stratify_by column or drop stratification."
            ),
        )

    rid = "__split_rid"
    while rid in src_cols:
        rid = "_" + rid
    pd_mod = _pd()
    assign_df = pd_mod.DataFrame(
        {"rid": np.arange(n, dtype=np.int64), "is_test": is_test}
    )
    view = "__data_analyst_split_assign"
    con.register(view, assign_df)
    try:
        base = (
            f"SELECT s.* EXCLUDE ({_quote(rid)}) FROM "
            f"(SELECT *, row_number() OVER () - 1 AS {_quote(rid)} FROM {src_q}) s "
            f"JOIN {view} a ON s.{_quote(rid)} = a.rid"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE {_quote(train_name)} AS {base} WHERE NOT a.is_test"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE {_quote(test_name)} AS {base} WHERE a.is_test"
        )
    finally:
        con.unregister(view)

    test_df = con.execute(f"SELECT * FROM {_quote(test_name)}").df()
    checksum = membership_checksum(test_df)

    common_opts: dict[str, Any] = {
        "source": payload.name,
        "seed": payload.seed,
        "test_fraction": payload.test_fraction,
        "stratify_by": payload.stratify_by,
        "train_name": train_name,
        "test_name": test_name,
        "rid_column": rid,
    }
    strata_out = _strata_counts(strata, is_test) if strata is not None else None
    for out_name, role, rows in (
        (train_name, "train", n - n_test),
        (test_name, "test", n_test),
    ):
        out_describe = con.execute(f"DESCRIBE {_quote(out_name)}").fetchall()
        out_columns = [{"name": str(r[0]), "dtype": str(r[1])} for r in out_describe]
        opts = {**common_opts, "role": role}
        if role == "test":
            opts["membership_checksum"] = checksum
        session.register(
            name=out_name,
            path="(split)",
            read_options=opts,
            format="split",
            rows=rows,
            columns=out_columns,
        )

    _record_split(payload, train_name, test_name, n - n_test, n_test, checksum, rid)

    return {
        "ok": True,
        "source": payload.name,
        "train": {"name": train_name, "rows": n - n_test},
        "test": {"name": test_name, "rows": n_test},
        "seed": payload.seed,
        "test_fraction": payload.test_fraction,
        "stratify_by": payload.stratify_by,
        "strata": strata_out,
        "warnings": warnings,
    }


def _strata_counts(strata: Any, is_test: Any) -> list[dict[str, Any]]:
    """Per-stratum train/test row counts, sorted-stratum order, NULL last."""
    null_mask: Any = strata.isna().to_numpy()
    values: list[Any] = sorted(strata[~strata.isna()].unique().tolist(), key=str)
    out: list[dict[str, Any]] = []
    for v in values:
        mask: Any = (strata == v).to_numpy() & ~null_mask
        out.append(
            {
                "value": v,
                "train_rows": int((mask & ~is_test).sum()),
                "test_rows": int((mask & is_test).sum()),
            }
        )
    if bool(null_mask.any()):
        out.append(
            {
                "value": None,
                "train_rows": int((null_mask & ~is_test).sum()),
                "test_rows": int((null_mask & is_test).sum()),
            }
        )
    return out


def _record_split(
    payload: SplitDatasetInput,
    train_name: str,
    test_name: str,
    n_train: int,
    n_test: int,
    checksum: str,
    rid: str,
) -> None:
    """Append the markdown + code cell pair for a successful split.

    Task 4 replaces the placeholder code body with
    ``recorder.split_replay_source`` so the per-call cell and the setup
    cell share one snippet.
    """
    md = (
        f"### Split `{payload.name}` into `{train_name}` / `{test_name}`\n\n"
        f"- seed={payload.seed}, test_fraction={payload.test_fraction}, "
        f"stratify_by={payload.stratify_by!r}\n"
        f"- train rows: {n_train}, test rows: {n_test}"
    )
    code = (
        f"# split_dataset({payload.name!r}) -> {train_name!r} / {test_name!r} "
        f"(replay source lands in Task 4)"
    )
    get_recorder().record(markdown=md, code=code, tool_name="split_dataset")
```

- [ ] **Step 5: Register the wrapper in `server.py`**

Add to the imports block at the top of `src/data_analyst_mcp/server.py` (next to `from data_analyst_mcp.tools import materialize as _materialize`):

```python
from data_analyst_mcp.tools import split as _split
```

Insert after the `materialize_query` wrapper (after current line 98):

```python
@mcp.tool()
def split_dataset(
    name: str,
    test_fraction: float = 0.25,
    seed: int = 42,
    stratify_by: str | None = None,
    train_name: str | None = None,
    test_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Partition a registered dataset into seeded train/test datasets.

    Membership is a deterministic function of (source rows, seed) via
    ``np.random.RandomState`` — the same seed always produces the same
    split, and the emitted notebook reproduces it exactly. Defaults:
    ``{name}_train`` / ``{name}_test`` (override with train_name /
    test_name; all three names must be pairwise distinct). Optional
    ``stratify_by`` keeps per-stratum proportions; strata with fewer
    than 2 rows go entirely to train (``small_strata`` warning). Both
    outputs register as first-class datasets every other tool can
    target by name. Errors: dataset_not_found, dataset_name_collision,
    invalid_name, split_name_conflict, test_fraction_out_of_range,
    stratify_column_missing, dataset_too_small, stratification_too_sparse.
    """
    try:
        payload = _split.SplitDatasetInput(
            name=name,
            test_fraction=test_fraction,
            seed=seed,
            stratify_by=stratify_by,
            train_name=train_name,
            test_name=test_name,
            overwrite=overwrite,
        )
        return _split.split_dataset(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("split_dataset failed")
        return build_error(type="internal", message=str(exc))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_split.py -q`
Expected: PASS (all 6). Then run the full gates (see Global Constraints).

- [ ] **Step 7: Commit green**

```bash
git add src/data_analyst_mcp/tools/split.py src/data_analyst_mcp/server.py
git commit -m "green: split_dataset partitions a dataset into seeded train/test derived datasets"
```

---

### Task 2: `split_dataset` validation and error paths

**Files:**
- Modify: `src/data_analyst_mcp/tools/split.py` (no changes expected — Task 1 already implements these paths; this task pins them)
- Modify: `tests/test_split.py`

**Interfaces:**
- Consumes: Task 1's tool.
- Produces: pinned error contracts for `dataset_not_found`, `test_fraction_out_of_range`, `invalid_name`, `split_name_conflict`, `dataset_name_collision` (atomic — nothing half-registered), `dataset_too_small`, overwrite semantics.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split.py`:

```python
import pytest


def test_split_dataset_unknown_source(call_tool: Any) -> None:
    result = call_tool("split_dataset", {"name": "nope"})
    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_not_found"


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_dataset_rejects_fraction_endpoints(
    call_tool: Any, load_df_into_session: Any, fraction: float
) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", "test_fraction": fraction})
    assert result["ok"] is False
    assert result["error"]["type"] == "test_fraction_out_of_range"


@pytest.mark.parametrize("bad", ["1train", "has space", "has-dash", ""])
def test_split_dataset_rejects_invalid_output_names(
    call_tool: Any, load_df_into_session: Any, bad: str
) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", "train_name": bad})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_name"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_name": "same", "test_name": "same"},
        {"train_name": "base"},
        {"test_name": "base"},
    ],
)
def test_split_dataset_rejects_name_conflicts(
    call_tool: Any, load_df_into_session: Any, kwargs: dict[str, str]
) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", **kwargs})
    assert result["ok"] is False
    assert result["error"]["type"] == "split_name_conflict"


def test_split_dataset_collision_is_atomic(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """If one output name collides, NEITHER table is created/registered."""
    import pandas as pd

    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    load_df_into_session("taken", pd.DataFrame({"z": [1]}))

    result = call_tool(
        "split_dataset", {"name": "base", "train_name": "fresh", "test_name": "taken"}
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_name_collision"
    assert "fresh" not in _session.get_datasets()
    con = _session.get_connection()
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "fresh" not in tables


def test_split_dataset_overwrite_replaces_existing(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    _load_ten_rows(load_df_into_session)
    load_df_into_session("taken", pd.DataFrame({"z": [1]}))

    result = call_tool(
        "split_dataset",
        {"name": "base", "train_name": "fresh", "test_name": "taken", "overwrite": True},
    )

    assert result["ok"] is True
    assert result["test"]["name"] == "taken"


def test_split_dataset_rejects_single_row_source(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session("tiny", pd.DataFrame({"x": [1]}))
    result = call_tool("split_dataset", {"name": "tiny"})
    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_too_small"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_split.py -q`
Expected: all PASS immediately if Task 1's implementation is complete. If any fail, the implementation has a gap — fix `split.py` until green. Either way the commit sequence below applies (red commit only if a fix was needed; when everything passes on first run, these are pinning tests and land as a single `test:` commit).

- [ ] **Step 3: Commit**

If all passed without code changes:

```bash
git add tests/test_split.py
git commit -m "test: pin split_dataset validation and error-path contracts"
```

If a fix was needed: commit the failing test first (`red: split_dataset <behavior>`), then the fix (`green: split_dataset <behavior>`).

---

### Task 3: `split_dataset` stratified mode

**Files:**
- Modify: `tests/test_split.py`
- Modify: `src/data_analyst_mcp/tools/split.py` (only if a gap surfaces — the stratified branch shipped in Task 1)

**Interfaces:**
- Consumes: `_assign_is_test` stratified branch, `_strata_counts`.
- Produces: pinned contracts for per-stratum proportions, NULL stratum, `small_strata` warning, `stratification_too_sparse`, `stratify_column_missing`, stratified determinism.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split.py`:

```python
def _load_strata(load_df_into_session: Any) -> None:
    import pandas as pd

    # 4×'a', 3×'b', 3×NULL — 10 rows.
    load_df_into_session(
        "strat",
        pd.DataFrame(
            {
                "g": ["a", "a", "a", "a", "b", "b", "b", None, None, None],
                "x": list(range(10)),
            }
        ),
    )


def test_split_dataset_stratified_counts_and_strata_table(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """fraction=0.25: 'a' (4 rows) → 1 test; 'b' (3) → 1 test; NULL (3) → 1 test."""
    _load_strata(load_df_into_session)

    result = call_tool(
        "split_dataset", {"name": "strat", "stratify_by": "g", "test_fraction": 0.25}
    )

    assert result["ok"] is True
    assert result["test"]["rows"] == 3
    assert result["train"]["rows"] == 7
    strata = result["strata"]
    assert [s["value"] for s in strata] == ["a", "b", None]
    assert [(s["train_rows"], s["test_rows"]) for s in strata] == [(3, 1), (2, 1), (2, 1)]


def test_split_dataset_stratified_is_deterministic(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_strata(load_df_into_session)
    call_tool("split_dataset", {"name": "strat", "stratify_by": "g"})
    call_tool(
        "split_dataset",
        {"name": "strat", "stratify_by": "g", "train_name": "t2", "test_name": "e2"},
    )
    con = _session.get_connection()
    a = sorted(r[0] for r in con.execute('SELECT x FROM "strat_test"').fetchall())
    b = sorted(r[0] for r in con.execute('SELECT x FROM "e2"').fetchall())
    assert a == b


def test_split_dataset_small_strata_go_to_train_with_warning(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session(
        "mixed",
        pd.DataFrame({"g": ["a"] * 8 + ["solo"], "x": list(range(9))}),
    )

    result = call_tool("split_dataset", {"name": "mixed", "stratify_by": "g"})

    assert result["ok"] is True
    assert "small_strata" in result["warnings"]
    solo = [s for s in result["strata"] if s["value"] == "solo"][0]
    assert solo == {"value": "solo", "train_rows": 1, "test_rows": 0}


def test_split_dataset_all_singleton_strata_rejected(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session(
        "singles", pd.DataFrame({"g": ["a", "b", "c"], "x": [1, 2, 3]})
    )

    result = call_tool("split_dataset", {"name": "singles", "stratify_by": "g"})

    assert result["ok"] is False
    assert result["error"]["type"] == "stratification_too_sparse"


def test_split_dataset_unknown_stratify_column(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base", "stratify_by": "ghost"})
    assert result["ok"] is False
    assert result["error"]["type"] == "stratify_column_missing"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_split.py -q`
Expected: PASS if Task 1's stratified branch is correct; treat failures as in Task 2 Step 2 (red/green a fix). Commit accordingly:

```bash
git add tests/test_split.py
git commit -m "test: pin split_dataset stratified proportions, NULL stratum, and sparse-strata guards"
```

---

### Task 4: Recorder replay for splits (setup cell + per-call cell + checksum guard)

**Files:**
- Modify: `src/data_analyst_mcp/recorder.py`
- Modify: `src/data_analyst_mcp/tools/split.py` (`_record_split` uses the shared snippet)
- Modify: `src/data_analyst_mcp/tools/materialize.py:116` (format guard tuple)
- Modify: `tests/test_split.py`

**Interfaces:**
- Consumes: `DatasetEntry` with `format="split"` and the `read_options` keys registered in Task 1 (`source`, `seed`, `test_fraction`, `stratify_by`, `train_name`, `test_name`, `rid_column`, `role`, `membership_checksum`).
- Produces: `recorder.split_replay_source(*, source: str, train_name: str, test_name: str, seed: int, test_fraction: float, stratify_by: str | None, rid_column: str, membership_checksum: str) -> str` — used by both the setup cell and `split.py`'s per-call cell. The setup cell's derived pass becomes a single merged registration-order loop over `derived` and `split` entries.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split.py`:

```python
def _setup_source(call_tool: Any) -> str:
    from data_analyst_mcp.recorder import get_recorder

    nb = get_recorder().to_notebook(include_setup=True)
    return nb.cells[0]["source"]


def test_split_setup_cell_recreates_both_tables_with_checksum_assert(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True

    src = _setup_source(call_tool)
    assert 'CREATE OR REPLACE TABLE "base_train"' in src
    assert 'CREATE OR REPLACE TABLE "base_test"' in src
    assert "RandomState(42)" in src
    checksum = _session.get_datasets()["base_test"].read_options["membership_checksum"]
    assert checksum in src
    assert "drifted at replay" in src


def test_split_percall_cell_uses_same_replay_source(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp.recorder import get_recorder

    _load_ten_rows(load_df_into_session)
    call_tool("split_dataset", {"name": "base"})

    code_cells = [
        c for c in get_recorder().cells
        if c["cell_type"] == "code" and c["metadata"]["tool_name"] == "split_dataset"
    ]
    assert len(code_cells) == 1
    assert "RandomState(42)" in code_cells[0]["source"]
    assert 'CREATE OR REPLACE TABLE "base_test"' in code_cells[0]["source"]


def test_split_replay_snippet_executes_and_reproduces_membership(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Execute the emitted snippet against a fresh DuckDB connection loaded
    with the same rows — the checksum assert inside the snippet must pass,
    proving the snippet's algorithm matches the live one."""
    import duckdb
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.recorder import split_replay_source

    _load_ten_rows(load_df_into_session)
    result = call_tool("split_dataset", {"name": "base"})
    assert result["ok"] is True
    entry = _session.get_datasets()["base_test"]

    snippet = split_replay_source(
        source="base",
        train_name="base_train",
        test_name="base_test",
        seed=42,
        test_fraction=0.25,
        stratify_by=None,
        rid_column=entry.read_options["rid_column"],
        membership_checksum=entry.read_options["membership_checksum"],
    )
    con = duckdb.connect()
    con.register("__base_src", pd.DataFrame({"x": list(range(10))}))
    con.execute('CREATE TABLE "base" AS SELECT * FROM __base_src')
    exec(snippet, {"con": con, "np": np, "pd": pd})  # noqa: S102 - replay snippet under test
    test_x = sorted(r[0] for r in con.execute('SELECT x FROM "base_test"').fetchall())
    assert test_x == [1, 8]


def test_split_stratified_setup_cell_replays(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_strata(load_df_into_session)
    result = call_tool("split_dataset", {"name": "strat", "stratify_by": "g"})
    assert result["ok"] is True
    src = _setup_source(call_tool)
    assert 'CREATE OR REPLACE TABLE "strat_test"' in src
    assert "isna" in src  # stratified branch emitted


def test_materialize_overwrite_of_split_entry_keeps_base_loader_none(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """A split entry has no file loader; materialize overwrite must not
    fabricate one with path '(split)'."""
    from data_analyst_mcp import session as _session

    _load_ten_rows(load_df_into_session)
    call_tool("split_dataset", {"name": "base"})
    result = call_tool(
        "materialize_query",
        {"sql": 'SELECT * FROM "base_train" WHERE x > 2', "name": "base_train",
         "overwrite": True},
    )
    assert result["ok"] is True
    assert _session.get_datasets()["base_train"].base_loader is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_split.py -q`
Expected: the five new tests FAIL (`split_replay_source` doesn't exist; setup cell has no split branch; materialize fabricates a base_loader with path `(split)`).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_split.py
git commit -m "red: emitted notebooks recreate split datasets behind a membership-checksum assert"
```

- [ ] **Step 4: Implement `split_replay_source` in `recorder.py`**

Add after `_file_load_stmt` (after current line 67):

```python
def _split_assignment_lines(
    source: str, seed: int, test_fraction: float, stratify_by: str | None
) -> list[str]:
    """Notebook lines that rebuild the boolean membership array.

    Algorithm-identical to ``tools.split._assign_is_test`` — the
    membership-checksum assert emitted below is the drift guard between
    the two implementations.
    """
    src_q = '"' + source.replace('"', '""') + '"'
    if stratify_by is None:
        return [
            f"_split_n = con.sql('SELECT COUNT(*) FROM {src_q}').fetchone()[0]",
            f"_split_rng = np.random.RandomState({seed})",
            "_split_is_test = np.zeros(_split_n, dtype=bool)",
            f"_split_n_test = min(max(int(round(_split_n * {test_fraction})), 1), _split_n - 1)",
            "_split_is_test[_split_rng.permutation(_split_n)[:_split_n_test]] = True",
        ]
    col_q = '"' + stratify_by.replace('"', '""') + '"'
    return [
        f"_split_labels = con.sql('SELECT {col_q} FROM {src_q}').df().iloc[:, 0]",
        f"_split_rng = np.random.RandomState({seed})",
        "_split_is_test = np.zeros(len(_split_labels), dtype=bool)",
        "_split_null = _split_labels.isna().to_numpy()",
        "_split_values = sorted(_split_labels[~_split_labels.isna()].unique().tolist(), key=str)",
        "_split_groups = [np.where((_split_labels == _v).to_numpy() & ~_split_null)[0] for _v in _split_values]",
        "if _split_null.any():",
        "    _split_groups.append(np.where(_split_null)[0])",
        "for _rids in _split_groups:",
        "    if len(_rids) < 2:",
        "        continue",
        f"    _n_t = min(max(int(round(len(_rids) * {test_fraction})), 1), len(_rids) - 1)",
        "    _split_is_test[_rids[_split_rng.permutation(len(_rids))[:_n_t]]] = True",
    ]


_SPLIT_CHECKSUM_DEF = """\
def _split_checksum(_df):
    import hashlib as _hl
    import math as _math
    _acc = 0
    for _row in _df.itertuples(index=False, name=None):
        _parts = []
        for _v in _row:
            try:
                _is_na = bool(pd.isna(_v))
            except (TypeError, ValueError):
                _is_na = False
            if _v is None or _is_na:
                _parts.append('<null>')
            elif isinstance(_v, (bool, np.bool_)):
                _parts.append('true' if bool(_v) else 'false')
            elif isinstance(_v, (float, np.floating)):
                _f = float(_v)
                _parts.append('<null>' if _math.isnan(_f) else repr(_f))
            elif isinstance(_v, (int, np.integer)):
                _parts.append(repr(int(_v)))
            elif isinstance(_v, str):
                _parts.append(_v)
            else:
                _parts.append(str(_v))
        _h = _hl.sha256('|'.join(_parts).encode('utf-8')).digest()
        _acc ^= int.from_bytes(_h[:16], 'big')
    return format(_acc, '032x')"""


def split_replay_source(
    *,
    source: str,
    train_name: str,
    test_name: str,
    seed: int,
    test_fraction: float,
    stratify_by: str | None,
    rid_column: str,
    membership_checksum: str,
) -> str:
    """Self-contained notebook snippet that recreates a train/test split.

    Rebuilds membership with the same ``RandomState`` algorithm the live
    tool used, recreates both tables, then asserts the order-independent
    membership checksum of the recreated test table. For file-backed
    sources the source hash assert upstream makes this deterministic; for
    derived sources whose SQL is not order-preserving, row-order drift
    fails loudly here instead of silently changing the split (spec §5.6b
    row-order tiers).
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
    message = f"Split membership for {test_name!r} drifted at replay (source row order changed)."
    lines.extend(
        [
            "_split_assign = pd.DataFrame({'rid': np.arange(len(_split_is_test), "
            "dtype=np.int64), 'is_test': _split_is_test})",
            "con.register('__data_analyst_split_assign', _split_assign)",
            f"con.execute({train_stmt!r})",
            f"con.execute({test_stmt!r})",
            "con.unregister('__data_analyst_split_assign')",
            _SPLIT_CHECKSUM_DEF,
            f"assert _split_checksum(con.sql('SELECT * FROM {test_q}').df()) == "
            f"{membership_checksum!r}, {message!r}",
        ]
    )
    return "\n".join(lines)
```

- [ ] **Step 5: Merge the setup-cell derived pass**

In `_build_setup_source`, replace the second pass (current lines 180-198, the `for name, entry ... if entry.format != "derived": continue` loop) with one merged loop:

```python
    # Second pass: derived and split datasets, in registration order.
    # Registration order IS topological order — a derived/split entry can
    # only reference earlier-registered tables — so one interleaved loop
    # replaces the old derived-only pass. Split blocks are emitted once
    # per pair, keyed off the test-role entry (which carries the
    # membership checksum). Overwriting a split output with
    # materialize_query drops that side's split recipe; replay then fails
    # loudly (missing table / checksum) rather than silently recomputing.
    for name, entry in _session.get_datasets().items():
        if entry.format == "derived":
            derived_sql = entry.read_options.get("sql", "")
            stmt = f'CREATE OR REPLACE TABLE "{name}" AS {derived_sql}'
            lines.append(f"con.execute({stmt!r})")
        elif entry.format == "split" and entry.read_options.get("role") == "test":
            opts = entry.read_options
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
                )
            )
```

Keep the existing comment about chained derived datasets; fold it into the new comment block. (`split_replay_source` is defined in this same module — no import needed.)

- [ ] **Step 6: Use the shared snippet in `split.py`'s per-call cell**

In `tools/split.py`, change the import to `from data_analyst_mcp.recorder import get_recorder, split_replay_source` and replace `_record_split`'s placeholder `code = ...` with:

```python
    code = split_replay_source(
        source=payload.name,
        train_name=train_name,
        test_name=test_name,
        seed=payload.seed,
        test_fraction=payload.test_fraction,
        stratify_by=payload.stratify_by,
        rid_column=rid,
        membership_checksum=checksum,
    )
```

Also update `_record_split`'s docstring (drop the Task 4 placeholder sentence).

- [ ] **Step 7: Fix the materialize format guard**

In `src/data_analyst_mcp/tools/materialize.py` line 116, change:

```python
        if existing.format not in ("derived", "dataframe"):
```

to:

```python
        if existing.format not in ("derived", "dataframe", "split"):
```

(Split entries have no file loader; without this, overwriting a split output would fabricate a base_loader with the unloadable path `"(split)"`.)

- [ ] **Step 8: Run tests + full gates**

Run: `uv run pytest tests/test_split.py tests/test_recorder.py tests/test_emit_notebook.py tests/test_materialize.py -q` then the full suite.
Expected: PASS.

- [ ] **Step 9: Commit green**

```bash
git add src/data_analyst_mcp/recorder.py src/data_analyst_mcp/tools/split.py src/data_analyst_mcp/tools/materialize.py
git commit -m "green: emitted notebooks recreate split datasets behind a membership-checksum assert"
```

---

### Task 5: Extract `fit_prepared` from `fit_model` (refactor, no behavior change)

**Files:**
- Modify: `src/data_analyst_mcp/tools/models.py`
- Modify: `src/data_analyst_mcp/tools/evaluate.py`

**Interfaces:**
- Produces (in `models.py`): `FormulaError = _FormulaError` (public alias) and:
  `fit_prepared(payload: FitModelInput, df: Any) -> dict[str, Any]` — negbin endog validation + bool coercion + fit dispatch; returns the result dict (with `"_result"` on success, or an error envelope); raises `FormulaError` on formula problems. `fit_model` delegates to it.
- Produces (in `evaluate.py`): public aliases `formula_outcome = _formula_outcome`, `validate_outcome_dtype = _validate_outcome_dtype`, `count_metrics = _count_metrics`.
- Task 6 consumes all of these.

- [ ] **Step 1: Extract `fit_prepared` in `models.py`**

Add after `fit_model` (after current line 175):

```python
# Public alias so sibling tools (cross_validate) can catch formula failures
# without reaching into a private name.
FormulaError = _FormulaError


def fit_prepared(payload: FitModelInput, df: Any) -> dict[str, Any]:
    """Shared fit path: negbin endog validation + dispatch, warnings merged.

    Returns the result dict — ``{"ok": True, ..., "_result": Results}`` on
    success, or a ``build_error`` envelope. Raises :class:`FormulaError`
    on patsy/column-binding failures. ``fit_model`` and ``cross_validate``
    both go through here so their validation/error taxonomy can never
    drift apart.
    """
    extra_warnings: list[str] = []
    if payload.kind == "negbin":
        df, validation_error, coerce_warning = _validate_negbin_endog(df, payload.formula)
        if validation_error is not None:
            return validation_error
        if coerce_warning is not None:
            extra_warnings.append(coerce_warning)
    result = _fit_dispatch(payload, df)
    if result.get("ok") and extra_warnings:
        existing: list[str] = list(result.get("warnings") or [])
        result["warnings"] = extra_warnings + existing
    return result
```

Then shrink `fit_model`'s body: replace current lines 135-155 (the `try:` block through the `result["warnings"] = extra_warnings + existing` merge) with:

```python
    try:
        df = _materialize_dataframe(payload.name)
        result = fit_prepared(payload, df)
    except _FormulaError as fe:
        return build_error(
            type="formula_error",
            message=str(fe),
            hint=("Verify column names exist and the formula parses, e.g. 'y ~ x + C(group)'."),
        )
```

- [ ] **Step 2: Add public aliases in `evaluate.py`**

At the bottom of `src/data_analyst_mcp/tools/evaluate.py`:

```python
# Public aliases shared with cross_validate (tools/crossval.py).
formula_outcome = _formula_outcome
validate_outcome_dtype = _validate_outcome_dtype
count_metrics = _count_metrics
```

- [ ] **Step 3: Run the full suite + gates**

Run: `uv run pytest tests/ -q`
Expected: PASS — zero behavior change (`tests/test_models.py`, `tests/test_model_registry.py` are the sentinels).

- [ ] **Step 4: Commit**

```bash
git add src/data_analyst_mcp/tools/models.py src/data_analyst_mcp/tools/evaluate.py
git commit -m "refactor: extract fit_prepared and public evaluate helpers for cross_validate"
```

---

### Task 6: `cross_validate` OLS happy path

**Files:**
- Create: `src/data_analyst_mcp/tools/crossval.py`
- Modify: `src/data_analyst_mcp/server.py` (import + wrapper after the `evaluate_model` wrapper, currently ending line 592)
- Create: `tests/test_crossval.py`

**Interfaces:**
- Consumes: `models.FitModelInput`, `models.fit_prepared`, `models.FormulaError`, `evaluate.formula_outcome`, `evaluate.validate_outcome_dtype`, `evaluate.count_metrics` (Task 5).
- Produces: `CrossValidateInput` (`name: str`, `formula: str`, `kind: Literal["ols","logistic","poisson","negbin"] = "ols"`, `robust: bool = False`, `k: int = 5`, `seed: int = 42`, `threshold: float = 0.5`) and `cross_validate(payload) -> dict[str, Any]`. Internal helpers Task 7/8 rely on: `_fold_ids(y: Any, k: int, seed: int, stratified: bool) -> Any`, `_fold_metrics(kind: str, y_true: Any, y_pred: Any, threshold: float) -> dict[str, float]`, `_classify_fold_failure(kind: str, exc: Exception | None, result: Any | None) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_crossval.py`:

```python
"""Tests for the ``cross_validate`` tool."""

from __future__ import annotations

from typing import Any

import numpy as np


def _load_linear(load_df_into_session: Any, n: int = 40) -> None:
    """y = 2x + noise — deterministic fixture."""
    import pandas as pd

    rng = np.random.RandomState(0)
    x = np.arange(n, dtype=float)
    y = 2.0 * x + rng.normal(0, 1.0, size=n)
    load_df_into_session("lin", pd.DataFrame({"x": x, "y": y}))


def test_cross_validate_ols_shape(call_tool: Any, load_df_into_session: Any) -> None:
    _load_linear(load_df_into_session)

    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})

    assert result["ok"] is True
    assert result["kind"] == "ols"
    assert result["k"] == 5
    assert result["seed"] == 42
    assert result["stratified"] is False
    assert result["n_obs"] == 40
    assert result["dropped_rows"] == 0
    assert result["fold_failures"] == []
    assert sorted(result["fold_sizes"]) == [8, 8, 8, 8, 8]
    metrics = result["metrics"]
    assert set(metrics.keys()) == {"rmse", "mae", "r_squared"}
    for m in metrics.values():
        assert set(m.keys()) == {"mean", "std", "per_fold"}
        assert len(m["per_fold"]) == 5
        assert all(isinstance(v, float) for v in m["per_fold"])
    assert isinstance(result["interpretation"], str) and result["interpretation"]
    assert "model_name" not in result  # fits are ephemeral


def test_cross_validate_never_touches_model_registry(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_linear(load_df_into_session)
    call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})
    assert _session.get_models() == {}


def test_cross_validate_ols_matches_manual_recompute(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Independent recompute: same RandomState fold assignment, statsmodels
    array fits, numpy metrics. Aggregates must match to 1e-10."""
    import pandas as pd
    import statsmodels.api as sm

    _load_linear(load_df_into_session)
    result = call_tool(
        "cross_validate", {"name": "lin", "formula": "y ~ x", "k": 4, "seed": 7}
    )
    assert result["ok"] is True

    from data_analyst_mcp import session as _session

    df = _session.get_connection().execute('SELECT * FROM "lin"').df()
    import statsmodels.formula.api as smf

    full = smf.ols("y ~ x", data=df).fit()
    y = np.asarray(full.model.endog)
    X = np.asarray(full.model.exog)
    n = len(y)
    rng = np.random.RandomState(7)
    fold = np.empty(n, dtype=int)
    perm = rng.permutation(n)
    fold[perm] = np.arange(n) % 4
    rmses = []
    for i in range(4):
        tr = fold != i
        res = sm.OLS(y[tr], X[tr]).fit()
        pred = np.asarray(res.predict(X[~tr]))
        rmses.append(float(np.sqrt(np.mean((y[~tr] - pred) ** 2))))
    assert abs(result["metrics"]["rmse"]["mean"] - float(np.mean(rmses))) < 1e-10
    assert abs(result["metrics"]["rmse"]["std"] - float(np.std(rmses, ddof=1))) < 1e-10


def test_cross_validate_same_seed_is_deterministic(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_linear(load_df_into_session)
    a = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})
    b = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})
    assert a["metrics"] == b["metrics"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_crossval.py -q`
Expected: FAIL (unknown tool `cross_validate`).

- [ ] **Step 3: Commit red**

```bash
git add tests/test_crossval.py
git commit -m "red: cross_validate returns k-fold OLS metrics with ephemeral fits"
```

- [ ] **Step 4: Implement `tools/crossval.py`**

Create `src/data_analyst_mcp/tools/crossval.py`:

```python
"""k-fold cross-validation — re-fitting complement to ``evaluate_model``.

Fits are ephemeral: there is no ``model_name`` parameter and the model
registry is never touched. A full-data preflight fit goes through
``models.fit_prepared`` (fit_model's exact validation path), surfacing
fit_model's whole error taxonomy before any fold work; the preflight's
patsy design matrices are then sliced per fold, so categorical levels
are encoded globally and a fold-local level can never crash scoring
(spec §5.11d).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder
from data_analyst_mcp.tools import evaluate as _evaluate
from data_analyst_mcp.tools import models as _models

logger = logging.getLogger(__name__)


def _sm() -> Any:
    """Return ``statsmodels.api`` as an untyped module."""
    import statsmodels.api as _sm_mod  # type: ignore[reportMissingTypeStubs]

    return _sm_mod


def _sklearn_metrics() -> Any:
    """Return ``sklearn.metrics`` as untyped — strict pyright + sklearn stubs are noisy."""
    import sklearn.metrics as _m  # type: ignore[reportMissingTypeStubs]

    return _m


def _materialize_dataframe(name: str) -> Any:
    """Materialize a registered dataset as a pandas DataFrame via DuckDB."""
    con = session.get_connection()
    quoted = '"' + name.replace('"', '""') + '"'
    return con.execute(f"SELECT * FROM {quoted}").df()


class CrossValidateInput(BaseModel):
    """Inputs for ``cross_validate``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    formula: str
    kind: Literal["ols", "logistic", "poisson", "negbin"] = "ols"
    robust: bool = False
    k: int = 5
    seed: int = 42
    threshold: float = 0.5


def _fold_ids(y: Any, k: int, seed: int, stratified: bool) -> Any:
    """Fold id per row via a RandomState permutation.

    Stratified mode (logistic) permutes within each outcome class so every
    fold keeps the class balance; callers must have verified each class
    has >= k members first.
    """
    n = len(y)
    fold = np.empty(n, dtype=int)
    rng = np.random.RandomState(seed)
    if stratified:
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            perm = idx[rng.permutation(len(idx))]
            fold[perm] = np.arange(len(perm)) % k
    else:
        perm = rng.permutation(n)
        fold[perm] = np.arange(n) % k
    return fold


def _fit_fold(kind: str, robust: bool, y_tr: Any, X_tr: Any) -> Any:
    """Array-interface statsmodels fit for one training slice."""
    sm = _sm()
    if kind == "ols":
        return sm.OLS(y_tr, X_tr).fit(cov_type="HC3" if robust else "nonrobust")
    if kind == "logistic":
        return sm.Logit(y_tr, X_tr).fit(disp=0)
    if kind == "poisson":
        return sm.Poisson(y_tr, X_tr).fit(disp=0)
    return sm.NegativeBinomial(y_tr, X_tr).fit(disp=0)


def _classify_fold_failure(kind: str, exc: Exception | None, result: Any | None) -> str:
    """Map a fold-local fit failure to fit_model's error-type strings."""
    if exc is not None:
        from statsmodels.tools.sm_exceptions import (  # type: ignore[reportMissingTypeStubs]
            PerfectSeparationError,
        )

        if kind == "logistic" and isinstance(exc, PerfectSeparationError):
            return "perfect_separation"
        return "convergence_failed"
    # Returned-but-degenerate fit (logistic/negbin non-convergence).
    if kind == "logistic" and result is not None:
        degenerate = _models._detect_logistic_separation(result)  # type: ignore[reportPrivateUsage]
        if degenerate is not None:
            return str(degenerate["error"]["type"])
    return "convergence_failed"


def _fold_converged(kind: str, result: Any) -> bool:
    """MLE-family fits must report convergence; OLS always converges."""
    if kind == "ols":
        return True
    return bool(result.mle_retvals.get("converged", False))


def _fold_metrics(kind: str, y_true: Any, y_pred: Any, threshold: float) -> dict[str, float]:
    """Held-out fold metrics — same families as evaluate_model."""
    if kind == "logistic":
        skm: Any = _sklearn_metrics()
        y_i: Any = y_true.astype(int)
        y_cls: Any = (y_pred >= threshold).astype(int)
        clipped: Any = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return {
            "roc_auc": float(skm.roc_auc_score(y_i, y_pred)),
            "pr_auc": float(skm.average_precision_score(y_i, y_pred)),
            "brier": float(skm.brier_score_loss(y_i, y_pred)),
            "log_loss": float(skm.log_loss(y_i, clipped, labels=[0, 1])),
            "accuracy": float(np.mean(np.asarray(y_cls == y_i, dtype=float))),
            "precision": float(skm.precision_score(y_i, y_cls, zero_division=0)),
            "recall": float(skm.recall_score(y_i, y_cls, zero_division=0)),
            "f1": float(skm.f1_score(y_i, y_cls, zero_division=0)),
        }
    if kind == "ols":
        err: Any = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        ss_res = float(np.sum(err * err))
        y_arr: Any = np.asarray(y_true, dtype=float)
        ss_tot = float(np.sum((y_arr - float(np.mean(y_arr))) ** 2))
        return {
            "rmse": float(np.sqrt(np.mean(err * err))),
            "mae": float(np.mean(np.abs(err))),
            "r_squared": 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        }
    return _evaluate.count_metrics(y_true=y_true, mu=y_pred, kind=kind)


def cross_validate(payload: CrossValidateInput) -> dict[str, Any]:
    """k-fold cross-validated metrics for a formula on a dataset."""
    if payload.name not in session.get_datasets():
        return build_error(
            type="dataset_not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    if payload.kind == "negbin" and payload.robust:
        return build_error(
            type="robust_not_supported",
            message="robust=True is not supported for kind='negbin'.",
            hint="NB2 robust SE is not implemented in this server. Set `robust=False`.",
        )
    if not (2 <= payload.k <= 20):
        return build_error(
            type="k_out_of_range",
            message=f"k must be in [2, 20]; got {payload.k}.",
            hint="Pick a fold count in [2, 20].",
        )
    if not (0.0 < payload.threshold < 1.0):
        return build_error(
            type="threshold_out_of_range",
            message=f"threshold must be in the open interval (0, 1); got {payload.threshold}.",
            hint="Pick a threshold strictly between 0 and 1.",
        )

    df: Any = _materialize_dataframe(payload.name)
    outcome = _evaluate.formula_outcome(payload.formula)
    warning_flags: list[str] = []
    if outcome in df.columns:
        dtype_error = _evaluate.validate_outcome_dtype(payload.kind, df[outcome], warning_flags)
        if dtype_error is not None:
            return dtype_error

    fm_payload = _models.FitModelInput(
        name=payload.name,
        formula=payload.formula,
        kind=payload.kind,
        robust=payload.robust,
        model_name=None,
    )
    try:
        full = _models.fit_prepared(fm_payload, df)
    except _models.FormulaError as fe:
        return build_error(
            type="formula_error",
            message=str(fe),
            hint=("Verify column names exist and the formula parses, e.g. 'y ~ x + C(group)'."),
        )
    if not full.get("ok"):
        return full  # perfect_separation / convergence_failed / negbin dtype
    m: Any = full.pop("_result")
    warning_flags.extend(list(full.get("warnings") or []))

    y: Any = np.asarray(m.model.endog, dtype=float)
    X: Any = np.asarray(m.model.exog, dtype=float)
    n = len(y)
    dropped_rows = int(len(df) - n)
    if n < payload.k:
        return build_error(
            type="k_out_of_range",
            message=(
                f"k={payload.k} exceeds the {n} usable rows after NaN drops "
                f"({dropped_rows} dropped)."
            ),
            hint="Lower k or clean the missing predictor/outcome rows.",
        )
    stratified = payload.kind == "logistic"
    if stratified:
        for cls in (0, 1):
            n_cls = int(np.sum(y == cls))
            if n_cls < payload.k:
                return build_error(
                    type="outcome_class_too_small",
                    message=(
                        f"Stratified {payload.k}-fold CV needs at least {payload.k} rows "
                        f"of each outcome class; class {cls} has {n_cls}."
                    ),
                    hint="Lower k or gather more minority-class rows.",
                )
    fold: Any = _fold_ids(y, payload.k, payload.seed, stratified)

    n_params = int(X.shape[1])
    for i in range(payload.k):
        train_size = int(np.sum(fold != i))
        if train_size <= n_params:
            return build_error(
                type="fold_too_small",
                message=(
                    f"Fold {i} would train on {train_size} rows but the design "
                    f"matrix has {n_params} parameters."
                ),
                hint="Lower k or simplify the formula.",
            )

    per_fold: list[dict[str, float] | None] = []
    fold_sizes: list[int] = []
    fold_failures: list[dict[str, Any]] = []
    for i in range(payload.k):
        tr: Any = fold != i
        te: Any = ~tr
        fold_sizes.append(int(np.sum(te)))
        try:
            res: Any = _fit_fold(payload.kind, payload.robust, y[tr], X[tr])
        except Exception as exc:
            fold_failures.append(
                {"fold": i, "error_type": _classify_fold_failure(payload.kind, exc, None)}
            )
            per_fold.append(None)
            continue
        if not _fold_converged(payload.kind, res):
            fold_failures.append(
                {"fold": i, "error_type": _classify_fold_failure(payload.kind, None, res)}
            )
            per_fold.append(None)
            continue
        y_pred: Any = np.asarray(res.predict(X[te]))
        per_fold.append(_fold_metrics(payload.kind, y[te], y_pred, payload.threshold))

    successes = [p for p in per_fold if p is not None]
    if not successes:
        return build_error(
            type="cv_fit_failed",
            message=f"All {payload.k} folds failed to fit.",
            hint="See fit_model on the full dataset for a diagnosable single-fit error.",
        )
    if fold_failures:
        warning_flags.append("fold_failures")

    metric_keys = list(successes[0].keys())
    metrics: dict[str, Any] = {}
    for key in metric_keys:
        vals = [p[key] for p in successes]
        metrics[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "per_fold": [None if p is None else p[key] for p in per_fold],
        }

    result: dict[str, Any] = {
        "ok": True,
        "name": payload.name,
        "formula": payload.formula,
        "kind": payload.kind,
        "k": payload.k,
        "seed": payload.seed,
        "stratified": stratified,
        "metrics": metrics,
        "fold_sizes": fold_sizes,
        "n_obs": n,
        "dropped_rows": dropped_rows,
        "fold_failures": fold_failures,
        "warnings": warning_flags,
        "interpretation": _interpretation(payload, metrics, fold_failures),
    }
    _record_cross_validate(payload, result)
    return result


_PRIMARY_METRIC = {"ols": "rmse", "logistic": "roc_auc", "poisson": "rmse", "negbin": "rmse"}


def _interpretation(
    payload: CrossValidateInput, metrics: dict[str, Any], fold_failures: list[dict[str, Any]]
) -> str:
    """2-3 sentence summary anchored on the kind's primary metric."""
    key = _PRIMARY_METRIC[payload.kind]
    m = metrics[key]
    text = (
        f"{payload.k}-fold cross-validation of {payload.kind} model "
        f"'{payload.formula}' on '{payload.name}': {key} = "
        f"{m['mean']:.4g} ± {m['std']:.4g} across folds."
    )
    if fold_failures:
        text += f" {len(fold_failures)} fold(s) failed to fit and were excluded."
    return text


def _record_cross_validate(payload: CrossValidateInput, result: dict[str, Any]) -> None:
    """Markdown + code cell — placeholder body, replaced in the recorder task."""
    key = _PRIMARY_METRIC[payload.kind]
    m = result["metrics"][key]
    md = (
        f"### {payload.k}-fold CV of {payload.kind} on `{payload.name}`\n\n"
        f"- Formula: `{payload.formula}`\n"
        f"- {key} = {m['mean']:.4g} ± {m['std']:.4g}\n"
        f"- {result['interpretation']}"
    )
    code = f"# cross_validate({payload.name!r}) — replay source lands in the recorder task"
    get_recorder().record(markdown=md, code=code, tool_name="cross_validate")
```

- [ ] **Step 5: Register the wrapper in `server.py`**

Add the import `from data_analyst_mcp.tools import crossval as _crossval` next to the other tool imports, then insert after the `evaluate_model` wrapper (after current line 592):

```python
@mcp.tool()
def cross_validate(
    name: str,
    formula: str,
    kind: str = "ols",
    robust: bool = False,
    k: int = 5,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """k-fold cross-validated metrics for a model formula on a dataset.

    The re-fitting complement to ``evaluate_model``: fits are ephemeral
    and the model registry is never touched. A full-data preflight fit
    surfaces fit_model's error taxonomy (formula_error,
    perfect_separation, convergence_failed, ...) before any fold work;
    the folds reuse its design matrices so categorical levels are
    encoded globally. Logistic folds are auto-stratified by outcome
    class (requires >= k rows per class). Metrics match
    evaluate_model's families; response reports mean, std (ddof=1), and
    per-fold values. Errors: dataset_not_found, formula_error,
    perfect_separation, convergence_failed, outcome_dtype_mismatch,
    outcome_class_too_small, k_out_of_range, fold_too_small,
    cv_fit_failed, robust_not_supported, threshold_out_of_range.
    """
    try:
        payload = _crossval.CrossValidateInput(
            name=name,
            formula=formula,
            kind=kind,  # type: ignore[arg-type]
            robust=robust,
            k=k,
            seed=seed,
            threshold=threshold,
        )
        return _crossval.cross_validate(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("cross_validate failed")
        return build_error(type="internal", message=str(exc))
```

(If pyright rejects the `kind` Literal narrowing, validate via pydantic by passing the raw string — pydantic raises ValidationError for a bad literal, which the wrapper's except maps to `internal`; match how other Literal-taking wrappers in `server.py` handle it — check `fit_model`'s wrapper at line 426 and copy its exact approach.)

- [ ] **Step 6: Run tests + gates**

Run: `uv run pytest tests/test_crossval.py -q` then full gates.
Expected: PASS.

- [ ] **Step 7: Commit green**

```bash
git add src/data_analyst_mcp/tools/crossval.py src/data_analyst_mcp/server.py
git commit -m "green: cross_validate returns k-fold OLS metrics with ephemeral fits"
```

---

### Task 7: `cross_validate` logistic — auto-stratification and class guards

**Files:**
- Modify: `tests/test_crossval.py`
- Modify: `src/data_analyst_mcp/tools/crossval.py` (only if gaps surface)

**Interfaces:**
- Consumes: Task 6's implementation (stratified branch already coded).
- Produces: pinned contracts for `stratified: true`, per-fold class balance, `outcome_class_too_small`, logistic metric keys, threshold effect.

- [ ] **Step 1: Write the tests**

Append to `tests/test_crossval.py`:

```python
def _load_binary(load_df_into_session: Any, n: int = 60) -> None:
    """Noisy logistic fixture — not separable, both classes populated."""
    import pandas as pd

    rng = np.random.RandomState(1)
    x = rng.normal(0, 1, size=n)
    p = 1.0 / (1.0 + np.exp(-1.5 * x))
    y = (rng.uniform(size=n) < p).astype(int)
    # Force both classes present regardless of draw.
    y[0], y[1] = 0, 1
    load_df_into_session("bin", pd.DataFrame({"x": x, "y": y}))


def test_cross_validate_logistic_is_stratified_with_full_metrics(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_binary(load_df_into_session)

    result = call_tool(
        "cross_validate", {"name": "bin", "formula": "y ~ x", "kind": "logistic", "k": 3}
    )

    assert result["ok"] is True
    assert result["stratified"] is True
    assert set(result["metrics"].keys()) == {
        "roc_auc", "pr_auc", "brier", "log_loss",
        "accuracy", "precision", "recall", "f1",
    }
    for m in result["metrics"].values():
        assert len(m["per_fold"]) == 3


def test_cross_validate_logistic_folds_keep_class_balance(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Every fold's minority-class count differs from the ideal share by
    at most 1 — the structural guarantee that makes single-class folds
    impossible."""
    import statsmodels.formula.api as smf

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.crossval import _fold_ids

    _load_binary(load_df_into_session)
    df = _session.get_connection().execute('SELECT * FROM "bin"').df()
    full = smf.logit("y ~ x", data=df).fit(disp=0)
    y = np.asarray(full.model.endog, dtype=float)
    fold = _fold_ids(y, 3, 42, stratified=True)
    n_pos = int(np.sum(y == 1))
    for i in range(3):
        pos_in_fold = int(np.sum(y[fold == i] == 1))
        assert abs(pos_in_fold - n_pos / 3) <= 1
        assert pos_in_fold >= 1


def test_cross_validate_minority_class_smaller_than_k(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    rng = np.random.RandomState(2)
    load_df_into_session(
        "rare",
        pd.DataFrame({"x": rng.normal(size=30), "y": [1, 1] + [0] * 28}),
    )

    result = call_tool(
        "cross_validate", {"name": "rare", "formula": "y ~ x", "kind": "logistic", "k": 5}
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "outcome_class_too_small"


def test_cross_validate_nonbinary_outcome_rejected(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session(
        "multi", pd.DataFrame({"x": range(20), "y": [0, 1, 2, 3] * 5})
    )
    result = call_tool(
        "cross_validate", {"name": "multi", "formula": "y ~ x", "kind": "logistic"}
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "outcome_dtype_mismatch"
```

- [ ] **Step 2: Run, fix gaps if any, commit**

Run: `uv run pytest tests/test_crossval.py -q`
Expected: PASS from Task 6's implementation. Commit (red/green split if a fix was needed):

```bash
git add tests/test_crossval.py
git commit -m "test: pin cross_validate logistic stratification, class guards, and metric surface"
```

---

### Task 8: `cross_validate` remaining error paths + count models

**Files:**
- Modify: `tests/test_crossval.py`
- Modify: `src/data_analyst_mcp/tools/crossval.py` (only if gaps surface)

**Interfaces:**
- Consumes: Tasks 6-7.
- Produces: pinned contracts for `k_out_of_range` (both static and post-drop), `fold_too_small`, `robust_not_supported`, `threshold_out_of_range`, `formula_error`, `perfect_separation` (preflight), poisson/negbin metric keys, `_classify_fold_failure` unit behavior.

- [ ] **Step 1: Write the tests**

Append to `tests/test_crossval.py`:

```python
import pytest


def test_cross_validate_unknown_dataset(call_tool: Any) -> None:
    result = call_tool("cross_validate", {"name": "ghost", "formula": "y ~ x"})
    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_not_found"


@pytest.mark.parametrize("k", [1, 0, 21, -3])
def test_cross_validate_k_static_range(
    call_tool: Any, load_df_into_session: Any, k: int
) -> None:
    _load_linear(load_df_into_session)
    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x", "k": k})
    assert result["ok"] is False
    assert result["error"]["type"] == "k_out_of_range"


def test_cross_validate_k_exceeds_post_drop_rows(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session(
        "holey",
        pd.DataFrame({"x": [1.0, 2.0, 3.0, None, None, None], "y": [1.0] * 6}),
    )
    result = call_tool("cross_validate", {"name": "holey", "formula": "y ~ x", "k": 5})
    assert result["ok"] is False
    assert result["error"]["type"] == "k_out_of_range"
    assert "after NaN drops" in result["error"]["message"]


def test_cross_validate_fold_too_small(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """n=8, k=4 → train slices of 6 rows vs 7 design params."""
    import pandas as pd

    rng = np.random.RandomState(3)
    cols = {f"x{i}": rng.normal(size=8) for i in range(6)}
    load_df_into_session("wide", pd.DataFrame({**cols, "y": rng.normal(size=8)}))
    formula = "y ~ x0 + x1 + x2 + x3 + x4 + x5"
    result = call_tool("cross_validate", {"name": "wide", "formula": formula, "k": 4})
    assert result["ok"] is False
    assert result["error"]["type"] == "fold_too_small"


def test_cross_validate_robust_negbin_rejected(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_linear(load_df_into_session)
    result = call_tool(
        "cross_validate",
        {"name": "lin", "formula": "y ~ x", "kind": "negbin", "robust": True},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "robust_not_supported"


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_cross_validate_threshold_endpoints_rejected(
    call_tool: Any, load_df_into_session: Any, threshold: float
) -> None:
    _load_linear(load_df_into_session)
    result = call_tool(
        "cross_validate", {"name": "lin", "formula": "y ~ x", "threshold": threshold}
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "threshold_out_of_range"


def test_cross_validate_formula_error_from_preflight(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_linear(load_df_into_session)
    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ ghost_col"})
    assert result["ok"] is False
    assert result["error"]["type"] == "formula_error"


def test_cross_validate_perfect_separation_from_preflight(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    x = np.arange(20, dtype=float)
    y = (x >= 10).astype(int)
    load_df_into_session("sep", pd.DataFrame({"x": x, "y": y}))
    result = call_tool(
        "cross_validate", {"name": "sep", "formula": "y ~ x", "kind": "logistic"}
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "perfect_separation"


def test_cross_validate_poisson_metric_keys(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    rng = np.random.RandomState(4)
    x = rng.normal(size=50)
    y = rng.poisson(np.exp(0.5 + 0.3 * x))
    load_df_into_session("counts", pd.DataFrame({"x": x, "y": y}))

    result = call_tool(
        "cross_validate", {"name": "counts", "formula": "y ~ x", "kind": "poisson", "k": 3}
    )

    assert result["ok"] is True
    assert set(result["metrics"].keys()) == {"rmse", "mae", "pearson_chi2", "deviance"}


def test_classify_fold_failure_maps_exceptions() -> None:
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    from data_analyst_mcp.tools.crossval import _classify_fold_failure

    assert (
        _classify_fold_failure("logistic", PerfectSeparationError("sep"), None)
        == "perfect_separation"
    )
    assert _classify_fold_failure("ols", ValueError("boom"), None) == "convergence_failed"
    assert _classify_fold_failure("poisson", RuntimeError("nan"), None) == "convergence_failed"
```

- [ ] **Step 2: Run, fix gaps if any, commit**

Run: `uv run pytest tests/test_crossval.py -q`
Expected: PASS from Task 6's implementation; red/green any fix. Commit:

```bash
git add tests/test_crossval.py
git commit -m "test: pin cross_validate error taxonomy and count-model metrics"
```

---

### Task 9: `cross_validate` recorder cell

**Files:**
- Modify: `src/data_analyst_mcp/tools/crossval.py` (`_record_cross_validate` real code body)
- Modify: `tests/test_crossval.py`

**Interfaces:**
- Consumes: recorder setup-cell conventions (`con`, `np`, `pd`, `sm`, `smf` are in scope from `_SETUP_IMPORTS`; sklearn imported inside the cell like `evaluate_model` cells do).
- Produces: a self-contained code cell reproducing fold assignment + per-fold metrics.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crossval.py`:

```python
def test_cross_validate_records_replayable_cell(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp.recorder import get_recorder

    _load_linear(load_df_into_session)
    call_tool("cross_validate", {"name": "lin", "formula": "y ~ x", "k": 4, "seed": 7})

    code_cells = [
        c for c in get_recorder().cells
        if c["cell_type"] == "code" and c["metadata"]["tool_name"] == "cross_validate"
    ]
    assert len(code_cells) == 1
    src = code_cells[0]["source"]
    assert "RandomState(7)" in src
    assert 'smf.ols("y ~ x"' in src
    assert "sm.OLS" in src
    assert "% 4" in src
    assert "pd.DataFrame(_cv_rows)" in src


def test_cross_validate_logistic_cell_is_stratified(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp.recorder import get_recorder

    _load_binary(load_df_into_session)
    call_tool(
        "cross_validate", {"name": "bin", "formula": "y ~ x", "kind": "logistic", "k": 3}
    )
    src = [
        c for c in get_recorder().cells
        if c["cell_type"] == "code" and c["metadata"]["tool_name"] == "cross_validate"
    ][0]["source"]
    assert "for _cls in (0, 1):" in src
    assert "sm.Logit" in src
    assert "roc_auc_score" in src
```

- [ ] **Step 2: Run to verify failure, commit red**

Run: `uv run pytest tests/test_crossval.py -q` — the two new tests FAIL (placeholder cell).

```bash
git add tests/test_crossval.py
git commit -m "red: cross_validate emits a replayable fold-loop notebook cell"
```

- [ ] **Step 3: Implement the cell body**

Replace `_record_cross_validate`'s `code = ...` placeholder in `crossval.py` with:

```python
    code = _cv_cell_source(payload)
```

and add:

```python
_SMF_FN = {"ols": "ols", "logistic": "logit", "poisson": "poisson", "negbin": "negativebinomial"}
_SM_CLASS = {"ols": "OLS", "logistic": "Logit", "poisson": "Poisson", "negbin": "NegativeBinomial"}


def _cv_cell_source(payload: CrossValidateInput) -> str:
    """Self-contained notebook cell reproducing the CV table.

    Rebuilds the design matrices via the same smf formula fit the live
    preflight ran, assigns folds with the identical RandomState calls,
    then loops array-interface fits + metrics. ``con`` / ``np`` / ``pd``
    / ``sm`` / ``smf`` come from the setup cell.
    """
    smf_fn = _SMF_FN[payload.kind]
    sm_cls = _SM_CLASS[payload.kind]
    if payload.kind == "ols":
        full_fit_args = 'cov_type="HC3"' if payload.robust else ""
        fold_fit_args = 'cov_type="HC3"' if payload.robust else ""
    else:
        full_fit_args = "disp=0"
        fold_fit_args = "disp=0"
    lines = [
        f"_cv_df = con.sql('SELECT * FROM \"{payload.name}\"').df()",
        f'_cv_full = smf.{smf_fn}("{payload.formula}", data=_cv_df).fit({full_fit_args})',
        "_cv_y = np.asarray(_cv_full.model.endog, dtype=float)",
        "_cv_X = np.asarray(_cv_full.model.exog, dtype=float)",
        f"_cv_rng = np.random.RandomState({payload.seed})",
        "_cv_fold = np.empty(len(_cv_y), dtype=int)",
    ]
    if payload.kind == "logistic":
        lines += [
            "for _cls in (0, 1):",
            "    _idx = np.where(_cv_y == _cls)[0]",
            "    _p = _idx[_cv_rng.permutation(len(_idx))]",
            f"    _cv_fold[_p] = np.arange(len(_p)) % {payload.k}",
        ]
    else:
        lines += [
            "_perm = _cv_rng.permutation(len(_cv_y))",
            f"_cv_fold[_perm] = np.arange(len(_cv_y)) % {payload.k}",
        ]
    lines += [
        "_cv_rows = []",
        f"for _i in range({payload.k}):",
        "    _tr = _cv_fold != _i",
        f"    _res = sm.{sm_cls}(_cv_y[_tr], _cv_X[_tr]).fit({fold_fit_args})",
        "    _te_y = _cv_y[~_tr]",
        "    _pred = np.asarray(_res.predict(_cv_X[~_tr]))",
    ]
    if payload.kind == "logistic":
        lines = [
            "from sklearn import metrics as _skm",
            *lines,
            "    _ti = _te_y.astype(int)",
            f"    _cls_pred = (_pred >= {payload.threshold}).astype(int)",
            "    _cv_rows.append({",
            "        'fold': _i,",
            "        'roc_auc': _skm.roc_auc_score(_ti, _pred),",
            "        'pr_auc': _skm.average_precision_score(_ti, _pred),",
            "        'brier': _skm.brier_score_loss(_ti, _pred),",
            "        'log_loss': _skm.log_loss(_ti, np.clip(_pred, 1e-15, 1 - 1e-15), labels=[0, 1]),",
            "        'accuracy': float(np.mean(_cls_pred == _ti)),",
            "        'precision': _skm.precision_score(_ti, _cls_pred, zero_division=0),",
            "        'recall': _skm.recall_score(_ti, _cls_pred, zero_division=0),",
            "        'f1': _skm.f1_score(_ti, _cls_pred, zero_division=0),",
            "    })",
        ]
    elif payload.kind == "ols":
        lines += [
            "    _err = _te_y - _pred",
            "    _ss_tot = float(np.sum((_te_y - np.mean(_te_y)) ** 2))",
            "    _cv_rows.append({",
            "        'fold': _i,",
            "        'rmse': float(np.sqrt(np.mean(_err ** 2))),",
            "        'mae': float(np.mean(np.abs(_err))),",
            "        'r_squared': 1.0 - float(np.sum(_err ** 2)) / _ss_tot if _ss_tot > 0 else 0.0,",
            "    })",
        ]
    else:  # poisson / negbin
        lines += [
            "    _err = _te_y - _pred",
            "    _safe_mu = np.where(_pred > 0, _pred, np.nan)",
            "    _safe_y = np.where(_te_y > 0, _te_y, np.nan)",
            "    _log_term = np.where(_te_y > 0, _safe_y * np.log(_safe_y / _safe_mu), 0.0)",
            "    _cv_rows.append({",
            "        'fold': _i,",
            "        'rmse': float(np.sqrt(np.mean(_err ** 2))),",
            "        'mae': float(np.mean(np.abs(_err))),",
            "        'pearson_chi2': float(np.nansum(_err ** 2 / _safe_mu)),",
            "        'deviance': float(2.0 * np.nansum(_log_term - (_te_y - _pred))),",
            "    })",
        ]
    lines += ["pd.DataFrame(_cv_rows).set_index('fold').agg(['mean', 'std'])"]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests + gates, commit green**

Run: `uv run pytest tests/test_crossval.py -q` — PASS.

```bash
git add src/data_analyst_mcp/tools/crossval.py
git commit -m "green: cross_validate emits a replayable fold-loop notebook cell"
```

---

### Task 10: Integration evals — replay and drift

**Files:**
- Create: `evals/eval_split_cv.py`

**Interfaces:**
- Consumes: `evals/conftest.py` (`mcp_session`, `call`, `FIXTURES_DIR`, `PROJECT_ROOT`) and the `_nbconvert` helper pattern from `evals/eval_materialize.py:21-30` (subprocess `uv run jupyter nbconvert --to notebook --execute`, artifacts under `evals/_artifacts/`). Copy that helper verbatim — read `evals/eval_materialize.py` first and mirror its structure exactly (including any `@pytest.mark` decorations and artifact-path conventions it uses).
- Produces: three evals — split+CV replay via nbconvert, derived-source split replay, drift failure.

- [ ] **Step 1: Write the evals**

Create `evals/eval_split_cv.py` (adapt imports/decorators to match `eval_materialize.py` exactly):

```python
"""End-to-end evals for the model workflow bundle: split + CV + replay.

The full session — load titanic → split_dataset → fit on train →
evaluate on test → cross_validate — is emitted as a notebook and
re-executed via ``jupyter nbconvert``. The membership-checksum assert
inside the notebook is the drift guard under test.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from evals.conftest import FIXTURES_DIR, PROJECT_ROOT, call, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"


def _nbconvert(nb_path: Path) -> subprocess.CompletedProcess[str]:
    """Execute a notebook in place; exit code 0 means clean replay."""
    return subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            nb_path.name,
            str(nb_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


@pytest.mark.anyio
async def eval_split_fit_evaluate_cv_replays_via_nbconvert(tmp_path: Path) -> None:
    """split → fit(train) → evaluate(test) → cross_validate → emit →
    nbconvert --execute exits 0 (checksum assert passes)."""
    async with mcp_session() as session:
        r = await call(
            session,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "titanic.csv"), "name": "titanic"},
        )
        assert r.get("ok"), r
        r = await call(session, "split_dataset", {"name": "titanic", "seed": 7})
        assert r.get("ok"), r
        assert r["train"]["rows"] + r["test"]["rows"] == 891
        r = await call(
            session,
            "fit_model",
            {
                "name": "titanic_train",
                "formula": "Survived ~ C(Sex) + C(Pclass)",
                "kind": "logistic",
                "model_name": "surv",
            },
        )
        assert r.get("ok"), r
        r = await call(
            session, "evaluate_model", {"model_name": "surv", "dataset": "titanic_test"}
        )
        assert r.get("ok"), r
        cv = await call(
            session,
            "cross_validate",
            {
                "name": "titanic_train",
                "formula": "Survived ~ C(Sex) + C(Pclass)",
                "kind": "logistic",
                "k": 3,
            },
        )
        assert cv.get("ok"), cv
        assert cv["stratified"] is True
        emitted = await call(session, "emit_notebook", {})
        assert emitted.get("ok"), emitted

    ARTIFACTS.mkdir(exist_ok=True)
    nb_path = ARTIFACTS / "eval_split_cv.ipynb"
    shutil.copy(emitted["path"], nb_path)
    result = _nbconvert(nb_path)
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


@pytest.mark.anyio
async def eval_split_of_derived_source_replays(tmp_path: Path) -> None:
    """materialize_query (join/filter) → split_dataset on the derived table
    → emit → replay. Covers the derived-source checksum path."""
    async with mcp_session() as session:
        r = await call(
            session,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "titanic.csv"), "name": "titanic"},
        )
        assert r.get("ok"), r
        r = await call(
            session,
            "materialize_query",
            {"sql": 'SELECT * FROM "titanic" WHERE "Age" IS NOT NULL', "name": "adults"},
        )
        assert r.get("ok"), r
        r = await call(session, "split_dataset", {"name": "adults"})
        assert r.get("ok"), r
        emitted = await call(session, "emit_notebook", {})
        assert emitted.get("ok"), emitted

    ARTIFACTS.mkdir(exist_ok=True)
    nb_path = ARTIFACTS / "eval_split_derived.ipynb"
    shutil.copy(emitted["path"], nb_path)
    result = _nbconvert(nb_path)
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


@pytest.mark.anyio
async def eval_split_source_drift_fails_replay(tmp_path: Path) -> None:
    """Edit the source CSV between session and replay → the setup cell's
    SHA-256 assert must fail before any split runs."""
    src = tmp_path / "drifting.csv"
    src.write_text("x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(20)) + "\n")

    async with mcp_session() as session:
        r = await call(session, "load_dataset", {"path": str(src), "name": "drift"})
        assert r.get("ok"), r
        r = await call(session, "split_dataset", {"name": "drift"})
        assert r.get("ok"), r
        emitted = await call(session, "emit_notebook", {})
        assert emitted.get("ok"), emitted

    # Mutate the source after the session.
    src.write_text("x,y\n" + "\n".join(f"{i},{i * 3}" for i in range(20)) + "\n")

    ARTIFACTS.mkdir(exist_ok=True)
    nb_path = ARTIFACTS / "eval_split_drift.ipynb"
    shutil.copy(emitted["path"], nb_path)
    result = _nbconvert(nb_path)
    assert result.returncode != 0, "replay should have failed on the drift assert"
    assert "AssertionError" in (result.stderr + result.stdout)
```

Before running: open `evals/eval_materialize.py` and align this file's decorator style (`@pytest.mark.anyio` vs whatever it actually uses), artifact handling, and `emitted["path"]` access with what that file does — the eval suite has one house pattern; match it.

- [ ] **Step 2: Run the evals**

Run: `uv run pytest evals/eval_split_cv.py -q` (slow — spawns subprocesses and executes notebooks).
Expected: 3 passed. Debug any replay failure by opening the artifact notebook under `evals/_artifacts/`.

- [ ] **Step 3: Commit**

```bash
git add evals/eval_split_cv.py
git commit -m "test: split/CV notebook replay evals — happy path, derived source, drift failure"
```

(These are integration pins on already-green behavior, so `test:` is the right prefix — mirroring how `12c47ec test: add dataset provenance replay drift integration tests` landed.)

---

### Task 11: Docs + release 1.3.0

**Files:**
- Modify: `docs/SPEC.md` (new §5.6b after §5.6a; new §5.11d after §5.11c)
- Modify: `README.md` (tool count + list at line 219; new short section after "Post-hoc pairwise comparisons"; Development test counts)
- Modify: `ROADMAP.md` (line 3 count narrative)
- Modify: `CHANGELOG.md`
- Create: `docs/proposals/2026-07-08-model-workflow-bundle.md`
- Modify: `pyproject.toml` (version 1.2.1 → 1.3.0)

**Interfaces:** none — documentation of everything above.

- [ ] **Step 1: SPEC sections**

Insert §5.6b after the §5.6a block (before `### 5.7 correlate`), and §5.11d after §5.11c (before `### 5.12 plot`). Content: condense the corresponding sections of `docs/superpowers/specs/2026-07-08-model-workflow-bundle-design.md` into the SPEC's house format (Purpose / Input / Behavior / Output / Errors) — carry over verbatim: the input defaults, the RandomState/NEP-19 determinism rationale, the row-order tiers + membership checksum, the name-preflight atomicity, stratification rules, the preflight-fit reuse of `fit_prepared`, fold construction, the error lists, and the output shapes. The design doc's wording is already spec-grade; trim the "Why" narrative, keep the contracts.

- [ ] **Step 2: README**

- Line 219: change "all 22 tools" to "all 24 tools" and append `split_dataset` and `cross_validate` to the enumeration as "the model-workflow bundle (`split_dataset`, `cross_validate`)".
- Add after the "Post-hoc pairwise comparisons" section:

```markdown
## Train/test splits and cross-validation

`split_dataset` closes the loop the model registry opened: a seeded,
optionally stratified partition registered as two first-class datasets.
The same seed always produces the same split — membership comes from
`np.random.RandomState`, not DuckDB sampling — and the emitted notebook
recreates it behind an order-independent membership checksum, so silent
drift is impossible.

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
```

- Update the worked example §4a lead-in to mention the split: change the comment `# 1. fit + register` block's intro sentence to note `titanic_train` / `titanic_test` can now be produced with `split_dataset(name="titanic")`.
- Update test counts in Development (`uv run pytest tests/` / `evals/` comment counts) to the real numbers after Tasks 1-10 (run the suites and read the totals).

- [ ] **Step 3: ROADMAP, CHANGELOG, proposal stub**

- `ROADMAP.md` line 3: extend the count narrative — "…`pairwise_comparisons` then shipped 21 → 22… The model-workflow bundle (`split_dataset`, `cross_validate`) shipped 22 → 24 via `docs/proposals/2026-07-08-model-workflow-bundle.md`."
- `CHANGELOG.md`: add a `## 1.3.0` entry listing both tools, the membership-checksum replay guard, the `fit_prepared` refactor, and the materialize split-format guard.
- Create `docs/proposals/2026-07-08-model-workflow-bundle.md` with a short header pointing at `docs/superpowers/specs/2026-07-08-model-workflow-bundle-design.md` as the design record and SPEC §5.6b/§5.11d as the folded-in contract (mirrors how `pairwise_comparisons` documented its flow).

- [ ] **Step 4: Version bump + commit + tag**

- `pyproject.toml` line 3: `version = "1.3.0"`.

```bash
uv run pytest tests/ evals/ -q && uv run ruff format --check . && uv run ruff check . && uv run pyright src/ && uv run python scripts/check_tdd_commits.py
git add docs/SPEC.md README.md ROADMAP.md CHANGELOG.md docs/proposals/2026-07-08-model-workflow-bundle.md pyproject.toml
git commit -m "docs: model workflow bundle — SPEC 5.6b/5.11d, README, ROADMAP, CHANGELOG"
git commit --allow-empty -m "chore: release 1.3.0"   # or fold the bump into a release commit per house habit — check how 676e04e did it and match
```

Check `git show 676e04e --stat` first and mirror exactly how the 1.2.1 release commit bundled the version bump.

---

## Self-Review Notes (already applied)

- Spec coverage: every §5.6b / §5.11d contract line maps to a task above; the spec's testing matrix items map to Tasks 2, 3, 4, 7, 8, 10. The `threads > 1` determinism check from the spec's testing section is covered implicitly by the nbconvert replays (the replay subprocess runs DuckDB with default threading); no separate test is planned — if reviewers want an explicit pin, add a `PRAGMA threads=4` variant to `eval_split_cv.py`.
- Type consistency: `split_replay_source` keyword names match between recorder.py, split.py, and tests; `_fold_ids` / `_fold_metrics` / `_classify_fold_failure` names match between crossval.py and tests.
- The two checksum implementations (`split.membership_checksum` and recorder's `_SPLIT_CHECKSUM_DEF`) are intentionally duplicated (module-layering: recorder cannot import tools). Their sync is guarded twice: `test_split_replay_snippet_executes_and_reproduces_membership` (unit) and the nbconvert evals (integration).
