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
from data_analyst_mcp.recorder import get_recorder, split_replay_source

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
        # int(round(...)) kept verbatim: round() uses banker's rounding
        # (round(2.5)==2), the pinned split contract. noqa suppresses
        # RUF046 so the expression stays algorithm-identical to Task 4's
        # replay snippet.
        n_test = min(max(int(round(n * test_fraction)), 1), n - 1)  # noqa: RUF046
        is_test[rng.permutation(n)[:n_test]] = True
        return is_test, warnings
    null_mask: Any = strata.isna().to_numpy()
    values: list[Any] = sorted(strata[~strata.isna()].unique().tolist(), key=str)
    groups: list[Any] = [np.where((strata == v).to_numpy() & ~null_mask)[0] for v in values]
    if bool(null_mask.any()):
        groups.append(np.where(null_mask)[0])
    saw_small = False
    for rids in groups:
        if len(rids) < 2:
            saw_small = True
            continue
        n_t = min(max(int(round(len(rids) * test_fraction)), 1), len(rids) - 1)  # noqa: RUF046
        is_test[rids[rng.permutation(len(rids))[:n_t]]] = True
    if saw_small:
        warnings.append("small_strata")
    return is_test, warnings


def membership_checksum(df: Any) -> str:
    """Order-independent digest of a DataFrame's row contents.

    ``count:xor:sum`` over truncated SHA-256 per-row digests — the row
    count plus a second (additive) accumulator means duplicate rows
    cannot cancel out of a pure XOR. Values are converted to builtins
    before ``repr`` so the digest is stable across DuckDB/numpy
    versions; strings are ``repr``'d so a ``|`` inside a value cannot
    collide with the field separator. Must stay algorithm-identical to
    the ``_split_checksum`` snippet emitted by
    ``recorder.split_replay_source``.
    """
    pd_mod = _pd()
    acc_xor = 0
    acc_sum = 0
    n_rows = 0
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
                parts.append("true" if bool(v) else "false")  # type: ignore[reportUnknownArgumentType]
            elif isinstance(v, (float, np.floating)):
                f = float(v)  # type: ignore[reportUnknownArgumentType]
                parts.append("<null>" if math.isnan(f) else repr(f))
            elif isinstance(v, (int, np.integer)):
                parts.append(repr(int(v)))  # type: ignore[reportUnknownArgumentType]
            elif isinstance(v, str):
                parts.append(repr(v))
            else:
                parts.append(str(v))
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        row_hash = int.from_bytes(digest[:16], "big")
        acc_xor ^= row_hash
        acc_sum = (acc_sum + row_hash) % (1 << 128)
        n_rows += 1
    return f"{n_rows:x}:{acc_xor:032x}:{acc_sum:032x}"


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
    # `is None` (not truthiness): an explicit "" is a caller-supplied name to
    # be rejected as invalid below, not a request for the default.
    train_name = f"{payload.name}_train" if payload.train_name is None else payload.train_name
    test_name = f"{payload.name}_test" if payload.test_name is None else payload.test_name
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
        strata_frame: Any = con.execute(f"SELECT {_quote(payload.stratify_by)} FROM {src_q}").df()
        strata = strata_frame.iloc[:, 0]

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
    assign_df = pd_mod.DataFrame({"rid": np.arange(n, dtype=np.int64), "is_test": is_test})
    view = "__data_analyst_split_assign"
    con.register(view, assign_df)
    try:
        base = (
            f"SELECT s.* EXCLUDE ({_quote(rid)}) FROM "
            f"(SELECT *, row_number() OVER () - 1 AS {_quote(rid)} FROM {src_q}) s "
            f"JOIN {view} a ON s.{_quote(rid)} = a.rid"
        )
        con.execute(f"CREATE OR REPLACE TABLE {_quote(train_name)} AS {base} WHERE NOT a.is_test")
        con.execute(f"CREATE OR REPLACE TABLE {_quote(test_name)} AS {base} WHERE a.is_test")
    finally:
        con.unregister(view)

    # One digest per side (one extra hashing pass over the train frame):
    # test-side drift and train-side drift are independent failure modes —
    # a split-source change that only moves train rows passes the test
    # checksum (spec S16), so replay asserts each side against its own.
    test_df = con.execute(f"SELECT * FROM {_quote(test_name)}").df()
    test_checksum = membership_checksum(test_df)
    train_df = con.execute(f"SELECT * FROM {_quote(train_name)}").df()
    train_checksum = membership_checksum(train_df)

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
        opts["membership_checksum"] = test_checksum if role == "test" else train_checksum
        session.register(
            name=out_name,
            path="(split)",
            read_options=opts,
            format="split",
            rows=rows,
            columns=out_columns,
        )

    _record_split(payload, train_name, test_name, n - n_test, n_test, test_checksum, rid)

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

    The code body is the shared ``recorder.split_replay_source`` snippet,
    so the per-call cell and the setup cell replay identically.
    """
    md = (
        f"### Split `{payload.name}` into `{train_name}` / `{test_name}`\n\n"
        f"- seed={payload.seed}, test_fraction={payload.test_fraction}, "
        f"stratify_by={payload.stratify_by!r}\n"
        f"- train rows: {n_train}, test rows: {n_test}"
    )
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
    get_recorder().record(markdown=md, code=code, tool_name="split_dataset")
