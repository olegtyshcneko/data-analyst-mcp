"""Resume manifest — build at emit time, validate strictly at resume time.

The manifest (spec v4 §2) lives in ``nb.metadata["data_analyst_mcp"]`` and
carries the operation journal, per-cell SHA-256 descriptors, final-state
digests, and a final-registry descriptor. ``validate_manifest`` is the
strict gate resume phase 1 runs: pydantic ``extra="forbid"`` throughout
plus semantic cross-checks (unique names, monotonic revisions, op/cell
binding). Structure it rejects is *malformed*, never "drifted" — drift is
phase 2's verdict.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MANIFEST_VERSION = 1
COMPARISON: dict[str, float] = {"rtol": 1e-7, "atol": 1e-12}

MAX_NOTEBOOK_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_CELLS = 2000
MAX_JOURNAL_OPS = 500
MAX_STRING_BYTES = 100 * 1024


class ManifestInvalid(Exception):
    """Manifest failed structural or semantic validation."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pair[T](_Strict):
    """A train/test-keyed value pair (split journal entries)."""

    train: T
    test: T


class JournalLoad(_Strict):
    op: Literal["load"]
    op_id: str
    name: str
    path: str
    format: str
    read_options: dict[str, Any]
    source_hash: str
    rows: int
    revision: int
    output_digest: str | None


class JournalMaterialize(_Strict):
    op: Literal["materialize"]
    op_id: str
    name: str
    sql: str
    overwrote: bool
    base_loader: dict[str, Any] | None
    split_overwrite: dict[str, Any] | None
    rows: int
    revision: int
    output_digest: str | None


class SplitParams(_Strict):
    test_fraction: float
    stratify_by: str | None
    rid_column: str


class JournalSplit(_Strict):
    op: Literal["split"]
    op_id: str
    source: str
    names: Pair[str]
    params: SplitParams
    seed: int
    membership_checksums: Pair[str]
    rows: Pair[int]
    revisions: Pair[int]
    output_digests: Pair[str | None]


class JournalFit(_Strict):
    op: Literal["fit"]
    op_id: str
    model_name: str
    dataset: str
    formula: str
    kind: str
    fit_options: dict[str, Any]
    n_obs: int
    design_columns: list[str]
    params: dict[str, float | str]
    bse: dict[str, float | str]
    dispersion: float | str | None
    training_dataset_hash: str
    training_dataset_revision: int
    training_loader: dict[str, Any]


JournalOp = Annotated[
    JournalLoad | JournalMaterialize | JournalSplit | JournalFit,
    Field(discriminator="op"),
]


class CellDescriptor(_Strict):
    index: int
    cell_type: Literal["markdown", "code"]
    tool_name: str
    op_id: str | None
    source_sha256: str


class FinalDataset(_Strict):
    name: str
    format: str
    read_options: dict[str, Any]
    path: str
    columns: list[dict[str, str]]
    rows: int
    source_hash: str
    revision: int
    base_loader: dict[str, Any] | None
    split_overwrite: dict[str, Any] | None


class FinalModel(_Strict):
    name: str
    kind: str
    formula: str
    fitted_on_dataset: str
    n_obs: int
    fit_options: dict[str, Any]
    training_dataset_hash: str
    training_dataset_revision: int
    training_loader: dict[str, Any] | None


class FinalRegistry(_Strict):
    datasets: list[FinalDataset]
    models: list[FinalModel]
    next_revision: int


class Manifest(_Strict):
    manifest_version: int
    digest_algorithm: str
    comparison: dict[str, float]
    producer: dict[str, str]
    resume_supported: bool
    resume_unsupported_reasons: list[str]
    notebook_replayable: bool
    journal: list[JournalOp]
    cells: list[CellDescriptor]
    setup_cell_sha256: str
    state_digests: dict[str, str | None]
    final_registry: FinalRegistry


def _journal_revisions(
    op: JournalLoad | JournalMaterialize | JournalSplit | JournalFit,
) -> list[int]:
    """Registration revisions an op mints, in mint order (fit mints none)."""
    if isinstance(op, JournalSplit):
        return [op.revisions.train, op.revisions.test]
    if isinstance(op, (JournalLoad, JournalMaterialize)):
        return [op.revision]
    return []


def validate_manifest(meta: dict[str, Any]) -> Manifest:
    """Strict-parse + semantic-check a manifest dict; raise ManifestInvalid."""
    try:
        manifest = Manifest.model_validate(meta)
    except ValidationError as exc:
        raise ManifestInvalid(
            [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        ) from exc

    reasons: list[str] = []

    names = [d.name for d in manifest.final_registry.datasets]
    if len(names) != len(set(names)):
        reasons.append("final_registry.datasets: duplicate dataset names")

    minted: list[int] = []
    for op in manifest.journal:
        minted.extend(_journal_revisions(op))
    if any(b <= a for a, b in pairwise(minted)):
        reasons.append("journal: registration revisions are not strictly increasing")

    op_ids = [op.op_id for op in manifest.journal]
    if len(op_ids) != len(set(op_ids)):
        reasons.append("journal: duplicate op_id")

    cells_by_op: dict[str, list[CellDescriptor]] = {}
    for cell in manifest.cells:
        if cell.op_id is not None:
            cells_by_op.setdefault(cell.op_id, []).append(cell)
    for op_id in op_ids:
        pair = cells_by_op.get(op_id, [])
        if [c.cell_type for c in pair] != ["markdown", "code"]:
            reasons.append(f"journal op {op_id}: expected exactly one markdown+code cell pair")

    digest_names = set(manifest.state_digests)
    for dataset in manifest.final_registry.datasets:
        if dataset.format != "dataframe" and dataset.name not in digest_names:
            reasons.append(f"state_digests: missing entry for dataset {dataset.name!r}")

    if minted and manifest.final_registry.next_revision <= max(minted):
        reasons.append("final_registry.next_revision must exceed every journal revision")

    if reasons:
        raise ManifestInvalid(reasons)
    return manifest


def _producer_versions() -> dict[str, str]:
    import platform

    import duckdb
    import numpy
    import pandas  # type: ignore[reportMissingTypeStubs]
    import statsmodels  # type: ignore[reportMissingTypeStubs]

    return {
        "duckdb": str(duckdb.__version__),
        "pandas": str(pandas.__version__),
        "numpy": str(numpy.__version__),
        "statsmodels": str(statsmodels.__version__),
        "python": platform.python_version(),
    }


def build_manifest(nb: Any) -> dict[str, Any]:
    """Build the resume manifest from live session + recorder + built cells."""
    from data_analyst_mcp import session
    from data_analyst_mcp.digest import DIGEST_ALGORITHM, digest_table

    con = session.get_connection()
    datasets = session.get_datasets()
    journal = [dict(e) for e in session.get_journal()]

    reasons: list[str] = []
    for name, entry in datasets.items():
        if entry.format == "dataframe":
            reasons.append(
                f"dataset {name!r} is in-memory (dataframe) — journal cannot recreate it"
            )
    for e in journal:
        if e["op"] == "split":
            digests: dict[str, Any] = dict(e.get("output_digests") or {})
        elif e["op"] in ("load", "materialize"):
            digests = {"": e.get("output_digest")}
        else:  # fit ops mint no table — nothing to digest
            continue
        if any(v is None for v in digests.values()):
            reasons.append(f"journal op {e['op_id']} produced an undigestable table")

    state_digests: dict[str, str | None] = {}
    for name, entry in datasets.items():
        if entry.format == "dataframe":
            continue
        try:
            state_digests[name] = digest_table(con, name)
        except Exception:
            # A registered entry without a live table (possible in synthetic
            # session states): emit must not crash, but resume evidence for
            # this dataset is gone.
            state_digests[name] = None
            reasons.append(f"dataset {name!r} has no live table — state digest unavailable")
    setup_src = str(nb.cells[0].source)
    replayable = not any(
        line.startswith("raise AssertionError(") for line in setup_src.splitlines()
    )
    cells = [
        {
            "index": i,
            "cell_type": str(c.cell_type),
            "tool_name": str(c.metadata.get("tool_name", "")),
            "op_id": c.metadata.get("op_id"),
            "source_sha256": hashlib.sha256(str(c.source).encode("utf-8")).hexdigest(),
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
        "producer": _producer_versions(),
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
