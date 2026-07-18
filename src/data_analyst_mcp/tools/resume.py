"""load_session_from_notebook — resume an emitted session (spec v4).

Three phases, all under ``session.state_lock()``:

1. *Validate* — empty session AND empty live catalog, notebook/manifest
   integrity (strict pydantic parse, caps, per-cell SHA-256), and a source
   preflight that accumulates every drifted file into one error.
2. *Replay* — ``BEGIN TRANSACTION``; apply journal ops in order comparing
   recorded evidence at every step (source hashes, membership checksums,
   table digests, named model params/bse). First divergence → ``ROLLBACK``
   with the op's index/id; downstream ops stay unverified by design.
3. *Commit* — ``COMMIT``; publish staged registries + recorder cells in one
   ``install_state`` swap. Any failure before this leaves the live session
   untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from data_analyst_mcp import session
from data_analyst_mcp.digest import digest_table
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.journal_evidence import evidence_equal, tag_nonfinite
from data_analyst_mcp.manifest import (
    MANIFEST_VERSION,
    MAX_CELLS,
    MAX_JOURNAL_OPS,
    MAX_MANIFEST_BYTES,
    MAX_NOTEBOOK_BYTES,
    MAX_STRING_BYTES,
    JournalFit,
    JournalLoad,
    JournalMaterialize,
    JournalSplit,
    Manifest,
    ManifestInvalid,
    validate_manifest,
)
from data_analyst_mcp.provenance import compute_source_hash
from data_analyst_mcp.recorder import get_recorder
from data_analyst_mcp.session import DatasetEntry, ModelEntry
from data_analyst_mcp.tools._sql_safety import contains_unsafe_semicolon, leading_keyword
from data_analyst_mcp.tools.datasets import _build_read_call  # pyright: ignore[reportPrivateUsage]
from data_analyst_mcp.tools.models import FitModelInput, fit_prepared
from data_analyst_mcp.tools.split import (
    _assign_is_test,  # pyright: ignore[reportPrivateUsage]
    membership_checksum,
)

logger = logging.getLogger(__name__)

RESUME_BUDGET_SECONDS = 300.0
# Module-level so tests can monkeypatch the cap without touching manifest.py.
MAX_JOURNAL_OPS_EFFECTIVE = MAX_JOURNAL_OPS

_REMOTE_PREFIXES = ("s3://", "http://", "https://")
_RESUME_VIEW = "__dam_resume_view"
_SPLIT_VIEW = "__data_analyst_split_assign"


def _np() -> Any:
    """Return ``numpy`` as untyped to keep strict pyright clean."""
    import numpy as _np_mod

    return _np_mod


def _pd() -> Any:
    """Return ``pandas`` as untyped to keep strict pyright clean."""
    import pandas as _pd_mod  # type: ignore[reportMissingTypeStubs]

    return _pd_mod


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'


def _read_notebook(path: str) -> Any:
    """Parse an .ipynb via nbformat (untyped — NotebookNode is dict-like)."""
    import nbformat

    return nbformat.read(path, as_version=4)  # type: ignore[reportUnknownMemberType]


class LoadSessionInput(BaseModel):
    """Inputs for ``load_session_from_notebook``."""

    model_config = ConfigDict(extra="forbid")

    path: str


class _Divergence(Exception):
    """Phase-2 fail-fast carrier; converts to a build_error envelope."""

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


def load_session_from_notebook(payload: LoadSessionInput) -> dict[str, Any]:
    """Resume a previously-emitted notebook into the (empty) live session."""
    con = session.get_connection()
    with session.state_lock():
        # ---------------- Phase 1: validate ----------------
        if (
            session.get_datasets()
            or session.get_models()
            or session.get_journal()
            or get_recorder().cells
        ):
            return build_error(
                type="session_not_empty",
                message="The live session already holds datasets, models, journal ops, or cells.",
                hint="Resume requires a fresh session — restart the server (or reset) first.",
            )
        tables = [
            str(r[0])
            for r in con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
        ]
        views = [
            str(r[0])
            for r in con.execute(
                "SELECT view_name FROM duckdb_views() WHERE schema_name = 'main' AND NOT internal"
            ).fetchall()
        ]
        if tables or views:
            return build_error(
                type="catalog_not_empty",
                message=f"The live DuckDB catalog is not empty: {sorted(tables + views)}.",
                hint=(
                    "Resume refuses to adopt or drop ambient tables — remove them "
                    "or restart the server, then retry."
                ),
            )
        if not os.path.isfile(payload.path):
            return build_error(
                type="notebook_not_found",
                message=f"No notebook file at {payload.path!r}.",
                hint="Check the path is absolute or relative to the server's cwd.",
            )
        if os.path.getsize(payload.path) > MAX_NOTEBOOK_BYTES:
            return build_error(
                type="manifest_invalid",
                message=f"Notebook exceeds the {MAX_NOTEBOOK_BYTES} byte cap.",
                hint="Resume refuses oversized notebooks (trust model, spec §6).",
            )
        try:
            nb = _read_notebook(payload.path)
        except Exception as exc:
            return build_error(
                type="notebook_invalid",
                message=f"Could not parse notebook: {exc}",
                hint="The file must be a valid nbformat v4 .ipynb.",
            )
        meta_raw = nb.metadata.get("data_analyst_mcp")
        if meta_raw is None:
            return build_error(
                type="manifest_missing",
                message="The notebook carries no data_analyst_mcp resume manifest.",
                hint=(
                    "Only notebooks emitted by emit_notebook on a manifest-aware "
                    "server can be resumed — re-emit the session with a current version."
                ),
            )
        try:
            meta: dict[str, Any] = json.loads(json.dumps(meta_raw))
        except (TypeError, ValueError) as exc:
            return build_error(
                type="manifest_invalid",
                message=f"Manifest is not JSON-serializable: {exc}",
                hint="Re-emit the notebook; its metadata was corrupted.",
            )
        if meta.get("manifest_version") != MANIFEST_VERSION:
            return build_error(
                type="manifest_version_unsupported",
                message=(
                    f"Manifest version {meta.get('manifest_version')!r} is not supported "
                    f"(this server reads version {MANIFEST_VERSION})."
                ),
                hint="Re-emit the notebook with a compatible server version.",
            )
        if len(json.dumps(meta).encode("utf-8")) > MAX_MANIFEST_BYTES:
            return build_error(
                type="manifest_invalid",
                message=f"Manifest exceeds the {MAX_MANIFEST_BYTES} byte cap.",
            )
        try:
            manifest = validate_manifest(meta)
        except ManifestInvalid as exc:
            return build_error(
                type="manifest_invalid",
                message="Manifest failed validation: " + "; ".join(exc.reasons),
                hint="Re-emit the notebook; do not hand-edit its metadata.",
            )
        cap_reasons: list[str] = []
        if len(nb.cells) > MAX_CELLS:
            cap_reasons.append(f"notebook has {len(nb.cells)} cells (cap {MAX_CELLS})")
        if len(manifest.journal) > MAX_JOURNAL_OPS_EFFECTIVE:
            cap_reasons.append(
                f"journal has {len(manifest.journal)} ops (cap {MAX_JOURNAL_OPS_EFFECTIVE})"
            )
        for op in manifest.journal:
            for value in _capped_strings(op):
                if len(value.encode("utf-8")) > MAX_STRING_BYTES:
                    cap_reasons.append(
                        f"journal op {op.op_id}: string exceeds {MAX_STRING_BYTES} bytes"
                    )
        if cap_reasons:
            return build_error(
                type="manifest_invalid", message="Caps exceeded: " + "; ".join(cap_reasons)
            )
        if not manifest.resume_supported:
            return build_error(
                type="unreplayable_dataset",
                message="This notebook was emitted as not resumable: "
                + "; ".join(manifest.resume_unsupported_reasons),
                hint=(
                    "Sessions holding in-memory (dataframe) datasets or undigestable "
                    "tables cannot be resumed from a journal."
                ),
            )
        if not nb.cells:
            return build_error(
                type="notebook_invalid", message="Notebook has no cells (expected a setup cell)."
            )
        setup_cell = nb.cells[0]
        body = list(nb.cells[1:])
        integrity: list[str] = []
        setup_sha = hashlib.sha256(str(setup_cell.source).encode("utf-8")).hexdigest()
        if setup_sha != manifest.setup_cell_sha256:
            integrity.append("setup cell source was modified")
        if len(body) != len(manifest.cells):
            integrity.append(
                f"notebook has {len(body)} body cells but the manifest describes "
                f"{len(manifest.cells)}"
            )
        else:
            for desc, cell in zip(manifest.cells, body, strict=True):
                sha = hashlib.sha256(str(cell.source).encode("utf-8")).hexdigest()
                if (
                    desc.cell_type != cell.cell_type
                    or desc.tool_name != str(cell.metadata.get("tool_name", ""))
                    or desc.source_sha256 != sha
                ):
                    integrity.append(f"body cell {desc.index} does not match its descriptor")
        if integrity:
            return build_error(
                type="notebook_modified",
                message="Notebook was modified since emit: " + "; ".join(integrity),
                hint="Resume only accepts byte-identical cells — re-emit instead of editing.",
            )
        drift_lines = _preflight_source_drift(manifest)
        if drift_lines:
            return build_error(
                type="source_drift",
                message="Source files changed since emit:\n" + "\n".join(drift_lines),
                hint="Reload the changed files in a fresh session instead of resuming.",
            )

        # ---------------- Phase 2: transactional replay ----------------
        staged_datasets: dict[str, DatasetEntry] = {}
        staged_models: dict[str, ModelEntry] = {}
        warnings: list[str] = []
        deadline = time.monotonic() + RESUME_BUDGET_SECONDS
        con.execute("BEGIN TRANSACTION")
        try:
            for op_index, op in enumerate(manifest.journal):
                if time.monotonic() > deadline:
                    raise _Divergence(
                        "resume_budget_exceeded",
                        f"Replay exceeded the {RESUME_BUDGET_SECONDS:.0f}s budget",
                        hint="Resume the notebook on a machine/session with more headroom.",
                        op_index=op_index,
                        op_id=op.op_id,
                    )
                if isinstance(op, JournalLoad):
                    _apply_load(con, op, op_index, staged_datasets, warnings)
                elif isinstance(op, JournalMaterialize):
                    _apply_materialize(con, op, op_index, staged_datasets)
                elif isinstance(op, JournalSplit):
                    _apply_split(con, op, op_index, staged_datasets)
                else:
                    _apply_fit(con, op, op_index, staged_models, manifest)
            _verify_final_state(con, manifest, staged_datasets, staged_models)
            con.execute("COMMIT")
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

        # ---------------- Phase 3: publish ----------------
        session.install_state(
            datasets=staged_datasets,
            models=staged_models,
            journal=[dict(e) for e in meta["journal"]],
            next_revision=manifest.final_registry.next_revision,
        )
        imported_cells = [
            {
                "cell_type": desc.cell_type,
                "source": str(cell.source),
                "metadata": {"tool_name": desc.tool_name},
                "op_id": desc.op_id,
            }
            for desc, cell in zip(manifest.cells, body, strict=True)
        ]
        get_recorder().install_cells(imported_cells)
        return {
            "ok": True,
            "path": payload.path,
            "datasets": [
                {"name": name, "rows": entry.rows, "format": entry.format}
                for name, entry in staged_datasets.items()
            ],
            "models": list(staged_models),
            "n_cells_imported": len(imported_cells),
            "n_journal_ops": len(manifest.journal),
            "warnings": warnings,
        }


def _capped_strings(op: JournalLoad | JournalMaterialize | JournalSplit | JournalFit) -> list[str]:
    """The op's cap-checked strings (spec §6: SQL / formula / path)."""
    if isinstance(op, JournalLoad):
        return [op.path]
    if isinstance(op, JournalMaterialize):
        return [op.sql]
    if isinstance(op, JournalFit):
        return [op.formula]
    return []


def _is_remote(path: str) -> bool:
    return path.startswith(_REMOTE_PREFIXES)


def _preflight_source_drift(manifest: Manifest) -> list[str]:
    """Hash every journal-referenced local source; return ALL mismatches."""
    lines: list[str] = []
    checked: set[tuple[str, str]] = set()
    for op in manifest.journal:
        candidates: list[tuple[str, str]] = []
        if isinstance(op, JournalLoad) and not _is_remote(op.path):
            candidates.append((op.path, op.source_hash))
        elif isinstance(op, JournalMaterialize) and op.base_loader is not None:
            base_path = str(op.base_loader.get("path", ""))
            base_hash = str(op.base_loader.get("source_hash", ""))
            if base_path and base_hash and not _is_remote(base_path):
                candidates.append((base_path, base_hash))
        for path, expected in candidates:
            key = (path, expected)
            if key in checked:
                continue
            checked.add(key)
            actual = compute_source_hash(path)
            if actual != expected:
                lines.append(f"{path}: expected {expected[:12]}…, found {actual[:12]}…")
    return lines


def _describe_columns(con: Any, name: str) -> list[dict[str, str]]:
    describe_rows = con.execute(f"DESCRIBE {_quote(name)}").fetchall()
    return [{"name": str(row[0]), "dtype": str(row[1])} for row in describe_rows]


def _apply_load(
    con: Any,
    op: JournalLoad,
    op_index: int,
    staged: dict[str, DatasetEntry],
    warnings: list[str],
) -> None:
    remote = _is_remote(op.path)
    pre: str | None = None
    if remote:
        warnings.append(f"{op.path} reloaded unguarded")
    else:
        pre = compute_source_hash(op.path)
        if pre != op.source_hash:
            raise _Divergence(
                "source_drift",
                f"{op.path} changed since emit (expected {op.source_hash[:12]}…, "
                f"found {pre[:12]}…)",
                op_index=op_index,
                op_id=op.op_id,
            )
    read_call = _build_read_call(op.path, op.format, dict(op.read_options))
    df = session.read_file_as_df(read_call)
    if not remote:
        # TOCTOU guard: the pre-hash passed, but the file may have changed
        # while (or after) the loader read it — pre/post agreement binds the
        # loaded bytes to the recorded evidence.
        post = compute_source_hash(op.path)
        if post != pre:
            raise _Divergence(
                "source_drift",
                f"{op.path} changed while being reloaded",
                op_index=op_index,
                op_id=op.op_id,
            )
    con.register(_RESUME_VIEW, df)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {_quote(op.name)} AS SELECT * FROM {_RESUME_VIEW}")
    finally:
        con.unregister(_RESUME_VIEW)
    digest = digest_table(con, op.name)
    if digest != op.output_digest:
        raise _Divergence(
            "state_digest_mismatch",
            f"Dataset {op.name!r} reloaded with different contents than recorded",
            op_index=op_index,
            op_id=op.op_id,
        )
    staged[op.name] = DatasetEntry(
        path=op.path,
        read_options=dict(op.read_options),
        format=op.format,
        rows=op.rows,
        columns=_describe_columns(con, op.name),
        source_hash=op.source_hash,
        revision=op.revision,
    )


def _apply_materialize(
    con: Any,
    op: JournalMaterialize,
    op_index: int,
    staged: dict[str, DatasetEntry],
) -> None:
    # Same gates the live tool enforces: a journal carrying SQL the live
    # tool would reject is malformed, not drifted.
    if leading_keyword(op.sql) not in ("SELECT", "WITH") or contains_unsafe_semicolon(op.sql):
        raise _Divergence(
            "manifest_invalid",
            f"Journal op for {op.name!r} carries SQL the live tool would reject",
            op_index=op_index,
            op_id=op.op_id,
        )
    con.execute(f"CREATE OR REPLACE TABLE {_quote(op.name)} AS {op.sql}")
    digest = digest_table(con, op.name)
    if digest != op.output_digest:
        raise _Divergence(
            "state_digest_mismatch",
            f"Derived dataset {op.name!r} produced different contents than recorded "
            "(non-deterministic SQL or upstream drift)",
            op_index=op_index,
            op_id=op.op_id,
        )
    staged[op.name] = DatasetEntry(
        path="(query)",
        read_options={"sql": op.sql},
        format="derived",
        rows=op.rows,
        columns=_describe_columns(con, op.name),
        base_loader=dict(op.base_loader) if op.base_loader is not None else None,
        split_overwrite=dict(op.split_overwrite) if op.split_overwrite is not None else None,
        source_hash=compute_source_hash("(query)"),
        revision=op.revision,
    )


def _apply_split(
    con: Any,
    op: JournalSplit,
    op_index: int,
    staged: dict[str, DatasetEntry],
) -> None:
    np = _np()
    pd_mod = _pd()
    src_q = _quote(op.source)
    n = int(con.execute(f"SELECT COUNT(*) FROM {src_q}").fetchone()[0])  # type: ignore[index]
    strata: Any | None = None
    if op.params.stratify_by is not None:
        strata_frame: Any = con.execute(f"SELECT {_quote(op.params.stratify_by)} FROM {src_q}").df()
        strata = strata_frame.iloc[:, 0]
    is_test, _ = _assign_is_test(n, op.params.test_fraction, op.seed, strata)
    rid = op.params.rid_column
    assign_df = pd_mod.DataFrame({"rid": np.arange(n, dtype=np.int64), "is_test": is_test})
    con.register(_SPLIT_VIEW, assign_df)
    try:
        base = (
            f"SELECT s.* EXCLUDE ({_quote(rid)}) FROM "
            f"(SELECT *, row_number() OVER () - 1 AS {_quote(rid)} FROM {src_q}) s "
            f"JOIN {_SPLIT_VIEW} a ON s.{_quote(rid)} = a.rid"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE {_quote(op.names.train)} AS {base} WHERE NOT a.is_test"
        )
        con.execute(f"CREATE OR REPLACE TABLE {_quote(op.names.test)} AS {base} WHERE a.is_test")
    finally:
        con.unregister(_SPLIT_VIEW)

    checksums = {"train": op.membership_checksums.train, "test": op.membership_checksums.test}
    digests = {"train": op.output_digests.train, "test": op.output_digests.test}
    rows = {"train": op.rows.train, "test": op.rows.test}
    revisions = {"train": op.revisions.train, "test": op.revisions.test}
    names = {"train": op.names.train, "test": op.names.test}
    for role in ("train", "test"):
        side_df = con.execute(f"SELECT * FROM {_quote(names[role])}").df()
        if membership_checksum(side_df) != checksums[role]:
            raise _Divergence(
                "split_drift",
                f"Recomputed {role} membership for split of {op.source!r} does not match "
                "the recorded checksum",
                op_index=op_index,
                op_id=op.op_id,
            )
        if digest_table(con, names[role]) != digests[role]:
            raise _Divergence(
                "state_digest_mismatch",
                f"Split output {names[role]!r} produced different contents than recorded",
                op_index=op_index,
                op_id=op.op_id,
            )
        staged[names[role]] = DatasetEntry(
            path="(split)",
            read_options={
                "source": op.source,
                "seed": op.seed,
                "test_fraction": op.params.test_fraction,
                "stratify_by": op.params.stratify_by,
                "train_name": op.names.train,
                "test_name": op.names.test,
                "rid_column": rid,
                "role": role,
                "membership_checksum": checksums[role],
            },
            format="split",
            rows=rows[role],
            columns=_describe_columns(con, names[role]),
            source_hash=compute_source_hash("(split)"),
            revision=revisions[role],
        )


def _apply_fit(
    con: Any,
    op: JournalFit,
    op_index: int,
    staged_models: dict[str, ModelEntry],
    manifest: Manifest,
) -> None:
    df = con.execute(f"SELECT * FROM {_quote(op.dataset)}").df()
    fit_payload = FitModelInput(
        name=op.dataset,
        formula=op.formula,
        kind=op.kind,  # type: ignore[arg-type]
        robust=bool(op.fit_options.get("robust", False)),
        model_name=op.model_name,
    )
    try:
        result = fit_prepared(fit_payload, df)
    except Exception as exc:
        raise _Divergence(
            "model_drift",
            f"Re-fit of {op.model_name!r} failed: {exc}",
            op_index=op_index,
            op_id=op.op_id,
        ) from exc
    live = result.pop("_result", None)
    if not result.get("ok") or live is None:
        detail = result.get("error", {}).get("message", "fit returned no result")
        raise _Divergence(
            "model_drift",
            f"Re-fit of {op.model_name!r} failed: {detail}",
            op_index=op_index,
            op_id=op.op_id,
        )
    rtol = float(manifest.comparison.get("rtol", 1e-7))
    atol = float(manifest.comparison.get("atol", 1e-12))
    n_obs = int(result["fit"]["n_obs"])
    design_columns = [str(k) for k in live.params.index]
    params_actual = {str(k): tag_nonfinite(float(v)) for k, v in live.params.items()}
    bse_actual = {str(k): tag_nonfinite(float(v)) for k, v in live.bse.items()}
    if n_obs != op.n_obs:
        raise _Divergence(
            "model_drift",
            f"n_obs mismatch for {op.model_name!r}: recorded {op.n_obs}, refit {n_obs}",
            op_index=op_index,
            op_id=op.op_id,
        )
    if design_columns != op.design_columns:
        raise _Divergence(
            "model_drift",
            f"design_columns mismatch for {op.model_name!r}",
            op_index=op_index,
            op_id=op.op_id,
        )
    if not evidence_equal(dict(op.params), params_actual, rtol=rtol, atol=atol):
        raise _Divergence(
            "model_drift",
            f"params mismatch for {op.model_name!r}: refit coefficients differ from "
            "the recorded evidence",
            op_index=op_index,
            op_id=op.op_id,
        )
    if not evidence_equal(dict(op.bse), bse_actual, rtol=rtol, atol=atol):
        raise _Divergence(
            "model_drift",
            f"bse mismatch for {op.model_name!r}: refit standard errors differ from "
            "the recorded evidence (covariance/fit_options drift)",
            op_index=op_index,
            op_id=op.op_id,
        )
    staged_models[op.model_name] = ModelEntry(
        name=op.model_name,
        kind=op.kind,
        formula=op.formula,
        fitted_on_dataset=op.dataset,
        n_obs=op.n_obs,
        fitted_at=datetime.now(UTC),
        training_dataset_hash=op.training_dataset_hash,
        _result=live,
        training_dataset_revision=op.training_dataset_revision,
        training_loader=dict(op.training_loader),
        fit_options=dict(op.fit_options),
    )


def _dataset_descriptor(name: str, entry: DatasetEntry) -> dict[str, Any]:
    return {
        "name": name,
        "format": entry.format,
        "read_options": dict(entry.read_options),
        "path": entry.path,
        "columns": list(entry.columns),
        "rows": entry.rows,
        "source_hash": entry.source_hash,
        "revision": entry.revision,
        "base_loader": entry.base_loader,
        "split_overwrite": entry.split_overwrite,
    }


def _model_descriptor(entry: ModelEntry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "kind": entry.kind,
        "formula": entry.formula,
        "fitted_on_dataset": entry.fitted_on_dataset,
        "n_obs": entry.n_obs,
        "fit_options": dict(entry.fit_options),
        "training_dataset_hash": entry.training_dataset_hash,
        "training_dataset_revision": entry.training_dataset_revision,
        "training_loader": entry.training_loader,
    }


def _verify_final_state(
    con: Any,
    manifest: Manifest,
    staged_datasets: dict[str, DatasetEntry],
    staged_models: dict[str, ModelEntry],
) -> None:
    """Replay finished: staged registry + final digests must match the manifest."""
    expected_datasets = {d.name: d.model_dump() for d in manifest.final_registry.datasets}
    actual_datasets = {n: _dataset_descriptor(n, e) for n, e in staged_datasets.items()}
    mismatched = sorted(
        set(expected_datasets) ^ set(actual_datasets)
        | {
            n
            for n in set(expected_datasets) & set(actual_datasets)
            if expected_datasets[n] != actual_datasets[n]
        }
    )
    expected_models = {m.name: m.model_dump() for m in manifest.final_registry.models}
    actual_models = {n: _model_descriptor(e) for n, e in staged_models.items()}
    model_mismatched = sorted(
        set(expected_models) ^ set(actual_models)
        | {
            n
            for n in set(expected_models) & set(actual_models)
            if expected_models[n] != actual_models[n]
        }
    )
    expected_next = max((e.revision for e in staged_datasets.values()), default=-1) + 1
    if manifest.final_registry.next_revision != expected_next:
        model_mismatched.append("next_revision")
    if mismatched or model_mismatched:
        raise _Divergence(
            "registry_mismatch",
            "Replayed registry does not match the manifest's final registry: "
            f"datasets {mismatched or 'ok'}, models/counters {model_mismatched or 'ok'}",
        )
    digest_mismatches = [
        name
        for name, expected in manifest.state_digests.items()
        if digest_table(con, name) != expected
    ]
    if digest_mismatches:
        raise _Divergence(
            "state_digest_mismatch",
            f"Final state digests diverge for: {sorted(digest_mismatches)}",
        )
