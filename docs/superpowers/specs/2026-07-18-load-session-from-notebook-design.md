# `load_session_from_notebook` — design (v4)

**Date:** 2026-07-18
**Status:** revised per third-pass external review (gpt-5.6-sol xhigh): v1
snapshot → "rethink"; v2 journal → "revise then ship" (1 blocker, 7 majors);
v3 → "revise then ship" (2 blockers, 5 majors); v4 folds all third-pass findings.
**Tool count:** 24 → 25 via explicit boundary waiver (precedent: `adjust_pvalues`,
`analyze_missingness`, model-registry trio — ROADMAP ¶1); target release 1.6.0

## Problem

Datasets, derived tables, splits, and fitted models are in-process state; restarting
the client drops all of it (README "Known gotchas"). The emitted notebook can replay
the work, but there is no way to *continue* it: a new session starts blank, and a
second `emit_notebook` produces a second, disjoint audit trail.

## Decision summary

| Question | Decision |
|---|---|
| Resume scope | **Full continuation** — restore live state (datasets, derived, splits, registered models) and import the old notebook's cells into the recorder, so the next `emit_notebook` produces one unified notebook. |
| Drift at load | **Atomic fail** — any verification failure aborts with a structured error; live session untouched. No partial load, no force flag. |
| Mechanism | **Operation journal, replayed transactionally on the live connection** — the session records a structured journal of state-changing operations as they happen; `emit_notebook` serializes it into notebook metadata; `load_session_from_notebook` replays the journal inside a DuckDB transaction with expected-evidence comparison at every step, then `COMMIT`s and publishes the Python session state in one swap. |

Prior-version rationale kept for the record: a final-registry snapshot (v1) cannot
reconstruct overwrite-chain state, integrity-bind cells, roll back, or bind
evidence; a staging-connection copy (v2) is not type-faithful (`DECIMAL(9,2)` →
`DOUBLE` through the pandas bridge, review-verified) and not atomic.

## Architecture

### 0. Session state and locking (new, foundational)

Datasets, models, recorder cells, journal, revision counter, and op sequence
become one composite, immutable **`SessionState`** value behind a session-wide
readers-writer lock. Every state-changing tool takes the write lock, opens a
DuckDB transaction, mutates tables, *prepares* its journal entry and cell pair,
commits, and publishes the replacement `SessionState` as a single reference swap.
Readers (`emit_notebook`, `list_*`, every stats tool) take the read lock and see
one consistent state. FastMCP currently serializes tool calls on its event loop,
but that is framework behavior, not an invariant — the lock is ours.
`load_session_from_notebook` holds the write lock from validation through
publication. This is also what makes the **operation-transaction invariant**
implementable: an exception at any boundary rolls back DuckDB and publishes
nothing, so live state can never exist without its history or vice versa. Tests
fault-inject at every mutation boundary.

### 1. Session-side: the operation journal

Alongside the recorder's cells, `SessionState` carries an append-only **journal**
of state-changing operations. Read-only tools do not journal. Each entry has an
`op_id` — a UUIDv4, generated at call time, validated unique across the manifest —
binding it to its markdown+code cell pair; emit-time validation asserts the
binding (every journal op has its cell pair, in order).

Journal entries (all carry `op_id`; digest fields per §3):

- `load_dataset` → `{op: "load", name, path, format, read_options, source_hash,
  rows, revision, output_digest}`
- `materialize_query` → `{op: "materialize", name, sql, overwrote: bool,
  base_loader?, split_overwrite?, rows, revision, output_digest}`
- `split_dataset` → `{op: "split", source, names: {train, test}, params, seed,
  membership_checksums: {train, test}, rows: {…}, revisions: {…},
  output_digests: {train, test}}`
- `fit_model` (with `model_name`) → `{op: "fit", model_name, dataset, formula,
  kind, fit_options, n_obs, design_columns: […], params: {name: value…},
  bse: {name: value…}, dispersion?: value, training_dataset_hash,
  training_dataset_revision, training_loader}`

`source_hash` uses the real provenance formats (bare hex, `fallback:`,
`sentinel:` — provenance.py). `params`/`bse` are *named* maps over
`design_columns`; `dispersion` is the NegBin alpha. Standard errors are mandatory
evidence — robust and plain OLS have identical coefficients (review-verified) and
differ only here. Nonfinite evidence values are serialized as tagged strings
(`"NaN"`, `"Infinity"`, `"-Infinity"`) so the manifest stays standard JSON;
comparison treats `NaN` as equal to `NaN` and infinities as sign-exact.
Model-evidence comparison requires **exact key-set equality** (a missing or extra
coefficient name is `model_drift`, not a skip).

**Prerequisite fix (phase 0, own red/green):** `robust=True` OLS re-fits today as
plain OLS in the setup cell — pre-existing silent replay drift. `ModelEntry` gains
`fit_options`; setup-cell generation and the journal both honor it.

### 2. Emit-side: manifest

`emit_notebook` writes `nb.metadata["data_analyst_mcp"]`:

```
{
  "manifest_version": 1,
  "digest_algorithm": "damcp-digest-v1",
  "comparison": {"rtol": 1e-7, "atol": 1e-12},
  "producer": {"duckdb": "…", "pandas": "…", "numpy": "…", "statsmodels": "…",
               "python": "…"},
  "resume_supported": true | false,
  "resume_unsupported_reasons": ["…"],
  "notebook_replayable": true | false,
  "journal": [ <ordered entries> ],
  "cells": [ {"index", "cell_type", "tool_name", "op_id?", "source_sha256"} … ],
  "setup_cell_sha256": "…",
  "state_digests": { "<dataset>": "<digest>" … },
  "final_registry": {
    "datasets": [ {name, format, read_options, path, columns, rows, source_hash,
                   revision, base_loader?, split_overwrite?} … ],
    "models":   [ {name, kind, formula, fitted_on_dataset, n_obs, fit_options,
                   training_dataset_hash, training_dataset_revision,
                   training_loader} … ],
    "next_revision": <int>
  }
}
```

`producer` versions are diagnostic only — never compared, but cited in
`model_drift`/`state_digest_mismatch` messages. `notebook_replayable` is false
when the emitted setup cell deliberately raises (e.g. a registered model fit on
since-overwritten state) — resume-only sessions. `resume_supported` is false when
the journal cannot recreate the session: any `dataframe`-format entry (recorder
already treats those as unreloadable) or any table containing a type
`damcp-digest-v1` does not cover (§3) — the reasons are listed. Per-op digests
are **captured at live execution time and serialized at emit**; `state_digests`
are computed at emit. Cell metadata carries `tool_name` (body) / `{"role":
"setup"}` (setup). Rendered content and replay behavior unchanged; manifest
invisible in Jupyter. Pre-manifest notebooks are not resumable (re-emit to
upgrade; no back-compat parser).

**Strict schema.** Pydantic discriminated union, `extra="forbid"`, semantic
validation (unique final names, revision monotonicity, split reciprocity,
op_id↔cell pairing and uniqueness, digest coverage of every final dataset) —
violation → `manifest_invalid`, before anything else runs.

### 3. The digest contract (`damcp-digest-v1`)

The primary state-equality algorithm, fixed here so implementations cannot
diverge and the encoding is injective:

- **Hash:** SHA-256 over a byte stream with single-byte domain separators —
  `0x01` schema part, `0x02` value part, then one tag byte per type (table fixed
  at implementation-plan time from the enumeration below, stable forever under
  this algorithm name). All length prefixes are unsigned 64-bit little-endian;
  all multi-byte integers little-endian.
- **Schema part:** per column in position order — position (u64), name
  (length-prefixed UTF-8), exact DuckDB logical type string (length-prefixed).
- **Values, exhaustive over DuckDB's logical types:** BOOLEAN 1 byte;
  TINYINT/SMALLINT/INTEGER/BIGINT/HUGEINT and unsigned widths plus BIGNUM as
  length-prefixed two's-complement little-endian; FLOAT as 4-byte and DOUBLE as
  8-byte raw IEEE bit patterns (covers NaN payloads, ±Inf, −0.0); DECIMAL as
  (width, scale) in the type string + unscaled integer value; VARCHAR/BLOB/BIT
  length-prefixed bytes; DATE days-since-epoch i32; TIME/TIME_TZ micros;
  **each TIMESTAMP variant in its native resolution with a distinct tag** —
  TIMESTAMP_S seconds, TIMESTAMP_MS millis, TIMESTAMP micros, TIMESTAMP_NS
  nanos, TIMESTAMPTZ UTC micros (review verified that routing timestamps
  through Python `datetime` silently truncates nanoseconds — the extraction
  path must preserve native resolution and raw float bits, e.g. DuckDB's
  Arrow-native chunk reader, never `datetime`/`float` coercion); INTERVAL as
  (months i32, days i32, micros i64); UUID 16 bytes; ENUM as its string value;
  LIST/STRUCT/MAP/UNION recursively with per-element tags; NULL as its own tag.
  **Any other type (VARIANT, extension types) is undigestable:** the live tool
  call still succeeds but the op records `output_digest: null` and the session
  becomes `resume_supported: false` with the reason — loud at resume, never
  breaking live work.
- **Computation:** streamed in chunks through the DuckDB connection (never a
  whole-table pandas materialization — SPEC §11). The digest is
  chunk-size-invariant by construction (pure byte-stream concatenation);
  conformance vectors assert it.
- **Row order** is physical scan order — deliberately order-sensitive (CV fold
  assignment is positional; ROADMAP's known hole). Review probes showed parallel
  CTAS can vary physical order between runs; pinning the digest scan cannot
  stabilize creation order. Consequence, stated honestly: a session whose
  recreated physical order differs fails loudly with `state_digest_mismatch` —
  unordered-output SQL is *allowed to fail*, not pre-declared non-resumable.
- **Settings hygiene:** any connection setting the digest or replay path changes
  (`threads`, and anything else touched) is snapshot before and restored in a
  `finally` on **every** exit — success, rollback, budget stop, commit failure —
  because PRAGMAs are not transactional (review verified `threads=1` survives
  `ROLLBACK`). `threads`, `memory_limit`, `preserve_insertion_order`, search
  path, and temp-directory settings are treated as observable session state;
  restoration is tested explicitly. Live per-op digesting restores settings the
  same way so it can never alter subsequent CTAS behavior.

Every `load`/`materialize`/`split` entry records its **operation-output digest**
at live execution time; replay compares immediately after re-executing that op.
Row counts are diagnostic only. This binds intermediate states final digests
cannot see (tables later overwritten mid-chain).

### 4. Load-side: phases

**Phase 1 — validate (no side effects; all independent failures accumulated).**
Under the write lock: `SessionState` must be empty — else `session_not_empty` —
**and the live user catalog must be empty of tables and views** (unregistered
objects deliberately survive `reset()`; review verified journal SQL can silently
*read* such ambient tables, so mere collision-freedom would let replay "succeed"
on state the notebook never created) — else `catalog_not_empty`, listing objects.
(An isolated-schema replay that tolerates ambient state was considered and
deferred; empty-catalog is the v1 contract.) Notebook readable/parseable
(`notebook_not_found` / `notebook_invalid`); manifest present, version known,
schema-valid (`manifest_missing` / `manifest_version_unsupported` /
`manifest_invalid`); `resume_supported` true — else `unreplayable_dataset` with
the recorded reasons. Cells match descriptors (count, order, type, tool_name,
SHA-256; setup cell vs `setup_cell_sha256`) — else `notebook_modified`. Resource
caps (§6). Source files preflight-hashed — mismatches → `source_drift`, all
listed.

**Phase 2 — transactional journal replay (fail-fast).**
`BEGIN` on the live connection; replay in order through **shared prepare/apply
primitives extracted from the existing tool internals** (tool functions refactor
to call them; recording and MCP wrapping stay in the tool layer). Each primitive
compares expected evidence before accepting:

- `load`: hash file → load through the trusted-loader path → hash again;
  pre/post must agree with each other and the journal (`source_drift`; closes
  load TOCTOU). `fallback:` entries compare `(path, mtime, size)`; `s3://`/
  `http` replay unguarded + response warning. Then compare `output_digest`.
  Historical loads verify per entry — 1.5.0 per-load-cell semantics.
- `materialize`: re-validate SQL through the live `_sql_safety` gate, re-run at
  the journal position on the external-access-disabled connection, compare
  `output_digest`.
- `split`: recompute membership from `(params, seed)`; compare per-side checksums
  (`split_drift`) and `output_digests`.
- `fit`: re-fit at the journal position (restores models fit on
  later-overwritten state with no special casing), honoring `fit_options`;
  compare `n_obs` exactly, then `params`, `bse`, `dispersion` name-by-name with
  exact key sets under the manifest tolerances (`model_drift`).

Replay re-executes the original registration sequence, so revisions reproduce
exactly; the rebuilt next revision is **computed and compared** to the manifest's
`next_revision` (`registry_mismatch`), not trusted. After the last op: compare
every final table digest (`state_digests`) and the rebuilt registries — including
`rows` — against `final_registry` as name-keyed maps (ordering is not part of the
contract), excluding timestamps (`registry_mismatch`). Failure → first divergent
op reported (`op_index`, `op_id`, expected/actual, downstream marked unverified)
→ `ROLLBACK`; settings restored; live Python state was never touched.

**Phase 3 — commit and publish.**
`COMMIT`, then publish the prepared `SessionState` (registries, models, recorder
cells — source + tool_name, setup excluded — journal, `next_revision`, op
sequence) as one reference swap; no fallible work after `COMMIT`. `resume_failed`
covers only a `COMMIT` that itself errors, reported honestly.

### 5. Contract: what "resume" guarantees

Resume succeeds **iff** transactional replay reproduces recorded evidence at
every step, every final table digest, and the final-registry descriptor.
Equivalence, precisely: tables digest-equal under `damcp-digest-v1`; registry
metadata field-equal (including `rows`, compared as name-keyed maps) excluding
`registered_at`/`fitted_at`; models evidence-equal (named params/SEs/dispersion
within manifest tolerances); live statsmodels Results objects are re-fit
artifacts outside the equality claim. Weaker evidence classes surface in
`warnings` (`fallback:` hashes, unguarded remote sources).

`resume_supported` and `notebook_replayable` are independent: a pre-overwrite
registered model is resumable while its notebook deliberately refuses nbconvert
replay (existing loud behavior, unchanged).

**Version envelope.** Correctness of rollback, type strings, and physical-order
behavior is DuckDB-version-sensitive; the design was runtime-verified on 1.5.2
while `pyproject.toml` currently allows `duckdb>=1.1.0`. Release 1.6.0 raises the
minimum to the tested line (`duckdb>=1.5.2`) and CI covers the supported range;
`producer` versions are recorded for diagnostics.

### 6. Trust model, security, and budgets

A resumable notebook is **trusted executable provenance**: resuming re-executes
recorded operations. That includes filesystem/network access through the trusted
loader (paths and schemes re-validated by the same policy as live
`load_dataset`), journal SQL under the same `enable_external_access=false`
sandbox and `_sql_safety` validation as live `materialize_query`, **and model
formulas, which are process-level arbitrary code execution** — Patsy evaluates
Python inside formulas (`I(__import__(…))`, review-verified). No sandbox contains
the formula path today (live `fit_model` has the same property); resume only
notebooks whose author you fully trust. SHA-256 descriptors are structural
consistency, not authentication; no signatures in v1.

Caps, enforced in phase 1: notebook ≤ 32 MB, manifest ≤ 8 MB, ≤ 2 000 cells,
≤ 500 journal ops, ≤ 100 KB per SQL/formula/path string (`manifest_invalid`).
The replay deadline (default 300 s) is **cooperative, best-effort**: checked
between operations, with `connection.interrupt()` where DuckDB supports it — a
single blocking operation (CTAS, statsmodels fit, loader read) can overshoot; a
hard guarantee would need process isolation, explicitly out of scope. On budget
stop: rollback + settings restoration as any phase-2 failure
(`resume_budget_exceeded`).

### 7. Recording, response, after resume

The tool records no cell and no journal entry of its own. Response:

```
{ "ok": true, "path": "…",
  "datasets": [{"name", "rows", "format"}…], "models": ["…"],
  "n_cells_imported": 14, "n_journal_ops": 9, "warnings": ["…"] }
```

Afterward tools target the restored state; new calls append cells/journal entries
(fresh UUIDv4 op_ids); the next `emit_notebook` regenerates a fresh setup cell
over imported + new cells with a new manifest that must itself validate and
resume (idempotence is tested).

## Error taxonomy (new `error.type` values)

`session_not_empty` · `catalog_not_empty` · `notebook_not_found` ·
`notebook_invalid` · `manifest_missing` · `manifest_version_unsupported` ·
`manifest_invalid` · `notebook_modified` · `unreplayable_dataset` ·
`source_drift` · `split_drift` · `model_drift` · `state_digest_mismatch` ·
`registry_mismatch` · `resume_budget_exceeded` · `resume_failed`

Phase 1 accumulates all independent failures; phase 2 fails fast at the first
divergent op.

## Governance — explicit fold checklist

1. SPEC §5: tool section for `load_session_from_notebook`; record contract
   replaced by the no-record enumeration — `list_datasets`, `list_models`,
   `describe_column` (already no-record today, review-verified), `emit_notebook`,
   `load_session_from_notebook`.
2. SPEC §5.11: manifest + journal subsection. SPEC §5.13 **and §12**: notebook
   execution criterion conditioned on `notebook_replayable`.
3. SPEC §11: 24-tool boundary waived for tool 25 (explicit waiver text,
   ROADMAP ¶1 precedent).
4. Quality-gate wording wherever it presumes unconditional notebook execution.
5. ROADMAP: replace the parked snapshot-flavored `load_session_from_notebook`
   entry with a pointer to this spec; update the tool count.
6. README: tool count, "every tool call appends a cell" language, "Known
   gotchas" restart paragraph, worked example addendum.
7. `pyproject.toml`: `duckdb>=1.5.2`. CHANGELOG under 1.6.0.

## Testing

Phase 0 (robust-OLS `fit_options`) first; red/green per repo rule. **Black-box
evals** (fresh server process per round-trip, via `mcp.client.stdio` like the
existing 54):

- **Round-trip (replayable):** overwrite chain + split + robust OLS →
  emit → new process → load → continue → emit → `nbconvert --execute` green;
  live values match (13-vs-7 chain case asserted).
- **Round-trip (resume-only):** pre-overwrite model resumes faithfully (params
  and SEs equal); notebook refuses nbconvert replay; manifest flags
  `notebook_replayable: false`.
- **Idempotence:** post-resume emit validates and resumes again.
- **Drift:** file mutated between emit and load → phase-1 `source_drift`;
  session and catalog unchanged.
- **Robust detection:** replay with HC3 stripped → `model_drift` on `bse`
  (coefficients alone would pass — the guard on the guard).
- **Integrity/rejections:** edited cell → `notebook_modified`; manifest-less →
  `manifest_missing`; malformed/oversize → `manifest_invalid`; dataframe entry
  or undigestable type → `unreplayable_dataset`; non-empty session →
  `session_not_empty`; ambient unregistered table → `catalog_not_empty` and
  rollback never drops it.
- **Determinism:** `random()` recipe → digest mismatch at its journal position.

**White-box unit / fault-injection tests** (in-process, hooks allowed):

- TOCTOU via an injected pre-load barrier (deterministic mutation between
  phase-1 hash and the load primitive) → `source_drift`.
- Fault-injection at every phase-2 op boundary and every live-tool mutation
  boundary → rollback leaves catalog, settings, and Python state untouched;
  no journal/cell/state mismatch ever observable.
- Commit-failure path; cooperative-deadline stop mid-journal; settings
  snapshot/restoration on all exit paths.
- Digest conformance vectors per type rule (every supported logical type, NULL,
  NaN payloads, ±Inf, −0.0, DECIMAL scales, all timestamp resolutions incl. a
  TIMESTAMP_NS value that would collide under microsecond truncation, INTERVAL,
  ENUM, nested types), chunk-size invariance, unsupported-type →
  `resume_supported: false` at emit.
- Ordered-SQL round-trip success and an explicitly-allowed unordered-CTAS
  failure; registry descriptor comparison incl. `rows` and `next_revision`;
  op_id uniqueness validation; nonfinite model-evidence serialization and
  equality; caps.

## Out of scope

- Resuming pre-manifest or hand-edited notebooks; signatures/authentication.
- Merging into a non-empty session; force/allow-drift flags; isolated-schema
  replay tolerating ambient catalog state (deferred; empty-catalog is v1).
- Hard (non-cooperative) budget enforcement / process isolation.
- Cross-machine float tolerance beyond the manifest's fixed `rtol`/`atol`.
- `compare_notebooks` diff (separate ROADMAP item).
- Changing `describe_column` to record (enumeration chosen instead).
