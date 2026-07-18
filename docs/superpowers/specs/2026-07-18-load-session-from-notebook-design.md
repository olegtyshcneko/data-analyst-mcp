# `load_session_from_notebook` — design (v3)

**Date:** 2026-07-18
**Status:** revised per second-pass external review (gpt-5.6-sol xhigh): v1
snapshot manifest → "rethink"; v2 journal architecture → "revise then ship";
v3 folds every revise-then-ship finding.
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
| Mechanism | **Operation journal, replayed transactionally on the live connection** — the session records a structured journal of state-changing operations as they happen; `emit_notebook` serializes it into notebook metadata; `load_session_from_notebook` replays the journal inside a DuckDB transaction with expected-evidence comparison at every step, then `COMMIT`s and publishes the Python session state in one pointer swap. |

Why not a final-registry snapshot (v1): it cannot reconstruct overwrite-chain state
(review reproduced 13-live vs 7-reconstructed on `y *= 2; y += 1`), cannot
integrity-bind imported cells, had no true rollback, and let recomputed evidence
silently replace expected evidence. Why not a staging-connection copy (v2): the
cross-connection bridge is not type-faithful (review verified `DECIMAL(9,2)` →
`DOUBLE` through the pandas bridge) and the copy step is itself non-atomic.

## Architecture

### 1. Session-side: the operation journal

Alongside the recorder's cell list, the session keeps an append-only **journal** of
state-changing operations. Read-only tools (`query`, `correlate`, plots, …) do not
journal; their cells re-derive from state.

**Operation transaction invariant.** Each state-changing call carries a unique
`op_id`. The journal entry and the markdown+code cell pair are *prepared* before
any state is published; table mutation, registry update, journal entry, and cells
then publish together — an exception mid-call must not leave live state with
missing history. The `op_id` binds the journal entry to its two cells; emit-time
validation asserts the binding (every journal op has its cell pair, in order).
Tests fault-inject after every mutation boundary.

Journal entries (all also carry `op_id` and the digest fields defined in §3):

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

`source_hash` values use the real provenance formats: bare hex, `fallback:` for
files over the 100 MB ceiling, `sentinel:` where unverifiable (provenance.py).
`params`/`bse` are *named* coefficient and standard-error maps over
`design_columns`; `dispersion` carries the NegBin alpha. Standard errors are what
distinguish robust from plain OLS (coefficients are identical), so they are
mandatory evidence, not optional.

**Prerequisite fix (phase 0, own red/green):** `robust=True` OLS re-fits today as
plain OLS in the setup cell — pre-existing silent replay drift found by the review.
`ModelEntry` gains `fit_options`; setup-cell generation and the journal both honor
it. Lands before the resume work.

### 2. Emit-side: manifest

`emit_notebook` writes `nb.metadata["data_analyst_mcp"]`:

```
{
  "manifest_version": 1,
  "digest_algorithm": "damcp-digest-v1",
  "comparison": {"rtol": 1e-7, "atol": 1e-12},
  "resume_supported": true,
  "notebook_replayable": true,
  "journal": [ <ordered entries> ],
  "cells": [ {"index", "cell_type", "tool_name", "op_id?", "source_sha256"} … ],
  "setup_cell_sha256": "…",
  "state_digests": { "<dataset>": "<digest>" … },
  "final_registry": {
    "datasets": [ {name, format, read_options, path, columns, source_hash,
                   revision, base_loader?, split_overwrite?} … ],
    "models":   [ {name, kind, formula, fitted_on_dataset, n_obs, fit_options,
                   training_dataset_hash, training_dataset_revision,
                   training_loader} … ],
    "next_revision": <int>
  }
}
```

`notebook_replayable` is false when the emitted setup cell deliberately raises
(e.g. a registered model fit on since-overwritten state) — such sessions are
resume-only. `resume_supported` is false when the journal cannot recreate the
session (any `dataframe`-format entry — recorder already treats those as
unreloadable). Cell-level metadata additionally carries `tool_name` (body) and
`{"role": "setup"}` (setup). Rendered content and replay behavior are unchanged;
the manifest is invisible in Jupyter.

Notebooks emitted before this feature have no manifest and are not resumable
(re-emit from a live session to upgrade; no back-compat parser).

**Strict schema.** The manifest is a Pydantic discriminated union with
`extra="forbid"` plus semantic validation (unique final names, revision
monotonicity, split reciprocity, op_id↔cell pairing, digest coverage of every
final dataset) — any violation → `manifest_invalid`, before anything else runs.

### 3. The digest contract (`damcp-digest-v1`)

The table digest is the primary equality algorithm, so it is fixed *now*, not at
implementation, and named in the manifest so future encodings version cleanly:

- **Schema part:** for each column in position order — position, name, exact
  DuckDB type string — length-prefixed and hashed first.
- **Value part:** rows in physical scan order, each value encoded type-tagged and
  length-prefixed, with explicit rules: NULL tag; IEEE doubles bit-pattern-encoded
  (covers NaN, ±Inf, −0.0); DECIMAL as unscaled integer + scale; timestamps as
  UTC epoch micros + type tag; VARCHAR/BLOB length-prefixed bytes; nested types
  recursively tagged.
- **Computation:** streamed in chunks through the DuckDB connection (never a
  whole-table pandas materialization — SPEC §11 prohibits it), with
  `PRAGMA threads=1` pinned for the digest scan so scan order is deterministic.
- **Row order** is physical order — deliberately order-sensitive, because CV fold
  assignment is positional (ROADMAP's known hole). A session whose recreated
  physical order differs (e.g. planner-order-unstable CTAS) fails verification
  loudly with `state_digest_mismatch`; unordered SQL is *allowed to fail*, not
  pre-declared non-resumable.
- Digests are computed on the live in-transaction tables at both ends: at emit
  (`state_digests`, per-op `output_digest`) and at resume immediately before
  `COMMIT` — the digested tables are the committed tables, with no transfer step
  in between.

Every `load`/`materialize`/`split` journal entry records its **operation-output
digest** at live execution time; replay compares it immediately after re-executing
that op. Row counts remain diagnostic only. This binds intermediate states that
final digests cannot see (a table later overwritten mid-chain).

### 4. Load-side: phases

**Phase 1 — validate (no side effects, all independent failures accumulated).**
Session must be empty — registries, recorder, journal all empty — else
`session_not_empty`. Every table name the journal would create is preflighted
against the **live DuckDB catalog** (not just the registry; unregistered tables
survive `reset()` by design) — collision → `catalog_collision`, listing names.
Notebook readable/parseable (`notebook_not_found` / `notebook_invalid`); manifest
present, version known, schema-valid (`manifest_missing` /
`manifest_version_unsupported` / `manifest_invalid`); `resume_supported` true —
else `unreplayable_dataset` naming the offending entries. Every body cell matches
its descriptor (count, order, type, tool_name, SHA-256) and the setup cell matches
`setup_cell_sha256` — else `notebook_modified`. Resource caps enforced (see §6).
Source files preflight-hashed against journal expectations — mismatches →
`source_drift`, all listed.

**Phase 2 — transactional journal replay (fail-fast).**
Under a session-wide lock: `BEGIN` on the live connection, then replay the journal
in order through **shared prepare/apply primitives extracted from the existing
tool internals** (current tool functions are refactored to call them; recording
and MCP wrapping stay in the tool layer). Each primitive takes the entry's
expected evidence and compares before accepting:

- `load`: hash the file, load through the trusted-loader path, then hash again —
  pre/post provenance must agree with each other and with the journal
  (`source_drift`; closes load-time TOCTOU). `fallback:` entries compare
  `(path, mtime, size)`; `s3://`/`http` sources replay unguarded and add a
  response warning. Then compare `output_digest`. Historical loads (same name,
  different bytes mid-session) each verify at their own entry — the 1.5.0
  per-load-cell guard semantics exactly.
- `materialize`: re-validate the SQL through the same `_sql_safety` gate as the
  live tool, re-run at the journal position (overwrite chains reproduce step by
  step) on the external-access-disabled connection, compare `output_digest`.
- `split`: recompute membership from `(params, seed)`, compare each side's
  checksum (`split_drift`) and `output_digest`.
- `fit`: re-fit at the journal position — on the table state as of that point,
  which restores models fit on later-overwritten state with no special casing —
  honoring `fit_options`; compare `n_obs` exactly, then `params` *and* `bse`
  (and `dispersion` where present) name-by-name under the manifest's
  `comparison` tolerances (`model_drift`). The robust-OLS eval deliberately
  replays once without HC3 and asserts rejection.

Replay re-executes the original registration sequence, so revisions reproduce
exactly. After the last op: compare every final table digest (`state_digests`),
and compare the rebuilt registries against `final_registry`
(`registry_mismatch`) — tables can be right while metadata that future guards and
setup generation depend on is wrong, so both are checked. Phase 2 **fails at the
first divergent operation** with `op_index`, `op_id`, expected/actual evidence,
and downstream ops marked unverified (dependent evidence is tainted; accumulation
would mislead) → `ROLLBACK`. DuckDB rolls back `CREATE TABLE` and
`CREATE OR REPLACE TABLE` (review-verified on DuckDB 1.5.2); live Python state
was never touched.

**Phase 3 — commit and publish.**
`COMMIT`, then install the Python state as one atomic publication: registries,
model registry, recorder cells (source + tool_name, setup excluded), journal, and
`next_revision` from `final_registry` (without it, the next registration would
mint duplicate revisions). Publication is a single reference swap of prepared
structures — no fallible work after `COMMIT`. A failure *before* `COMMIT` is
phase 2's rollback; `resume_failed` remains only for the narrow window of a
`COMMIT` that itself errors, after which the connection state is reported
honestly rather than guessed at.

### 5. Contract: what "resume" guarantees

Resume succeeds **iff** transactional journal replay reproduces the recorded
evidence at every step, every final table digest, and the final registry
descriptor. Equivalence is defined precisely — not "bit-equal": tables are
digest-equal under `damcp-digest-v1`; dataset/model registry metadata is
field-equal excluding `registered_at`/`fitted_at` timestamps; fitted models are
evidence-equal (named params, SEs, dispersion within manifest tolerances); live
statsmodels Results objects are re-fit artifacts, intentionally outside the
equality claim. Weaker evidence classes surface in `warnings`: `fallback:`
hashes and unguarded remote sources are continuity-degraded, not silently equal.

`resume_supported` and `notebook_replayable` are independent: a session with a
pre-overwrite registered model is resumable while its emitted notebook
deliberately refuses nbconvert replay at the setup cell (existing loud behavior,
unchanged). The SPEC fold amends §5.13's acceptance criterion ("emitted notebooks
execute successfully") to apply when `notebook_replayable` is true, and README
language distinguishes the two.

### 6. Trust model, security, and budgets

A resumable notebook is **trusted executable provenance**, not passive data:
resuming re-executes its recorded operations — file loads touch the filesystem
(or network for s3/http) through the same trusted-loader path as live
`load_dataset`, and journal SQL executes under the same
`enable_external_access=false` sandbox and `_sql_safety` validation as live
`materialize_query`. The SHA-256 descriptors are **structural consistency**
checks, not authentication: an editor who consistently rewrites cells, journal,
and hashes produces a different valid trusted input, same as handing the server a
different file. No signature scheme in v1.

Guardrails: journal paths obey the same path/scheme policy as `load_dataset`;
all journal inputs re-run the live tools' validators; caps enforced in phase 1 —
notebook ≤ 32 MB, manifest ≤ 8 MB, ≤ 2 000 cells, ≤ 500 journal ops, ≤ 100 KB
per SQL/formula/path string (`manifest_invalid`) — and phase 2 runs under a
wall-clock budget (default 300 s, `resume_budget_exceeded`).

### 7. Recording, response, after resume

The tool records no cell and no journal entry of its own — the imported history
is its record. The SPEC fold replaces the "every successful tool records a cell"
contract with an explicit no-record enumeration reflecting reality —
`list_datasets`, `list_models`, `describe_column` (already no-record today,
review-verified), `emit_notebook`, `load_session_from_notebook`. Response:

```
{ "ok": true, "path": "…",
  "datasets": [{"name", "rows", "format"}…], "models": ["…"],
  "n_cells_imported": 14, "n_journal_ops": 9, "warnings": ["…"] }
```

Afterward tools target the restored state, new calls append cells and journal
entries (op_ids continue uniquely), and the next `emit_notebook` regenerates a
fresh setup cell over imported + new cells with a new manifest.

## Error taxonomy (new `error.type` values)

`session_not_empty` · `catalog_collision` · `notebook_not_found` ·
`notebook_invalid` · `manifest_missing` · `manifest_version_unsupported` ·
`manifest_invalid` · `notebook_modified` · `unreplayable_dataset` ·
`source_drift` · `split_drift` · `model_drift` · `state_digest_mismatch` ·
`registry_mismatch` · `resume_budget_exceeded` · `resume_failed`

Phase 1 accumulates all independent failures; phase 2 fails fast at the first
divergent op (`op_index`, `op_id`, expected/actual, downstream marked unverified).

## Governance

Tool 25 crosses the declared 24-tool v2 boundary: shipped as an explicit waivered
addition (ROADMAP ¶1 precedent). Additive release 1.6.0. SPEC folds: new tool
section; §5.11 manifest subsection; §5.13 replayability criterion conditioned on
`notebook_replayable`; recorder-contract no-record enumeration; README count +
"Known gotchas" restart paragraph; CHANGELOG.

## Testing

Phase 0 (robust-OLS `fit_options` fix) first, then red/green per repo rule. Evals:

- **Round-trip (replayable):** session with an overwrite chain, a split, and a
  robust OLS model → emit → reset → load → continue → emit → `nbconvert
  --execute` green; live values match the original session (the 13-vs-7 chain
  case asserted explicitly).
- **Round-trip (resume-only):** session with a pre-overwrite model resumes
  faithfully (params and SEs equal), while its emitted notebook still refuses
  nbconvert replay at the setup cell; manifest flags
  `notebook_replayable: false` — all asserted.
- **Drift:** file mutated between emit and load → `source_drift` in phase 1;
  file mutated between phase-1 preflight and the load op → `source_drift` from
  the pre/post-load agreement check; session and catalog unchanged after both.
- **Robust detection:** journal replayed once with HC3 stripped → `model_drift`
  on `bse` (coefficients alone would pass — asserted as a guard on the guard).
- **Integrity:** edited body cell → `notebook_modified`; manifest-less →
  `manifest_missing`; malformed → `manifest_invalid`; oversize journal →
  `manifest_invalid`.
- **Determinism:** `random()` recipe → op `output_digest` mismatch at its
  journal position (`state_digest_mismatch` family), not a downstream cascade.
- **Rejections:** dataframe dataset → `unreplayable_dataset`; non-empty session →
  `session_not_empty`; unregistered live table colliding with a journal name →
  `catalog_collision`, and rollback never drops it.
- **Atomicity:** fault-injection at every phase-2 op boundary → `ROLLBACK`
  leaves catalog and Python state untouched; op-transaction fault-injection in
  live tools leaves no journal/cell/state mismatch; emit-time op_id↔cell binding
  validated.
- **Continuation:** post-resume `next_revision` correct (new registration does
  not collide); post-resume emit's manifest passes its own validation and
  resumes again (idempotence).
- **Unit:** digest vectors for every type rule (NULL, NaN, ±Inf, −0.0, DECIMAL,
  timestamps, nested); journal recording per tool; per-primitive evidence
  comparison; registry descriptor comparison; caps.

## Out of scope

- Resuming pre-manifest or hand-edited notebooks; signatures/authentication.
- Merging into a non-empty session; force/allow-drift flags.
- Cross-machine float tolerance beyond the manifest's fixed `rtol`/`atol`
  (revisit only if a real user hits it).
- `compare_notebooks` diff (separate ROADMAP item).
- Changing `describe_column` to record (enumeration chosen instead).
