# `load_session_from_notebook` — design (v2, journal architecture)

**Date:** 2026-07-18
**Status:** approved direction; v2 redesign after external review (gpt-5.6-sol xhigh,
session 019f75a4) returned "rethink" on the v1 snapshot-manifest design
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
| Mechanism | **Operation journal, replayed in staging** — the session records a structured journal of state-changing operations as they happen; `emit_notebook` serializes it into notebook metadata; `load_session_from_notebook` replays the journal in an isolated staging context with expected-evidence comparison at every step and final state digests, then commits in one swap. |

### Why not the v1 snapshot manifest (review blockers, all accepted)

1. A final-registry snapshot cannot reconstruct live state: `materialize_query`
   overwrite chains keep only the latest recipe, so snapshot reconstruction of
   `y *= 2; y += 1` yields 7 where the live session holds 13 — resume would bless
   wrong state (verified by review).
2. Historical per-load hash guards (1.5.0) live only inside generated cell code;
   without per-cell integrity binding, resume could succeed on a notebook whose own
   body would fail replay, or on an edited notebook.
3. `session.reset()` is cleanup, not rollback — it drops only registry-known tables
   and never touches the recorder; mid-apply failure would leak orphan tables.
4. Existing load/materialize/split internals *compute and store* fresh evidence;
   they never *compare against expected* evidence, so verify-then-apply had a
   TOCTOU laundering hole.

The journal design dissolves all four: replaying the full ordered operation history
reproduces overwrite chains and pre-overwrite model fits exactly; cell descriptors
integrity-bind the imported history; staging + single-swap commit gives real
atomicity; and every replayed step compares computed evidence against recorded
evidence before proceeding.

## Architecture

### 1. Session-side: the operation journal (new in-process structure)

Alongside the recorder's cell list, the session keeps an append-only **journal**.
Every state-changing tool appends one structured entry at call time:

- `load_dataset` → `{op: "load", name, path, format, read_options, source_hash,
  rows, revision}`
- `materialize_query` → `{op: "materialize", name, sql, overwrote: bool,
  base_loader?, split_overwrite?, rows, revision}`
- `split_dataset` → `{op: "split", source, names: [train, test], params, seed,
  membership_checksums: {train, test}, rows: {...}, revisions: {...}}`
- `fit_model` (with `model_name`) → `{op: "fit", model_name, dataset, formula,
  kind, fit_options: {robust, ...}, n_obs, params: [...], training_dataset_hash,
  training_dataset_revision, training_loader}`

`source_hash` values use the real provenance formats: bare hex, `fallback:` for
files over the 100 MB ceiling, `sentinel:` where unverifiable (provenance.py).
`params` is the fitted coefficient vector, recorded for post-refit comparison.
Read-only tools (query, describe, correlate, plots, …) do **not** journal — their
cells re-derive from state.

**Prerequisite fix (phase 0, own red/green):** `robust=True` OLS re-fits today as
plain OLS in the setup cell — pre-existing silent replay drift found by the review.
`ModelEntry` gains `fit_options`; setup-cell generation and the journal both honor
it. This lands before the resume work.

### 2. Emit-side: manifest = journal + cell descriptors

`emit_notebook` writes `nb.metadata["data_analyst_mcp"]`:

```
{
  "manifest_version": 1,
  "journal": [ <ordered entries as above> ],
  "cells": [ {"index", "cell_type", "tool_name", "source_sha256"} ... ],
  "setup_cell_sha256": "...",
  "state_digests": { "<dataset>": "<order-sensitive table digest>" ... },
  "final_models": ["<model_name>", ...]
}
```

`state_digests` are order-sensitive SHA-256 digests over a canonical row/column
serialization of each live table at emit time (exact encoding fixed at
implementation; row order = physical order). They are the ground truth that journal
replay must reproduce. Cell descriptors cover every body cell in order; the setup
cell is digested separately. Rendered notebook content and replay behavior are
unchanged; the manifest is invisible in Jupyter. Cell-level metadata additionally
carries `tool_name` (body) and `{"role": "setup"}` (setup).

Notebooks emitted before this feature have no manifest and are **not resumable**
(re-emit from a live session to upgrade; no back-compat parser).

**Strict schema.** The manifest is validated as a Pydantic discriminated union with
`extra="forbid"` plus semantic checks (unique names per final state, revision
monotonicity, split reciprocity, cell pairing, digest coverage of every final
dataset). Any violation → `manifest_invalid`, before anything else runs.

**Non-serializable state.** A session containing a dataset the journal cannot
recreate (e.g. `dataframe`-format entries, which the recorder already treats as
unreloadable) still emits — replay behavior today — but the manifest marks those
entries `resumable: false`; `load_session_from_notebook` rejects such notebooks
with `unreplayable_dataset`, naming them. Format enum covers the real loader set:
`csv, tsv, parquet, xlsx, json, jsonl, derived, split, dataframe`.

### 3. Load-side: three phases

**Phase 1 — validate (no side effects).**
Session must be empty (no datasets, no models, no recorded cells, no journal) —
else `session_not_empty`. Notebook readable/parseable — else `notebook_not_found`
/ `notebook_invalid`. Manifest present, version known, schema-valid —
`manifest_missing` / `manifest_version_unsupported` / `manifest_invalid`. Every
body cell matches its descriptor (count, order, type, `tool_name`, SHA-256) and
the setup cell matches `setup_cell_sha256` — any mismatch → `notebook_modified`
(edited notebooks are non-resumable, by design). No `resumable: false` entries —
else `unreplayable_dataset`.

**Phase 2 — staged journal replay (no live-session mutation).**
A staging context — its own DuckDB connection with
`enable_external_access=false` (file loads use the same short-lived trusted-loader
path as live tools; journal SQL runs only under the same sandbox as
`materialize_query`) plus staged dataset/model registries — replays the journal in
order through **shared prepare/apply primitives extracted from the existing tool
internals** (the current monolithic tool functions are refactored to call them;
recording and MCP wrapping stay in the tool layer). Each primitive takes the
journal entry's *expected evidence* and compares before accepting:

- `load`: hash the file, compare to the entry's `source_hash` (fallback entries
  compare `(path, mtime, size)`; `s3://`/`http` sources replay unguarded and add a
  response warning, mirroring replay) — mismatch → `source_drift`. This check runs
  per journal entry, so historical loads (same name, different bytes mid-session)
  are each verified — matching the 1.5.0 per-load-cell guard semantics exactly.
- `materialize`: re-run the recipe at its journal position (overwrite chains thus
  reproduce step by step); compare resulting row count.
- `split`: recompute membership from `(params, seed)`, compare each side's
  checksum — mismatch → `split_drift`.
- `fit`: re-fit at the journal position — i.e. on the table state as of that point,
  which restores models fit on later-overwritten state without any special casing —
  honoring `fit_options`; compare `n_obs` exactly and coefficients via
  `np.allclose` against recorded `params` — mismatch → `model_drift` (catches
  statsmodels/library version drift).

Because replay re-executes the same registration sequence, staged revisions
reproduce the original numbering exactly — no revision mapping. After the journal,
compute each final table's digest and compare to `state_digests` — mismatch →
`state_digest_mismatch`. This is what turns the ROADMAP's known nondeterminism
holes (`random()` in recipes, order drift) into loud resume failures. Any failure
in this phase discards staging; the live session was never touched.

**Phase 3 — commit (single swap, tracked rollback).**
Move staged state into the live session: create each table on the live connection
(tracking every name created), install the staged registries, append the imported
cells (source + `tool_name`, setup excluded) and the journal to the live
recorder/session. Commit operations are mechanical (pre-validated data only); if
one nonetheless fails, roll back by dropping exactly the tracked created tables and
clearing registries, recorder, and journal — complete, because the session was
empty at entry — then raise `resume_failed`.

### Contract: what "resume" guarantees

Resume succeeds **iff** staged journal replay reproduces the recorded evidence at
every step and the recorded final state digests. This is deliberately *stronger*
than notebook replayability: nbconvert replay of the emitted notebook remains the
human-facing contract with its documented residual gaps (ROADMAP); resume adds
digest-verified state equality on top. One divergence, made explicit: a session
whose registered model was fit on since-overwritten state emits a notebook whose
setup cell refuses replay (existing loud behavior, unchanged), yet such a session
**is** resumable — journal replay recreates the fit faithfully at its position.
Weaker evidence classes surface in the response `warnings`: `fallback:` hash
entries and unguarded remote sources are continuity-degraded, not silently equal.

### Recording, response, after resume

The tool records no cell and no journal entry of its own — the imported history is
its record (SPEC's "every successful tool records a cell" contract gains named
exceptions: `emit_notebook`, `load_session_from_notebook`). Response:

```
{ "ok": true, "path": "...",
  "datasets": [{"name", "rows", "format"}...], "models": ["..."],
  "n_cells_imported": 14, "n_journal_ops": 9, "warnings": ["..."] }
```

Afterward the session is (verified-)bit-equal to the emitting session's final
state: tools target restored data, new calls append cells and journal entries, and
the next `emit_notebook` regenerates a fresh setup cell over imported + new cells
with a new manifest.

## Error taxonomy (new `error.type` values)

`session_not_empty` · `notebook_not_found` · `notebook_invalid` ·
`manifest_missing` · `manifest_version_unsupported` · `manifest_invalid` ·
`notebook_modified` · `unreplayable_dataset` · `source_drift` · `split_drift` ·
`model_drift` · `state_digest_mismatch` · `resume_failed`

Phase-1/2 errors report **all** failing items of their category, not the first.

## Governance

Tool 25 crosses the declared 24-tool v2 boundary: shipped as an explicit waivered
addition (ROADMAP ¶1 precedent), recorded in ROADMAP and SPEC §5 at fold time.
Additive release 1.6.0. SPEC folds: new tool section; §5.11 manifest subsection;
recorder-contract no-record exceptions; README count + "Known gotchas" restart
paragraph; CHANGELOG.

## Testing

Phase 0 (robust-OLS fix), then red/green per repo rule. Evals:

- **Round-trip (replayable):** session with an overwrite chain, a split, and a
  robust OLS model → emit → reset → load → continue → emit → `nbconvert
  --execute` green; live values match the original session (the 13-vs-7 chain
  case asserted explicitly).
- **Round-trip (ephemeral model):** session with a pre-overwrite model resumes
  faithfully (coefficients equal), while its emitted notebook still refuses
  nbconvert replay at the setup cell — both asserted.
- **Drift:** mutate a file between emit and load → `source_drift` naming it;
  session empty after; no orphan tables on the live connection.
- **Integrity:** edit one body cell → `notebook_modified`; manifest-less notebook →
  `manifest_missing`; malformed manifest → `manifest_invalid`.
- **Determinism:** `random()` recipe → `state_digest_mismatch`.
- **Rejections:** dataframe dataset → `unreplayable_dataset`; non-empty session →
  `session_not_empty`.
- **Fidelity:** resumed robust-OLS SEs equal originals; resumed pre-overwrite model
  coefficients equal originals; historical two-hash load sequence verifies per
  entry.
- **Unit:** journal recording per tool; manifest build/validation; per-primitive
  evidence comparison; staged-replay isolation (live session untouched on every
  phase-2 failure); commit rollback drops tracked tables only.

## Out of scope

- Resuming pre-manifest or hand-edited notebooks.
- Merging into a non-empty session; force/allow-drift flags.
- Cross-machine float tolerance policy beyond `np.allclose` defaults (revisit only
  if a real user hits it).
- `compare_notebooks` diff (separate ROADMAP item).
