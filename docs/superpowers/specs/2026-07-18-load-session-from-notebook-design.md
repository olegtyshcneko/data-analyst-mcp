# `load_session_from_notebook` — design

**Date:** 2026-07-18
**Status:** approved design, pre-implementation
**Tool count:** 24 → 25 (first post-1.5.0 tool; ROADMAP "Reproducibility" bucket, first item)

## Problem

Datasets, derived tables, splits, and fitted models are in-process state; restarting
the client drops all of it (README "Known gotchas"). The emitted notebook can replay
the work, but there is no way to *continue* it: a new session starts blank, and a
second `emit_notebook` produces a second, disjoint audit trail.

## Decision summary

| Question | Decision |
|---|---|
| Resume scope | **Full continuation** — registry (datasets, derived, splits) + re-fit registered models + import the old notebook's cells into the recorder, so the next `emit_notebook` produces one unified replayable notebook. |
| Drift at load | **Atomic fail** — any guard failure aborts the whole load with a structured error; session state untouched. No partial load, no force flag. |
| Mechanism | **Emit-time manifest** — `emit_notebook` embeds a schema-versioned manifest in notebook metadata; `load_session_from_notebook` reads only the manifest. No parsing of generated code, no execution of notebook code. |

Rejected alternatives: parsing the setup cell (permanently couples a parser to the
code generator; silently mis-parses hand-edited notebooks) and executing the notebook
via nbclient (arbitrary code execution inside the server; state lands in the wrong
connection).

## Governing principle: resume = trusted replay of the manifest

`load_session_from_notebook` succeeds exactly when the notebook's own setup-cell
guards would pass at replay time against the current filesystem. Every existing
drift guard applies; no new guarantee class is invented. A notebook that is
non-replayable by design — e.g. it contains a model fit on since-overwritten
("ephemeral") table state, which the setup cell refuses to re-fit — fails resume
with the same explanation.

## Emit-side changes (additive, invisible in Jupyter)

`NotebookRecorder.to_notebook()` gains:

1. **Notebook-level manifest** at `nb.metadata["data_analyst_mcp"]`:

   ```json
   {
     "manifest_version": 1,
     "datasets": [
       {
         "name": "...", "format": "csv|parquet|xlsx|json|derived",
         "path": "...", "read_options": {...},
         "source_hash": "sha256:...|sentinel:...",
         "revision": 3,
         "recreate": { "recipe_sql": "...", "split": {...}, "base_loader": {...},
                       "split_overwrite": {...} }
       }
     ],
     "models": [
       {
         "name": "...", "kind": "ols|logistic|poisson|negbin", "formula": "...",
         "fitted_on_dataset": "...", "training_dataset_hash": "...",
         "training_dataset_revision": 5, "training_loader": {...}
       }
     ]
   }
   ```

   `datasets` is ordered by registration revision. The `recreate` block carries
   exactly the recreation inputs the setup-cell builder already draws on; fields
   absent when not applicable. Model entries mirror `ModelEntry`'s guard fields
   verbatim (minus the live `_result`).

2. **Cell-level metadata:** every body cell gets `{"tool_name": ...}` (the value
   the recorder already holds in memory and currently drops at emit); the setup
   cell gets `{"role": "setup"}`.

Rendered notebook content and replay behavior are unchanged. Notebooks emitted
before this change have no manifest and are **not resumable**; the upgrade path is
re-emitting from a live session. No back-compat parser will be built.

## Tool: `load_session_from_notebook(path: str)`

### Phase 1 — verify (no session mutation)

Checks in order; first category to fail aborts:

1. Session must be empty — no datasets, no models, no recorded cells — else
   `session_not_empty`. (Primary use-case is a fresh session after client restart.)
2. File readable and valid nbformat — else `notebook_not_found` / `notebook_invalid`.
3. Manifest present — else `manifest_missing` (message directs: emitted before this
   feature; re-emit from a live session).
4. `manifest_version` known — else `manifest_version_unsupported`.
5. Every guard the setup cell would check, evaluated across the *whole* manifest so
   the error lists **every** failing item, not just the first:
   - file-backed sources re-hashed against `source_hash`; entries that used the
     >100 MB `(path, mtime, size)` fallback verify the same way; `s3://`/`http`
     sources skip verification and add a warning (mirrors replay behavior);
   - every model's training state replayable per the existing revision/loader
     guards — else `ephemeral_training_state`.

   Hash mismatches → `source_drift` with per-file detail.

### Phase 2 — apply

Walk `datasets` in manifest order through the **existing internal code paths**
(`load_dataset` / `materialize_query` / `split_dataset` internals, not the MCP tool
wrappers — nothing records cells during apply). Order preservation means fresh
revisions stamp consistently with inter-entry references. Then re-fit each model
via the existing fit internals (statsmodels is deterministic on identical input),
which stamp fresh revision/loader identity from the rehydrated registry — the
manifest's recorded revisions are verify-phase inputs only and never enter the
new session's state.
Finally import every body cell — source + `tool_name`, in order, `role: "setup"`
excluded — verbatim into the recorder.

A mid-apply failure (e.g. a file changed between verify and apply) raises
`resume_failed` and calls `session.reset()` — acceptable precisely because the
session was empty at entry; no user state can be lost.

### Recording and response

The tool records **no cell of its own** — the imported history is its record.
Response:

```json
{ "ok": true, "path": "...",
  "datasets": [{"name": "...", "rows": 5000, "format": "csv"}],
  "models": ["titanic_logit"],
  "n_cells_imported": 14,
  "warnings": ["s3://... reloaded unguarded"] }
```

### After resume

The session behaves as if it never restarted: tools target the restored registry;
new calls append after the imported cells; `emit_notebook` builds a fresh setup
cell from current state plus imported + new cells — one unified notebook, with a
new manifest, replayable end-to-end.

## Error taxonomy (new `error.type` values)

`session_not_empty` · `notebook_not_found` · `notebook_invalid` ·
`manifest_missing` · `manifest_version_unsupported` · `source_drift` ·
`ephemeral_training_state` · `resume_failed`

## Testing

Red/green TDD per repo rule, plus evals:

- **Round-trip eval:** session → emit → reset → load → continue with new tool calls
  → emit → `jupyter nbconvert --execute` exits 0; unified notebook contains old +
  new cells and one setup cell.
- **Drift eval:** mutate a source file between emit and load → `source_drift`
  listing the file; session still empty afterward.
- **Manifest-less notebook** → `manifest_missing`. **Non-empty session** →
  `session_not_empty`. **Ephemeral-model notebook** → `ephemeral_training_state`.
- **Unit:** manifest build (each dataset kind: file, derived, split, overwrite
  chains), cell metadata emission, verify-phase error accumulation (multiple
  failures all reported), revision/guard consistency of re-fit models after apply,
  mid-apply failure resets cleanly.

## Out of scope

- Resuming pre-manifest notebooks (re-emit instead).
- Merging into a non-empty session.
- A `force`/`allow_drift` escape hatch (launders drift into a notebook that
  claims continuity).
- Notebook diff, `compare_notebooks` (separate ROADMAP item).

## SPEC fold

At implementation: SPEC §5 gains the tool section; §5.11 (emit) gains the manifest
subsection; README tool count and "Known gotchas" (restart paragraph) updated;
ROADMAP reproducibility bucket updated; CHANGELOG under 1.6.0.
