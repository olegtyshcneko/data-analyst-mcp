# Dataset provenance hashes — design

**Date:** 2026-07-05
**Status:** approved (brainstorm session)
**ROADMAP origin:** Reproducibility → "Provenance hashes"

## Problem

The emitted notebook's setup cell reloads every file-backed dataset with a bare
`CREATE OR REPLACE TABLE ... FROM read_*_auto(path)` — no guard. If the source
file is edited between session and replay, every downstream number silently
changes. Models already solve this for *their* slice of the problem: the
setup cell asserts a SHA-256 of the training file before re-fitting. Datasets
that never fed a model have no protection at all.

There is also a latent correctness quirk: in a live session `fit_model` trains
on the DuckDB table that was populated at `load_dataset` time, but it hashes
the **file at fit time** (`tools/models.py`). If the file changes mid-session,
the recorded hash describes bytes the model was never trained on. A load-time
hash is the truthful provenance anchor.

## Decision summary

- Hash every dataset at registration time; store on `DatasetEntry`.
- Emit hash asserts in the setup cell for every file-backed dataset (and
  `base_loader` re-creates), mirroring the model guard's three shapes.
- Unify: `fit_model` reads the dataset entry's hash instead of recomputing;
  the model block's own assert lines and `ModelEntry.training_dataset_hash`
  are removed as redundant.
- Internal only: no tool-response schema changes, no new tool. The 22-tool
  boundary is untouched.

Alternatives considered: (A) purely additive parallel guard leaving
`fit_model`'s recompute in place — rejected as it preserves the mid-session
quirk and double-asserts the same file; (C) hashing at `emit_notebook` time —
rejected because a mid-session edit would be baked in as the "expected" hash,
defeating the feature.

## Architecture

New module `src/data_analyst_mcp/provenance.py`:

- `compute_source_hash(path: str) -> str` — moved verbatim from
  `tools/models.py::compute_training_dataset_hash`. Semantics unchanged:
  content SHA-256 streamed in 1 MB chunks for files up to the ceiling;
  `fallback:<sha256 of "path|mtime|size">` above it; `sentinel:no-file:<path>`
  / `sentinel:stat-failed:<path>` for anything that is not a stat-able local
  file (in-memory `(dataframe)`, derived `(query)`, `s3://`, …).
- `_HASH_CONTENT_CEILING_BYTES` (100 MB) moves with it.
- `tools/models.py` keeps a thin re-export alias during the migration so the
  existing unit tests pass until the slice that repoints them.

`session.DatasetEntry` gains `source_hash: str`. It is computed inside
`session.register()` from `path` — no call-site changes needed:

| Caller | `path` | Resulting hash |
|---|---|---|
| `load_dataset` (local file) | real path | content or `fallback:` |
| `load_dataset` (s3:// etc.) | URL | `sentinel:` |
| `materialize_query` | `"(query)"` | `sentinel:` |
| direct `register()` with `format="dataframe"` (test fixtures only — no public tool registers dataframes today) | `"(dataframe)"` | `sentinel:` |

`materialize_query`, when overwriting a file-backed entry, copies
`existing.source_hash` into the `base_loader` dict (alongside `path`,
`format`, `read_options`) so the replay guard survives overwrite chains.

## Data flow

1. **Load:** `register()` hashes the file once, right after the successful
   read. Accepted limitation: a race window between read and hash exists but
   is strictly smaller than today's load→fit window.
2. **Fit:** `fit_model` uses `dataset_entry.source_hash`.
   `ModelEntry.training_dataset_hash` and the corresponding
   `register_model(...)` parameter are deleted — nothing reads them after
   unification (verified: no tool response, eval, or SPEC reference).
3. **Emit:** for each file-backed dataset the setup cell writes, *before* its
   `CREATE OR REPLACE TABLE` line, one of:
   - content shape (single-line assert, matching the existing model-block
     emission style):
     ```python
     expected_hash_ds_<var> = '<hex>'
     actual_hash_ds_<var> = hashlib.sha256(open('<path>', 'rb').read()).hexdigest()
     assert actual_hash_ds_<var> == expected_hash_ds_<var>, "Source file for dataset '<name>' changed since the session was recorded."
     ```
   - fallback shape: recompute `'fallback:' + sha256(f"{path}|{mtime}|{size}")`
     via `os.stat`, then the same hard assert (mirrors today's model fallback).
   - sentinel shape: a `# Note: no provenance hash for <name> (...)` comment,
     no assert.
   `base_loader` re-creates (first pass of the derived-emission logic) get the
   same treatment using the carried hash. Derived `CREATE` lines themselves
   carry no assert — their recipe is the SQL plus the upstream guards.
4. **Model block:** keeps `# --- Re-fit model ---`, the `<name>_df`
   materialization, and the `smf.<kind>(...).fit()` lines; loses its assert
   lines (subsumed by the dataset guards upstream).

Assert variable naming: dataset names are **not** validated as Python
identifiers (`load_dataset` defaults the name from the file basename, so
`my-data.csv` → `my-data`). The existing `<name>_df` emission silently assumes
identifier-safety — a pre-existing limitation, out of scope here. The new
assert variables avoid compounding it: `<var>` is the dataset name with
non-identifier characters replaced by `_`, suffixed with the emission index
(`expected_hash_ds_my_data_0`) — always valid Python, collision-free even when
two names sanitize identically.

## Error handling / edge cases

- **Mismatch at replay** → hard `AssertionError`:
  `"Source file for dataset '<name>' changed since the session was recorded."`
  Loud beats silent drift; same philosophy as the model guard it replaces.
- **s3:// / unstattable** → sentinel → comment, no assert (today's model
  behavior, now applied at the dataset level).
- **> ceiling files** → fallback assert; weaker guarantee stays documented in
  ROADMAP/README wording.
- **Same file under two names** → two hashes, two asserts; harmless.
- **Reload same name mid-session** → `register()` replaces the entry and the
  hash refreshes; previously-fit models hold no stale copy because the field
  is gone.
- No new error types. `load_dataset` failure modes unchanged — hashing runs
  only after a successful read, and sentinel paths never raise.

## TDD slices (red → green each, per repo convention)

1. `DatasetEntry.source_hash` equals the file's SHA-256 after `load_dataset`
   (tmp CSV fixture).
2. Derived (`materialize_query`) and dataframe registrations get `sentinel:`
   hashes.
3. `materialize_query` overwrite of a file-backed entry carries
   `source_hash` into `base_loader`.
4. Setup cell emits the content-shape assert before the CREATE line for a
   file-backed dataset.
5. Fallback shape when file size exceeds a monkeypatched ceiling; sentinel →
   comment and no assert.
6. `fit_model` records the load-time hash: edit the file between load and
   fit; the stored hash must equal the load-time value (the behavioral fix).
7. Model-rehydration block emits no assert lines; `ModelEntry` field removed;
   `tests/test_model_registry.py` / `tests/test_recorder.py` updated in the
   same slices.
8. Integration eval: emit → `jupyter nbconvert --execute` passes untouched;
   append one byte to the source CSV → replay raises `AssertionError`.
9. `refactor:` commit for the `provenance.py` extraction (no behavior
   change); final docs slice syncs SPEC §5 (emit_notebook setup-cell
   contract, fit_model hash wording), README (gotchas + worked example if
   touched), ROADMAP (remove the parked bullet; update the Phase 5
   reproducibility caveat to point at dataset-level guards).

## Acceptance criteria

- Every file-backed dataset in an emitted notebook is guarded by a hash
  assert at replay; editing any source file between session and replay fails
  the setup cell loudly.
- `fit_model`'s recorded provenance equals the load-time hash of its
  training dataset.
- No tool-response schema changes; tool count stays 22.
- Full gates pass: pytest (tests + evals), ruff format/check, pyright strict,
  `check_tdd_commits.py`.

## ROADMAP impact

Removes "Provenance hashes" from Reproducibility (shipped). The remaining
bucket entries (`load_session_from_notebook`, notebook diff) become easier:
both can trust `DatasetEntry.source_hash` as the identity anchor.
