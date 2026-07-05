# Dataset provenance hashes — design

**Date:** 2026-07-05
**Status:** approved (brainstorm session); revised after one headless
codex-review round (kept the per-model hash capture — see Decision summary)
**ROADMAP origin:** Reproducibility → "Provenance hashes"

## Problem

The emitted notebook's setup cell reloads every file-backed dataset with a bare
`CREATE OR REPLACE TABLE <name> AS SELECT * FROM <format-reader>(path)` (reader
per `_FORMAT_TO_READER`: `read_csv_auto` / `read_parquet` / `read_json` /
`read_xlsx`) — no guard. If the source
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
- Unify the hash *source*, keep the per-model *capture*: `fit_model` stops
  re-hashing the file and instead copies the dataset entry's load-time hash
  into `ModelEntry.training_dataset_hash`. The model block's assert lines
  stay in the setup cell — they are not redundant: they diverge from the
  dataset assert exactly when the same dataset name is reloaded/overwritten
  after a fit, which is the drift case they guard.
- Setup-cell file loads (first-pass and `base_loader` re-creates) render
  `read_options` via the same trusted call renderer `load_dataset` uses, so
  a passing hash also implies the same parse at replay.
- Internal only: no tool-response schema changes, no new tool. The 22-tool
  boundary is untouched.

Alternatives considered: (A) purely additive parallel guard leaving
`fit_model`'s recompute in place — rejected as it preserves the mid-session
quirk and double-asserts the same file; (B-full) additionally deleting
`ModelEntry.training_dataset_hash` and the model assert lines as redundant —
rejected in review: a same-name dataset reload after a fit would then replay
silently against data the model was never trained on, a guard today's
fit-time hash does provide; (C) hashing at `emit_notebook` time — rejected
because a mid-session edit would be baked in as the "expected" hash,
defeating the feature.

## Architecture

New module `src/data_analyst_mcp/provenance.py`:

- `compute_source_hash(path: str) -> str` — moved from
  `tools/models.py::compute_training_dataset_hash` with one addition.
  Semantics: content SHA-256 streamed in 1 MB chunks for files up to the
  ceiling; `fallback:<sha256 of "path|mtime|size">` above it;
  `sentinel:no-file:<path>` / `sentinel:stat-failed:<path>` for anything that
  is not a stat-able local file (in-memory `(dataframe)`, derived `(query)`,
  `s3://`, …). New: `OSError` during the content read (file vanished or
  became unreadable between load and hash) yields
  `sentinel:read-failed:<path>` instead of propagating — hashing must never
  add a failure mode to `load_dataset`.
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

Setup-cell loader fidelity: the recorder's `_file_load_stmt` today renders
only format + path, silently dropping `read_options` that the live load used
(`load_dataset` renders them via `_build_read_call`). This design extends the
setup-cell emission to render `read_options` through that same trusted
builder (shared, not duplicated), for both first-pass file-backed CREATEs and
`base_loader` re-creates. Without this, a passing hash would prove same bytes
while replay could still parse them differently (e.g. `header=false`,
`delim=";"`) — undermining the guarantee this feature exists to give.

## Data flow

1. **Load:** `register()` hashes the file once, right after the successful
   read. Accepted limitation: a race window between read and hash exists but
   is strictly smaller than today's load→fit window.
2. **Fit:** `fit_model` copies `dataset_entry.source_hash` into
   `ModelEntry.training_dataset_hash` — no re-hash of the file. The field and
   the `register_model(...)` parameter stay: the captured value is the
   model's own provenance anchor, and it intentionally goes stale if the
   dataset name is later reloaded/overwritten (that staleness is the guard).
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
   same treatment using the carried hash. All file-backed `CREATE` lines
   render `read_options` via the shared builder (see Architecture). Derived
   `CREATE` lines themselves carry no assert — their recipe is the SQL plus
   the upstream guards.
4. **Model block:** unchanged in structure — `# --- Re-fit model ---`, the
   `<name>_df` materialization, the assert lines, and the
   `smf.<kind>(...).fit()` lines all stay. The asserts now check the
   load-time capture rather than a fit-time re-hash. In the common no-drift
   case they duplicate the dataset assert (accepted; the notebook is an
   audit trail and explicit is fine). In the drift case — dataset reloaded
   from an edited file after the fit — the dataset assert passes (current
   file matches current entry) while the model assert fails loudly, which is
   exactly the distinction that makes both guards necessary.

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
  Loud beats silent drift; same philosophy as the existing model guard it
  now sits alongside.
- **s3:// / unstattable** → sentinel → comment, no assert (today's model
  behavior, now applied at the dataset level).
- **> ceiling files** → fallback assert; weaker guarantee stays documented in
  ROADMAP/README wording.
- **Same file under two names** → two hashes, two asserts; harmless.
- **Reload same name mid-session** → `register()` replaces the entry and the
  hash refreshes. Models fit *before* the reload keep their captured
  load-time hash; at replay their assert fails loudly because the current
  file no longer matches what they trained on — the guard full-unification
  would have lost.
- No new error types. `load_dataset` failure modes unchanged — hashing runs
  only after a successful read and never raises: stat failures and
  mid-hash `OSError`s both collapse to `sentinel:` values (comment, no
  assert, at emit time).

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
6. `compute_source_hash` returns `sentinel:read-failed:<path>` when the
   content read raises `OSError` (monkeypatched `open`); `load_dataset`
   still succeeds.
7. `fit_model` stores the load-time hash without re-reading the file: edit
   the file between load and fit; `ModelEntry.training_dataset_hash` must
   equal the load-time value (the behavioral fix).
8. Setup cell renders `read_options` in file-backed CREATE lines (e.g.
   `header=false`, `delim=";"`) — first-pass entries and `base_loader`
   re-creates both.
9. `base_loader` guard emission: load file-backed → `materialize_query`
   overwrite → emit; the carried-hash assert precedes the base CREATE;
   a second overwrite (derived over derived) still carries the guard.
10. Integration eval: emit → `jupyter nbconvert --execute` passes untouched;
    append one byte to the source CSV → replay raises `AssertionError` —
    once for the plain file-backed case, once for the overwrite-chain case
    (drift caught by the `base_loader` guard).
11. `refactor:` commit for the `provenance.py` extraction (no behavior
    change; `tests/test_model_registry.py` hash tests repointed here);
    final docs slice syncs SPEC §5 (emit_notebook setup-cell contract,
    fit_model hash wording), README (gotchas + worked example if touched),
    ROADMAP (remove the parked bullet; update the Phase 5 reproducibility
    caveat to point at dataset-level guards).

## Acceptance criteria

- Every file-backed dataset in an emitted notebook is guarded by a hash
  assert at replay; editing any source file between session and replay fails
  the setup cell loudly — including files reachable only via `base_loader`
  after overwrite chains.
- `fit_model`'s recorded provenance equals the load-time hash of its
  training dataset, and a same-name reload after a fit still fails loudly at
  replay via the model assert.
- Setup-cell file loads reproduce the live session's `read_options` — a
  passing hash implies the same parse.
- No tool-response schema changes; tool count stays 22.
- Full gates pass: pytest (tests + evals), ruff format/check, pyright strict,
  `check_tdd_commits.py`.

## ROADMAP impact

Removes "Provenance hashes" from Reproducibility (shipped). The remaining
bucket entries (`load_session_from_notebook`, notebook diff) become easier:
both can trust `DatasetEntry.source_hash` as the identity anchor.
