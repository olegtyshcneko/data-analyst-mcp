# Fit-then-overwrite replay fix — design

**Status:** accepted (design approved 2026-07-05)
**Scope:** `src/data_analyst_mcp/recorder.py` model-rehydration block only. No tool-surface change; no schema change.

## Problem

A model fitted on a file-backed dataset that is *later* overwritten by
`materialize_query` (transform-in-place) emits a broken model block in the
notebook's setup cell:

1. **Crash at replay, even with zero drift.** The block reads the *current*
   registry entry's `path` — the literal `"(query)"` placeholder — while
   `ModelEntry.training_dataset_hash` still holds the real fit-time hash.
   The content shape emits `open('(query)', 'rb')` (FileNotFoundError); the
   fallback shape emits `_os.stat('(query)')` (same failure).
2. **Wrong re-fit target.** The rehydration line fits against
   `<dataset>_df`, which the setup cell materializes *after* the second-pass
   `CREATE OR REPLACE` applies the derived SQL — i.e. the post-transform
   table, not the table the model was trained on.

Everything needed for a correct fix is already in the registry:
`materialize_query` carries
`base_loader = {"path", "format", "read_options", "source_hash"}` through
overwrites, including chains of overwrites (`tools/materialize.py`).

## Decision (approved)

**Re-fit from the original source file** — not skip-with-comment. The
notebook must keep replaying end-to-end; a skipped re-fit would leave the
model variable undefined and NameError every downstream predict/evaluate
cell.

## Design

### Detection

In `_build_setup_source`'s model loop, a model predates an overwrite of its
training dataset iff:

```python
ds_entry is not None
and ds_entry.format == "derived"
and ds_entry.base_loader is not None
and model_entry.training_dataset_hash != ds_entry.source_hash
```

Rationale: a derived entry's own `source_hash` is always
`"sentinel:no-file:(query)"` (computed from the placeholder path at
registration). A model fit *after* the overwrite copied exactly that value
at fit time — equal, so it keeps the current behavior (comment + re-fit on
`<dataset>_df`, which is correct: the model trained on the derived table).
A model fit *before* the overwrite carries the original file's hash (or
`fallback:` / a path-bearing sentinel for s3/http) — different, so it takes
the new path.

### Guard fix

In the detected case the guard's recompute path becomes
`base_loader["path"]`; the expected value stays
`model_entry.training_dataset_hash`. The three existing guard shapes are
reused unchanged (content assert / `fallback:` recompute assert for
above-ceiling files / comment when the hash is a sentinel, e.g. s3 bases).

Property preserved: if the user re-loaded the dataset from a *different*
file before overwriting, expected ≠ actual and the assert fires loudly —
consistent with the shipped same-name-reload semantics.

### Re-fit fix

The detected case emits a dedicated training frame from the carried base
loader, then fits against it:

```python
# --- Re-fit model 'm' (kind=ols) ---
# Dataset 'X' was overwritten by materialize_query after this model was fit;
# re-fitting from the original source file, not the current derived table.
expected_hash_m = 'a3f0…'
actual_hash_m = hashlib.sha256(open('/data/x.csv', 'rb').read()).hexdigest()
assert actual_hash_m == expected_hash_m, "Training data for 'm' changed since the session was recorded."
m_train_df = con.sql("SELECT * FROM read_csv_auto('/data/x.csv', SAMPLE_SIZE=-1)").df()
m = smf.ols("y ~ x", data=m_train_df).fit()
```

- The frame is named `<model_name>_train_df`. Model names are already
  identifier-validated (the block emits `expected_hash_<model_name>`
  variables today), so no sanitization is needed.
- `<dataset>_df` is left untouched — post-transform — because the
  predict/evaluate scoring cells reference it and that is the table they saw
  live.
- The reader expression is shared with `_file_load_stmt` via a new helper so
  `read_options` render identically at reload and at re-fit:

```python
def _file_select_expr(fmt: str, path: str, read_options: dict[str, Any] | None = None) -> str:
    reader = _FORMAT_TO_READER.get(fmt, "read_csv_auto")
    path_lit = repr(path)
    extra = render_read_options_fragment(read_options or {})
    if reader == "read_csv_auto":
        return f"SELECT * FROM {reader}({path_lit}, SAMPLE_SIZE=-1{extra})"
    return f"SELECT * FROM {reader}({path_lit}{extra})"
```

`_file_load_stmt` becomes
`f"CREATE OR REPLACE TABLE {name} AS {_file_select_expr(fmt, path, read_options)}"` —
a pure refactor, byte-identical output.

### Edge cases

| Case | Behavior |
|---|---|
| Model fit on derived table (post-overwrite) | Not detected (hashes equal) — unchanged: sentinel comment + re-fit on `<dataset>_df`. Correct. |
| Chain of overwrites | `base_loader` is carried forward by `materialize_query`; same handling. |
| Reload from different file, then overwrite | Guard compares fit-time hash to base file → assert fires loudly. Intended. |
| s3/http base | Training hash is a path-bearing sentinel ≠ `"sentinel:no-file:(query)"` → detected; guard emits the sentinel comment shape; re-fit loads from the s3/http loader — mirrors the unguarded s3 reload semantics. |
| >100 MB base | `fallback:` shape recomputed against the base path. |
| In-memory base (`base_loader is None`) | Not detected — unchanged (pre-existing gap: no reproducible source exists). |
| Dataset deleted from registry | `ds_entry is None` → unchanged comment branch. |

### Out of scope (documented limitation)

Body cells recorded *before* an overwrite replay against final-state tables
(the setup cell pre-applies derived SQL so `<name>_df` frames exist for the
model block). This is a broader pre-existing ordering semantic affecting
`query`/`plot`/`predict` cells alike, independent of the model-block crash
fixed here. It stays on the ROADMAP if it ever earns a fix.

## Testing

- **Unit (recorder):** build session state directly and assert on
  `_build_setup_source` output — (a) content-hash case: guard targets the
  base path, `<model>_train_df` line present, re-fit references it, and
  `<dataset>_df` still post-transform; (b) fallback-hash case emits
  `_os.stat(<base path>)`; (c) sentinel case emits the comment shape and
  still re-fits from the base loader; (d) post-overwrite-fit model keeps the
  current emission byte-for-byte; (e) `_file_load_stmt` refactor is
  output-identical (existing tests already pin this).
- **Integration (nbconvert):** fit → overwrite (self-referencing SQL) →
  `emit_notebook` → `jupyter nbconvert --execute` exits 0; mutate the source
  file → replay fails with the AssertionError. Follow the existing drift
  integration-test pattern in `tests/test_recorder.py`.

## Docs

- ROADMAP: remove the "Fit-then-overwrite replay" bullet.
- README gotchas / SPEC drift-guard wording: "reloaded" may again say
  models re-fit correctly across `materialize_query` overwrites.
- CHANGELOG: `### Fixed` entry under a new `[Unreleased]` section.
