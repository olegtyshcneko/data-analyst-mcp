# Ephemeral-fit replay provenance

Design draft for the ROADMAP § Reproducibility item "Ephemeral-fit replay
provenance." Not a new tool — the surface stays at 24 — so nothing folds
into SPEC §5 as a new entry; the accepted design lands as amendments to
§5.11 / §5.11d / §6 and this file is deleted, per the proposals
convention.

## Purpose

Close the last named replay failure class from the 1.4.0 guard work.
`cross_validate` and `fit_model(model_name=None)` re-fit inside their
per-call cells with no fit-time provenance guard: the cells read whatever
table state the setup cell recreates at replay, with no connection to the
table state that existed at call time. Registered models are covered by
the setup-cell dispatch (revision → base-loader → content+loader
equivalence); ephemeral fits have no equivalent, so a training source
mutated *and reloaded* between the call and replay silently reports
different CV/fit numbers.

## Failure class (concrete)

1. `load_dataset("sales.csv", name="sales")` → entry revision 3, hash H1.
2. `cross_validate(name="sales", formula=..., k=5)` → cell begins
   `_cv_df = con.sql('SELECT * FROM "sales"').df()`.
3. `sales.csv` is edited; `load_dataset` runs again → revision 7, hash H2.
4. `emit_notebook` → the setup cell asserts H2 against the edited file
   (passes — the assert anchors to the *latest* load), recreates `sales`
   from the new bytes, and the CV cell recomputes on data the CV never
   saw. Exit code 0, different numbers, no error.

The same holds for `fit_model` without `model_name` (cell begins
`df = con.sql("SELECT * FROM sales").df()`), and for replacement via
`materialize_query` / `split_dataset` overwrites, not just reloads.

## Scope decision

Guard exactly the two ephemeral surfaces: `cross_validate` (always) and
`fit_model` when `model_name is None`. Registered `fit_model` per-call
cells stay unguarded — they are transitively safe: the setup cell's model
block raises before any per-call cell runs whenever the training state is
gone, and when it passes, the recreated table *is* the fit-time state.
Other analytic per-call cells (`correlate`, `compare_groups`, …) share
the replay property in principle but are out of scope (see ROADMAP
impact).

## Design

### 1. Call-time capture

`NotebookRecorder.record()` gains an optional `guard: dict | None = None`
parameter, stored on the **code** cell's metadata as `"replay_guard"`
(cells already carry `metadata: {"tool_name": ...}`; this is additive and
invisible to replay). `_record_cross_validate` always stamps it;
`_record_fit_model` stamps it only when `payload.model_name is None`.
Captured from the session entry for `payload.name` at call time:

```python
{
    "tool": "cross_validate" | "fit_model",
    "dataset": <name>,
    "revision": entry.revision,
    "source_hash": entry.source_hash,
    "loader": {"path": entry.path, "format": entry.format,
               "read_options": entry.read_options},
    "data_var": "_cv_df" | "df",
    "table_read_line": <the exact table-read line the cell emitted>,
}
```

`data_var` / `table_read_line` exist so the emit-time rewrite (verdict
REFIT_FROM_BASE below) can replace the read line without parsing the
cell.

### 2. Shared verdict helper

The registered-model decision tree in `_build_setup_source`
(recorder.py, the `rev != ds_entry.revision` dispatch) is extracted into
a helper — `_resolve_fit_provenance(*, revision, source_hash, loader,
entry) -> verdict` — returning one of three verdicts:

- **UNCHANGED** — `entry.revision == revision`, or the entry is
  file-backed (format is a file format — remote s3/http sources
  included; their path-keyed sentinels compare equal exactly when the
  path matches, as in the model block today) and provably the same
  loading semantics: `source_hash == entry.source_hash` **and** `loader`
  equals the entry's current `{path, format, read_options}` (the
  innocent same-file reload; hash alone cannot see re-parsing under
  changed read options).
- **REFIT_FROM_BASE** — the entry is derived, carries a `base_loader`,
  and `revision == base_loader["revision"]`: the fit-time state was the
  pre-overwrite file-backed state, still reachable on disk (pinned
  original revision survives chained overwrites).
- **RAISE(cause)** — everything else: replaced derived / split /
  dataframe state, file reload with different content or read options,
  or a missing entry (defensive — no tool deregisters).

The model block is refactored to consume the same helper. The helper
returns verdicts only; each consumer renders its own lines, so the
setup-cell output stays byte-identical (existing tests pin it). One
deliberate consumer-side difference: the **per-call resolver** maps a
revision-equal *dataframe-format* entry to RAISE (the table is never
recreated at replay; see Edges), while the model block keeps its
existing note-comment behavior for dataframe-trained models — pushing
that case into the helper would change setup-cell output. Guard
strength is uniform by construction: an ephemeral fit is replayable
exactly when a registered model fit on the same state would be.

### 3. Emit-time resolution

`to_notebook()` resolves each `replay_guard`-carrying cell against the
live registry:

- **UNCHANGED** → cell emitted exactly as today. No in-cell hash assert
  is added: the setup cell's dataset-level `_hash_guard_lines` already
  guard the source file before the table is recreated, and revision
  equality (checked here, at emit time) guards call-to-emit replacement.
  Happy-path notebooks are **byte-identical to 1.4.0**.
- **RAISE** → the cell is prefixed with a single
  `raise AssertionError(<message>)` line; the original computation is
  retained below (unreachable — the markdown cell plus the dead code
  keep the audit trail of what was computed live). Messages are
  cause-specific and name the tool, mirroring the model-block wording,
  e.g. "The cross-validation in this cell ran against a state of dataset
  'sales' that was later replaced; that table state no longer exists at
  replay, so the cell cannot be replayed faithfully."
- **REFIT_FROM_BASE** → `table_read_line` is replaced by: an explanatory
  comment (mirroring the model block's "was overwritten by
  materialize_query" comment), fit-time hash-guard lines against
  `guard["source_hash"]` at the base path (content / fallback /
  sentinel-comment shapes, reusing the `_hash_guard_lines` dispatch with
  a per-cell collision-free variable stem), and
  `<data_var> = con.sql(<_file_select_expr(base)>).df()` re-reading the
  original file with the fit-time read options. CV/fit numbers reproduce
  faithfully from the original bytes.

### 4. Edges and non-changes

- **Dataframe-format sources.** Today a guarded-tool cell on an
  in-memory dataset fails replay with a bare `CatalogException` (the
  setup cell only emits a comment for dataframe datasets; the table
  never exists). These now get the RAISE verdict with an explanatory
  message — strictly better than today, still loud.
- **Split-backed sources.** Unreplaced split sides are UNCHANGED — the
  setup cell recreates them behind per-side membership checksums, so the
  guarded cell reads a provably-faithful table. A re-split gets a fresh
  revision → RAISE.
- **Registered `fit_model` per-call cells**: untouched.
- **Setup cell**: byte-identical (helper extraction is behavior-neutral).
- **Known hole, explicitly not closed.** A derived dataset whose
  upstream base is reloaded *after* materialization keeps its own
  revision; replay recreates it from the recipe over the new base, and
  neither this guard nor the registered-model guard can see it. Shared,
  pre-existing, filed as its own ROADMAP item (see below).
- **Version**: 1.5.0. Emitted notebooks change shape only for sessions
  that were already silently unreplayable (drifted) or loudly broken
  (dataframe sources).

## TDD slices

Each slice red → green per the repo convention; the shared-helper
extraction rides inside slice 4 (the first consumer), pinned by the
existing setup-cell tests.

1. `NotebookRecorder.record` accepts `guard` and stores it as
   `replay_guard` on the code cell; absent by default.
2. `cross_validate` stamps `replay_guard` with the call-time entry state.
3. Ephemeral `fit_model` stamps it; registered `fit_model` does not.
4. Emit with revision-equal guard → cell byte-identical to an unguarded
   emit (covers UNCHANGED; helper extraction + model-block refactor land
   here behavior-neutrally).
5. Replaced derived / split state at emit → RAISE prefix with
   cause-specific message; original code retained below.
6. Provably-identical file reload (equal hash + loader) → UNCHANGED.
7. Same-path reload with different content, and same-content reload with
   different read options → RAISE (two red cases, one slice).
8. Dataframe-backed guarded cell → RAISE with the in-memory explanation.
9. Derived overwrite of the file-backed training state →
   REFIT_FROM_BASE: read line replaced, hash assert present, replayed
   numbers equal the live run.
10. Eval (`evals/`): end-to-end session — load, `cross_validate`, edit
    file, reload, `emit_notebook`, execute → replay raises
    `AssertionError` naming cross_validate. A sibling case covers
    ephemeral `fit_model`.

## Acceptance criteria

- Sessions whose CV / ephemeral-fit datasets survive to emit unreplaced
  produce **byte-identical notebooks to 1.4.0**.
- Every replacement of a guarded cell's dataset after the call —
  re-load with different content or options, re-materialize, re-split —
  yields a replay that raises `AssertionError` with a cause-specific,
  tool-naming message before recomputing.
- A `materialize_query` overwrite of a file-backed training state
  replays the guarded cell faithfully from the original file behind a
  fit-time hash assert.
- Setup-cell output is byte-identical before/after the refactor.
- All gates pass: `pytest tests/`, `pytest evals/`, `ruff format
  --check`, `ruff check`, `pyright src/`, `check_tdd_commits.py`.

## ROADMAP impact

- Remove "Ephemeral-fit replay provenance" from § Reproducibility.
- Add a new parked item: **Derived-upstream drift.** A derived dataset's
  replay recipe runs against whatever its upstream tables contain at
  replay; an upstream reload after materialization changes the derived
  content without touching the derived revision, and both the
  registered-model and ephemeral-fit guards are blind to it. Closing it
  likely means content checksums captured at materialize time.

## Docs impact

- SPEC §5.11 (`fit_model`) and §5.11d (`cross_validate`): recorder-cell
  paragraphs note the fit-time `replay_guard` and the three verdicts.
- SPEC §6 (recorder): `replay_guard` metadata contract and emit-time
  resolution.
- README: reproducibility caveat paragraph and "Known gotchas" bullet
  extend to ephemeral fits.
- CHANGELOG: 1.5.0 entry.
