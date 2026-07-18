# Ephemeral-fit replay provenance — prefix-guard realization

Design draft for the ROADMAP § Reproducibility item "Ephemeral-fit replay
provenance." Second revision: the first draft's emit-time revision
dispatch was invalidated in design review (it reasoned against the wrong
replay model) and this rework replaces it. Not a new tool — the surface
stays at 24; the accepted design lands as amendments to SPEC §5.1 /
§5.11 / §5.11d / §6 and this file is deleted, per the proposals
convention.

## The replay model (corrected)

`to_notebook` emits the setup cell first, then **every recorded cell in
call order** (`recorder.py`, `to_notebook`). The state-mutating tools
all record cells that re-execute their state change at replay:
`load_dataset` records `CREATE OR REPLACE TABLE <name> AS SELECT * FROM
<read_call>` (`tools/datasets.py`), `materialize_query` records its
`CREATE` (`tools/materialize.py`), `split_dataset` records the full
membership-recipe snippet (`tools/split.py`). A per-call cell therefore
reads the table state recreated by its **historical prefix** — the last
state-recreating cell for that name before it — not the final session
state. Registered models are the exception: their re-fits run inside the
setup cell, against the final state, behind the revision/hash/loader
dispatch.

Consequences that shape this design:

- The recorded prefix always *structurally* reproduces the fit-time
  state — it is, by construction, the history that produced it. What
  can drift is the **content** flowing through re-executed loads: a
  historical `load_dataset` cell re-reads its file with no guard, and
  the setup cell's hash asserts anchor only to each dataset's **latest**
  registration.
- Hence the ROADMAP failure class: load `sales.csv` → `cross_validate`
  → edit the file → reload → emit. Setup asserts the *new* hash
  (passes); the first load cell re-reads the edited bytes; the CV cell
  recomputes on data it never saw. Exit 0, different numbers.
- Conversely, histories that merely *replace* states are faithful at
  replay: `materialize d=S1 → CV(d) → materialize d=S2` recreates S1
  before the CV cell. A guard must not raise on these.

## Purpose

Close the failure class above: make every file-content drift between a
session's loads and replay fail loudly at the cell that re-reads the
file, and give `cross_validate` / ephemeral `fit_model` cells on
unreplayable (in-memory) sources an explained failure instead of a bare
`duckdb.CatalogException`.

## Mechanism

### 1. Hash-guarded load cells (the substantive fix)

Every `load_dataset` recorded cell gains drift-guard lines **before**
its `CREATE`, anchored to the load-time `source_hash` of the entry just
registered (already computed by `session.register`). The three shapes
mirror the setup cell's `_hash_guard_lines` exactly, and the helper is
reused, not duplicated:

- content hash (file ≤ 100 MB): `expected/actual` SHA-256 assert;
- fallback (file > 100 MB): recompute the `(path, mtime, size)` digest
  and assert — the documented weaker guarantee, still a hard assert;
- sentinel (remote s3/http, unstattable): an explanatory comment; the
  reload stays unguarded, as in the setup cell today.

Each load cell asserts **its own** load-time hash. A reloaded name emits
one guard per load cell, so the ROADMAP scenario dies at the *first*
load cell (fit-time bytes gone) even though the setup cell's
latest-registration assert passes. Guard-variable stems come from the
existing sanitized-name convention plus a per-record ordinal, so
repeated loads of one name cannot collide (assignment-then-assert within
a single cell makes collisions harmless, but the stems keep cells
self-describing).

Because every file enters the session through `load_dataset`, the
recorded prefix of **any** per-call cell — not just fits — now reads
provably load-time-faithful content for every file-backed root,
covering the whole session→replay window. Derived recipes re-execute
over those guarded roots; split recreation already asserts per-side
membership checksums.

### 2. Explained failure for in-memory-backed guarded tools

`_record_cross_validate`, and `_record_fit_model` when
`payload.model_name is None`, check the source entry's format **at
record time**. If it is `"dataframe"` (in-memory registration — the
setup cell emits only a comment and the table never exists at replay),
the recorded code cell is prefixed with a single
`raise AssertionError(...)` line naming the tool and dataset,
explaining that the in-memory source is not recreated at replay. The
original computation stays below the raise (unreachable — the markdown
cell plus the dead code preserve the audit trail). Today these cells
fail replay with a bare `CatalogException`; loud already, now explained.

This is stamped at record time from state that cannot change
retroactively; there is **no emit-time resolution step, no cell
metadata contract, and no recorder API change** anywhere in this
design.

### 3. Considered and dropped: emit-time re-hash of fit-time lineage

The reworked direction as originally chosen included re-hashing the
guarded cell's fit-time source file at `emit_notebook` and prefixing a
raise on drift. Drafting showed it is strictly dominated by mechanism 1:
every real firing (file drifted before emit and still drifted at
replay) is also caught by the load-cell assert, while a file edited
*and reverted* between emit and replay would bake a false-positive raise
into a notebook whose replay is faithful. It also needs lineage tracing
for derived sources that mechanism 1 gets for free at the roots.
Dropped; flagged here for reviewer veto since it changes the approved
shape.

## Explicitly out of scope (parked, not silently ignored)

- **Row-order drift through order-preserving multisets.** Split
  membership checksums are deliberately order-independent (SPEC §5.6b
  tiers) while CV fold assignment is positional: a source edit that
  permutes rows within a split side passes every checksum yet changes
  folds. Mechanism 1 closes the file-backed root cause (the permuted
  file fails its load-cell assert); the residual exposure is derived
  recipes whose SQL is not order-preserving over unchanged inputs.
  Parked as its own ROADMAP item.
- **Remote (s3/http) and above-ceiling sources.** Path-keyed sentinels
  and `(path, mtime, size)` fallbacks cannot prove content; guard
  strength for these matches the setup cell's documented weakness.
- **Registered models fit on in-memory datasets** replay as a
  `NameError` (the `<name>_df` frame is never materialized) — a
  pre-existing rough edge of the setup cell, out of this proposal's
  scope.
- **Recorder cells surviving `session.reset()`.** Revisions restart at
  zero while the recorder is cleared separately; mixed-epoch notebooks
  are a pre-existing oddity, unaffected by this design.

## TDD slices

Each slice red → green per the repo convention.

1. `load_dataset` cell for a small file carries the content-hash assert
   before its `CREATE` (red: current cell has none).
2. Above-ceiling load emits the fallback-recompute assert (threshold
   injected via the existing `HASH_CONTENT_CEILING_BYTES` seam).
3. Remote/sentinel load emits the explanatory comment, no assert.
4. Reloading a name records a second load cell asserting the *new*
   hash; the first cell's assert is unchanged.
5. Guard-variable stems are unique across repeated loads of one name.
6. `cross_validate` on a dataframe-backed dataset records a raise-prefix
   cell with the original computation retained below.
7. Ephemeral `fit_model` on a dataframe-backed dataset: same;
   registered `fit_model` on the same dataset: cell unchanged.
8. Eval (`evals/`): the ROADMAP scenario — load, `cross_validate`, edit
   file, reload, emit, execute — fails at the first load cell with the
   drift message. Sibling case: ephemeral `fit_model`.
9. Eval: faithful replacement histories replay cleanly end-to-end with
   matching numbers — `materialize S1 → CV → materialize S2
   (overwrite)`, and re-split-under-same-names after CV (the design
   review's false-positive counterexamples, pinned as regressions).

## Acceptance criteria

"Unchanged"/"identical" below means cell-source string equality
(nbformat-generated cell ids excluded); no cell metadata is added or
relied on.

- Setup-cell source is identical before/after this change.
- The only per-call cells whose source changes are `load_dataset` cells
  (guard lines added) and `cross_validate` / ephemeral `fit_model`
  cells recorded against in-memory datasets (raise prefix). All other
  recorded cells are unchanged.
- A file whose content at replay differs from what *any* session load
  of it saw fails replay at that load cell with a message naming the
  dataset — including the mutate-and-reload scenario, where the final
  registration's setup assert passes.
- Faithful replacement histories (slice 9) replay cleanly and reproduce
  the live numbers.
- Above-ceiling loads assert the fallback digest; remote loads emit the
  unguarded-comment shape. Both match the setup cell's behavior for the
  same sources.
- All gates pass: `pytest tests/`, `pytest evals/`, `ruff format
  --check`, `ruff check`, `pyright src/`, `check_tdd_commits.py`.

## ROADMAP impact

- Remove "Ephemeral-fit replay provenance" from § Reproducibility on
  ship.
- Add parked item: **Row-order drift under order-independent
  checksums** (split sides + non-order-preserving derived recipes vs
  positional CV folds; closing it needs an order-sensitive digest).
- Fold the "emit-time re-hash" rationale (§ Mechanism 3) into the
  ROADMAP note so the idea is not re-proposed without the
  false-positive analysis.

## Docs impact

- SPEC §5.1 (`load_dataset`): recorder-cell paragraph gains the guard
  shapes.
- SPEC §5.11 / §5.11d: note the in-memory raise-prefix for ephemeral
  fits / CV.
- SPEC §6 (recorder): corrected replay-model paragraph (setup + ordered
  prefix) and the load-cell guard contract.
- README: reproducibility caveat + "Known gotchas" — emitted notebooks
  now guard every load cell, not only the setup cell.
- CHANGELOG: 1.5.0 entry (emitted-notebook shape change for every
  session containing `load_dataset` calls).
