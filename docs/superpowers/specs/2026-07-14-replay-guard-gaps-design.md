# Replay-guard gaps: registration revisions + split-overwrite provenance + symmetric split replay

**Date:** 2026-07-14
**Status:** draft v3 — two adversarial review rounds (headless Codex,
gpt-5.6-sol @ xhigh); each round's confirmed findings are folded in. See
"Review deltas" at the bottom.
**Target release:** 1.4.0 (see "Docs & release impact" for the 1.3.2
alternative and the rationale)

## Problem

Two documented ROADMAP gaps — plus adjacent defects confirmed during review —
break the project's core replay promise ("silent drift is impossible;
failures are loud *and explained*"):

1. **Pure-query fit-then-overwrite (silent wrong numbers; ROADMAP).** Every
   derived dataset registers with placeholder path `"(query)"`, so
   `session.register` (`session.py:137`) computes the *same* constant
   `source_hash = "sentinel:no-file:(query)"` for every materialization.
   `fit_model` copies that hash (`tools/models.py:158`); the recorder's
   overwrite detection (`recorder.py:467,474`) compares hashes and cannot
   tell materializations apart, so a model fit on materialization #1 silently
   re-fits on materialization #2's table at replay.
2. **Double split-side overwrite (loud but unexplained; ROADMAP).**
   `_split_side_overwritten` (`recorder.py:255`) detects a split-side
   overwrite from the *surviving sibling* split entry. When both sides are
   overwritten with recipes that depend on missing pre-overwrite split
   tables, no split entry survives, detection returns `None`, and replay
   fails with a raw `duckdb.CatalogException` instead of 1.3.1's friendly
   chained `AssertionError`.
3. **Split-over-split (silent; review r1, reproduced).** `split_dataset`
   with `overwrite=True` (`split.py:181`) replaces both sides under the same
   names. Old and new entries share the constant `sentinel:no-file:(split)`
   hash, so a model fit on the seed-1 train side silently re-fits on the
   seed-2 membership at replay.
4. **`load_dataset` over a non-file state (silent; review r1, reproduced).**
   `load_dataset` unconditionally replaces the table and entry
   (`datasets.py:600,610`). A model fit on the pre-load derived / split /
   dataframe state keeps a sentinel hash; the new entry is file-backed, no
   overwrite branch applies, and the recorder re-fits on the new file with
   only a "no hash assert possible" comment.
5. **Same-name reload with different loading semantics (silent; review r2,
   reproduced).** Reloading the *same bytes* under the same name with
   different `read_options` (e.g. `nullstr`) keeps the SHA-256 identical but
   parses a different table (Titanic fixture: `n_obs` 887 → 848). A content
   hash proves identical bytes, not identical loading semantics; the same
   applies to identical bytes at a different path/format.
6. **Test-side overwrite asymmetry (loud/broken; review r1, reproduced).**
   The split replay block is emitted only from a surviving `role == "test"`
   entry (`recorder.py:407`). If the *test* side is overwritten — even with
   independently replayable SQL — no split block is emitted and the
   surviving **train** table is never recreated: replay hits a raw
   `CatalogException` in the model frame loop (`recorder.py:443`) or
   wherever the train table is first referenced.
7. **Split-source drift invisible to the test checksum (silent; review r2,
   reproduced).** Overwriting the split *source* so that only train-side
   rows change passes the test-side membership checksum; replay completed
   with drifted OLS coefficients. Only a train-side checksum catches this.
8. **Second-pass emission order is wrong for overwrites (review r2,
   reproduced).** The recorder assumes dict insertion order is registration
   order (`recorder.py:354-360`), but re-assigning an existing key does not
   move it: with pre-registered output names and `split_dataset(...,
   overwrite=True)` the derived test-side CREATE can be emitted *before* the
   train-only split block it depends on.

### Why content hashes cannot fix 1/3/4/5

Derived, split, and dataframe states have no file to hash — their sentinels
are constants per format — and even for file-backed states a content hash
cannot see loading semantics (5). Identity of the *registration*, plus
loader identity for file-backed states, is what the model guard needs.
`source_hash` keeps exactly one job: content provenance for the emitted file
guards.

## Design

### Part 1 — registration revisions and fit-time loader identity
(`session.py`, `tools/models.py`, `tools/materialize.py`)

- `session` gains a module-level monotonic counter; **every** `register()`
  call stamps the new entry with a unique `DatasetEntry.revision: int`
  (default `-1` for direct constructions in tests). `reset()` zeroes the
  counter.
- `ModelEntry` gains:
  - `training_dataset_revision: int` — copied from `entry.revision` at fit
    time (next to the existing hash copy, `models.py:158`);
  - `training_loader: dict[str, Any]` — fit-time loader identity
    `{"path", "format", "read_options"}` copied from the entry, used only by
    the file-backed mismatch rule in Part 2.
- When `materialize_query` captures a `base_loader` (`materialize.py:114-125`)
  from a *file-backed* entry, the dict additionally records that entry's
  `revision`. On chained derived overwrites the dict is carried forward
  **unchanged** — `base_loader["revision"]` is always the original file
  entry's revision (R0 in `file@R0 → derived@R1 → derived@R2`), never an
  intermediate derived revision.
- `source_hash` semantics are unchanged.

### Part 2 — model re-fit block: exhaustive cases (`recorder.py:435-558`)

For each registered model, with `rev` = `training_dataset_revision`,
`hash_val` = `training_dataset_hash`, `entry` = current dataset entry.
Stated invariants (unreachable via tools, asserted nowhere): a current
derived/split/dataframe entry always has a `sentinel:` hash; real/fallback
hashes occur only on file-backed entries.

- **`entry is None`** (reachable only by direct registry mutation): emit an
  unconditional `raise AssertionError(...)` — "training dataset is no longer
  registered".
- **`entry.format == "derived"`:**
  1. `rev == entry.revision` → fit on the current materialization: re-fit on
     the recreated table (unchanged normal path).
  2. `entry.base_loader is not None and rev == base_loader["revision"]` →
     fit on the pre-overwrite *file-backed* state: re-fit from the original
     file, hash-guarded exactly as today per the base loader's hash kind
     (content assert / fallback assert / sentinel comment for remote bases —
     the unguarded-remote contract is pinned at `test_recorder.py:884`).
  3. otherwise → the fit-time table state no longer exists anywhere
     reachable: emit the loud `raise AssertionError(...)` naming the model,
     the dataset, and that the dataset was replaced after the fit.
- **`entry.format == "split"`:** `rev == entry.revision` → normal path (the
  split block recreates the table behind its checksum); otherwise → the loud
  raise (covers split-over-split).
- **`entry.format == "dataframe"`:** `rev == entry.revision` → unchanged
  today-path; otherwise → the loud raise.
- **file-backed formats:**
  - `rev == entry.revision` → unchanged path (content guard / fallback
    guard / unguarded remote reload with comment).
  - `rev != entry.revision` → allow the unchanged path **only** when the
    reload is provably the same loading semantics:
    `hash_val == entry.source_hash` **and** `training_loader` equals the
    entry's current `(path, format, read_options)`. This keeps the innocent
    same-file re-load working (including remote URLs, whose path-keyed
    sentinels compare equal) while closing problem 5. Anything else — hash
    mismatch, changed read options, changed path/format, or a
    `sentinel:` fit-time hash from a replaced non-file state (problem 4) —
    emits the loud raise.

The emitted `raise` halts the setup cell at the first unreplayable model —
same convention as the existing split-source raise (`recorder.py:486`).

### Part 3 — recorded split-overwrite provenance (`session.py`,
`tools/materialize.py`, `recorder.py`)

- `DatasetEntry` gains `split_overwrite: dict[str, Any] | None = None` with
  keys `{"side": "train" | "test", "source": <split source name>}`;
  `register()` gains the matching optional parameter (copied defensively).
- `materialize_query` overwrite path: `existing.format == "split"` → record
  side + source from `existing.read_options`; `existing.format == "derived"`
  → carry `existing.split_overwrite` forward (chained overwrites); otherwise
  `None`.
- Recorder: the derived-CREATE wrap keys off `entry.split_overwrite`; the
  sibling-scanning `_split_side_overwritten()` is deleted.
- **Message contract:** the wrap message MUST begin with the existing pinned
  phrasing `Dataset {name!r} was created by overwriting the {side} side of
  the split of {source!r}.` (tests pin the substrings
  `overwriting the train side` / `overwriting the test side`,
  `test_split.py:561,591`). Only the trailing explanation generalizes: the
  split's pre-overwrite tables are not recreated at replay, so the recipe
  must be rematerialized from a table that exists at replay. The original
  `CatalogException` stays chained.
- The wrap only converts `CatalogException` → explained `AssertionError`;
  independently replayable overwrite recipes execute transparently,
  including the both-sides-replayable case, which is a *success* scenario
  (S14). In a double failure only the first wrapper's error is observable
  (the cell halts there).

### Part 4 — revision-ordered emission + symmetric split replay with
per-side checksums (`tools/split.py`, `recorder.py`)

- **Second-pass ordering:** iterate
  `sorted(datasets.items(), key=lambda kv: kv[1].revision)` instead of raw
  dict order. Revisions are true temporal order, which dict insertion order
  is not for overwrites (problem 8); this also makes the existing
  "registration order is topological order" comment actually true for
  overwrite chains. (First-pass file loads have no inter-dependencies; their
  order is unchanged.)
- `split_dataset` computes a membership checksum for **both** sides and
  stores each side's value under the *same key*,
  `read_options["membership_checksum"]`, in its own entry (keeping the
  existing test-side key untouched — renaming would break
  `test_split.py:75,375`). One extra hashing pass over the train frame at
  split time.
- `split_replay_source()` becomes side-symmetric: `include_train` /
  `include_test` flags; each recreated table is asserted against its own
  checksum. The existing `membership_checksum` parameter keeps meaning the
  test-side value; a new optional train-side checksum parameter is added.
  The direct-call test at `test_split.py:375` is updated deliberately.
- **Emission keying contract** (a "matching" sibling means *reciprocal pair
  metadata* — each entry is `format == "split"`, has the expected role, and
  each names the other via `train_name` / `test_name`; a name merely reused
  by a different split does not match):

  | Surviving matching sides | Block owner | include_train, include_test |
  |---|---|---|
  | train + test | test entry | True, True |
  | test only | test entry | False, True |
  | train only | train entry | True, False |
  | neither | — | no block |

- When both sides are overwritten no split block is emitted (unchanged);
  both derived CREATEs carry provenance wraps.
- Per-side checksums also close problem 7: a split-source overwrite that
  changes only train-side rows now fails the train checksum loudly instead
  of replaying with drifted numbers.

## Behavior matrix

"Loud raise" = the Part-2 model `AssertionError`; "wrapped" = Part-3
provenance `AssertionError` with chained `CatalogException`.

| # | Scenario | Today at replay | After fix |
|---|----------|-----------------|-----------|
| S1 | fit on file-backed dataset, never replaced | guarded re-fit from file | unchanged |
| S2 | fit on file-backed dataset → `materialize_query` overwrite (any chain depth) | re-fit from carried base file behind hash guard | unchanged (identified by `base_loader["revision"]` == R0) |
| S3 | fit on derived dataset, never replaced (recipe replayable) | re-fit on recreated table | unchanged |
| S4 | fit on pure-query derived → overwrite, final recipe replayable (**ROADMAP gap**) | **silent** re-fit on post-transform table | loud raise |
| S4b | same, but final recipe self-/cross-references missing tables | raw `CatalogException` at the CREATE | unchanged if plain derived (loud, unexplained — parked, see Non-goals); wrapped if the entry carries split provenance |
| S5 | fit on base-carrying derived → overwrite (adjacent gap) | **silent** re-fit on post-transform table | loud raise (not a silent file re-fit) |
| S6 | fit on split output → that side overwritten | loud raise (1.3.0) | unchanged mechanism-wise; note the Part-3 wrapper fires first when the overwrite recipe itself is unreplayable |
| S7a | fit on in-memory dataframe dataset → `materialize_query` overwrite | loud raise | unchanged |
| S7b | fit on in-memory dataframe dataset → `load_dataset` replacement | **silent** re-fit on new file | loud raise |
| S8 | one split side self-referentially overwritten | wrapped (1.3.1) | unchanged incl. pinned message prefix |
| S9 | both split sides overwritten, ≥1 recipe depends on missing split tables (**ROADMAP gap**) | raw `CatalogException` | wrapped (first failing wrapper halts the cell) |
| S10 | derived overwritten twice; model fit on middle materialization | **silent** re-fit on latest table | loud raise (same S4b qualification) |
| S11 | split-over-split, same names, model fit on old side (**review r1**) | **silent** re-fit on new membership | loud raise |
| S12 | `load_dataset` over derived/split name, model fit pre-load (**review r1**) | **silent** re-fit on new file | loud raise |
| S13 | test side overwritten with replayable SQL, train survives (**review r1**) | train table never recreated → raw `CatalogException` downstream | train-only split block (revision-ordered before the derived CREATE) recreates train behind its own checksum; replay proceeds |
| S14 | both split sides overwritten with independently replayable SQL | replay succeeds | unchanged — wrappers transparent (pinned as success test) |
| S15 | same-name reload, same bytes, same `(path, format, read_options)` | hash-guarded re-fit passes | unchanged |
| S15b | same-name reload, same bytes, **different** read options / path / format (**review r2**) | **silent** re-fit on differently-parsed table | loud raise |
| S15c | same-name reload, changed file content | fails at the model hash assert | loud raise (same outcome, clearer message) |
| S16 | split source overwritten so only train rows change (**review r2**) | test checksum passes; **silent** drifted re-fit | train checksum fails loudly |
| S17 | remote (s3/http) dataset reloaded, same URL and options | unguarded re-fit + comment | unchanged (path-keyed sentinels + loader identity compare equal) |

Deliberate conservatism: any replacement bumps the revision, so re-running
byte-identical SQL (or re-splitting with the same seed) over an unchanged
source still fails the pre-replacement model's replay loudly, even though a
faithful re-fit might have been possible. Determinism of the recipe is not
verifiable; loud beats silently-maybe-right.

## Error handling

No new tool-level error types; live-session behavior is untouched (every
problem above is emit/replay-side only). New failure modes are replay-time
`AssertionError`s; where an original exception exists it is chained.

## Test plan (TDD, red:/green: per slice)

1. `session`: every `register()` bumps the revision; `reset()` restarts;
   `base_loader` records the file entry's revision and still equals R0 after
   a second chained overwrite (pin R0 explicitly).
2. `models`: `fit_model` stamps `training_dataset_revision` and
   `training_loader`.
3. `recorder` + eval: S4 emits the loud raise; notebook replay fails with
   the explanatory `AssertionError`.
4. `recorder`: S5 raises loudly (no silent base-file re-fit); S2 regression
   pin (still re-fits from base file, incl. chained overwrite).
5. `recorder`: S10 direct test; S11 split-over-split; S12 load-over-derived
   and load-over-split; S7b load-over-dataframe.
6. `recorder`: S15 same-loader reload stays guarded-pass; S15b changed
   read-options reload raises; S15c changed-content contract stays green
   (`test_recorder.py:750`); S17 remote-reload comment path unchanged;
   `entry=None` raises "no longer registered".
7. `session`/`materialize`: split-overwrite provenance recorded; carried
   forward on chained overwrites.
8. `recorder` + eval: S9 double self-referential overwrite → wrapped;
   existing S8 tests stay green unmodified (message prefix contract).
9. `split`/`recorder`: per-side checksums stored under
   `membership_checksum`; S13 emits a train-only block asserting the train
   checksum and replay succeeds; symmetric test-only path still green;
   stratified splits exercised in both asymmetric directions; the
   reversed-name-order `overwrite=True` case (problem 8) replays correctly.
10. `recorder` + eval: S14 both-sides-replayable overwrite replays
    successfully; S16 split-source train-only drift fails at the train
    checksum.
11. Emission ordering: second pass sorted by revision — pre-registered
    output names + `split_dataset(overwrite=True)` + test-side overwrite
    emits the train-only block before the derived CREATE.

Gates: `pytest tests/ evals/`, `ruff format --check`, `ruff check`,
`pyright src/`, `scripts/check_tdd_commits.py`.

## Docs & release impact

- ROADMAP: remove both Reproducibility entries (lines 27–28); add parked
  notes for (a) S4b — plain-derived self-referential overwrite still fails
  with an unexplained-but-loud `CatalogException`; (b) ephemeral-fit
  provenance — `cross_validate` and `fit_model(model_name=None)` body cells
  re-fit without fit-time guards, so a source mutated *and reloaded* after
  the call replays silently different CV numbers (review r2; separate
  failure class: per-call cells, not the setup cell).
- CHANGELOG: `Fixed` (S4/S5/S7b/S9/S10/S11/S12/S13/S15b/S16) + `Changed`
  (revision plumbing, loader-identity capture, per-side split checksums,
  split replay symmetry, revision-ordered emission).
- README: `Known gotchas` (`README.md:454`) — models re-fit from the
  original file only when fit *on* the file-backed state; split section
  (`README.md:378`) — per-side checksums; test counts (`README.md:461`).
- `docs/SPEC.md`: model-refit claim (`SPEC.md:537`), §5.6b checksum
  semantics (`SPEC.md:372`), emit-behavior section (`SPEC.md:674`).
- `evals/README.md:27`: already stale (omits `eval_split_cv.py`); refresh
  totals while touching it.
- Recorder docstring (`recorder.py:301`).
- Release plumbing: `pyproject.toml`, `uv.lock`, `__init__.__version__`,
  `test_smoke.py`, CHANGELOG release link.
- **Version:** 1.4.0. Repo precedent (1.2.1, 1.3.1) used patch bumps for
  single replay fixes, and review r2 argued 1.3.2 is defensible; 1.4.0 is
  chosen because per-side checksums and revision plumbing change emitted
  notebook shape and introduce deliberate new loud-failure behavior
  (identical-recipe re-fits now fail conservatively). Downgrading to 1.3.2
  at release time changes nothing else in this design.

## Non-goals

- Explaining plain-derived self-referential overwrite CREATE failures (S4b):
  replay stays loud-but-raw there; wrapping every derived CREATE risks
  churning emitted-notebook shape and pinned tests for marginal benefit.
  Parked in ROADMAP.
- Ephemeral-fit provenance (CV / unregistered fits) — parked in ROADMAP
  (see Docs & release impact).
- Verifying SQL/seed determinism to allow "identical recipe" re-fits after
  replacement — conservative loud failure instead.
- Concurrency-safe revision counters — the whole session is a single-client
  singleton by design (spec §1); the counter inherits that assumption.

## Review deltas

### v1 → v2 (review round 1)

1. v1's "split outputs cannot be overwritten" non-goal was false
   (`split.py:181`) → split-over-split in scope (S11, revisions).
2. v1's sentinel discriminator missed `load_dataset` replacement (S12) →
   revisions cover every registration surface uniformly.
3. v1's S4/S9/S10 rows overstated coverage for unreplayable final recipes →
   qualified (S4b), S9 scoped, S14 added as an explicit success case.
4. The `include_train` asymmetry v1 called "correct" is a real defect (S13)
   → symmetric replay with per-side checksums.
5. v1 under-specified the wrap message and the non-derived fallthrough →
   message prefix contract pinned; Part-2 cases made exhaustive.

### v2 → v3 (review round 2)

1. v2's file-backed mismatch rule was unsafe: identical bytes with changed
   `read_options` silently re-parse (reproduced, `n_obs` 887 → 848) →
   fit-time loader identity (`training_loader`) added; rule now requires
   hash *and* loader equality (S15/S15b/S15c/S17).
2. v2's "registration order" assumption is false for overwrites (dict key
   re-assignment keeps position; reproduced) → second pass sorted by
   `revision` (problem 8, slice 11).
3. Part-2 state space made exhaustive: `entry=None` raise, sentinel-on-
   remote-file distinction, unguarded-remote base contract
   (`test_recorder.py:884`), stated invariants.
4. `cross_validate` confirmed outside Part 2; its separate ephemeral-fit
   replay gap parked explicitly.
5. Base-loader revision carry-forward confirmed; R0-after-chain pinned in
   the test plan.
6. Split keying contract made exact (reciprocal pair metadata, block-owner
   table); checksum key name pinned to avoid breaking
   `test_split.py:75,375`.
7. Matrix corrections: S6 wrapper-precedence note, S7 split into S7a/S7b,
   S15 family rewritten, S16/S17 added.
8. Docs impact extended (SPEC §5.6b + emit section, README split section +
   test counts, stale `evals/README.md`, release plumbing); version
   trade-off documented.
