# Replay-guard gaps: registration revisions + split-overwrite provenance + symmetric split replay

**Date:** 2026-07-14
**Status:** draft v2 — v1's sentinel-discriminator mechanism was replaced after
an adversarial review (headless Codex, gpt-5.6-sol @ xhigh) empirically
disproved three of v1's claims; see "Review deltas" at the bottom.
**Target release:** 1.4.0 (bug-fix cluster + emitted-notebook shape changes;
no tool-surface change)

## Problem

Two documented ROADMAP gaps — plus three adjacent defects confirmed during
review — break the project's core replay promise ("silent drift is
impossible; failures are loud *and explained*"):

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
3. **Split-over-split (silent; review finding, reproduced).** `split_dataset`
   with `overwrite=True` (`split.py:181`) replaces both sides under the same
   names. Old and new entries share the constant `sentinel:no-file:(split)`
   hash, so a model fit on the seed-1 train side silently re-fits on the
   seed-2 membership at replay.
4. **`load_dataset` over a derived/split name (silent; review finding,
   reproduced).** `load_dataset` unconditionally replaces the table and entry
   (`datasets.py:600,610`). A model fit on the pre-load derived/split state
   keeps a sentinel hash; the new entry is file-backed, no overwrite branch
   applies, and the recorder re-fits on the new file with only a "no hash
   assert possible" comment.
5. **Test-side overwrite asymmetry (loud/broken; review finding,
   reproduced).** The split replay block is emitted only from a surviving
   `role == "test"` entry (`recorder.py:407`). If the *test* side is
   overwritten — even with independently replayable SQL — no split block is
   emitted and the surviving **train** table is never recreated: replay hits
   a raw `CatalogException` in the model frame loop (`recorder.py:443`) or
   wherever the train table is first referenced.

### Why content hashes cannot fix 1/3/4

Derived, split, and dataframe states have no file to hash — their sentinels
are constants per format, and any per-state string stuffed into `source_hash`
still can't distinguish "fit on the current state" from "fit on a
same-format predecessor" across *all* replacement surfaces
(`materialize_query`, `split_dataset(overwrite=True)`, `load_dataset`).
Identity, not content, is what the model guard needs.

## Design

### Part 1 — registration revisions (`session.py`, `tools/models.py`,
`tools/materialize.py`)

- `session` gains a module-level monotonic counter; **every**
  `register()` call stamps the new entry with a unique
  `DatasetEntry.revision: int` (default `-1` for direct constructions in
  tests). `reset()` zeroes the counter.
- `ModelEntry` gains `training_dataset_revision: int`; `fit_model` copies
  `entry.revision` at fit time (next to the existing hash copy,
  `models.py:158`). `cross_validate`'s ephemeral fits are unaffected.
- When `materialize_query` captures/carries a `base_loader`
  (`materialize.py:114-125`), the dict additionally records the replaced
  file-backed entry's `revision`.
- `source_hash` semantics are **unchanged** — it remains file provenance for
  the emitted hash guards. Revisions are identity; hashes are content.

### Part 2 — model re-fit block: exhaustive cases (`recorder.py:435-558`)

For each registered model, with `rev` = `training_dataset_revision`,
`hash_val` = `training_dataset_hash`, `entry` = current dataset entry:

- **`entry.format == "derived"`:**
  1. `rev == entry.revision` → fit on the current materialization: re-fit on
     the recreated table (unchanged normal path).
  2. `entry.base_loader is not None and rev == base_loader["revision"]` →
     fit on the pre-overwrite *file-backed* state: guarded re-fit from the
     original file (unchanged emitted code, now positively identified).
  3. otherwise → the fit-time table state no longer exists anywhere
     reachable: emit an unconditional `raise AssertionError(...)` naming the
     model, the dataset, and that the dataset was replaced after the fit.
- **`entry.format == "split"`:** `rev == entry.revision` → normal path (the
  split block recreates the table behind its checksum); otherwise → the loud
  raise (covers split-over-split).
- **`entry.format == "dataframe"`:** `rev == entry.revision` → unchanged
  today-path; otherwise → the loud raise.
- **file-backed formats:** `rev == entry.revision` → unchanged hash-guarded
  path. `rev != entry.revision`:
  - `hash_val` is a real content/fallback hash → **keep today's behavior**
    (emit the hash guard against the current path): a same-file reload
    passes and re-fits faithfully; a changed file fails at the assert. This
    preserves the pinned contract in `test_recorder.py:750`.
  - `hash_val` is `sentinel:`-prefixed (model fit on a derived/split/
    dataframe state whose name was later `load_dataset`-ed) → the loud
    raise (closes problem 4).

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
  independently replayable overwrite recipes execute transparently (their
  wrapped CREATE succeeds), including the both-sides-replayable case, which
  is a *success* scenario (S14).

### Part 4 — symmetric split replay with per-side checksums
(`tools/split.py`, `recorder.py`)

- `split_dataset` computes a membership checksum for **both** sides
  (`membership_checksum(train_df)` alongside the existing test-side one) and
  stores each side's checksum in its own entry's `read_options`
  (one extra hashing pass at split time; same cost family as the existing
  test-side pass).
- `split_replay_source()` becomes side-symmetric: `include_train` and
  `include_test` flags, each recreated table asserted against its own
  checksum. (Today only the test table is created-and-asserted
  unconditionally, `recorder.py:194-204`.)
- Emission logic in `_build_setup_source`: a split block is emitted for a
  surviving split *pair* keyed off the test entry (as today, with
  `include_train` mirroring the current check), **and additionally** for a
  surviving train-role entry whose test sibling was overwritten
  (`include_test=False`) — closing problem 5: the surviving train table is
  recreated behind its own checksum, and the overwritten test side's derived
  CREATE (wrapped per Part 3) follows in registration order.
- When both sides are overwritten no split block is emitted (unchanged);
  both derived CREATEs carry provenance wraps.

## Behavior matrix

"Loud raise" = the Part-2 model `AssertionError`; "wrapped" = Part-3
provenance `AssertionError` with chained `CatalogException`.

| # | Scenario | Today at replay | After fix |
|---|----------|-----------------|-----------|
| S1 | fit on file-backed dataset, never replaced | guarded re-fit from file | unchanged |
| S2 | fit on file-backed dataset → `materialize_query` overwrite | re-fit from carried base file behind hash guard | unchanged (identified by base revision) |
| S3 | fit on derived dataset, never replaced (recipe replayable) | re-fit on recreated table | unchanged |
| S4 | fit on pure-query derived → overwrite, final recipe replayable (**ROADMAP gap**) | **silent** re-fit on post-transform table | loud raise |
| S4b | same, but final recipe self-/cross-references missing tables | raw `CatalogException` at the CREATE | unchanged if plain derived (loud, unexplained — parked, see Non-goals); wrapped if the entry carries split provenance |
| S5 | fit on base-carrying derived → overwrite (adjacent gap) | **silent** re-fit on post-transform table | loud raise (not a silent file re-fit) |
| S6 | fit on split output → that side overwritten | loud raise (1.3.0) | unchanged (now via revision mismatch) |
| S7 | fit on in-memory dataframe dataset → overwrite | loud raise | unchanged |
| S8 | one split side self-referentially overwritten | wrapped (1.3.1) | unchanged incl. pinned message prefix |
| S9 | both split sides overwritten, ≥1 recipe depends on missing split tables (**ROADMAP gap**) | raw `CatalogException` | wrapped |
| S10 | derived overwritten twice; model fit on middle materialization | **silent** re-fit on latest table | loud raise (same S4b qualification) |
| S11 | split-over-split, same names, model fit on old side (**review**) | **silent** re-fit on new membership | loud raise |
| S12 | `load_dataset` over derived/split name, model fit pre-load (**review**) | **silent** re-fit on new file | loud raise |
| S13 | test side overwritten with replayable SQL, train survives (**review**) | train table never recreated → raw `CatalogException` downstream | train recreated behind its own checksum; replay proceeds |
| S14 | both split sides overwritten with independently replayable SQL | replay succeeds | unchanged — wrappers transparent (pinned as success test) |
| S15 | same-name file reload after fit | model hash guard: identical file passes, changed file fails loudly | unchanged (`test_recorder.py:750` contract) |

Deliberate conservatism: any replacement bumps the revision, so re-running
byte-identical SQL (or re-splitting with the same seed) over an unchanged
source still fails the pre-replacement model's replay loudly, even though a
faithful re-fit might have been possible. Determinism of the recipe is not
verifiable; loud beats silently-maybe-right.

## Error handling

No new tool-level error types; live-session behavior is untouched (all five
problems are emit/replay-side only). New failure modes are replay-time
`AssertionError`s; where an original exception exists it is chained.

## Test plan (TDD, red:/green: per slice)

1. `session`: every `register()` bumps the revision; `reset()` restarts;
   `base_loader` carries the replaced entry's revision.
2. `models`: `fit_model` stamps `training_dataset_revision`.
3. `recorder` + eval: S4 (replayable final recipe) emits the loud raise;
   notebook replay fails with the explanatory `AssertionError`.
4. `recorder`: S5 raises loudly — does **not** re-fit from the base file;
   S2 regression pin (still re-fits from base file).
5. `recorder`: S10 direct test (double derived overwrite, fit on middle).
6. `recorder`: S11 split-over-split raises loudly.
7. `recorder`: S12 load-over-derived (and over-split) raises loudly; S15
   regression pin stays green (`test_recorder.py:750`).
8. `session`/`materialize`: split-overwrite provenance recorded; carried
   forward on chained overwrites.
9. `recorder` + eval: S9 double self-referential overwrite → wrapped, both
   sides; existing S8 tests stay green unmodified (message prefix contract).
10. `split`/`recorder`: per-side checksums stored; S13 emits a train-only
    split block asserting the train checksum, replay succeeds; symmetric
    test-only path (existing behavior) still green.
11. `recorder` + eval: S14 both-sides-replayable overwrite replays
    successfully end to end.

Gates: `pytest tests/ evals/`, `ruff format --check`, `ruff check`,
`pyright src/`, `scripts/check_tdd_commits.py`.

## Docs & release impact

- ROADMAP: remove both Reproducibility entries (lines 27–28); add a parked
  note for S4b (plain-derived self-referential overwrite still fails with an
  unexplained-but-loud `CatalogException`; extending Part-3-style wrapping to
  ordinary derived CREATEs is future work).
- CHANGELOG: `1.4.0` — `Fixed` (S4/S5/S9/S10/S11/S12/S13) + `Changed`
  (revision plumbing, per-side split checksums, split replay symmetry).
- README `Known gotchas` (`README.md:454`): correct the claim that models
  registered before an overwrite re-fit from the original file — that now
  holds only when the model was fit *on* the file-backed state; otherwise
  replay fails loudly.
- `docs/SPEC.md:537` (same claim) and the recorder docstring
  (`recorder.py:301`): same correction.

## Non-goals

- Explaining plain-derived self-referential overwrite CREATE failures (S4b):
  replay stays loud-but-raw there; wrapping every derived CREATE risks
  churning emitted-notebook shape and pinned tests for marginal benefit.
  Parked in ROADMAP.
- Verifying SQL/seed determinism to allow "identical recipe" re-fits after
  replacement — conservative loud failure instead.
- Concurrency-safe revision counters — the whole session is a single-client
  singleton by design (spec §1); the counter inherits that assumption.

## Review deltas (v1 → v2)

Adversarial review (headless Codex, gpt-5.6-sol @ xhigh, 2026-07-14)
reproduced five defects in v1; all are addressed above:

1. v1's "split outputs cannot be overwritten" non-goal was false
   (`split.py:181`) → split-over-split is now in scope (S11, revisions).
2. v1's sentinel discriminator missed `load_dataset` replacement (S12) →
   revisions cover every registration surface uniformly.
3. v1's S4/S9/S10 rows overstated coverage for unreplayable final recipes →
   matrix rows qualified (S4b), S9 requires dependence on missing tables,
   S14 added as an explicit success case.
4. The `include_train` asymmetry v1 called "correct" is a real defect (S13)
   → Part 4 symmetric replay with per-side checksums.
5. v1 under-specified the wrap message and the non-derived fallthrough →
   message prefix contract pinned; Part-2 cases made exhaustive with S15
   preserved.
