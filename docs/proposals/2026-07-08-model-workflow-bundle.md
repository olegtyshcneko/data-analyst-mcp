# Model workflow bundle: `split_dataset` + `cross_validate`

**Status:** shipped in 1.3.0 — folded into `docs/SPEC.md`.

This is a pointer stub, not the design record. The full design conversation
lives in
[`docs/superpowers/specs/2026-07-08-model-workflow-bundle-design.md`](../superpowers/specs/2026-07-08-model-workflow-bundle-design.md)
(purpose, rejected alternatives, determinism rationale, recorder/replay
contract, TDD slices). The accepted contracts are folded into `docs/SPEC.md`
as **§5.6b `split_dataset`** and **§5.11d `cross_validate`**.

## What shipped

The Phase-5 model registry (`fit_model(model_name=...)` / `predict` /
`evaluate_model`) assumes a held-out dataset exists but offered no way to make
one, and `evaluate_model` is single-holdout only. This two-tool cohort closes
both gaps and composes with the existing registry and recorder:

- **`split_dataset`** — a seeded, optionally stratified train/test partition
  registered as two first-class `format="split"` datasets, replayed in the
  emitted notebook behind an order-independent membership checksum (SPEC §5.6b).
- **`cross_validate`** — k-fold cross-validated metrics for a formula; a
  full-data preflight fit through `fit_prepared` surfaces `fit_model`'s error
  taxonomy before any fold work, fits are ephemeral, and the registry is never
  touched (SPEC §5.11d).

## Flow

Same proposal flow `pairwise_comparisons` established: issue → this
`docs/proposals/` draft → design conversation (the design doc above) → fold
into `docs/SPEC.md` §5 → release. The tool surface grew **22 → 24**; the
boundary note is updated in `ROADMAP.md`.
