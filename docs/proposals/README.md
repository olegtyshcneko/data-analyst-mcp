# Proposals

Design drafts for tools / features parked in `ROADMAP.md`. Each file is the
*starting point* for the design conversation required by the contributing
flow — not the final spec. When a proposal is accepted, fold the relevant
sections into `docs/SPEC.md` §5 and delete the proposal.

Style convention: mirror the section structure of `docs/SPEC.md` §5
(purpose, input, behavior, output, errors, recorder cells, TDD slices,
acceptance criteria, ROADMAP impact).

## Current proposals

- [`2026-07-18-ephemeral-fit-replay-provenance.md`](2026-07-18-ephemeral-fit-replay-provenance.md) — prefix replay guards (ROADMAP § Reproducibility): hash asserts in every `load_dataset` per-call cell plus explained failures for `cross_validate` / ephemeral `fit_model` on in-memory sources. Not a new tool; lands as amendments to SPEC §5.1 / §5.11 / §5.11d / §6.

The tier-1 feature bundle (`materialize_query`, `find_outliers`, `power_analysis`, `regression_line` + `residual_diagnostic`) shipped and was folded into `docs/SPEC.md` §5 (entries 5.5a, 5.6a, 5.10a, 5.12a, 5.12b); `pairwise_comparisons` shipped next and was folded into §5.9a (the post-hoc follow-up to `compare_groups`, growing the tool surface 21 → 22). Per the convention above, the proposal files themselves were deleted once folded; the git history retains the design context.
