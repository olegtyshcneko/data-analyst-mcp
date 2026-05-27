# Proposals

Design drafts for tools / features parked in `ROADMAP.md`. Each file is the
*starting point* for the design conversation required by the contributing
flow — not the final spec. When a proposal is accepted, fold the relevant
sections into `docs/SPEC.md` §5 and delete the proposal.

Style convention: mirror the section structure of `docs/SPEC.md` §5
(purpose, input, behavior, output, errors, recorder cells, TDD slices,
acceptance criteria, ROADMAP impact).

## Current proposals

Tier 1 feature bundle — four new tools, drafted ahead of TDD implementation:

- [`materialize_query`](./materialize_query.md) — persist a `SELECT` / `WITH` result as a named DuckDB table and register it in the session, unblocking cohort / funnel / multi-step join workflows.
- [`find_outliers`](./find_outliers.md) — multi-column anomaly detection with four methods (`iqr`, `zscore`, `mahalanobis`, `isolation_forest`).
- [`power_analysis`](./power_analysis.md) — sample-size / MDE / achieved-power solver across five test families (`two_sample_t`, `one_sample_t`, `paired_t`, `two_proportion_z`, `anova_oneway`).
- [`regression_diagnostic_plots`](./regression_diagnostic_plots.md) — OLS-only `regression_line` (scatter + fitted line + 95 % mean-CI band) and `residual_diagnostic` (residuals-vs-fitted, Q-Q, scale-location, residuals-vs-leverage).
