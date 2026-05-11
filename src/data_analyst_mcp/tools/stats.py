"""Statistical tools — correlate, compare_groups, test_hypothesis."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

_NUMERIC_DTYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
}


def _is_numeric_dtype(dtype: str) -> bool:
    """True if a DuckDB dtype represents a numeric column."""
    base = dtype.split("(")[0].strip().upper()
    return base in _NUMERIC_DTYPES


def _quote(name: str) -> str:
    """Quote a SQL identifier safely for DuckDB."""
    return '"' + name.replace('"', '""') + '"'

logger = logging.getLogger(__name__)


class CorrelateInput(BaseModel):
    """Inputs for ``correlate``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of column names to correlate. When omitted, every "
            "numeric column in the dataset is used."
        ),
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description="Correlation method: pearson, spearman, or kendall.",
    )
    plot: bool = Field(
        default=True,
        description="When true, include a base64-encoded PNG heatmap in the response.",
    )


def correlate(payload: CorrelateInput) -> dict[str, Any]:
    """Compute a correlation matrix across numeric columns."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    available = {c["name"] for c in entry.columns}
    if payload.columns is not None:
        missing = [c for c in payload.columns if c not in available]
        if missing:
            return build_error(
                type="column_not_found",
                message=f"Columns not in dataset {payload.name!r}: {missing}.",
                hint=f"Available: {sorted(available)}",
            )
        chosen = list(payload.columns)
    else:
        chosen = [c["name"] for c in entry.columns if _is_numeric_dtype(c["dtype"])]
        if not chosen:
            return build_error(
                type="no_numeric_columns",
                message=f"Dataset {payload.name!r} has no numeric columns.",
                hint="Pass an explicit `columns` list, or cast columns to numeric first.",
            )

    matrix = _build_corr_matrix(payload.name, chosen, payload.method)

    out: dict[str, Any] = {
        "ok": True,
        "method": payload.method,
        "labels": list(chosen),
        "matrix": matrix,
    }
    if payload.plot:
        out["heatmap_png_base64"] = _render_heatmap_png(chosen, matrix)

    md = (
        f"### Correlation matrix on `{payload.name}` ({payload.method})\n"
        f"- Columns: {', '.join(chosen)}"
    )
    cols_list = ", ".join(f'"{c}"' for c in chosen)
    code = (
        f"from scipy import stats\n"
        f"df = con.sql('SELECT {cols_list} FROM {payload.name}').df()\n"
        f"df.corr(method='{payload.method}')"
    )
    get_recorder().record(markdown=md, code=code, tool_name="correlate")
    return out


class _TwoSampleColumns(BaseModel):
    """Shared fields for kinds that compare two groups within one dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    group_column: str = Field(..., description="Column holding the group labels.")
    metric_column: str = Field(..., description="Numeric metric column to test.")
    group_a: str = Field(..., description="Label of the first group.")
    group_b: str = Field(..., description="Label of the second group.")


class TTestInput(_TwoSampleColumns):
    """Student's t-test (equal variances assumed)."""

    kind: Literal["t_test"] = "t_test"


class WelchInput(_TwoSampleColumns):
    """Welch's t-test (unequal variances)."""

    kind: Literal["welch"] = "welch"


class MannWhitneyInput(_TwoSampleColumns):
    """Mann-Whitney U (non-parametric two-sample)."""

    kind: Literal["mann_whitney"] = "mann_whitney"


class KSInput(_TwoSampleColumns):
    """Kolmogorov-Smirnov two-sample distribution test."""

    kind: Literal["ks"] = "ks"


class AnovaInput(BaseModel):
    """One-way ANOVA across all distinct group labels in ``group_column``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["anova"] = "anova"
    name: str = Field(..., description="Registered dataset name.")
    group_column: str = Field(..., description="Column holding the group labels.")
    metric_column: str = Field(..., description="Numeric metric column to test.")


class KruskalInput(BaseModel):
    """Kruskal-Wallis H test (non-parametric many-sample)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["kruskal"] = "kruskal"
    name: str = Field(..., description="Registered dataset name.")
    group_column: str = Field(..., description="Column holding the group labels.")
    metric_column: str = Field(..., description="Numeric metric column to test.")


class ChiSquareInput(BaseModel):
    """Chi-square test of independence on an explicit contingency table."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["chi_square"] = "chi_square"
    table: list[list[int]] = Field(
        ..., description="r×c integer contingency table as a list of rows."
    )


class FisherInput(BaseModel):
    """Fisher's exact test on an explicit 2×2 contingency table."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["fisher"] = "fisher"
    table: list[list[int]] = Field(
        ..., description="2×2 integer contingency table as a list of two rows."
    )


TestHypothesisInput = Annotated[
    Union[  # noqa: UP007
        TTestInput,
        WelchInput,
        MannWhitneyInput,
        KSInput,
        AnovaInput,
        KruskalInput,
        ChiSquareInput,
        FisherInput,
    ],
    Field(discriminator="kind"),
]


ALLOWED_KINDS: frozenset[str] = frozenset(
    {
        "t_test",
        "welch",
        "mann_whitney",
        "ks",
        "anova",
        "kruskal",
        "chi_square",
        "fisher",
    }
)


def _materialize_group(name: str, group_col: str, metric_col: str, label: str) -> Any:
    """Return a numpy array of the metric for rows where group equals label."""
    con = session.get_connection()
    table = _quote(name)
    rel = con.execute(
        f"SELECT {_quote(metric_col)} FROM {table} WHERE {_quote(group_col)} = ?",
        [label],
    )
    df = rel.df()
    return df[metric_col].to_numpy()


def _cohens_d(a: Any, b: Any) -> float:
    """Pooled Cohen's d (ddof=1) for two independent samples."""
    import math

    n1, n2 = len(a), len(b)
    s1 = float(a.std(ddof=1))
    s2 = float(b.std(ddof=1))
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0.0:
        return 0.0
    return (float(a.mean()) - float(b.mean())) / sp


def _materialize_two(payload: _TwoSampleColumns) -> tuple[Any, Any]:
    """Materialize the metric arrays for ``group_a`` and ``group_b``."""
    a = _materialize_group(payload.name, payload.group_column, payload.metric_column, payload.group_a)
    b = _materialize_group(payload.name, payload.group_column, payload.metric_column, payload.group_b)
    return a, b


def _run_t_test(payload: TTestInput) -> dict[str, Any]:
    """Student's t-test (equal_var=True)."""
    from scipy import stats as _sps

    a, b = _materialize_two(payload)
    r = _sps.ttest_ind(a, b, equal_var=True)
    return {
        "ok": True,
        "test": "t_test",
        "statistic": float(r.statistic),
        "p_value": float(r.pvalue),
        "df": float(r.df),
        "effect_size": {"name": "cohens_d", "value": _cohens_d(a, b)},
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "interpretation": _interpret_two_sample(
            payload.group_a, payload.group_b, float(r.pvalue)
        ),
    }


def _interpret_two_sample(a: str, b: str, p: float) -> str:
    """One-sentence plain-English interpretation of a two-sample p-value."""
    if p < 0.05:
        return (
            f"Groups `{a}` and `{b}` differ at α=0.05 (p={p:.4f})."
        )
    return (
        f"No statistically significant difference between `{a}` and `{b}` at α=0.05 (p={p:.4f})."
    )


def _run_welch(payload: WelchInput) -> dict[str, Any]:
    """Welch's t-test (equal_var=False)."""
    from scipy import stats as _sps

    a, b = _materialize_two(payload)
    r = _sps.ttest_ind(a, b, equal_var=False)
    return {
        "ok": True,
        "test": "welch",
        "statistic": float(r.statistic),
        "p_value": float(r.pvalue),
        "df": float(r.df),
        "effect_size": {"name": "cohens_d", "value": _cohens_d(a, b)},
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "interpretation": _interpret_two_sample(
            payload.group_a, payload.group_b, float(r.pvalue)
        ),
    }


def _run_mann_whitney(payload: MannWhitneyInput) -> dict[str, Any]:
    """Mann-Whitney U (two-sided)."""
    from scipy import stats as _sps

    a, b = _materialize_two(payload)
    r = _sps.mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = len(a), len(b)
    rbis = 1.0 - 2.0 * float(r.statistic) / (n_a * n_b)
    interp = _interpret_two_sample(payload.group_a, payload.group_b, float(r.pvalue))
    return {
        "ok": True,
        "test": "mann_whitney",
        "statistic": float(r.statistic),
        "p_value": float(r.pvalue),
        "df": None,
        "effect_size": {"name": "rank_biserial", "value": rbis},
        "n_a": int(n_a),
        "n_b": int(n_b),
        "interpretation": interp,
    }


def _run_chi_square(payload: ChiSquareInput) -> dict[str, Any]:
    """Chi-square test of independence on an explicit contingency table."""
    import math

    import numpy as np
    from scipy import stats as _sps

    table = np.array(payload.table, dtype=float)
    chi2, p, dof, _exp = _sps.chi2_contingency(table)
    n = float(table.sum())
    cv_denom = n * max(min(table.shape) - 1, 1)
    cv = math.sqrt(float(chi2) / cv_denom) if cv_denom > 0 else 0.0
    n_a = int(table[0].sum()) if table.shape[0] >= 1 else 0
    n_b = int(table[1].sum()) if table.shape[0] >= 2 else 0
    interp = (
        f"Variables are not independent at α=0.05 (chi-square, p={p:.4f})."
        if p < 0.05
        else f"No evidence against independence at α=0.05 (chi-square, p={p:.4f})."
    )
    return {
        "ok": True,
        "test": "chi_square",
        "statistic": float(chi2),
        "p_value": float(p),
        "df": int(dof),
        "effect_size": {"name": "cramers_v", "value": float(cv)},
        "n_a": n_a,
        "n_b": n_b,
        "interpretation": interp,
    }


def _run_fisher(payload: FisherInput) -> dict[str, Any]:
    """Fisher's exact test on a 2x2 contingency table."""
    import numpy as np
    from scipy import stats as _sps

    table = np.array(payload.table, dtype=float)
    r = _sps.fisher_exact(table)
    odds = float(r.statistic)
    p = float(r.pvalue)
    interp = (
        f"Variables are not independent at α=0.05 (Fisher exact, p={p:.4f})."
        if p < 0.05
        else f"No evidence against independence at α=0.05 (Fisher exact, p={p:.4f})."
    )
    n_a = int(table[0].sum()) if table.shape[0] >= 1 else 0
    n_b = int(table[1].sum()) if table.shape[0] >= 2 else 0
    return {
        "ok": True,
        "test": "fisher",
        "statistic": odds,
        "p_value": p,
        "df": None,
        "effect_size": {"name": "odds_ratio", "value": odds},
        "n_a": n_a,
        "n_b": n_b,
        "interpretation": interp,
    }


def _materialize_groups(
    name: str, group_col: str, metric_col: str
) -> tuple[list[str], list[Any]]:
    """Return all distinct group labels and the per-group metric arrays."""
    con = session.get_connection()
    table = _quote(name)
    label_rows = con.execute(
        f"SELECT DISTINCT {_quote(group_col)} FROM {table} "
        f"WHERE {_quote(group_col)} IS NOT NULL "
        f"ORDER BY {_quote(group_col)}"
    ).fetchall()
    labels = [str(r[0]) for r in label_rows]
    groups = [
        _materialize_group(name, group_col, metric_col, lab) for lab in labels
    ]
    return labels, groups


def _eta_squared(groups: list[Any]) -> float:
    """Compute η² = SS_between / SS_total across a list of group arrays."""
    import numpy as np

    all_vals = np.concatenate(groups)
    gm = float(all_vals.mean())
    ssb = sum(len(g) * (float(g.mean()) - gm) ** 2 for g in groups)
    sst = float(((all_vals - gm) ** 2).sum())
    return ssb / sst if sst > 0 else 0.0


def _run_anova(payload: AnovaInput) -> dict[str, Any]:
    """One-way ANOVA across every distinct group label."""
    from scipy import stats as _sps

    labels, groups = _materialize_groups(payload.name, payload.group_column, payload.metric_column)
    r = _sps.f_oneway(*groups)
    eta = _eta_squared(groups)
    p = float(r.pvalue)
    interp = (
        f"At least one of {labels} differs at α=0.05 (one-way ANOVA, p={p:.4f})."
        if p < 0.05
        else f"No evidence of group differences at α=0.05 (one-way ANOVA, p={p:.4f})."
    )
    return {
        "ok": True,
        "test": "anova",
        "statistic": float(r.statistic),
        "p_value": p,
        "df": None,
        "effect_size": {"name": "eta_squared", "value": eta},
        "n_a": int(len(groups[0])) if groups else 0,
        "n_b": int(len(groups[1])) if len(groups) > 1 else 0,
        "interpretation": interp,
    }


def _run_kruskal(payload: KruskalInput) -> dict[str, Any]:
    """Kruskal-Wallis H test across distinct group labels."""
    from scipy import stats as _sps

    labels, groups = _materialize_groups(payload.name, payload.group_column, payload.metric_column)
    r = _sps.kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    eps = float(r.statistic) / (n_total - 1) if n_total > 1 else 0.0
    p = float(r.pvalue)
    interp = (
        f"At least one of {labels} differs at α=0.05 (Kruskal-Wallis, p={p:.4f})."
        if p < 0.05
        else f"No evidence of group differences at α=0.05 (Kruskal-Wallis, p={p:.4f})."
    )
    return {
        "ok": True,
        "test": "kruskal",
        "statistic": float(r.statistic),
        "p_value": p,
        "df": None,
        "effect_size": {"name": "epsilon_squared", "value": eps},
        "n_a": int(len(groups[0])) if groups else 0,
        "n_b": int(len(groups[1])) if len(groups) > 1 else 0,
        "interpretation": interp,
    }


def _run_ks(payload: KSInput) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov."""
    from scipy import stats as _sps

    a, b = _materialize_two(payload)
    r = _sps.ks_2samp(a, b)
    p = float(r.pvalue)
    interp = (
        f"Distributions of `{payload.group_a}` and `{payload.group_b}` differ at "
        f"α=0.05 (KS 2-sample, p={p:.4f})."
        if p < 0.05
        else f"No evidence of distributional difference at α=0.05 (KS 2-sample, p={p:.4f})."
    )
    return {
        "ok": True,
        "test": "ks",
        "statistic": float(r.statistic),
        "p_value": p,
        "df": None,
        "effect_size": {"name": "ks_d", "value": float(r.statistic)},
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "interpretation": interp,
    }


_DATASET_KINDS = {"t_test", "welch", "mann_whitney", "ks", "anova", "kruskal"}


_KIND_HANDLERS: dict[str, Any] = {
    "t_test": _run_t_test,
    "welch": _run_welch,
    "mann_whitney": _run_mann_whitney,
    "chi_square": _run_chi_square,
    "fisher": _run_fisher,
    "anova": _run_anova,
    "kruskal": _run_kruskal,
    "ks": _run_ks,
}


def test_hypothesis(payload: Any) -> dict[str, Any]:
    """Dispatch to a kind-specific handler. ``payload`` is the validated union."""
    kind = payload.kind
    if kind in _DATASET_KINDS:
        ds_name = payload.name
        if ds_name not in session.get_datasets():
            return build_error(
                type="not_found",
                message=f"No dataset named {ds_name!r} registered.",
                hint="Call list_datasets to see what is available.",
            )
    handler = _KIND_HANDLERS.get(kind)
    if handler is None:
        return build_error(
            type="invalid_kind",
            message=f"Unknown kind {kind!r}.",
            hint=f"Allowed kinds: {sorted(ALLOWED_KINDS)}.",
        )
    result = handler(payload)
    _record_test_hypothesis(payload, result)
    return result


def _record_test_hypothesis(payload: Any, result: dict[str, Any]) -> None:
    """Append a markdown+code cell pair describing the test_hypothesis call."""
    if not result.get("ok"):
        return
    kind = payload.kind
    md = (
        f"### Hypothesis test (`{kind}`)\n"
        f"- statistic = {result['statistic']:.4f}\n"
        f"- p_value = {result['p_value']:.4f}\n"
        f"- effect_size ({result['effect_size']['name']}) = {result['effect_size']['value']:.4f}\n"
        f"- {result['interpretation']}"
    )
    code = _code_for_kind(payload)
    get_recorder().record(markdown=md, code=code, tool_name="test_hypothesis")


def _code_for_kind(payload: Any) -> str:
    """Render a reproducible scipy snippet for the given payload."""
    kind = payload.kind
    if kind in {"t_test", "welch"}:
        equal_var = "True" if kind == "t_test" else "False"
        return (
            f"from scipy import stats\n"
            f"a = con.sql(\"SELECT {payload.metric_column} FROM {payload.name} "
            f"WHERE {payload.group_column} = '{payload.group_a}'\").df()['{payload.metric_column}']\n"
            f"b = con.sql(\"SELECT {payload.metric_column} FROM {payload.name} "
            f"WHERE {payload.group_column} = '{payload.group_b}'\").df()['{payload.metric_column}']\n"
            f"stats.ttest_ind(a, b, equal_var={equal_var})"
        )
    if kind == "mann_whitney":
        return (
            f"from scipy import stats\n"
            f"a = con.sql(\"SELECT {payload.metric_column} FROM {payload.name} "
            f"WHERE {payload.group_column} = '{payload.group_a}'\").df()['{payload.metric_column}']\n"
            f"b = con.sql(\"SELECT {payload.metric_column} FROM {payload.name} "
            f"WHERE {payload.group_column} = '{payload.group_b}'\").df()['{payload.metric_column}']\n"
            f"stats.mannwhitneyu(a, b, alternative='two-sided')"
        )
    if kind == "ks":
        return (
            f"from scipy import stats\n"
            f"a = con.sql(\"SELECT {payload.metric_column} FROM {payload.name} "
            f"WHERE {payload.group_column} = '{payload.group_a}'\").df()['{payload.metric_column}']\n"
            f"b = con.sql(\"SELECT {payload.metric_column} FROM {payload.name} "
            f"WHERE {payload.group_column} = '{payload.group_b}'\").df()['{payload.metric_column}']\n"
            f"stats.ks_2samp(a, b)"
        )
    if kind in {"anova", "kruskal"}:
        fname = "f_oneway" if kind == "anova" else "kruskal"
        return (
            f"from scipy import stats\n"
            f"df = con.sql('SELECT * FROM {payload.name}').df()\n"
            f"groups = [g['{payload.metric_column}'].to_numpy() "
            f"for _, g in df.groupby('{payload.group_column}')]\n"
            f"stats.{fname}(*groups)"
        )
    if kind == "chi_square":
        return (
            f"from scipy import stats\n"
            f"stats.chi2_contingency({payload.table!r})"
        )
    return f"from scipy import stats\nstats.fisher_exact({payload.table!r})"


class CompareGroupsInput(BaseModel):
    """Inputs for ``compare_groups``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered dataset name.")
    group_column: str = Field(..., description="Column holding the group labels.")
    metric_column: str = Field(
        ..., description="Column holding the metric to compare across groups."
    )
    groups: list[str] | None = Field(
        default=None,
        description=(
            "Subset of group labels to include. When omitted, default to "
            "the top-2 most-frequent labels (2-sample path) or all distinct "
            "labels (>2-sample path)."
        ),
    )


def _top_n_labels(name: str, group_col: str, n: int) -> list[str]:
    """Return the top-N most-frequent (non-null) labels of ``group_col``."""
    con = session.get_connection()
    rows = con.execute(
        f"SELECT {_quote(group_col)} AS lab, COUNT(*) AS c "
        f"FROM {_quote(name)} "
        f"WHERE {_quote(group_col)} IS NOT NULL "
        f"GROUP BY lab ORDER BY c DESC, lab ASC LIMIT {int(n)}"
    ).fetchall()
    return [str(r[0]) for r in rows]


def _all_labels(name: str, group_col: str) -> list[str]:
    """Return every distinct (non-null) label of ``group_col``."""
    con = session.get_connection()
    rows = con.execute(
        f"SELECT DISTINCT {_quote(group_col)} FROM {_quote(name)} "
        f"WHERE {_quote(group_col)} IS NOT NULL ORDER BY 1"
    ).fetchall()
    return [str(r[0]) for r in rows]


def _shapiro_p(arr: Any) -> float | None:
    """Return Shapiro-Wilk p-value, sampling at most 5000 elements; None if n<3."""
    import numpy as np
    from scipy import stats as _sps

    if len(arr) < 3:
        return None
    if len(arr) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(arr), size=5000, replace=False)
        arr = arr[idx]
    return float(_sps.shapiro(arr).pvalue)


def _levene_p(*groups: Any) -> float:
    """Return Levene's test p-value across the given groups."""
    from scipy import stats as _sps

    return float(_sps.levene(*groups).pvalue)


def _select_two_sample_continuous(p_norm: list[float | None], p_levene: float) -> str:
    """Pick student_t, welch_t, or mann_whitney for a 2-sample continuous compare."""
    normality_violated = any(p is not None and p < 0.05 for p in p_norm)
    if normality_violated:
        return "mann_whitney"
    if p_levene < 0.05:
        return "welch_t"
    return "student_t"


def compare_groups(payload: CompareGroupsInput) -> dict[str, Any]:
    """Pick an appropriate statistical test for two-or-more groups."""
    entries = session.get_datasets()
    if payload.name not in entries:
        return build_error(
            type="not_found",
            message=f"No dataset named {payload.name!r} registered.",
            hint="Call list_datasets to see what is available.",
        )
    entry = entries[payload.name]
    available = {c["name"]: c["dtype"] for c in entry.columns}
    for col in (payload.group_column, payload.metric_column):
        if col not in available:
            return build_error(
                type="column_not_found",
                message=f"Column {col!r} is not in dataset {payload.name!r}.",
                hint=f"Available columns: {', '.join(sorted(available))}",
            )

    metric_dtype = available[payload.metric_column]
    if not _is_numeric_dtype(metric_dtype):
        return _compare_groups_categorical(payload)

    # Continuous metric path.
    if payload.groups is not None:
        labels = list(payload.groups)
    else:
        all_labels = _all_labels(payload.name, payload.group_column)
        if len(all_labels) > 2:
            labels = all_labels
        else:
            labels = _top_n_labels(payload.name, payload.group_column, 2)

    arrays = [
        _materialize_group(payload.name, payload.group_column, payload.metric_column, lab)
        for lab in labels
    ]

    if len(labels) == 2:
        a, b = arrays
        p_norm = [_shapiro_p(a), _shapiro_p(b)]
        p_lev = _levene_p(a, b)
        test = _select_two_sample_continuous(p_norm, p_lev)
        from scipy import stats as _sps

        if test == "mann_whitney":
            mw = _sps.mannwhitneyu(a, b, alternative="two-sided")
            stat, p = float(mw.statistic), float(mw.pvalue)
            df = float("nan")
            rbis = 1.0 - 2.0 * stat / (len(a) * len(b))
            effect = {"name": "rank_biserial", "value": rbis}
        else:
            if test == "welch_t":
                r = _sps.ttest_ind(a, b, equal_var=False)
            else:
                r = _sps.ttest_ind(a, b, equal_var=True)
            effect = {"name": "cohens_d", "value": _cohens_d(a, b)}
            stat, p, df = float(r.statistic), float(r.pvalue), float(r.df)

        return _build_two_sample_response(
            test=test,
            stat=stat,
            p=p,
            df=df,
            effect=effect,
            labels=labels,
            arrays=arrays,
            p_norm=p_norm,
            p_lev=p_lev,
        )

    # >2 groups: ANOVA if normality holds, else Kruskal-Wallis.
    p_norm_many = [_shapiro_p(arr) for arr in arrays]
    p_lev_many = _levene_p(*arrays)
    from scipy import stats as _sps

    if any(pn is not None and pn < 0.05 for pn in p_norm_many):
        r = _sps.kruskal(*arrays)
        n_total = sum(len(g) for g in arrays)
        eps = float(r.statistic) / (n_total - 1) if n_total > 1 else 0.0
        return _build_many_sample_response(
            test="kruskal_wallis",
            stat=float(r.statistic),
            p=float(r.pvalue),
            df=None,
            effect={"name": "epsilon_squared", "value": eps},
            labels=labels,
            arrays=arrays,
            p_norm=p_norm_many,
            p_lev=p_lev_many,
        )
    r = _sps.f_oneway(*arrays)
    eta = _eta_squared(arrays)
    return _build_many_sample_response(
        test="anova",
        stat=float(r.statistic),
        p=float(r.pvalue),
        df=None,
        effect={"name": "eta_squared", "value": eta},
        labels=labels,
        arrays=arrays,
        p_norm=p_norm_many,
        p_lev=p_lev_many,
    )


def _compare_groups_categorical(payload: CompareGroupsInput) -> dict[str, Any]:
    """Categorical-metric path: build contingency table → chi-square."""
    import math

    import numpy as np
    from scipy import stats as _sps

    con = session.get_connection()
    rows = con.execute(
        f"SELECT {_quote(payload.group_column)} AS g, "
        f"{_quote(payload.metric_column)} AS m, COUNT(*) AS c "
        f"FROM {_quote(payload.name)} "
        f"WHERE {_quote(payload.group_column)} IS NOT NULL "
        f"  AND {_quote(payload.metric_column)} IS NOT NULL "
        f"GROUP BY g, m ORDER BY g, m"
    ).fetchall()
    g_labels = sorted({str(r[0]) for r in rows})
    m_labels = sorted({str(r[1]) for r in rows})
    cell: dict[tuple[str, str], int] = {(str(r[0]), str(r[1])): int(r[2]) for r in rows}
    if payload.groups is not None:
        g_labels = [g for g in g_labels if g in set(payload.groups)]
    table = np.array(
        [[cell.get((g, m), 0) for m in m_labels] for g in g_labels], dtype=float
    )
    chi2, p, dof, _exp = _sps.chi2_contingency(table)
    n = float(table.sum())
    denom = n * max(min(table.shape) - 1, 1)
    cv = math.sqrt(float(chi2) / denom) if denom > 0 else 0.0
    arrays = [table[i] for i in range(len(g_labels))]

    # 2x2 with any expected cell < 5 → Fisher's exact (more accurate at low n)
    if table.shape == (2, 2) and float(_exp.min()) < 5:
        fr = _sps.fisher_exact(table)
        odds = float(fr.statistic)
        p_f = float(fr.pvalue)
        interp_f = (
            f"`{payload.group_column}` and `{payload.metric_column}` are associated "
            f"at α=0.05 (Fisher exact, p={p_f:.4f}, odds_ratio={odds:.3f})."
            if p_f < 0.05
            else f"No evidence of association between `{payload.group_column}` and "
            f"`{payload.metric_column}` at α=0.05 (Fisher exact, p={p_f:.4f})."
        )
        return {
            "ok": True,
            "test": "fisher_exact",
            "statistic": odds,
            "p_value": p_f,
            "df": None,
            "effect_size": {"name": "odds_ratio", "value": odds},
            "groups": [
                {"name": g_labels[i], "n": int(arrays[i].sum())} for i in range(len(g_labels))
            ],
            "assumption_checks": {
                "expected_cell_min": {
                    "value": float(_exp.min()),
                    "violated": True,
                    "consequence": "Switched to Fisher's exact (2×2 with expected<5).",
                }
            },
            "interpretation": interp_f,
        }

    interp = (
        f"`{payload.group_column}` and `{payload.metric_column}` are associated "
        f"at α=0.05 (chi-square, p={float(p):.4f}, cramers_v={cv:.3f})."
        if float(p) < 0.05
        else f"No evidence of association between `{payload.group_column}` and "
        f"`{payload.metric_column}` at α=0.05 (chi-square, p={float(p):.4f})."
    )
    return {
        "ok": True,
        "test": "chi_square",
        "statistic": float(chi2),
        "p_value": float(p),
        "df": int(dof),
        "effect_size": {"name": "cramers_v", "value": float(cv)},
        "groups": [{"name": g_labels[i], "n": int(arrays[i].sum())} for i in range(len(g_labels))],
        "assumption_checks": {
            "expected_cell_min": {
                "value": float(_exp.min()),
                "violated": float(_exp.min()) < 5,
                "consequence": (
                    "Expected cell < 5 — consider Fisher's exact for 2×2."
                    if float(_exp.min()) < 5
                    else "Chi-square expected-count assumption holds."
                ),
            }
        },
        "interpretation": interp,
    }


def _build_two_sample_response(
    *,
    test: str,
    stat: float,
    p: float,
    df: float,
    effect: dict[str, Any],
    labels: list[str],
    arrays: list[Any],
    p_norm: list[float | None],
    p_lev: float,
) -> dict[str, Any]:
    """Assemble the standard compare_groups envelope for the 2-sample path."""
    normality_violated = any(pn is not None and pn < 0.05 for pn in p_norm)
    equal_var_violated = p_lev < 0.05
    if test == "student_t":
        norm_consequence = "Normality holds; pooled variance acceptable."
        var_consequence = "Pooled variance acceptable."
    elif test == "welch_t":
        norm_consequence = "Normality holds."
        var_consequence = "Switched from Student to Welch's t."
    else:
        norm_consequence = "Switched from t-test to Mann-Whitney U."
        var_consequence = (
            "Welch-vs-Student moot — switched to rank-based test."
        )
    assumption_checks: dict[str, Any] = {
        "normality_test": {
            "name": "shapiro",
            "p": _min_non_none(p_norm),
            "violated": normality_violated,
            "consequence": norm_consequence,
        },
        "equal_variances_test": {
            "name": "levene",
            "p": p_lev,
            "violated": equal_var_violated,
            "consequence": var_consequence,
        },
    }
    interp = _build_interpretation_two(test, labels, p, effect)
    return {
        "ok": True,
        "test": test,
        "statistic": stat,
        "p_value": p,
        "df": df if df == df else None,  # NaN → None
        "effect_size": effect,
        "groups": [{"name": labels[i], "n": int(len(arrays[i]))} for i in range(len(labels))],
        "assumption_checks": assumption_checks,
        "interpretation": interp,
    }


def _build_many_sample_response(
    *,
    test: str,
    stat: float,
    p: float,
    df: float | None,
    effect: dict[str, Any],
    labels: list[str],
    arrays: list[Any],
    p_norm: list[float | None],
    p_lev: float,
) -> dict[str, Any]:
    """Assemble the standard compare_groups envelope for the >2-sample path."""
    normality_violated = any(pn is not None and pn < 0.05 for pn in p_norm)
    equal_var_violated = p_lev < 0.05
    if test == "anova":
        norm_consequence = "Normality holds; one-way ANOVA applies."
        var_consequence = "Levene's test did not reject equal variances."
    else:
        norm_consequence = "Switched from ANOVA to Kruskal-Wallis."
        var_consequence = "Rank-based test handles unequal variances."
    assumption_checks: dict[str, Any] = {
        "normality_test": {
            "name": "shapiro",
            "p": _min_non_none(p_norm),
            "violated": normality_violated,
            "consequence": norm_consequence,
        },
        "equal_variances_test": {
            "name": "levene",
            "p": p_lev,
            "violated": equal_var_violated,
            "consequence": var_consequence,
        },
    }
    pretty = {"anova": "one-way ANOVA", "kruskal_wallis": "Kruskal-Wallis"}.get(test, test)
    if p < 0.05:
        interp = (
            f"At least one of {labels} differs significantly ({pretty}, "
            f"p={p:.4f}, {effect['name']}={effect['value']:.3f})."
        )
    else:
        interp = (
            f"No evidence of group differences at α=0.05 ({pretty}, "
            f"p={p:.4f}, {effect['name']}={effect['value']:.3f})."
        )
    return {
        "ok": True,
        "test": test,
        "statistic": stat,
        "p_value": p,
        "df": df,
        "effect_size": effect,
        "groups": [{"name": labels[i], "n": int(len(arrays[i]))} for i in range(len(labels))],
        "assumption_checks": assumption_checks,
        "interpretation": interp,
    }


def _min_non_none(vals: list[float | None]) -> float | None:
    """Smallest finite value in ``vals`` or None when all are None."""
    finite = [v for v in vals if v is not None]
    return min(finite) if finite else None


def _build_interpretation_two(test: str, labels: list[str], p: float, effect: dict[str, Any]) -> str:
    """One-sentence plain-English interpretation for a 2-sample result."""
    a, b = labels[0], labels[1]
    name_h = {
        "student_t": "Student's t",
        "welch_t": "Welch's t",
        "mann_whitney": "Mann-Whitney U",
    }.get(test, test)
    if p < 0.05:
        return (
            f"Groups `{a}` and `{b}` differ significantly ({name_h}, p={p:.4f}, "
            f"{effect['name']}={effect['value']:.3f})."
        )
    return (
        f"No statistically significant difference between `{a}` and `{b}` at α=0.05 "
        f"({name_h}, p={p:.4f}, {effect['name']}={effect['value']:.3f})."
    )


def _render_heatmap_png(labels: list[str], matrix: list[list[float]]) -> str:
    """Render the correlation matrix as a base64-encoded PNG heatmap."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data_analyst_mcp.formatting import png_to_base64

    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(labels), 4 + 0.4 * len(labels)))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return png_to_base64(buf.getvalue())


def _pairwise_corr(a: Any, b: Any, method: str) -> float:
    """Return the requested pairwise correlation coefficient as a float."""
    from scipy import stats as _sps

    if method == "spearman":
        return float(_sps.spearmanr(a, b).statistic)
    if method == "kendall":
        return float(_sps.kendalltau(a, b).statistic)
    return float(_sps.pearsonr(a, b).statistic)


def _build_corr_matrix(
    dataset_name: str, columns: list[str], method: str
) -> list[list[float]]:
    """Materialize columns then compute the correlation matrix."""
    con = session.get_connection()
    table = _quote(dataset_name)
    select_cols = ", ".join(_quote(c) for c in columns)
    df = con.execute(f"SELECT {select_cols} FROM {table}").df()
    n = len(columns)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            r = _pairwise_corr(df[columns[i]].to_numpy(), df[columns[j]].to_numpy(), method)
            matrix[i][j] = r
            matrix[j][i] = r
    return matrix
