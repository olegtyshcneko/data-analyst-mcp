"""Result-formatting helpers (truncation, dict conversion, base64)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb

    from data_analyst_mcp.tools.posthoc import PairwiseComparisonsInput


_METHOD_PRETTY: dict[str, str] = {
    "bonferroni": "Bonferroni",
    "sidak": "Šidák",
    "holm": "Holm",
    "bh": "Benjamini–Hochberg",
    "by": "Benjamini–Yekutieli",
}


def png_to_base64(png_bytes: bytes) -> str:
    """Encode raw PNG bytes as a base64 ASCII string (no data: prefix)."""
    import base64

    return base64.b64encode(png_bytes).decode("ascii")


def rows_to_dicts(rel: duckdb.DuckDBPyRelation) -> list[dict[str, Any]]:
    """Materialize a DuckDB relation as a list of column-name-keyed dicts.

    Goes through the relation's column names + ``fetchall()`` so we never
    rely on pandas dtype coercion for plain row materialization.
    """
    columns = rel.columns
    return [dict(zip(columns, row, strict=True)) for row in rel.fetchall()]


def format_adjust_pvalues_markdown(
    output: dict[str, Any],
    *,
    method: str,
    alpha: float,
) -> str:
    """Render the recorder markdown summary for ``adjust_pvalues``.

    Rules (see proposal §Recorder cells):

    - Title: "Adjusted M p-values via <pretty method> (α=…)".
    - "N / M hypotheses rejected after correction" — always present.
    - "Largest adjusted-p still significant: <value> (`label`)" — only if
      ``n_rejected > 0``. The label is omitted when ``label is None``.
    - "Not rejected: `lbl1`, `lbl2`, `lbl3` and K more" — name up to 3
      non-rejected labels, with an "and K more" suffix beyond that. Skipped
      entirely when every test was rejected, or when there are no rows.
    """
    pretty = _METHOD_PRETTY.get(method, method)
    results: list[dict[str, Any]] = list(output.get("results", []))
    n_tests: int = int(output.get("n_tests", len(results)))
    n_rejected: int = int(output.get("n_rejected", 0))

    lines: list[str] = [f"### Adjusted {n_tests} p-values via {pretty} (α={alpha})"]
    lines.append(f"- {n_rejected} / {n_tests} hypotheses rejected after correction")

    if n_rejected > 0:
        rejected_rows = [r for r in results if r.get("rejected")]
        largest = max(rejected_rows, key=lambda r: float(r["p_adj"]))
        p_adj_val = float(largest["p_adj"])
        label = largest.get("label")
        label_suffix = f" (`{label}`)" if label is not None else ""
        lines.append(f"- Largest adjusted-p still significant: {p_adj_val:.4g}{label_suffix}")

    not_rejected = [r for r in results if not r.get("rejected")]
    if not_rejected:
        named: list[str] = []
        for row in not_rejected[:3]:
            lbl = row.get("label")
            named.append(f"`{lbl}`" if lbl is not None else "(unlabeled)")
        suffix = ""
        extra = len(not_rejected) - 3
        if extra > 0:
            suffix = f" and {extra} more"
        lines.append(f"- Not rejected: {', '.join(named)}{suffix}")

    return "\n".join(lines)


def format_pairwise_comparisons_markdown(
    output: dict[str, Any],
    *,
    payload: PairwiseComparisonsInput,
) -> str:
    """Render the recorder markdown summary for ``pairwise_comparisons``.

    Rules (see proposal §Recorder cells):

    - Title names the engine and the metric/group columns.
    - "Engine: **<engine>** (<adjustment note>)" — Tukey controls FWER
      internally; Dunn reuses ``_METHOD_PRETTY`` for the correction name.
    - "N / M pairs differ at α=…" — the rejected-pair ratio.
    - "Omnibus <test>: statistic=…, p=… (significant | not significant)".
    - "Largest difference: A vs B (<estimate_name>=…)" — the widest-estimate
      pair, skipped when there are no comparisons.
    """
    method = output.get("method")
    alpha = float(output.get("alpha", 0.05))
    comparisons: list[dict[str, Any]] = list(output.get("comparisons", []))
    n_comparisons = int(output.get("n_comparisons", len(comparisons)))
    n_rejected = int(output.get("n_rejected", 0))
    estimate_name = str(output.get("estimate_name", "estimate"))
    omnibus: dict[str, Any] = dict(output.get("omnibus", {}))

    if method == "tukey":
        engine = "Tukey HSD"
        adjust_note = "controls the family-wise error rate internally"
    else:
        engine = "Dunn's test"
        p_adjust = str(output.get("p_adjust") or "holm")
        adjust_note = f"{_METHOD_PRETTY.get(p_adjust, p_adjust)}-adjusted"

    lines: list[str] = [
        f"### Pairwise comparisons — {engine} "
        f"(`{payload.metric_column}` across `{payload.group_column}`)",
        f"- Engine: **{engine}** ({adjust_note})",
        f"- {n_rejected} / {n_comparisons} pairs differ at α={alpha:g}",
    ]

    significance = "significant" if omnibus.get("significant") else "not significant"
    lines.append(
        f"- Omnibus {omnibus.get('test')}: statistic={float(omnibus.get('statistic', 0.0)):.4f}, "
        f"p={float(omnibus.get('p_value', 0.0)):.4f} ({significance})"
    )

    if comparisons:
        largest = max(comparisons, key=lambda c: abs(float(c["estimate"])))
        lines.append(
            f"- Largest difference: {largest['group_a']} vs {largest['group_b']} "
            f"({estimate_name}={float(largest['estimate']):.4f})"
        )

    return "\n".join(lines)


def truncate_rows(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    """Return the standard truncation envelope for a row list.

    Always populates ``rows``, ``total_rows``, ``truncated``, ``cursor`` so
    every row-returning tool can spread this directly into its response.
    """
    total = len(rows)
    if total <= limit:
        return {
            "rows": rows,
            "total_rows": total,
            "truncated": False,
            "cursor": None,
        }
    return {
        "rows": rows[:limit],
        "total_rows": total,
        "truncated": True,
        "cursor": limit,
    }
