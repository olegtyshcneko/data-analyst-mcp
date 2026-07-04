"""Tests for the result-formatting helpers."""

from __future__ import annotations

import base64

import pytest


def test_truncate_rows_under_limit_returns_full_rows() -> None:
    from data_analyst_mcp.formatting import truncate_rows

    rows = [{"a": 1}, {"a": 2}, {"a": 3}]
    out = truncate_rows(rows, limit=5)

    assert out["rows"] == rows
    assert out["total_rows"] == 3
    assert out["truncated"] is False
    assert out["cursor"] is None


def test_truncate_rows_over_limit_truncates_with_cursor() -> None:
    from data_analyst_mcp.formatting import truncate_rows

    rows = [{"i": i} for i in range(10)]
    out = truncate_rows(rows, limit=3)

    assert out["rows"] == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert out["total_rows"] == 10
    assert out["truncated"] is True
    assert out["cursor"] == 3


def test_rows_to_dicts_converts_duckdb_relation_to_list_of_dicts() -> None:
    import duckdb

    from data_analyst_mcp.formatting import rows_to_dicts

    con = duckdb.connect()
    rel = con.sql("SELECT 1 AS a, 'x' AS b UNION ALL SELECT 2, 'y' ORDER BY a")

    out = rows_to_dicts(rel)

    assert out == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_png_to_base64_round_trips() -> None:
    from data_analyst_mcp.formatting import png_to_base64

    raw = bytes(range(256))
    encoded = png_to_base64(raw)

    assert isinstance(encoded, str)
    assert base64.b64decode(encoded) == raw
    # Sanity: pytest.approx is unused here intentionally to keep this exact.
    _ = pytest  # keep import non-unused for downstream tests


def _pairwise_payload(**overrides):
    from data_analyst_mcp.tools.posthoc import PairwiseComparisonsInput

    kwargs = {"name": "ds", "group_column": "grp", "metric_column": "val"}
    kwargs.update(overrides)
    return PairwiseComparisonsInput(**kwargs)


def test_format_pairwise_comparisons_markdown_tukey() -> None:
    from data_analyst_mcp.formatting import format_pairwise_comparisons_markdown

    payload = _pairwise_payload(method="tukey")
    output = {
        "ok": True,
        "method": "tukey",
        "p_adjust": None,
        "alpha": 0.05,
        "estimate_name": "mean_diff",
        "omnibus": {
            "test": "anova",
            "statistic": 8.0,
            "p_value": 0.0061963978,
            "significant": True,
        },
        "comparisons": [
            {"group_a": "A", "group_b": "B", "estimate": 2.0, "reject": False},
            {"group_a": "A", "group_b": "C", "estimate": 4.0, "reject": True},
            {"group_a": "B", "group_b": "C", "estimate": 2.0, "reject": False},
        ],
        "n_comparisons": 3,
        "n_rejected": 1,
    }

    md = format_pairwise_comparisons_markdown(output, payload=payload)

    assert "Tukey HSD" in md
    assert "1 / 3 pairs" in md
    assert "α=0.05" in md
    # Omnibus line names the test.
    assert "anova" in md
    # Largest difference names the widest-estimate pair with the estimate name.
    assert "A vs C" in md
    assert "mean_diff=4.0000" in md


def test_format_pairwise_comparisons_markdown_dunn_uses_method_pretty() -> None:
    from data_analyst_mcp.formatting import format_pairwise_comparisons_markdown

    payload = _pairwise_payload(method="dunn", p_adjust="bh")
    output = {
        "ok": True,
        "method": "dunn",
        "p_adjust": "bh",
        "alpha": 0.01,
        "estimate_name": "mean_rank_diff",
        "omnibus": {
            "test": "kruskal_wallis",
            "statistic": 0.3239344262,
            "p_value": 0.8504690883,
            "significant": False,
        },
        "comparisons": [
            {"group_a": "A", "group_b": "B", "estimate": 1.0, "reject": False},
            {"group_a": "A", "group_b": "C", "estimate": 3.0, "reject": False},
            {"group_a": "B", "group_b": "C", "estimate": 2.0, "reject": False},
        ],
        "n_comparisons": 3,
        "n_rejected": 0,
    }

    md = format_pairwise_comparisons_markdown(output, payload=payload)

    assert "Dunn's test" in md
    assert "0 / 3 pairs" in md
    # Reuses _METHOD_PRETTY for the correction name.
    assert "Benjamini–Hochberg" in md
    # A non-significant omnibus is flagged.
    assert "not significant" in md
