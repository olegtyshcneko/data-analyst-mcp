"""Tests for the dataset tools (load_dataset, list_datasets, profile, describe)."""

from __future__ import annotations

import os
from typing import Any

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
MESSY_CSV = os.path.join(FIXTURE_DIR, "messy.csv")


def test_load_dataset_rejects_unsupported_extension(call_tool: Any) -> None:
    result = call_tool("load_dataset", {"path": "/tmp/nope.xyz"})

    assert result["ok"] is False
    assert result["error"]["type"] == "unsupported_format"
    assert ".xyz" in result["error"]["message"] or "xyz" in result["error"]["message"]


def test_load_dataset_reports_file_not_found(call_tool: Any) -> None:
    result = call_tool("load_dataset", {"path": "/tmp/does_not_exist_12345.csv"})

    assert result["ok"] is False
    assert result["error"]["type"] == "file_not_found"


def test_load_dataset_registers_csv_with_rows_and_columns(call_tool: Any) -> None:
    result = call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    assert result["ok"] is True
    assert result["name"] == "messy"
    assert result["rows"] == 5000
    # 12 columns per fixture spec.
    assert len(result["columns"]) == 12
    # Each column entry has the canonical shape.
    for col in result["columns"]:
        assert set(col.keys()) >= {"name", "dtype"}


def test_load_dataset_records_markdown_and_code_cell_pair(call_tool: Any) -> None:
    from data_analyst_mcp.recorder import get_recorder

    rec = get_recorder()
    assert rec.cells == []

    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    assert len(rec.cells) == 2
    assert rec.cells[0]["cell_type"] == "markdown"
    assert rec.cells[1]["cell_type"] == "code"
    assert "CREATE OR REPLACE TABLE" in rec.cells[1]["source"]
    assert "messy" in rec.cells[1]["source"]


def test_load_dataset_records_nothing_on_error(call_tool: Any) -> None:
    from data_analyst_mcp.recorder import get_recorder

    rec = get_recorder()
    call_tool("load_dataset", {"path": "/tmp/nope.xyz"})

    assert rec.cells == []


def test_load_dataset_supports_parquet(call_tool: Any, tmp_path: Any) -> None:
    import duckdb

    parquet_path = tmp_path / "tiny.parquet"
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT 1 AS a, 'x' AS b UNION ALL SELECT 2, 'y' UNION ALL SELECT 3, 'z') "
        f"TO '{parquet_path}' (FORMAT PARQUET)"
    )
    con.close()

    result = call_tool("load_dataset", {"path": str(parquet_path), "name": "tiny"})

    assert result["ok"] is True
    assert result["rows"] == 3
    assert {c["name"] for c in result["columns"]} == {"a", "b"}


def test_list_datasets_returns_empty_on_fresh_session(call_tool: Any) -> None:
    result = call_tool("list_datasets", {})

    assert result == {"ok": True, "datasets": []}


def test_profile_dataset_errors_when_name_missing(call_tool: Any) -> None:
    result = call_tool("profile_dataset", {"name": "nope"})

    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_profile_dataset_reports_summary_totals(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    assert result["ok"] is True
    assert result["summary"]["total_rows"] == 5000
    assert result["summary"]["total_columns"] == 12


def test_profile_dataset_reports_null_counts_per_column(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    by_name = {c["name"]: c for c in result["columns"]}
    # email has exactly 3900 empty cells per the fixture spec; DuckDB reads
    # empty CSV cells as NULLs by default.
    assert by_name["email"]["null_count"] == 3900
    # customer_id is never null.
    assert by_name["customer_id"]["null_count"] == 0


def test_profile_dataset_flags_mostly_null_when_over_50_percent(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    by_name = {c["name"]: c for c in result["columns"]}
    # email is 78% null → mostly_null.
    assert by_name["email"]["flags"]["mostly_null"] is True
    # customer_id is 0% null → not mostly_null.
    assert by_name["customer_id"]["flags"]["mostly_null"] is False


def test_profile_dataset_returns_numeric_stats(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    by_name = {c["name"]: c for c in result["columns"]}
    age = by_name["age"]
    assert "numeric" in age
    stats = age["numeric"]
    for key in ("min", "max", "mean", "median", "std", "p25", "p75", "p99",
                "zeros", "negatives"):
        assert key in stats
    # age values are in [18, 80] roughly per the generator.
    assert stats["min"] >= 0
    assert stats["max"] <= 200
    assert stats["zeros"] == 0
    assert stats["negatives"] == 0


def test_profile_dataset_returns_string_stats(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    by_name = {c["name"]: c for c in result["columns"]}
    cust = by_name["customer_id"]
    assert "string" in cust
    s = cust["string"]
    for key in ("min_length", "max_length", "mean_length", "empty_count", "whitespace_count"):
        assert key in s
    # CUST-NNNNNN-NNNN is always 16 chars long.
    assert s["min_length"] == 16
    assert s["max_length"] == 16


def test_profile_dataset_returns_temporal_stats(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    by_name = {c["name"]: c for c in result["columns"]}
    # DuckDB trims the trailing whitespace from the header so the column
    # surfaces as `last_login`.
    login = by_name["last_login"]
    assert "temporal" in login
    t = login["temporal"]
    for key in ("min", "max", "range_days", "null_count", "modal_weekday"):
        assert key in t
    assert t["range_days"] >= 0
    assert t["modal_weekday"] in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


def test_profile_dataset_returns_top5_most_frequent_per_column(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    by_name = {c["name"]: c for c in result["columns"]}
    country = by_name["country"]
    assert "top_values" in country
    top = country["top_values"]
    assert len(top) <= 5
    # Each entry has value + count.
    for item in top:
        assert "value" in item
        assert "count" in item
    # The leading country is one of the high-frequency cases (PL/US/DE/UA).
    assert top[0]["value"] in {"PL", "US", "DE", "UA"}


def test_profile_dataset_emits_heuristic_flags(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    by_name = {c["name"]: c for c in result["columns"]}
    cust = by_name["customer_id"]
    # customer_id is unique → looks_like_id + high_cardinality.
    assert cust["flags"]["looks_like_id"] is True
    assert cust["flags"]["high_cardinality"] is True
    assert cust["flags"]["looks_like_categorical"] is False

    # country is low-cardinality categorical.
    country = by_name["country"]
    assert country["flags"]["looks_like_categorical"] is True
    assert country["flags"]["looks_like_id"] is False

    # last_login is timestamp → looks_like_timestamp.
    login = by_name["last_login"]
    assert login["flags"]["looks_like_timestamp"] is True


def test_profile_dataset_returns_head_sample(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy", "sample_rows": 3})

    assert "head" in result
    assert len(result["head"]) == 3
    # Each row is a dict keyed by column name.
    assert "customer_id" in result["head"][0]


def test_profile_dataset_returns_suggestions(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("profile_dataset", {"name": "messy"})

    assert "suggestions" in result
    assert isinstance(result["suggestions"], list)
    assert 1 <= len(result["suggestions"]) <= 3
    for s in result["suggestions"]:
        assert isinstance(s, str)


def test_describe_column_errors_on_missing_column(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("describe_column", {"name": "messy", "column": "nope"})

    assert result["ok"] is False
    assert result["error"]["type"] == "column_not_found"


def test_describe_column_numeric_returns_quantiles_skew_kurt_iqr(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("describe_column", {"name": "messy", "column": "score"})

    assert result["ok"] is True
    q = result["quantiles"]
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        assert p in q or str(p) in q
    assert "skewness" in result
    assert "kurtosis" in result
    assert "iqr" in result
    assert result["iqr"] > 0


def test_describe_column_numeric_returns_histogram_with_bins(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("describe_column", {"name": "messy", "column": "score", "bins": 10})

    assert "histogram" in result
    hist = result["histogram"]
    assert "bin_edges" in hist
    assert "counts" in hist
    assert len(hist["counts"]) == 10
    assert len(hist["bin_edges"]) == 11
    assert sum(hist["counts"]) > 0


def test_describe_column_numeric_returns_outliers(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("describe_column", {"name": "messy", "column": "score"})

    assert "outliers" in result
    out = result["outliers"]
    assert "iqr_count" in out
    assert "zscore_count" in out
    assert "examples" in out
    # The fixture plants 20 IQR outliers.
    assert out["iqr_count"] >= 20
    assert len(out["examples"]) <= 5


def test_describe_column_categorical_returns_value_counts_and_entropy(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("describe_column", {"name": "messy", "column": "country"})

    assert "value_counts" in result
    vc = result["value_counts"]
    assert isinstance(vc, list)
    # Total values per the fixture has 8-ish distinct values; capped at 50.
    assert len(vc) <= 50
    for item in vc:
        assert "value" in item
        assert "count" in item
    assert "entropy" in result
    assert result["entropy"] > 0


def test_profile_dataset_records_cell_pair_on_success(call_tool: Any) -> None:
    from data_analyst_mcp.recorder import get_recorder

    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})
    rec = get_recorder()
    cells_before = len(rec.cells)

    call_tool("profile_dataset", {"name": "messy"})

    assert len(rec.cells) == cells_before + 2
    assert rec.cells[-2]["cell_type"] == "markdown"
    assert rec.cells[-1]["cell_type"] == "code"
    assert rec.cells[-1]["metadata"]["tool_name"] == "profile_dataset"


def test_list_datasets_reports_registered_entries(call_tool: Any) -> None:
    call_tool("load_dataset", {"path": MESSY_CSV, "name": "messy"})

    result = call_tool("list_datasets", {})

    assert result["ok"] is True
    assert len(result["datasets"]) == 1
    entry = result["datasets"][0]
    assert entry["name"] == "messy"
    assert entry["rows"] == 5000
    assert entry["columns"] == 12
    assert "registered_at" in entry


def test_load_dataset_supports_jsonl(call_tool: Any, tmp_path: Any) -> None:
    p = tmp_path / "tiny.jsonl"
    p.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

    result = call_tool("load_dataset", {"path": str(p), "name": "tinyj"})

    assert result["ok"] is True
    assert result["rows"] == 2
    assert {c["name"] for c in result["columns"]} == {"a", "b"}
