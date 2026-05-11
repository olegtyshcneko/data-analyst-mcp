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
