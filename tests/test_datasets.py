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
