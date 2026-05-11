"""Tests for the NotebookRecorder."""

from __future__ import annotations


def test_fresh_recorder_has_zero_cells() -> None:
    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()

    assert rec.cells == []


def test_record_appends_one_markdown_and_one_code_cell() -> None:
    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()
    rec.record(
        markdown="### Loaded dataset `raw`",
        code="con.execute('SELECT 1')",
        tool_name="load_dataset",
    )

    assert len(rec.cells) == 2
    md, code = rec.cells
    assert md["cell_type"] == "markdown"
    assert md["source"] == "### Loaded dataset `raw`"
    assert code["cell_type"] == "code"
    assert code["source"] == "con.execute('SELECT 1')"
    # tool_name preserved in metadata for downstream consumers.
    assert code["metadata"]["tool_name"] == "load_dataset"
