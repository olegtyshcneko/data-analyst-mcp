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


def test_to_notebook_without_setup_returns_recorded_cells() -> None:
    import nbformat

    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()
    rec.record(markdown="### M", code="x = 1", tool_name="load_dataset")

    nb = rec.to_notebook(include_setup=False)

    assert isinstance(nb, nbformat.NotebookNode)
    nbformat.validate(nb)
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == "markdown"
    assert nb.cells[0].source == "### M"
    assert nb.cells[1].cell_type == "code"
    assert nb.cells[1].source == "x = 1"
