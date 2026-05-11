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


def test_to_notebook_with_setup_prepends_imports_and_reload_statements() -> None:
    import nbformat

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    session.register(
        name="raw",
        path="fixtures/messy.csv",
        read_options={},
        format="csv",
        rows=5000,
        columns=[],
    )

    rec = NotebookRecorder()
    rec.record(markdown="### m", code="x = 1", tool_name="t")

    nb = rec.to_notebook(include_setup=True)
    nbformat.validate(nb)

    # First cell is the setup cell.
    setup = nb.cells[0]
    assert setup.cell_type == "code"
    assert "import duckdb" in setup.source
    assert "import pandas as pd" in setup.source
    assert "from scipy import stats" in setup.source
    assert "import statsmodels.api as sm" in setup.source
    assert "import matplotlib.pyplot as plt" in setup.source
    assert "con = duckdb.connect()" in setup.source
    # The reload statement for the `raw` table is rebuilt from the registry.
    assert "CREATE OR REPLACE TABLE raw" in setup.source
    assert "read_csv_auto('fixtures/messy.csv', SAMPLE_SIZE=-1)" in setup.source

    # Recorded cells follow the setup cell.
    assert len(nb.cells) == 3
    assert nb.cells[1].cell_type == "markdown"
    assert nb.cells[2].cell_type == "code"


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
