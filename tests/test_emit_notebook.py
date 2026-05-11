"""Tests for the ``emit_notebook`` tool.

The acceptance criterion the project lives or dies on is the integration
test in :func:`test_six_step_workflow_round_trip` — it emits a real session
to disk and re-executes it via ``jupyter nbconvert``. Everything else is
scaffolding around that single observable behavior.
"""

from __future__ import annotations

from pathlib import Path


def test_emit_notebook_on_empty_session_writes_a_valid_notebook(
    call_tool, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    r = call_tool("emit_notebook", {})
    assert r["ok"] is True
    import nbformat as nbf

    nb = nbf.read(r["path"], as_version=4)
    nbf.validate(nb)


def test_emit_notebook_returns_absolute_path(call_tool, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = call_tool("emit_notebook", {"path": "relative.ipynb"})
    assert r["ok"] is True
    assert Path(r["path"]).is_absolute()


def test_emit_notebook_reports_n_cells_matching_file(call_tool, tmp_path):
    target = tmp_path / "out.ipynb"
    r = call_tool("emit_notebook", {"path": str(target)})
    assert r["ok"] is True
    import nbformat as nbf

    nb = nbf.read(r["path"], as_version=4)
    assert r["n_cells"] == len(nb.cells)
    # Fresh recorder + setup cell = exactly one cell.
    assert r["n_cells"] == 1


def test_emit_notebook_errors_when_parent_directory_missing(call_tool, tmp_path):
    target = tmp_path / "does_not_exist" / "out.ipynb"
    r = call_tool("emit_notebook", {"path": str(target)})
    assert r["ok"] is False
    assert r["error"]["type"] == "write_failed"


def test_emit_notebook_errors_when_path_is_a_directory(call_tool, tmp_path):
    # Passing an existing directory as `path` should fail cleanly.
    r = call_tool("emit_notebook", {"path": str(tmp_path)})
    assert r["ok"] is False
    assert r["error"]["type"] == "write_failed"


def _read_nb(path: str):
    import nbformat as nbf

    return nbf.read(path, as_version=4)


def test_setup_cell_contains_canonical_imports(call_tool, tmp_path):
    r = call_tool("emit_notebook", {"path": str(tmp_path / "out.ipynb")})
    assert r["ok"] is True
    nb = _read_nb(r["path"])
    setup = nb.cells[0]
    assert setup.cell_type == "code"
    src = setup.source
    for expected in (
        "import duckdb",
        "import pandas as pd",
        "import numpy as np",
        "from scipy import stats",
        "import statsmodels.api as sm",
        "import matplotlib.pyplot as plt",
    ):
        assert expected in src, f"missing canonical import: {expected!r}"


def test_setup_cell_creates_duckdb_connection_exactly_once(call_tool, tmp_path):
    r = call_tool("emit_notebook", {"path": str(tmp_path / "out.ipynb")})
    assert r["ok"] is True
    setup_src = _read_nb(r["path"]).cells[0].source
    assert setup_src.count("con = duckdb.connect()") == 1


def test_setup_cell_reloads_each_registered_dataset(call_tool, tmp_path):
    import duckdb

    # one csv (already on disk) + one parquet (generated via DuckDB to avoid
    # pulling pyarrow into our dev dependencies just for this test)
    parquet = tmp_path / "tiny.parquet"
    duckdb.sql(f"COPY (SELECT 1 AS a, 4 AS b UNION ALL SELECT 2, 5) TO '{parquet}' (FORMAT PARQUET)")
    csv = Path(__file__).parent.parent / "fixtures" / "messy.csv"

    r = call_tool("load_dataset", {"path": str(csv), "name": "raw"})
    assert r["ok"]
    r = call_tool("load_dataset", {"path": str(parquet), "name": "tiny"})
    assert r["ok"]

    r = call_tool("emit_notebook", {"path": str(tmp_path / "out.ipynb")})
    assert r["ok"]
    setup_src = _read_nb(r["path"]).cells[0].source

    assert "CREATE OR REPLACE TABLE raw AS" in setup_src
    assert "read_csv_auto(" in setup_src
    assert "CREATE OR REPLACE TABLE tiny AS" in setup_src
    assert "read_parquet(" in setup_src


def test_setup_cell_skips_in_memory_datasets(call_tool, tmp_path, load_df_into_session):
    import pandas as pd

    load_df_into_session("inmem", pd.DataFrame({"x": [1, 2, 3]}))
    r = call_tool("emit_notebook", {"path": str(tmp_path / "out.ipynb")})
    assert r["ok"]
    setup_src = _read_nb(r["path"]).cells[0].source
    # No reload line for in-memory datasets — they have no path to read from.
    assert "CREATE OR REPLACE TABLE inmem" not in setup_src
    # We do leave a hint so the reader knows the table existed in the live session.
    assert "inmem" in setup_src


def test_cells_appear_in_recording_order_with_setup_first(call_tool, tmp_path):
    csv = Path(__file__).parent.parent / "fixtures" / "messy.csv"
    call_tool("load_dataset", {"path": str(csv), "name": "raw"})
    call_tool("query", {"sql": "SELECT COUNT(*) AS n FROM raw"})
    r = call_tool("emit_notebook", {"path": str(tmp_path / "out.ipynb")})
    assert r["ok"]
    cells = _read_nb(r["path"]).cells
    # setup, md(load), code(load), md(query), code(query)  => 5 cells
    assert [c.cell_type for c in cells] == ["code", "markdown", "code", "markdown", "code"]
    assert "Loaded dataset" in cells[1].source
    assert "Query" in cells[3].source


def test_three_recorded_operations_produce_seven_cells(call_tool, tmp_path):
    csv = Path(__file__).parent.parent / "fixtures" / "messy.csv"
    call_tool("load_dataset", {"path": str(csv), "name": "raw"})
    call_tool("query", {"sql": "SELECT COUNT(*) AS n FROM raw"})
    call_tool("query", {"sql": "SELECT COUNT(*) AS n FROM raw WHERE score IS NOT NULL"})
    r = call_tool("emit_notebook", {"path": str(tmp_path / "out.ipynb")})
    assert r["ok"]
    # 1 setup + 3 markdown + 3 code = 7
    assert r["n_cells"] == 7


def test_include_outputs_is_a_documented_noop(call_tool, tmp_path):
    csv = Path(__file__).parent.parent / "fixtures" / "messy.csv"
    call_tool("load_dataset", {"path": str(csv), "name": "raw"})
    a = tmp_path / "a.ipynb"
    b = tmp_path / "b.ipynb"
    call_tool("emit_notebook", {"path": str(a), "include_outputs": False})
    call_tool("emit_notebook", {"path": str(b), "include_outputs": True})
    nb_a = _read_nb(str(a))
    nb_b = _read_nb(str(b))
    code_a = [c for c in nb_a.cells if c.cell_type == "code"]
    code_b = [c for c in nb_b.cells if c.cell_type == "code"]
    # Outputs are never captured at record time, so both flavours emit empty
    # outputs lists on every code cell.
    for cell in code_a + code_b:
        assert cell.outputs == []




