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




