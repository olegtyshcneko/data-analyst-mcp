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


