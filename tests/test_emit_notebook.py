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
