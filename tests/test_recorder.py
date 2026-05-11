"""Tests for the NotebookRecorder."""

from __future__ import annotations


def test_fresh_recorder_has_zero_cells() -> None:
    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()

    assert rec.cells == []
