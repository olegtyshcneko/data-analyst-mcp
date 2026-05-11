"""NotebookRecorder — accumulates markdown+code cell pairs per tool call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import nbformat


class NotebookRecorder:
    """Stub — exposes cells as a sentinel so the empty-list test fails."""

    cells: list[Any] = [object()]
