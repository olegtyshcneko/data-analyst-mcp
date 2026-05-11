"""NotebookRecorder — accumulates markdown+code cell pairs per tool call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import nbformat  # noqa: F401  (used by type-hint of NotebookNode)


class NotebookRecorder:
    """Records one markdown + one code cell per successful tool call."""

    def __init__(self) -> None:
        self.cells: list[dict[str, Any]] = []

    def record(self, *, markdown: str, code: str, tool_name: str) -> None:
        """Stub — records nothing so the cell-count assertion fails."""
        return None


_recorder = NotebookRecorder()


def get_recorder() -> NotebookRecorder:
    """Return the module-level singleton recorder."""
    return _recorder
