"""NotebookRecorder — accumulates markdown+code cell pairs per tool call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import nbformat


class NotebookRecorder:
    """Records one markdown + one code cell per successful tool call."""

    def __init__(self) -> None:
        self.cells: list[dict[str, Any]] = []

    def to_notebook(self, include_setup: bool = True) -> nbformat.NotebookNode:
        """Stub — returns an empty notebook so the cell-count test fails."""
        import nbformat as _nbformat

        return _nbformat.v4.new_notebook()

    def record(self, *, markdown: str, code: str, tool_name: str) -> None:
        """Append one markdown + one code cell describing a tool invocation."""
        self.cells.append(
            {
                "cell_type": "markdown",
                "source": markdown,
                "metadata": {"tool_name": tool_name},
            }
        )
        self.cells.append(
            {
                "cell_type": "code",
                "source": code,
                "metadata": {"tool_name": tool_name},
            }
        )


_recorder = NotebookRecorder()


def get_recorder() -> NotebookRecorder:
    """Return the module-level singleton recorder."""
    return _recorder
