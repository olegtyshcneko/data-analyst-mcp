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
        """Render the recorded cells as a ``nbformat.v4`` notebook.

        When ``include_setup`` is true a setup cell with imports + a DuckDB
        connection + ``CREATE OR REPLACE TABLE`` statements for every
        currently-registered dataset is prepended (added in a later cycle).
        """
        import nbformat as _nbformat

        nb = _nbformat.v4.new_notebook()
        for cell in self.cells:
            if cell["cell_type"] == "markdown":
                nb.cells.append(_nbformat.v4.new_markdown_cell(cell["source"]))
            else:
                nb.cells.append(_nbformat.v4.new_code_cell(cell["source"]))
        return nb

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
