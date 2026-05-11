"""NotebookRecorder — accumulates markdown+code cell pairs per tool call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import nbformat


_SETUP_IMPORTS = """\
import duckdb
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

con = duckdb.connect()
"""


_FORMAT_TO_READER: dict[str, str] = {
    "csv": "read_csv_auto",
    "tsv": "read_csv_auto",
    "parquet": "read_parquet",
    "json": "read_json",
    "jsonl": "read_json",
    "xlsx": "read_xlsx",
}


def _build_setup_source() -> str:
    """Compose the setup-cell body from the live session registry."""
    from data_analyst_mcp import session as _session

    lines = [_SETUP_IMPORTS]
    for name, entry in _session.get_datasets().items():
        reader = _FORMAT_TO_READER.get(entry.format, "read_csv_auto")
        if reader == "read_csv_auto":
            call = f"{reader}('{entry.path}', SAMPLE_SIZE=-1)"
        else:
            call = f"{reader}('{entry.path}')"
        lines.append(f'con.execute("""CREATE OR REPLACE TABLE {name} AS SELECT * FROM {call}""")')
    return "\n".join(lines)


class NotebookRecorder:
    """Records one markdown + one code cell per successful tool call."""

    def __init__(self) -> None:
        self.cells: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Empty the recorded cell list."""
        self.cells.clear()

    def to_notebook(self, include_setup: bool = True) -> nbformat.NotebookNode:
        """Render the recorded cells as a ``nbformat.v4`` notebook.

        When ``include_setup`` is true a setup cell with imports + a DuckDB
        connection + ``CREATE OR REPLACE TABLE`` statements for every
        currently-registered dataset is prepended (added in a later cycle).
        """
        import nbformat as _nbformat

        nb: Any = _nbformat.v4.new_notebook()  # type: ignore[reportUnknownMemberType]
        if include_setup:
            nb.cells.append(
                _nbformat.v4.new_code_cell(_build_setup_source())  # type: ignore[reportUnknownMemberType]
            )
        for cell in self.cells:
            if cell["cell_type"] == "markdown":
                nb.cells.append(
                    _nbformat.v4.new_markdown_cell(cell["source"])  # type: ignore[reportUnknownMemberType]
                )
            else:
                nb.cells.append(
                    _nbformat.v4.new_code_cell(cell["source"])  # type: ignore[reportUnknownMemberType]
                )
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
