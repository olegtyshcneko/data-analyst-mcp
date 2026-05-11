"""FastMCP server entry point.

Stdio transport only. Stdout is reserved for the MCP protocol — every log
record goes to stderr via the module-level handler configured below.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from data_analyst_mcp.errors import build_error
from data_analyst_mcp.tools import datasets as _datasets

_handler = logging.StreamHandler(stream=sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("data_analyst_mcp")

mcp: FastMCP = FastMCP("data-analyst-mcp")


@mcp.tool()
def load_dataset(
    path: str,
    name: str | None = None,
    read_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a file as a named dataset queryable by all other tools.

    The path extension picks the DuckDB reader (``.csv``/``.tsv``/``.parquet``/
    ``.json``/``.jsonl``/``.xlsx``). ``read_options`` is forwarded into the
    reader for cases where auto-detection fails (e.g. semicolon-delimited
    files). Reports row count, column names + dtypes, file size, and any
    parser warnings.
    """
    try:
        payload = _datasets.LoadDatasetInput(
            path=path, name=name, read_options=read_options
        )
        return _datasets.load_dataset(payload)
    except Exception as exc:  # pragma: no cover - tools must not raise
        logger.exception("load_dataset failed")
        return build_error(type="internal", message=str(exc))


def main() -> None:  # pragma: no cover - exercised by the console-script smoke test
    """Run the MCP server on stdio. Console-script entry point."""
    logger.info("starting data-analyst-mcp on stdio")
    mcp.run()
