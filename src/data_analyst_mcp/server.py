"""FastMCP server entry point.

Stdio transport only. Stdout is reserved for the MCP protocol — every log
record goes to stderr via the module-level handler configured below.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

_handler = logging.StreamHandler(stream=sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("data_analyst_mcp")

mcp: FastMCP = FastMCP("data-analyst-mcp")


def main() -> None:  # pragma: no cover - exercised by the console-script smoke test
    """Run the MCP server on stdio. Console-script entry point."""
    logger.info("starting data-analyst-mcp on stdio")
    mcp.run()
