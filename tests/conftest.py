"""Shared test fixtures.

The `server` fixture is the FastMCP instance from `data_analyst_mcp.server`.
The `call_tool` helper invokes tools through the registered FastMCP entry —
not by direct function import — so tests exercise the same input-validation,
serialization, and error-formatting path that the live MCP stdio transport
uses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def server() -> Iterator[Any]:
    """Yield the FastMCP instance with a fresh per-test session."""
    from data_analyst_mcp import server as server_module

    yield server_module.mcp


@pytest.fixture
def call_tool(server: Any):
    """Call a registered MCP tool by name with a dict of arguments.

    Returns the parsed dict the tool produced (FastMCP serializes to text by
    default; we decode JSON on the way out so tests can assert on the
    structured shape).
    """
    import json

    async def _invoke(name: str, args: dict[str, Any]) -> Any:
        result = await server.call_tool(name, args)
        if hasattr(result, "content"):
            content = result.content
        else:
            content = result
        if isinstance(content, list) and content:
            first = content[0]
            text = getattr(first, "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        if isinstance(content, dict):
            return content
        return content

    def _call(name: str, args: dict[str, Any] | None = None) -> Any:
        return asyncio.run(_invoke(name, args or {}))

    return _call
