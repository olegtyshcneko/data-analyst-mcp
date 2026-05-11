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


@pytest.fixture(autouse=True)
def _reset_session_and_recorder() -> Iterator[None]:
    """Wipe the module-level session + recorder state before every test."""
    from data_analyst_mcp import session

    session.reset()
    try:
        from data_analyst_mcp import recorder as _recorder

        _recorder.get_recorder().reset()
    except (ImportError, AttributeError):
        # recorder module may not exist yet during early-phase TDD cycles
        pass
    yield
    session.reset()


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
        # FastMCP returns (content_list, structured_dict). Prefer structured.
        if isinstance(result, tuple) and len(result) == 2:
            content_list, structured = result
            if isinstance(structured, dict) and structured:
                return structured
            content = content_list
        elif hasattr(result, "content"):
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


@pytest.fixture
def load_df_into_session():
    """Register an in-memory pandas DataFrame as a DuckDB table + session entry."""

    def _load(name: str, df: Any) -> None:
        from data_analyst_mcp import session as _session

        con = _session.get_connection()
        con.register(f"__df_{name}", df)
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM __df_{name}')
        con.unregister(f"__df_{name}")
        describe_rows = con.execute(f'DESCRIBE "{name}"').fetchall()
        cols = [{"name": str(r[0]), "dtype": str(r[1])} for r in describe_rows]
        _session.register(
            name=name,
            path="(dataframe)",
            read_options={},
            format="dataframe",
            rows=int(len(df)),
            columns=cols,
        )

    return _load
