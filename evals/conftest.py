"""Shared fixtures and helpers for the eval suite.

Every eval spawns a fresh ``data-analyst-mcp`` subprocess via
:func:`mcp.client.stdio.stdio_client` and drives it through an initialized
:class:`mcp.ClientSession`. That subprocess boundary is deliberate — the
protocol path is the contract under test, not the in-process function calls
exercised in ``tests/``.

The session is opened as an ``asynccontextmanager`` inside each test body
rather than as a pytest-asyncio yield-fixture; anyio's cancel-scope
discipline rejects fixture teardown that crosses task boundaries, so the
``async with`` lives in the test's own task.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
CRM_DIR = FIXTURES_DIR / "synthetic_crm"


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="uv",
        args=["run", "data-analyst-mcp"],
        cwd=str(PROJECT_ROOT),
        env=None,
    )


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """Spawn a fresh data-analyst-mcp subprocess and yield an initialized client."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call(session: ClientSession, name: str, args: dict[str, Any] | None = None) -> Any:
    """Invoke ``name`` on the live session and decode the structured payload.

    FastMCP serializes a tool's dict return either as the protocol
    ``structuredContent`` envelope or as a single ``TextContent`` JSON
    string. We accept either and return a plain ``dict``.
    """
    result = await session.call_tool(name, args or {})
    structured: Any = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        # FastMCP wraps non-BaseModel returns under a "result" key.
        if set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        text = getattr(first, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return content


async def load_crm(session: ClientSession) -> dict[str, Any]:
    """Load the three synthetic_crm CSVs into an opened session."""
    loaded: dict[str, Any] = {}
    for name in ("accounts", "contacts", "opportunities"):
        r = await call(
            session,
            "load_dataset",
            {"path": str(CRM_DIR / f"{name}.csv"), "name": name},
        )
        assert r.get("ok"), f"failed to load {name}: {r}"
        loaded[name] = r
    return loaded
