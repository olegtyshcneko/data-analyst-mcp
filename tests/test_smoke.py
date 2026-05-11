"""Phase 0 smoke test — proves the package imports and the FastMCP instance exists.

Replaced piece-by-piece in Phase 2+ as real TDD-driven tests come online.
"""

from __future__ import annotations


def test_package_version_is_set() -> None:
    import data_analyst_mcp

    assert data_analyst_mcp.__version__ == "0.1.0"


def test_server_module_exposes_fastmcp_instance() -> None:
    from data_analyst_mcp import server

    assert server.mcp.name == "data-analyst-mcp"


def test_main_entry_is_callable() -> None:
    from data_analyst_mcp.server import main

    assert callable(main)
