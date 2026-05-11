"""Tests for the structured error builder."""

from __future__ import annotations


def test_build_error_returns_canonical_envelope() -> None:
    from data_analyst_mcp.errors import build_error

    err = build_error(type="not_found", message="dataset missing", hint="call load_dataset first")

    assert err == {
        "ok": False,
        "error": {
            "type": "not_found",
            "message": "dataset missing",
            "hint": "call load_dataset first",
        },
    }
