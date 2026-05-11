"""Result-formatting helpers (truncation, dict conversion, base64)."""

from __future__ import annotations

from typing import Any


def truncate_rows(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    """Return the standard truncation envelope for a row list.

    Always populates ``rows``, ``total_rows``, ``truncated``, ``cursor`` so
    every row-returning tool can spread this directly into its response.
    """
    total = len(rows)
    if total <= limit:
        return {
            "rows": rows,
            "total_rows": total,
            "truncated": False,
            "cursor": None,
        }
    return {
        "rows": rows[:limit],
        "total_rows": total,
        "truncated": True,
        "cursor": limit,
    }
