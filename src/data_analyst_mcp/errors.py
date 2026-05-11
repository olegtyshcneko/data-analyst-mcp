"""Structured error envelope builder."""

from __future__ import annotations

from typing import Any


def build_error(*, type: str, message: str, hint: str | None = None) -> dict[str, Any]:
    """Return a structured ``{"ok": False, "error": {...}}`` envelope.

    Always wraps the error payload so callers can ``return build_error(...)``
    directly from a tool body. The ``hint`` field is always present (set to
    ``None`` when no hint is provided) so consumers can rely on a stable
    shape.
    """
    return {
        "ok": False,
        "error": {
            "type": type,
            "message": message,
            "hint": hint,
        },
    }
