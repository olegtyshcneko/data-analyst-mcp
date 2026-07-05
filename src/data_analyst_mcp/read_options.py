"""Rendering of DuckDB ``read_*`` option fragments.

Shared by the live loader (``tools.datasets``) and the notebook recorder,
so the emitted setup cell reproduces exactly the options the live load
used. Keys are identifier-validated and values rendered as DuckDB
literals — SQL injection via the option dict stays impossible.
"""

from __future__ import annotations

import re
from typing import Any, cast

_READ_OPTION_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _render_read_option(value: Any) -> str:
    """Render a Python value as a DuckDB literal for the reader option list.

    Bools → ``TRUE``/``FALSE``; ints/floats unchanged; strings single-quoted
    with embedded quotes doubled; lists rendered as ``[a, b, c]`` (used for
    ``names``, ``columns``, etc.). Unknown shapes raise — the caller surfaces
    them as a ``bad_read_option`` error rather than producing broken SQL.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, list):
        items = cast(list[Any], value)
        return "[" + ", ".join(_render_read_option(v) for v in items) + "]"
    raise TypeError(f"unsupported read_option value type: {type(value).__name__}")


def render_read_options_fragment(options: dict[str, Any]) -> str:
    """Render ``read_options`` as a leading ``, key=value, ...`` fragment.

    Keys must be identifier-like (``[A-Za-z_][A-Za-z0-9_]*``) to keep SQL
    injection impossible via the option dict. Returns an empty string when
    ``options`` is empty.
    """
    if not options:
        return ""
    parts: list[str] = []
    for key, val in options.items():
        if not _READ_OPTION_KEY_RE.match(key):
            raise ValueError(f"read_options key {key!r} is not a valid identifier")
        parts.append(f"{key}={_render_read_option(val)}")
    return ", " + ", ".join(parts)
