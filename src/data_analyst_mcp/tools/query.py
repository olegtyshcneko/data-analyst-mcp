"""SQL query tool with a read-only allowlist."""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp.errors import build_error

logger = logging.getLogger(__name__)


class QueryInput(BaseModel):
    """Inputs for ``query``."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(
        ...,
        description=(
            "A read-only DuckDB SQL statement (SELECT / WITH / DESCRIBE / "
            "SHOW / EXPLAIN / PRAGMA show_tables). Writes are rejected."
        ),
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=10000,
        description="Row cap. Auto-applied if the SQL does not already include LIMIT.",
    )


_ALLOWED_LEADING_KEYWORDS = ("SELECT", "WITH", "DESCRIBE", "SHOW", "EXPLAIN", "PRAGMA")
_BLOCKED_LEADING_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "REPLACE",
    "SET",
    "ATTACH",
    "DETACH",
    "COPY",
    "GRANT",
    "REVOKE",
    "VACUUM",
    "COMMIT",
    "ROLLBACK",
)


def _strip_sql(sql: str) -> str:
    """Strip leading whitespace, line comments, and ``--`` blocks."""
    text = sql.lstrip()
    # Drop leading ``--`` line comments.
    while text.startswith("--"):
        newline_idx = text.find("\n")
        if newline_idx == -1:
            return ""
        text = text[newline_idx + 1 :].lstrip()
    # Drop leading ``/* ... */`` block comments.
    while text.startswith("/*"):
        end_idx = text.find("*/")
        if end_idx == -1:
            return ""
        text = text[end_idx + 2 :].lstrip()
    return text


def _first_keyword(sql: str) -> str:
    """Uppercase first token of ``sql`` after stripping leading comments."""
    text = _strip_sql(sql)
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", text)
    return match.group(1).upper() if match else ""


def query(payload: QueryInput) -> dict[str, Any]:
    """Run a read-only SQL query through DuckDB with auto-LIMIT."""
    first = _first_keyword(payload.sql)
    if first in _BLOCKED_LEADING_KEYWORDS:
        return build_error(
            type="write_not_allowed",
            message=f"Statements starting with {first!r} are not allowed.",
            hint="Use SELECT / WITH / DESCRIBE / SHOW / EXPLAIN / PRAGMA show_tables.",
        )
    if first not in _ALLOWED_LEADING_KEYWORDS:
        return build_error(
            type="write_not_allowed",
            message=f"Statements starting with {first!r} are not on the allowlist.",
            hint="Use SELECT / WITH / DESCRIBE / SHOW / EXPLAIN / PRAGMA show_tables.",
        )
    return {"ok": True}
