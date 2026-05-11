"""SQL query tool with a read-only allowlist."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

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


_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _has_explicit_limit(sql: str) -> bool:
    """True if the SQL already contains a ``LIMIT N`` clause."""
    return bool(_LIMIT_PATTERN.search(sql))


def _apply_limit(sql: str, limit: int) -> str:
    """Append a ``LIMIT limit`` clause when none is present."""
    stripped = sql.rstrip().rstrip(";")
    return stripped + f" LIMIT {int(limit)}"


def _total_rows(con: Any, sql: str) -> int:
    """Run a separate COUNT(*) over the original SQL (without auto-LIMIT)."""
    base = sql.rstrip().rstrip(";")
    count_row = con.execute(f"SELECT COUNT(*) FROM ({base})").fetchone()
    return int(count_row[0]) if count_row else 0


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

    con = session.get_connection()
    auto_limited = not _has_explicit_limit(payload.sql)
    final_sql = _apply_limit(payload.sql, payload.limit) if auto_limited else payload.sql

    started = time.perf_counter()
    cursor = con.execute(final_sql)
    rows_raw = cursor.fetchall()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    columns = [d[0] for d in cursor.description] if cursor.description else []
    rows = [dict(zip(columns, row, strict=True)) for row in rows_raw]

    try:
        total = _total_rows(con, payload.sql)
    except Exception:
        total = len(rows)

    truncated = auto_limited and len(rows) >= payload.limit and total > len(rows)

    md = f"### Query\n\n```\n{payload.sql.strip()}\n```\n\n- {len(rows)} rows returned"
    code = final_sql
    get_recorder().record(markdown=md, code=code, tool_name="query")

    return {
        "ok": True,
        "rows": rows,
        "columns": columns,
        "total_rows": total,
        "execution_time_ms": elapsed_ms,
        "truncated": truncated,
    }
