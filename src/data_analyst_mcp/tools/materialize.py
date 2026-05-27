"""Materialize a SELECT/WITH query as a named DuckDB table + session entry."""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error

logger = logging.getLogger(__name__)


class MaterializeQueryInput(BaseModel):
    """Inputs for ``materialize_query``."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1)
    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    overwrite: bool = False


# materialize_query's allowlist is intentionally narrower than ``query``'s.
# ``DESCRIBE`` / ``SHOW`` / ``PRAGMA`` are valid read-only statements but
# yield metadata, not a table source — persisting them as a derived dataset
# is meaningless, so they are rejected with ``write_not_allowed``.
_ALLOWED_LEADING_KEYWORDS = ("SELECT", "WITH")
_META_KEYWORDS = ("DESCRIBE", "SHOW", "PRAGMA", "EXPLAIN")


def _first_keyword(sql: str) -> str:
    """Uppercase first token of ``sql`` (no comment stripping)."""
    match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)", sql)
    return match.group(1).upper() if match else ""


def materialize_query(payload: MaterializeQueryInput) -> dict[str, Any]:
    """Persist a SELECT/WITH result as a registered derived dataset.

    On success, returns ``{ok, name, rows, columns, total_rows}`` where
    ``columns`` is a list of ``{"name": str, "dtype": str}`` records
    produced by ``DESCRIBE "<name>"`` on the freshly-materialized table.
    """
    first = _first_keyword(payload.sql)
    if first not in _ALLOWED_LEADING_KEYWORDS:
        if first in _META_KEYWORDS:
            hint = (
                f"{first} returns metadata, not a table source. Wrap the "
                "result in a SELECT or pick a different tool (e.g. `query`)."
            )
        else:
            hint = (
                "Use SELECT or WITH — materialize_query persists the query "
                "result as a table."
            )
        return build_error(
            type="write_not_allowed",
            message=f"Statements starting with {first!r} are not allowed.",
            hint=hint,
        )

    if not payload.overwrite and payload.name in session.get_datasets():
        return build_error(
            type="dataset_name_collision",
            message=f"Dataset {payload.name!r} is already registered.",
            hint="Pass overwrite=True to replace it, or choose a different name.",
        )

    con = session.get_connection()
    con.execute(f'CREATE OR REPLACE TABLE "{payload.name}" AS {payload.sql}')

    rows = int(con.execute(f'SELECT COUNT(*) FROM "{payload.name}"').fetchone()[0])  # type: ignore[index]
    describe_rows = con.execute(f'DESCRIBE "{payload.name}"').fetchall()
    columns = [{"name": str(row[0]), "dtype": str(row[1])} for row in describe_rows]

    session.register(
        name=payload.name,
        path="(query)",
        read_options={"sql": payload.sql},
        format="derived",
        rows=rows,
        columns=columns,
    )

    return {
        "ok": True,
        "name": payload.name,
        "rows": rows,
        "columns": columns,
        "total_rows": rows,
    }
