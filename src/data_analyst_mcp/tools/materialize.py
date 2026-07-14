"""Materialize a SELECT/WITH query as a named DuckDB table + session entry."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder
from data_analyst_mcp.tools._sql_safety import contains_unsafe_semicolon, leading_keyword

logger = logging.getLogger(__name__)


class MaterializeQueryInput(BaseModel):
    """Inputs for ``materialize_query``."""

    model_config = ConfigDict(extra="forbid")

    # No min_length on sql: an empty/whitespace string flows to the
    # leading-keyword guard, which rejects it with the typed
    # ``write_not_allowed`` error (consistent with ``query``). A min_length
    # constraint here would raise a pydantic ValidationError on ``("sql",)``
    # that the server wrapper only special-cases for ``name``, leaking the
    # rest as the generic ``internal`` envelope.
    sql: str
    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    overwrite: bool = False


# materialize_query's allowlist is intentionally narrower than ``query``'s.
# ``DESCRIBE`` / ``SHOW`` / ``PRAGMA`` are valid read-only statements but
# yield metadata, not a table source — persisting them as a derived dataset
# is meaningless, so they are rejected with ``write_not_allowed``.
_ALLOWED_LEADING_KEYWORDS = ("SELECT", "WITH")
_META_KEYWORDS = ("DESCRIBE", "SHOW", "PRAGMA", "EXPLAIN")


def materialize_query(payload: MaterializeQueryInput) -> dict[str, Any]:
    """Persist a SELECT/WITH result as a registered derived dataset.

    On success, returns ``{ok, name, rows, columns, total_rows}`` where
    ``columns`` is a list of ``{"name": str, "dtype": str}`` records
    produced by ``DESCRIBE "<name>"`` on the freshly-materialized table.
    When ``payload.overwrite`` is true and ``payload.name`` is already
    registered, the existing entry (loaded *or* derived) is replaced —
    both the DuckDB table (via ``CREATE OR REPLACE TABLE``) and the
    session registry (via ``session.register``, which overwrites by
    name).
    """
    first = leading_keyword(payload.sql)
    if first not in _ALLOWED_LEADING_KEYWORDS:
        if first in _META_KEYWORDS:
            hint = (
                f"{first} returns metadata, not a table source. Wrap the "
                "result in a SELECT or pick a different tool (e.g. `query`)."
            )
        else:
            hint = "Use SELECT or WITH — materialize_query persists the query result as a table."
        return build_error(
            type="write_not_allowed",
            message=f"Statements starting with {first!r} are not allowed.",
            hint=hint,
        )
    # Defence-in-depth: the leading-keyword allowlist doesn't catch
    # multi-statement payloads like ``SELECT 1; DROP TABLE base``. Reject
    # any ``;`` that lives outside string literals / comments / trailing
    # whitespace — see ``_sql_safety.contains_unsafe_semicolon``.
    if contains_unsafe_semicolon(payload.sql):
        return build_error(
            type="write_not_allowed",
            message="Multi-statement SQL is not allowed.",
            hint=(
                "materialize_query accepts a single SELECT or WITH statement. "
                "Remove any embedded `;` followed by another statement."
            ),
        )

    if not payload.overwrite and payload.name in session.get_datasets():
        return build_error(
            type="dataset_name_collision",
            message=f"Dataset {payload.name!r} is already registered.",
            hint="Pass overwrite=True to replace it, or choose a different name.",
        )

    con = session.get_connection()
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{payload.name}" AS {payload.sql}')
    except Exception as exc:
        # DuckDB raises one of several subclasses of duckdb.Error (Catalog,
        # Parser, Binder, …). Any of them maps to query_error so callers
        # see the structured envelope rather than an internal stack trace.
        return build_error(
            type="query_error",
            message=str(exc),
            hint="Check the SQL — verify referenced tables/columns exist and the syntax is valid.",
        )

    rows = int(con.execute(f'SELECT COUNT(*) FROM "{payload.name}"').fetchone()[0])  # type: ignore[index]
    describe_rows = con.execute(f'DESCRIBE "{payload.name}"').fetchall()
    columns = [{"name": str(row[0]), "dtype": str(row[1])} for row in describe_rows]

    # When overwriting a file-backed dataset with a derived query, retain the
    # original file loader so the recorder can re-create the base table at
    # replay time — the derived SQL frequently self-references the same name
    # (transform-in-place: ``... AS SELECT ... FROM data`` over ``data``). If
    # overwriting an already-derived entry, carry its base_loader forward so a
    # chain of overwrites still rehydrates the original file. In-memory
    # (dataframe) bases are not reloadable, so they leave base_loader None —
    # as do split outputs, whose "(split)" path is a placeholder, not a file.
    existing = session.get_datasets().get(payload.name)
    base_loader: dict[str, Any] | None = None
    if existing is not None:
        if existing.format not in ("derived", "dataframe", "split"):
            base_loader = {
                "path": existing.path,
                "format": existing.format,
                "read_options": dict(existing.read_options),
                "source_hash": existing.source_hash,
                "revision": existing.revision,
            }
        elif existing.format == "derived":
            base_loader = existing.base_loader

    # Record split-overwrite provenance on the entry itself: the recorder's
    # replay wrap must work even when the sibling split entry is gone too
    # (double overwrite), so sibling inference is not enough. Chained derived
    # overwrites carry the original provenance forward.
    split_overwrite: dict[str, Any] | None = None
    if existing is not None:
        if existing.format == "split":
            split_overwrite = {
                "side": str(existing.read_options["role"]),
                "source": str(existing.read_options["source"]),
            }
        elif existing.format == "derived":
            split_overwrite = existing.split_overwrite

    session.register(
        name=payload.name,
        path="(query)",
        read_options={"sql": payload.sql},
        format="derived",
        rows=rows,
        columns=columns,
        base_loader=base_loader,
        split_overwrite=split_overwrite,
    )

    md = f"### Materialize query as dataset `{payload.name}`\n\n```sql\n{payload.sql}\n```"
    stmt = f'CREATE OR REPLACE TABLE "{payload.name}" AS {payload.sql}'
    code = f"con.execute({stmt!r})"
    get_recorder().record(markdown=md, code=code, tool_name="materialize_query")

    return {
        "ok": True,
        "name": payload.name,
        "rows": rows,
        "columns": columns,
        "total_rows": rows,
    }
