"""SQL query tool with a read-only allowlist."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


def query(payload: QueryInput) -> dict[str, Any]:
    """Stub — always returns ok=True so the write-not-allowed test fails."""
    return {"ok": True}
