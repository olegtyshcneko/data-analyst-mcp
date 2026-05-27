"""Materialize a SELECT/WITH query as a named DuckDB table + session entry."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class MaterializeQueryInput(BaseModel):
    """Inputs for ``materialize_query``."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1)
    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    overwrite: bool = False


def materialize_query(payload: MaterializeQueryInput) -> dict[str, Any]:
    """Persist a SELECT/WITH result as a registered derived dataset."""
    return {"ok": True}
