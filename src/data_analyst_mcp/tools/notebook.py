"""Notebook emission tool — serialize the recorded session to .ipynb."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


def _nbf() -> Any:
    """Return ``nbformat`` as an untyped module so strict pyright stays clean."""
    import nbformat as _nbformat

    return _nbformat


class EmitNotebookInput(BaseModel):
    """Inputs for ``emit_notebook``."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        description=(
            "Destination .ipynb path. When omitted the notebook is written to "
            "./session_<YYYYmmdd_HHMMSS>.ipynb in the current working directory."
        ),
    )
    include_outputs: bool = Field(
        default=False,
        description=(
            "API-stability placeholder: cell outputs are never recorded during "
            "tool calls, so the emitted cells always have empty outputs lists. "
            "Re-execute the notebook with ``jupyter nbconvert --execute`` to "
            "populate outputs."
        ),
    )


def _default_path() -> str:
    """Default destination: ``./session_<timestamp>.ipynb`` in the cwd."""
    return datetime.now().strftime("session_%Y%m%d_%H%M%S.ipynb")


def emit_notebook(payload: EmitNotebookInput) -> dict[str, Any]:
    """Serialize the recorded session to a runnable .ipynb file."""
    nbf = _nbf()
    nb: Any = get_recorder().to_notebook(include_setup=True)
    target = payload.path if payload.path is not None else _default_path()
    with open(target, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)  # type: ignore[reportUnknownMemberType]
    return {"ok": True, "path": target, "n_cells": len(nb.cells)}
