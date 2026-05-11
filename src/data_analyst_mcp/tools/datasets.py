"""Dataset registration + EDA tools."""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


class LoadDatasetInput(BaseModel):
    """Inputs for ``load_dataset``."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "Local filesystem path or s3:// URL to a CSV, TSV, Parquet, "
            "Excel, JSON, or JSONL file. The extension determines the "
            "DuckDB reader used."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Name to register the dataset under so other tools can refer to "
            "it. When omitted a slugified version of the filename stem is used."
        ),
    )
    read_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional DuckDB reader options forwarded into the read_* call, "
            "e.g. {\"header\": false, \"delim\": \";\"} for CSV. Use this "
            "only when auto-detection produces wrong results."
        ),
    )


_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".json", ".jsonl"}

_EXT_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".xlsx": "xlsx",
    ".json": "json",
    ".jsonl": "jsonl",
}


def _extension(path: str) -> str:
    """Lowercase extension including the leading dot."""
    return os.path.splitext(path)[1].lower()


def _build_read_call(path: str, fmt: str) -> str:
    """Render the DuckDB read_* call used in the CREATE TABLE statement."""
    if fmt == "parquet":
        return f"read_parquet('{path}')"
    if fmt in {"json", "jsonl"}:
        return f"read_json('{path}')"
    return f"read_csv_auto('{path}', SAMPLE_SIZE=-1)"


def list_datasets() -> dict[str, Any]:
    """Return every registered dataset with name, rows, columns, registered_at."""
    return {"ok": True, "datasets": []}


def load_dataset(payload: LoadDatasetInput) -> dict[str, Any]:
    """Register a file as a DuckDB table in the session."""
    ext = _extension(payload.path)
    if ext not in _SUPPORTED_EXTENSIONS:
        return build_error(
            type="unsupported_format",
            message=f"Extension {ext!r} is not a supported tabular format.",
            hint="Use one of .csv, .tsv, .parquet, .xlsx, .json, .jsonl.",
        )
    is_remote = payload.path.startswith(("s3://", "http://", "https://"))
    if not is_remote and not os.path.exists(payload.path):
        return build_error(
            type="file_not_found",
            message=f"No file at {payload.path!r}.",
            hint="Check the path is absolute or relative to the server's cwd.",
        )

    fmt = _EXT_TO_FORMAT[ext]
    name = payload.name or os.path.splitext(os.path.basename(payload.path))[0]
    con = session.get_connection()
    read_call = _build_read_call(payload.path, fmt)
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM {read_call}')

    describe_rows = con.execute(f'DESCRIBE "{name}"').fetchall()
    columns = [{"name": str(row[0]), "dtype": str(row[1])} for row in describe_rows]
    rows = int(con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])  # type: ignore[index]

    session.register(
        name=name,
        path=payload.path,
        read_options=payload.read_options or {},
        format=fmt,
        rows=rows,
        columns=columns,
    )

    md = (
        f"### Loaded dataset `{name}`\n"
        f"- Source: `{payload.path}`\n"
        f"- {rows} rows x {len(columns)} columns"
    )
    code = (
        f'con.execute("""\n'
        f"    CREATE OR REPLACE TABLE {name} AS\n"
        f"    SELECT * FROM {read_call}\n"
        f'""")\n'
        f'{name}_df = con.sql("SELECT * FROM {name}").df()\n'
        f"{name}_df.head()"
    )
    get_recorder().record(markdown=md, code=code, tool_name="load_dataset")

    return {
        "ok": True,
        "name": name,
        "rows": rows,
        "columns": columns,
        "warnings": [],
    }
