"""NotebookRecorder — accumulates markdown+code cell pairs per tool call."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from data_analyst_mcp.read_options import render_read_options_fragment

if TYPE_CHECKING:
    import nbformat


_SETUP_IMPORTS = """\
import duckdb
import hashlib
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

con = duckdb.connect()
"""


_KIND_TO_SMF = {
    "ols": "ols",
    "logistic": "logit",
    "poisson": "poisson",
    "negbin": "negativebinomial",
}


_FORMAT_TO_READER: dict[str, str] = {
    "csv": "read_csv_auto",
    "tsv": "read_csv_auto",
    "parquet": "read_parquet",
    "json": "read_json",
    "jsonl": "read_json",
    "xlsx": "read_xlsx",
}


def _file_load_stmt(
    name: str, fmt: str, path: str, read_options: dict[str, Any] | None = None
) -> str:
    """Build the ``CREATE OR REPLACE TABLE`` line that reloads a file-backed
    dataset from disk via the format-appropriate DuckDB reader.

    ``repr()`` quotes the path safely — embedded ``'`` / ``"`` / ``\"\"\"`` no
    longer break out of the host literal. ``read_options`` is rendered via the
    same fragment builder the live load used, so replay parses identically.
    """
    reader = _FORMAT_TO_READER.get(fmt, "read_csv_auto")
    path_lit = repr(path)
    extra = render_read_options_fragment(read_options or {})
    if reader == "read_csv_auto":
        call = f"{reader}({path_lit}, SAMPLE_SIZE=-1{extra})"
    else:
        call = f"{reader}({path_lit}{extra})"
    return f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {call}"


_IDENT_SANITIZE_RE = re.compile(r"\W")


def _sanitized_guard_var(name: str, idx: int) -> str:
    """Guard-variable stem for a dataset: sanitized name + emission index.

    Dataset names are not validated as Python identifiers, and two names may
    sanitize identically — the index keeps the variables collision-free.
    """
    return f"{_IDENT_SANITIZE_RE.sub('_', name)}_{idx}"


def _hash_guard_lines(var: str, display_name: str, path: str, hash_val: str) -> list[str]:
    """Drift-guard lines emitted before one file-backed dataset reload.

    Three shapes keyed off the stored hash: content assert, ``(path, mtime,
    size)`` fallback assert, or (next cycle) a comment when only a sentinel
    is available. The message is built as data and emitted with ``!r`` so
    quote-containing dataset names cannot break the emitted literal.
    """
    message = f"Source file for dataset {display_name!r} changed since the session was recorded."
    if hash_val.startswith("fallback:"):
        # Above-ceiling files use a (path, mtime, size) fallback. Recompute
        # the same fallback at replay time; the assert remains hard, but the
        # weaker guarantee is documented.
        return [
            "import os as _os",
            f"_st = _os.stat({path!r})",
            f"expected_hash_ds_{var} = {hash_val!r}",
            f"actual_hash_ds_{var} = 'fallback:' + hashlib.sha256("
            f"f'{{{path!r}}}|{{_st.st_mtime}}|{{_st.st_size}}'.encode('utf-8')"
            f").hexdigest()",
            f"assert actual_hash_ds_{var} == expected_hash_ds_{var}, {message!r}",
        ]
    return [
        f"expected_hash_ds_{var} = {hash_val!r}",
        f"actual_hash_ds_{var} = hashlib.sha256(open({path!r}, 'rb').read()).hexdigest()",
        f"assert actual_hash_ds_{var} == expected_hash_ds_{var}, {message!r}",
    ]


def _build_setup_source() -> str:
    """Compose the setup-cell body from the live session registry.

    In-memory datasets (``format == "dataframe"``) have no on-disk path and
    cannot be reloaded inside the notebook — we emit a commented note so the
    reader knows the table was present in the live session but must be
    rematerialized by hand.

    When any models are registered we additionally:

      1. Materialize a ``<name>_df`` DataFrame for every reloadable dataset
         (predict / evaluate recorder cells reference this name).
      2. For each model, emit a SHA-256 assert against the training file
         and a ``smf.<kind>(...).fit(disp=False)`` rehydration line. The
         hash assert is *hard*: silent drift between training data and
         replay is worse than a loud AssertionError.

    Datasets with a sentinel hash (in-memory / unstattable path) skip the
    hash assert and emit a comment instead — there's no file to verify.
    """
    from data_analyst_mcp import session as _session

    lines = [_SETUP_IMPORTS]
    guard_idx = 0
    # First pass: file-backed datasets. Derived datasets are emitted in a
    # second pass so that their CREATE OR REPLACE TABLE lines land *after*
    # the base tables they SELECT from — otherwise DuckDB would fail at
    # replay because the base table wouldn't exist yet.
    for name, entry in _session.get_datasets().items():
        if entry.format == "dataframe":
            lines.append(
                f"# Note: in-memory dataset {name!r} was registered live and is "
                f"not reloaded here — rematerialize it from your own source."
            )
            continue
        if entry.format == "derived":
            # A derived entry that overwrote a file-backed dataset retains the
            # original loader in base_loader; emit it here (first pass) so the
            # second-pass derived CREATE — which may self-reference this same
            # name (transform-in-place) — has its base table at replay.
            base = entry.base_loader
            if base is not None:
                stmt = _file_load_stmt(name, base["format"], base["path"], base.get("read_options"))
                lines.append(f"con.execute({stmt!r})")
            continue
        var = _sanitized_guard_var(name, guard_idx)
        guard_idx += 1
        lines.extend(_hash_guard_lines(var, name, entry.path, entry.source_hash))
        stmt = _file_load_stmt(name, entry.format, entry.path, entry.read_options)
        lines.append(f"con.execute({stmt!r})")

    # Second pass: derived datasets, materialized via their recorded SQL.
    # No hash assert — the recipe is the SQL plus the upstream datasets,
    # which already carry their own asserts via the model rehydration
    # block when models depend on them.
    #
    # Chained derived datasets (derived_b SELECTs FROM derived_a) work
    # because ``_session.get_datasets()`` is a regular dict and Python
    # dicts preserve insertion order — so a derived dataset registered
    # after its upstream derived dataset will also be emitted after it
    # here.
    for name, entry in _session.get_datasets().items():
        if entry.format != "derived":
            continue
        derived_sql = entry.read_options.get("sql", "")
        stmt = f'CREATE OR REPLACE TABLE "{name}" AS {derived_sql}'
        # repr() escapes embedded quotes (including ``\"\"\"``) so the
        # emitted Python source compiles regardless of what SQL the user
        # passed to materialize_query.
        lines.append(f"con.execute({stmt!r})")

    models = _session.get_models()
    if models:
        # Materialize a DataFrame per reloadable dataset so the model
        # rehydration line has something to fit against. Derived datasets
        # (format == "derived") are reloadable too — they exist as DuckDB
        # tables once the second-pass CREATE OR REPLACE TABLE lines above
        # have run.
        for name, entry in _session.get_datasets().items():
            if entry.format == "dataframe":
                continue
            lines.append(f'{name}_df = con.sql("SELECT * FROM {name}").df()')

        for model_name, model_entry in models.items():
            lines.append("")
            lines.append(f"# --- Re-fit model {model_name!r} (kind={model_entry.kind}) ---")
            ds_path = (
                _session.get_datasets()[model_entry.fitted_on_dataset].path
                if model_entry.fitted_on_dataset in _session.get_datasets()
                else None
            )
            hash_val = model_entry.training_dataset_hash
            if (
                ds_path is not None
                and not hash_val.startswith("sentinel:")
                and not hash_val.startswith("fallback:")
            ):
                lines.append(f"expected_hash_{model_name} = {hash_val!r}")
                lines.append(
                    f"actual_hash_{model_name} = hashlib.sha256("
                    f"open({ds_path!r}, 'rb').read()).hexdigest()"
                )
                lines.append(
                    f"assert actual_hash_{model_name} == expected_hash_{model_name}, "
                    f'"Training data for {model_name!r} changed since the session was recorded."'
                )
            elif ds_path is not None and hash_val.startswith("fallback:"):
                # Above-ceiling files use a (path, mtime, size) fallback. Re-
                # compute the same fallback at replay time; the assert remains
                # hard, but the weaker guarantee is documented.
                lines.append("import os as _os")
                lines.append(f"_st = _os.stat({ds_path!r})")
                lines.append(f"expected_hash_{model_name} = {hash_val!r}")
                lines.append(
                    f"actual_hash_{model_name} = 'fallback:' + hashlib.sha256("
                    f"f'{{{ds_path!r}}}|{{_st.st_mtime}}|{{_st.st_size}}'.encode('utf-8')"
                    f").hexdigest()"
                )
                lines.append(
                    f"assert actual_hash_{model_name} == expected_hash_{model_name}, "
                    f'"Training data for {model_name!r} changed since the session was recorded."'
                )
            else:
                lines.append(
                    f"# Note: model {model_name!r} was fit on an in-memory or "
                    f"non-file dataset; no hash assert is possible."
                )

            smf_fn = _KIND_TO_SMF.get(model_entry.kind, "ols")
            fit_args = "disp=False" if model_entry.kind in ("logistic", "poisson", "negbin") else ""
            lines.append(
                f'{model_name} = smf.{smf_fn}("{model_entry.formula}", '
                f"data={model_entry.fitted_on_dataset}_df).fit({fit_args})"
            )
    return "\n".join(lines)


class NotebookRecorder:
    """Records one markdown + one code cell per successful tool call."""

    def __init__(self) -> None:
        self.cells: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Empty the recorded cell list."""
        self.cells.clear()

    def to_notebook(self, include_setup: bool = True) -> nbformat.NotebookNode:
        """Render the recorded cells as a ``nbformat.v4`` notebook.

        When ``include_setup`` is true a setup cell with imports + a DuckDB
        connection + ``CREATE OR REPLACE TABLE`` statements for every
        currently-registered dataset is prepended (added in a later cycle).
        """
        import nbformat as _nbformat

        nb: Any = _nbformat.v4.new_notebook()  # type: ignore[reportUnknownMemberType]
        if include_setup:
            nb.cells.append(
                _nbformat.v4.new_code_cell(_build_setup_source())  # type: ignore[reportUnknownMemberType]
            )
        for cell in self.cells:
            if cell["cell_type"] == "markdown":
                nb.cells.append(
                    _nbformat.v4.new_markdown_cell(cell["source"])  # type: ignore[reportUnknownMemberType]
                )
            else:
                nb.cells.append(
                    _nbformat.v4.new_code_cell(cell["source"])  # type: ignore[reportUnknownMemberType]
                )
        return nb

    def record(self, *, markdown: str, code: str, tool_name: str) -> None:
        """Append one markdown + one code cell describing a tool invocation."""
        self.cells.append(
            {
                "cell_type": "markdown",
                "source": markdown,
                "metadata": {"tool_name": tool_name},
            }
        )
        self.cells.append(
            {
                "cell_type": "code",
                "source": code,
                "metadata": {"tool_name": tool_name},
            }
        )


_recorder = NotebookRecorder()


def get_recorder() -> NotebookRecorder:
    """Return the module-level singleton recorder."""
    return _recorder
