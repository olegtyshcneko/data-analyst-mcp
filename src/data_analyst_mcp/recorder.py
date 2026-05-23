"""NotebookRecorder — accumulates markdown+code cell pairs per tool call."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    for name, entry in _session.get_datasets().items():
        if entry.format == "dataframe":
            lines.append(
                f"# Note: in-memory dataset {name!r} was registered live and is "
                f"not reloaded here — rematerialize it from your own source."
            )
            continue
        reader = _FORMAT_TO_READER.get(entry.format, "read_csv_auto")
        if reader == "read_csv_auto":
            call = f"{reader}('{entry.path}', SAMPLE_SIZE=-1)"
        else:
            call = f"{reader}('{entry.path}')"
        lines.append(f'con.execute("""CREATE OR REPLACE TABLE {name} AS SELECT * FROM {call}""")')

    models = _session.get_models()
    if models:
        # Materialize a DataFrame per reloadable dataset so the model
        # rehydration line has something to fit against.
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
