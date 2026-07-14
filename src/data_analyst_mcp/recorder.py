"""NotebookRecorder — accumulates markdown+code cell pairs per tool call."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from data_analyst_mcp.read_options import render_read_options_fragment

if TYPE_CHECKING:
    import nbformat

    from data_analyst_mcp.session import DatasetEntry


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


def _file_select_expr(fmt: str, path: str, read_options: dict[str, Any] | None = None) -> str:
    """``SELECT * FROM <reader>(...)`` expression for a file-backed source.

    ``repr()`` quotes the path safely — embedded ``'`` / ``"`` / ``\"\"\"`` no
    longer break out of the host literal. ``read_options`` is rendered via the
    same fragment builder the live load used, so replay parses identically.
    """
    reader = _FORMAT_TO_READER.get(fmt, "read_csv_auto")
    path_lit = repr(path)
    extra = render_read_options_fragment(read_options or {})
    if reader == "read_csv_auto":
        return f"SELECT * FROM {reader}({path_lit}, SAMPLE_SIZE=-1{extra})"
    return f"SELECT * FROM {reader}({path_lit}{extra})"


def _file_load_stmt(
    name: str, fmt: str, path: str, read_options: dict[str, Any] | None = None
) -> str:
    """Build the ``CREATE OR REPLACE TABLE`` line that reloads a file-backed
    dataset from disk via the format-appropriate DuckDB reader.
    """
    return f"CREATE OR REPLACE TABLE {name} AS {_file_select_expr(fmt, path, read_options)}"


def _split_assignment_lines(
    source: str, seed: int, test_fraction: float, stratify_by: str | None
) -> list[str]:
    """Notebook lines that rebuild the boolean membership array.

    Algorithm-identical to ``tools.split._assign_is_test`` — the
    membership-checksum assert emitted below is the drift guard between
    the two implementations.
    """
    src_q = '"' + source.replace('"', '""') + '"'
    if stratify_by is None:
        return [
            f"_split_n = con.sql('SELECT COUNT(*) FROM {src_q}').fetchone()[0]",
            f"_split_rng = np.random.RandomState({seed})",
            "_split_is_test = np.zeros(_split_n, dtype=bool)",
            f"_split_n_test = min(max(int(round(_split_n * {test_fraction})), 1), _split_n - 1)",
            "_split_is_test[_split_rng.permutation(_split_n)[:_split_n_test]] = True",
        ]
    col_q = '"' + stratify_by.replace('"', '""') + '"'
    return [
        f"_split_labels = con.sql('SELECT {col_q} FROM {src_q}').df().iloc[:, 0]",
        f"_split_rng = np.random.RandomState({seed})",
        "_split_is_test = np.zeros(len(_split_labels), dtype=bool)",
        "_split_null = _split_labels.isna().to_numpy()",
        "_split_values = sorted(_split_labels[~_split_labels.isna()].unique().tolist(), key=str)",
        "_split_groups = [np.where((_split_labels == _v).to_numpy() & ~_split_null)[0] for _v in _split_values]",
        "if _split_null.any():",
        "    _split_groups.append(np.where(_split_null)[0])",
        "for _rids in _split_groups:",
        "    if len(_rids) < 2:",
        "        continue",
        f"    _n_t = min(max(int(round(len(_rids) * {test_fraction})), 1), len(_rids) - 1)",
        "    _split_is_test[_rids[_split_rng.permutation(len(_rids))[:_n_t]]] = True",
    ]


_SPLIT_CHECKSUM_DEF = """\
def _split_checksum(_df):
    import hashlib as _hl
    import math as _math
    _acc_xor = 0
    _acc_sum = 0
    _n_rows = 0
    for _row in _df.itertuples(index=False, name=None):
        _parts = []
        for _v in _row:
            try:
                _is_na = bool(pd.isna(_v))
            except (TypeError, ValueError):
                _is_na = False
            if _v is None or _is_na:
                _parts.append('<null>')
            elif isinstance(_v, (bool, np.bool_)):
                _parts.append('true' if bool(_v) else 'false')
            elif isinstance(_v, (float, np.floating)):
                _f = float(_v)
                _parts.append('<null>' if _math.isnan(_f) else repr(_f))
            elif isinstance(_v, (int, np.integer)):
                _parts.append(repr(int(_v)))
            elif isinstance(_v, str):
                _parts.append(repr(_v))
            else:
                _parts.append(str(_v))
        _h = _hl.sha256('|'.join(_parts).encode('utf-8')).digest()
        _row_hash = int.from_bytes(_h[:16], 'big')
        _acc_xor ^= _row_hash
        _acc_sum = (_acc_sum + _row_hash) % (1 << 128)
        _n_rows += 1
    return f'{_n_rows:x}:{_acc_xor:032x}:{_acc_sum:032x}'"""


def split_replay_source(
    *,
    source: str,
    train_name: str,
    test_name: str,
    seed: int,
    test_fraction: float,
    stratify_by: str | None,
    rid_column: str,
    membership_checksum: str,
    include_train: bool = True,
) -> str:
    """Self-contained notebook snippet that recreates a train/test split.

    Rebuilds membership with the same ``RandomState`` algorithm the live
    tool used, recreates both tables, then asserts the order-independent
    membership checksum of the recreated test table. For file-backed
    sources the source hash assert upstream makes this deterministic; for
    derived sources whose SQL is not order-preserving, row-order drift
    fails loudly here instead of silently changing the split (spec §5.6b
    row-order tiers).

    ``include_train`` (setup-cell only): when the train side was later
    overwritten by ``materialize_query`` its split recipe is gone, so the
    train ``CREATE`` is skipped — re-creating it here would clobber the
    derived table the second pass already built. The per-call cell keeps
    the default (both sides are fresh at call time); the test ``CREATE`` and
    the membership checksum assert are always emitted.
    """
    src_q = '"' + source.replace('"', '""') + '"'
    train_q = '"' + train_name.replace('"', '""') + '"'
    test_q = '"' + test_name.replace('"', '""') + '"'
    rid_q = '"' + rid_column.replace('"', '""') + '"'
    lines = [
        f"# --- split_dataset: {source} -> {train_name} / {test_name} "
        f"(seed={seed}, test_fraction={test_fraction}) ---",
    ]
    lines.extend(_split_assignment_lines(source, seed, test_fraction, stratify_by))
    base = (
        f"SELECT s.* EXCLUDE ({rid_q}) FROM "
        f"(SELECT *, row_number() OVER () - 1 AS {rid_q} FROM {src_q}) s "
        f"JOIN __data_analyst_split_assign a ON s.{rid_q} = a.rid"
    )
    train_stmt = f"CREATE OR REPLACE TABLE {train_q} AS {base} WHERE NOT a.is_test"
    test_stmt = f"CREATE OR REPLACE TABLE {test_q} AS {base} WHERE a.is_test"
    message = f"Split membership for {test_name!r} drifted at replay (source row order changed)."
    exec_lines = [
        "_split_assign = pd.DataFrame({'rid': np.arange(len(_split_is_test), "
        "dtype=np.int64), 'is_test': _split_is_test})",
        "con.register('__data_analyst_split_assign', _split_assign)",
    ]
    if include_train:
        exec_lines.append(f"con.execute({train_stmt!r})")
    exec_lines.extend(
        [
            f"con.execute({test_stmt!r})",
            "con.unregister('__data_analyst_split_assign')",
            _SPLIT_CHECKSUM_DEF,
            f"assert _split_checksum(con.sql('SELECT * FROM {test_q}').df()) == "
            f"{membership_checksum!r}, {message!r}",
        ]
    )
    lines.extend(exec_lines)
    return "\n".join(lines)


_IDENT_SANITIZE_RE = re.compile(r"\W")


def _sanitized_guard_var(name: str, idx: int) -> str:
    """Guard-variable stem for a dataset: sanitized name + emission index.

    Dataset names are not validated as Python identifiers, and two names may
    sanitize identically — the index keeps the variables collision-free.
    """
    return f"{_IDENT_SANITIZE_RE.sub('_', name)}_{idx}"


def _hash_guard_lines(var: str, display_name: str, path: str, hash_val: str) -> list[str]:
    """Drift-guard lines emitted before one file-backed dataset reload.

    Three shapes keyed off the stored hash: a content assert, a ``(path,
    mtime, size)`` fallback recompute assert, or a comment when only a
    sentinel is available. The message is built as data and emitted with
    ``!r`` so quote-containing dataset names cannot break the emitted
    literal.
    """
    if not hash_val or hash_val.startswith("sentinel:"):
        return [
            f"# Note: dataset {display_name!r} has no verifiable source hash; reload is unguarded."
        ]
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


def _split_side_overwritten(name: str, datasets: dict[str, DatasetEntry]) -> tuple[str, str] | None:
    """Did the derived entry ``name`` overwrite one side of a still-live split?

    A ``materialize_query`` overwrite of a split output turns that side into a
    ``format == "derived"`` entry while its sibling stays ``format == "split"``.
    The surviving sibling still points back at the overwritten name, so the
    overwrite is detectable from the live registry alone:

    - a surviving ``role == "test"`` split entry whose ``train_name == name``
      means ``name`` overwrote the **train** side of that split;
    - a surviving ``role == "train"`` split entry whose ``test_name == name``
      means ``name`` overwrote the **test** side.

    Returns ``(side, source)`` — ``side`` is ``"train"``/``"test"`` and
    ``source`` is the surviving split entry's original source dataset — or
    ``None`` when ``name`` did not overwrite a split side (an ordinary derived
    dataset), in which case its CREATE is emitted bare.
    """
    for other in datasets.values():
        if other.format != "split":
            continue
        opts = other.read_options
        role = opts.get("role")
        if role == "test" and opts.get("train_name") == name:
            return "train", str(opts["source"])
        if role == "train" and opts.get("test_name") == name:
            return "test", str(opts["source"])
    return None


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
         If the model's training dataset was later overwritten by
         ``materialize_query``, the guard and the re-fit both target the
         carried base loader (original file) instead of the post-transform
         table.

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
        if entry.format == "split":
            # Split datasets are rehydrated by the merged second pass below
            # (RandomState recipe + membership-checksum assert), never by a
            # file reload — "(split)" is a placeholder, not a path.
            continue
        if entry.format == "derived":
            # A derived entry that overwrote a file-backed dataset retains the
            # original loader in base_loader; emit it here (first pass) so the
            # second-pass derived CREATE — which may self-reference this same
            # name (transform-in-place) — has its base table at replay. The
            # carried load-time hash guards the base file exactly like a
            # first-class file-backed entry.
            base = entry.base_loader
            if base is not None:
                var = _sanitized_guard_var(name, guard_idx)
                guard_idx += 1
                lines.extend(
                    _hash_guard_lines(
                        var, name, base["path"], base.get("source_hash", "sentinel:unset")
                    )
                )
                stmt = _file_load_stmt(name, base["format"], base["path"], base.get("read_options"))
                lines.append(f"con.execute({stmt!r})")
            continue
        var = _sanitized_guard_var(name, guard_idx)
        guard_idx += 1
        lines.extend(_hash_guard_lines(var, name, entry.path, entry.source_hash))
        stmt = _file_load_stmt(name, entry.format, entry.path, entry.read_options)
        lines.append(f"con.execute({stmt!r})")

    # Second pass: derived and split datasets, in registration order.
    # Registration order IS topological order — a derived/split entry can
    # only reference earlier-registered tables, and ``_session.get_datasets()``
    # is a regular dict whose insertion order Python preserves, so chained
    # derived datasets (derived_b SELECTs FROM derived_a) are emitted after
    # their upstream — one interleaved loop replaces the old derived-only
    # pass. Derived recipes get no hash assert: the recipe is the SQL plus
    # the upstream datasets, which already carry their own asserts via the
    # model rehydration block when models depend on them. Split blocks are
    # emitted once per pair, keyed off the test-role entry (which carries
    # the membership checksum) and are *side-aware*: the block always
    # re-creates the test table (behind the checksum assert), but re-creates
    # the train table only while the train side still belongs to this split.
    # Overwriting a split output with materialize_query drops that side's
    # split recipe — the overwritten side becomes a derived entry emitted
    # above, and the split block must NOT clobber it. For a train-side
    # overwrite we therefore skip the train CREATE (``include_train=False``);
    # for a test-side overwrite the test-keyed branch below never fires, so
    # no split block is emitted at all. Either way replay honors the derived
    # CREATE and fails loudly (missing table / checksum) rather than silently
    # recomputing the split.
    datasets = _session.get_datasets()
    for name, entry in datasets.items():
        if entry.format == "derived":
            derived_sql = entry.read_options.get("sql", "")
            stmt = f'CREATE OR REPLACE TABLE "{name}" AS {derived_sql}'
            # repr() escapes embedded quotes (including ``\"\"\"``) so the
            # emitted Python source compiles regardless of what SQL the user
            # passed to materialize_query.
            overwrite = _split_side_overwritten(name, datasets)
            if overwrite is None:
                lines.append(f"con.execute({stmt!r})")
            else:
                # This derived entry overwrote one side of a still-live split.
                # The pre-overwrite split table is deliberately NOT recreated
                # at replay (that would clobber this derived table), so a
                # self-referential overwrite SQL (``SELECT ... FROM "{name}"``)
                # hits a DuckDB catalog error at its own CREATE. Wrap it so the
                # replay explains why the table is missing instead of surfacing
                # a bare CatalogException; the CREATE is otherwise unchanged, so
                # a non-self-referential overwrite still succeeds transparently.
                side, split_source = overwrite
                msg = (
                    f"Dataset {name!r} was created by overwriting the {side} side "
                    f"of the split of {split_source!r}. Its pre-overwrite split "
                    f"table is deliberately not recreated at replay (that would "
                    f"clobber this dataset), so this SQL cannot read from {name!r} "
                    f"itself; rematerialize it from a table that exists at replay."
                )
                lines.append("try:")
                lines.append(f"    con.execute({stmt!r})")
                lines.append("except duckdb.CatalogException as exc:")
                lines.append(f"    raise AssertionError({msg!r}) from exc")
        elif entry.format == "split" and entry.read_options.get("role") == "test":
            opts = entry.read_options
            # The train side still belongs to this split only if its current
            # registry entry is the matching train-role split output. If a
            # materialize_query overwrite replaced it with a derived entry,
            # its split recipe is gone and re-creating it here would clobber
            # the derived table emitted above.
            train_entry = datasets.get(str(opts["train_name"]))
            include_train = (
                train_entry is not None
                and train_entry.format == "split"
                and train_entry.read_options.get("role") == "train"
                and train_entry.read_options.get("test_name") == name
            )
            lines.append(
                split_replay_source(
                    source=str(opts["source"]),
                    train_name=str(opts["train_name"]),
                    test_name=str(opts["test_name"]),
                    seed=int(opts["seed"]),
                    test_fraction=float(opts["test_fraction"]),
                    stratify_by=opts.get("stratify_by"),
                    rid_column=str(opts["rid_column"]),
                    membership_checksum=str(opts["membership_checksum"]),
                    include_train=include_train,
                )
            )

    models = _session.get_models()
    if models:
        # Materialize a DataFrame per reloadable dataset so the model
        # rehydration line has something to fit against. Derived datasets
        # (format == "derived") are reloadable too — they exist as DuckDB
        # tables once the second-pass CREATE OR REPLACE TABLE lines above
        # have run.
        frame_names: set[str] = set()
        for name, entry in _session.get_datasets().items():
            if entry.format == "dataframe":
                continue
            frame_names.add(f"{name}_df")
            lines.append(f'{name}_df = con.sql("SELECT * FROM {name}").df()')

        for model_name, model_entry in models.items():
            lines.append("")
            lines.append(f"# --- Re-fit model {model_name!r} (kind={model_entry.kind}) ---")
            ds_entry = _session.get_datasets().get(model_entry.fitted_on_dataset)
            hash_val = model_entry.training_dataset_hash
            rev = model_entry.training_dataset_revision
            # The fit-time registration REVISION — not the content hash —
            # decides whether the current entry is the same table state the
            # model was fit on: derived/split/dataframe states share constant
            # per-format sentinel hashes, so hash comparison cannot see a
            # replacement. A fit on the pre-overwrite file-backed state is
            # recognized by the carried base_loader's pinned revision (always
            # the original file entry's, unchanged across chained overwrites)
            # and re-fits from the original file behind the fit-time hash
            # guard. Any other revision mismatch on a derived entry means the
            # fit-time table state no longer exists anywhere reachable.
            overwritten_base: dict[str, Any] | None = None
            if (
                ds_entry is not None
                and ds_entry.format == "derived"
                and rev != ds_entry.revision
            ):
                base = ds_entry.base_loader
                if base is not None and rev == base.get("revision"):
                    overwritten_base = base
                else:
                    msg = (
                        f"Model {model_name!r} was fit on dataset "
                        f"{model_entry.fitted_on_dataset!r}, which was later "
                        f"replaced; the table state it was fit on no longer "
                        f"exists anywhere reachable, so the re-fit cannot be "
                        f"replayed faithfully."
                    )
                    lines.append(f"raise AssertionError({msg!r})")
                    continue
            if overwritten_base is not None:
                ds_path = overwritten_base["path"]
                # A dataset named <model>_train would make the train frame
                # collide with that dataset's <name>_df scoring frame — and
                # the model block emits after the frames loop, so the
                # assignment would clobber the post-transform frame that
                # predict/evaluate cells reference. Prefix underscores until
                # the name is free.
                train_var = f"{model_name}_train_df"
                while train_var in frame_names:
                    train_var = f"_{train_var}"
                frame_names.add(train_var)
                data_ref = train_var
                lines.append(
                    f"# Dataset {model_entry.fitted_on_dataset!r} was overwritten by "
                    f"materialize_query after this model was fit; re-fitting from "
                    f"the original source file, not the current derived table."
                )
            else:
                ds_path = ds_entry.path if ds_entry is not None else None
                data_ref = f"{model_entry.fitted_on_dataset}_df"
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

            if overwritten_base is not None:
                select = _file_select_expr(
                    overwritten_base["format"],
                    overwritten_base["path"],
                    overwritten_base.get("read_options"),
                )
                lines.append(f"{data_ref} = con.sql({select!r}).df()")

            smf_fn = _KIND_TO_SMF.get(model_entry.kind, "ols")
            fit_args = "disp=False" if model_entry.kind in ("logistic", "poisson", "negbin") else ""
            lines.append(
                f'{model_name} = smf.{smf_fn}("{model_entry.formula}", '
                f"data={data_ref}).fit({fit_args})"
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
