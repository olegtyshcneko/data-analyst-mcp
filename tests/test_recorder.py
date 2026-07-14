"""Tests for the NotebookRecorder."""

from __future__ import annotations


def test_fresh_recorder_has_zero_cells() -> None:
    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()

    assert rec.cells == []


def test_record_appends_one_markdown_and_one_code_cell() -> None:
    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()
    rec.record(
        markdown="### Loaded dataset `raw`",
        code="con.execute('SELECT 1')",
        tool_name="load_dataset",
    )

    assert len(rec.cells) == 2
    md, code = rec.cells
    assert md["cell_type"] == "markdown"
    assert md["source"] == "### Loaded dataset `raw`"
    assert code["cell_type"] == "code"
    assert code["source"] == "con.execute('SELECT 1')"
    # tool_name preserved in metadata for downstream consumers.
    assert code["metadata"]["tool_name"] == "load_dataset"


def test_to_notebook_with_setup_prepends_imports_and_reload_statements() -> None:
    import nbformat

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    session.register(
        name="raw",
        path="fixtures/messy.csv",
        read_options={},
        format="csv",
        rows=5000,
        columns=[],
    )

    rec = NotebookRecorder()
    rec.record(markdown="### m", code="x = 1", tool_name="t")

    nb = rec.to_notebook(include_setup=True)
    nbformat.validate(nb)

    # First cell is the setup cell.
    setup = nb.cells[0]
    assert setup.cell_type == "code"
    assert "import duckdb" in setup.source
    assert "import pandas as pd" in setup.source
    assert "from scipy import stats" in setup.source
    assert "import statsmodels.api as sm" in setup.source
    assert "import matplotlib.pyplot as plt" in setup.source
    assert "con = duckdb.connect()" in setup.source
    # The reload statement for the `raw` table is rebuilt from the registry.
    assert "CREATE OR REPLACE TABLE raw" in setup.source
    assert "read_csv_auto('fixtures/messy.csv', SAMPLE_SIZE=-1)" in setup.source

    # Recorded cells follow the setup cell.
    assert len(nb.cells) == 3
    assert nb.cells[1].cell_type == "markdown"
    assert nb.cells[2].cell_type == "code"


def test_reset_empties_the_cell_list() -> None:
    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()
    rec.record(markdown="### m", code="x = 1", tool_name="t")
    assert rec.cells != []

    rec.reset()

    assert rec.cells == []


def test_to_notebook_without_setup_returns_recorded_cells() -> None:
    import nbformat

    from data_analyst_mcp.recorder import NotebookRecorder

    rec = NotebookRecorder()
    rec.record(markdown="### M", code="x = 1", tool_name="load_dataset")

    nb = rec.to_notebook(include_setup=False)

    assert isinstance(nb, nbformat.NotebookNode)
    nbformat.validate(nb)
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == "markdown"
    assert nb.cells[0].source == "### M"
    assert nb.cells[1].cell_type == "code"
    assert nb.cells[1].source == "x = 1"


# ---- Model-registry recorder slices (Phase 5c) --------------------------


def test_setup_cell_emits_hash_assert_for_registered_model(tmp_path) -> None:
    """The setup cell for a session with a registered model contains a
    SHA-256 assert against the training CSV bytes plus a smf.<kind>(...)
    rehydration line keyed on the model's formula."""
    import hashlib

    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder
    from data_analyst_mcp.tools import models as _models

    session.reset()
    csv = tmp_path / "train.csv"
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    df.to_csv(csv, index=False)
    expected_hash = hashlib.sha256(csv.read_bytes()).hexdigest()

    # Register the dataset via the normal load_dataset path so its `path`
    # is the absolute CSV location.
    session.register(
        name="train",
        path=str(csv),
        read_options={},
        format="csv",
        rows=5,
        columns=[{"name": "y", "dtype": "DOUBLE"}, {"name": "x", "dtype": "DOUBLE"}],
    )
    # Materialize the table from the in-memory frame (same rows as the CSV).
    # The session connection is sandboxed (enable_external_access=false), so a
    # direct read_csv_auto('...') here would (correctly) be blocked; the file
    # still exists on disk for the recorder's hash assertion below.
    con = session.get_connection()
    con.register("__train_df", df)
    con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM __train_df")
    con.unregister("__train_df")

    # Register a model whose training_dataset_hash matches the file.
    payload = _models.FitModelInput(
        name="train", formula="y ~ x", kind="ols", robust=False, model_name="m"
    )
    _models.fit_model(payload)

    rec = NotebookRecorder()
    nb = rec.to_notebook(include_setup=True)
    setup_src = nb.cells[0].source

    assert "import hashlib" in setup_src
    assert "import statsmodels.formula.api as smf" in setup_src
    assert expected_hash in setup_src
    assert "actual_hash_m = hashlib.sha256(" in setup_src
    assert "assert actual_hash_m == expected_hash_m" in setup_src
    # Re-fit line.
    assert 'm = smf.ols("y ~ x"' in setup_src
    assert "data=train_df" in setup_src


def test_setup_cell_skips_hash_assert_for_in_memory_model(
    load_df_into_session,
) -> None:
    """Models fit on an in-memory dataset (no file) skip the hash assert
    but still emit a re-fit line — the rehydration just runs against the
    same in-memory DataFrame the reader has to provide."""
    import pandas as pd

    from data_analyst_mcp.recorder import NotebookRecorder
    from data_analyst_mcp.tools import models as _models

    load_df_into_session("tiny", pd.DataFrame({"y": [1, 2, 3, 4, 5], "x": [0, 1, 2, 3, 4]}))
    _models.fit_model(
        _models.FitModelInput(
            name="tiny", formula="y ~ x", kind="ols", robust=False, model_name="m"
        )
    )

    rec = NotebookRecorder()
    setup_src = rec.to_notebook(include_setup=True).cells[0].source
    # In-memory dataset → no hash assert, but a comment is left behind.
    assert "assert actual_hash_m" not in setup_src
    assert "non-file dataset" in setup_src
    # The re-fit line still has to be emitted so downstream tool cells
    # can reference `m`. We emit it after the note.
    assert "smf.ols" in setup_src


def test_emitted_notebook_with_fit_predict_evaluate_runs_via_nbconvert(tmp_path, call_tool) -> None:
    """End-to-end: emit a notebook covering fit_model → predict →
    evaluate_model and run it via ``jupyter nbconvert --execute`` on a
    fresh kernel. Confirms the hash-asserted rehydration story holds
    end-to-end (verification step 9)."""
    import os
    import subprocess

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 0.5 * x1 + 0.8 * x2
    from scipy.special import expit

    y = rng.binomial(1, expit(eta))
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2", "kind": "logistic", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool("predict", {"model_name": "m", "dataset": "train", "limit": 10})
    assert r["ok"], r
    r = call_tool("evaluate_model", {"model_name": "m", "dataset": "train"})
    assert r["ok"], r

    nb_path = tmp_path / "fit_predict_eval.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


def test_emitted_notebook_replays_overwrite_of_file_backed_dataset(tmp_path, call_tool) -> None:
    """materialize_query(overwrite=True) over a file-backed dataset replaces
    its registry entry with a derived one whose SQL self-references the same
    name. The emitted setup cell must still load the original file as the
    base table first, otherwise the derived CREATE OR REPLACE TABLE has no
    source at replay and nbconvert dies with
    ``CatalogException: Table with name data does not exist``.
    """
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    # Transform-in-place: overwrite the file-backed `data` with a derived
    # query that reads from `data` itself.
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r

    nb_path = tmp_path / "overwrite.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


def test_setup_cell_compiles_when_derived_dataset_sql_contains_triple_quotes() -> None:
    """Derived datasets whose SQL contains ``\"\"\"`` (e.g. inside a block
    comment) must not break the generated setup cell. Naive f-string
    interpolation would terminate the host triple-quoted string early
    and produce a SyntaxError; ``repr()`` escaping avoids it.
    """
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    # Register a derived dataset whose recorded SQL embeds a triple-quote.
    session.register(
        name="weird",
        path="(query)",
        read_options={"sql": 'SELECT 1 /* """ */ AS x'},
        format="derived",
        rows=1,
        columns=[{"name": "x", "dtype": "INTEGER"}],
    )

    rec = NotebookRecorder()
    setup_src = rec.to_notebook(include_setup=True).cells[0].source
    # Currently this raises SyntaxError because the embedded ``"""``
    # closes the host triple-quoted CREATE OR REPLACE TABLE string.
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_compiles_when_file_dataset_path_contains_quotes(tmp_path) -> None:
    """File-backed datasets whose ``path`` contains embedded triple-quotes
    must still produce a syntactically valid setup cell. The legacy
    interpolation embedded ``path`` directly inside ``\"\"\"...\"\"\"``,
    so any embedded ``\"\"\"`` terminated the host literal early.
    """
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    weird_path = str(tmp_path / 'weird"""name.csv')
    session.register(
        name="weird",
        path=weird_path,
        read_options={},
        format="csv",
        rows=3,
        columns=[{"name": "x", "dtype": "INTEGER"}],
    )

    rec = NotebookRecorder()
    setup_src = rec.to_notebook(include_setup=True).cells[0].source
    compile(setup_src, "<setup>", "exec")


def test_emitted_notebook_hash_assert_fires_when_training_csv_is_mutated(
    tmp_path, call_tool
) -> None:
    """Mutate the training CSV between fit and replay → setup cell raises
    AssertionError (verification step 9 negative branch)."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1, 0, 1], "x": [-2, -1, -0.5, 0.5, 1, 2, 3, 4]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    call_tool("load_dataset", {"path": str(csv), "name": "train"})
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "logistic", "model_name": "m"},
    )
    assert r["ok"], r

    nb_path = tmp_path / "drift.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    # Mutate the CSV — append a row. Hash now differs from fit-time hash.
    with open(csv, "a") as fh:
        fh.write("0,99\n")

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    # nbconvert returns non-zero on cell errors. The error message should
    # mention our descriptive AssertionError text.
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "AssertionError" in combined or "changed since the session was recorded" in combined


def test_setup_cell_renders_read_options_in_reload_statements() -> None:
    """The live load honored read_options; the replay reload must too,
    otherwise a passing hash can still parse differently (e.g. header or
    delimiter divergence). Covers both first-pass file-backed entries and
    base_loader re-creates."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    session.register(
        name="raw",
        path="fixtures/messy.csv",
        read_options={"header": False, "delim": ";"},
        format="csv",
        rows=10,
        columns=[],
    )
    session.register(
        name="data",
        path="(query)",
        read_options={"sql": "SELECT 1 AS a"},
        format="derived",
        rows=1,
        columns=[],
        base_loader={
            "path": "fixtures/messy.csv",
            "format": "csv",
            "read_options": {"delim": "|"},
            "source_hash": "sentinel:unset",
        },
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source

    assert "header=FALSE" in setup_src
    assert "delim=';'" in setup_src
    assert "delim='|'" in setup_src
    # Derived read_options ({"sql": ...}) must NOT leak into a reader call.
    assert "sql=" not in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_emits_source_hash_assert_for_file_backed_dataset(tmp_path) -> None:
    """Every file-backed dataset reload is preceded by a content-hash assert
    so an edited source file fails replay loudly (dataset-level counterpart
    of the model-block guard)."""
    import hashlib

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    csv = tmp_path / "tiny.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    session.reset()
    session.register(name="tiny", path=str(csv), read_options={}, format="csv", rows=1, columns=[])

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source

    assert f"expected_hash_ds_tiny_0 = '{expected}'" in setup_src
    assert "actual_hash_ds_tiny_0 = hashlib.sha256(" in setup_src
    assert (
        "assert actual_hash_ds_tiny_0 == expected_hash_ds_tiny_0, "
        "\"Source file for dataset 'tiny' changed since the session was recorded.\"" in setup_src
    )
    # Guard precedes the reload it protects.
    assert setup_src.index("expected_hash_ds_tiny_0") < setup_src.index(
        "CREATE OR REPLACE TABLE tiny"
    )
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_guard_variable_sanitizes_non_identifier_names(tmp_path) -> None:
    """Dataset names are not validated as Python identifiers (load_dataset
    defaults the name from the file basename, so ``my-data.csv`` → ``my-data``).
    Guard variables sanitize the name and add an emission index so the setup
    cell stays valid Python and collision-free."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    csv = tmp_path / "my-data.csv"
    csv.write_bytes(b"a\n1\n")

    session.reset()
    session.register(
        name="my-data", path=str(csv), read_options={}, format="csv", rows=1, columns=[]
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert "expected_hash_ds_my_data_0" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_guard_stays_valid_python_for_quote_containing_names(tmp_path) -> None:
    """Dataset names may contain quote characters; the guard's assert
    message must be emitted as a repr'd literal so the setup cell still
    compiles (the model guard's inline-quoting style would break here)."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    csv = tmp_path / "q.csv"
    csv.write_bytes(b"a\n1\n")

    session.reset()
    session.register(
        name='he said "hi"',
        path=str(csv),
        read_options={},
        format="csv",
        rows=1,
        columns=[],
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_emits_fallback_guard_above_content_ceiling(tmp_path, monkeypatch) -> None:
    """Files above the content-hash ceiling carry a (path, mtime, size)
    fallback hash; the emitted guard recomputes the same tuple at replay.
    Weaker guarantee, still a hard assert."""
    from data_analyst_mcp import provenance, session
    from data_analyst_mcp.recorder import NotebookRecorder

    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 4)
    csv = tmp_path / "big.csv"
    csv.write_bytes(b"a,b\n1,2\n3,4\n")

    session.reset()
    session.register(name="big", path=str(csv), read_options={}, format="csv", rows=2, columns=[])
    entry = session.get_datasets()["big"]
    assert entry.source_hash.startswith("fallback:")

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_ds_big_0 = '{entry.source_hash}'" in setup_src
    assert "_os.stat(" in setup_src
    assert "'fallback:' + hashlib.sha256(" in setup_src
    assert "assert actual_hash_ds_big_0 == expected_hash_ds_big_0" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_emitted_fallback_guard_executes_and_detects_drift(tmp_path, monkeypatch) -> None:
    """Executing (not just compiling) the fallback guard: it must pass on an
    unchanged file and fire after mutation — pinning that the emitted
    recompute is byte-identical to provenance.compute_source_hash's fallback."""
    import hashlib

    from data_analyst_mcp import provenance, session
    from data_analyst_mcp.recorder import NotebookRecorder

    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 4)
    csv = tmp_path / "big.csv"
    csv.write_bytes(b"a,b\n1,2\n3,4\n")

    session.reset()
    session.register(name="big", path=str(csv), read_options={}, format="csv", rows=2, columns=[])
    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    guard_lines = [ln for ln in setup_src.splitlines() if "_os" in ln or "hash_ds_big_0" in ln]
    guard_src = "\n".join(guard_lines)

    exec(guard_src, {"hashlib": hashlib})  # unchanged file: must not raise

    import os
    import time

    csv.write_bytes(b"a,b\n9,9\n9,9\n99\n")  # different size -> different fallback
    os.utime(csv, (time.time() + 10, time.time() + 10))
    try:
        exec(guard_src, {"hashlib": hashlib})
    except AssertionError:
        pass
    else:
        raise AssertionError("fallback guard did not fire after mutation")


def test_setup_cell_emits_comment_not_assert_for_sentinel_hash() -> None:
    """Datasets without a verifiable local file (s3://, http) reload
    unguarded — the setup cell says so in a comment instead of asserting."""
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder

    session.reset()
    session.register(
        name="remote",
        path="s3://bucket/file.csv",
        read_options={},
        format="csv",
        rows=1,
        columns=[],
    )

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert "# Note: dataset 'remote' has no verifiable source hash" in setup_src
    assert "expected_hash_ds_remote_0" not in setup_src
    assert "CREATE OR REPLACE TABLE remote" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_guards_base_loader_reload_after_overwrite(call_tool, tmp_path) -> None:
    """Overwriting a file-backed dataset moves its reload into the
    base_loader branch — the carried load-time hash must guard that reload,
    and a second overwrite must keep the guard."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    csv.write_bytes(b"a\n1\n2\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_ds_data_0 = '{expected}'" in setup_src
    assert setup_src.index("expected_hash_ds_data_0") < setup_src.index(
        "CREATE OR REPLACE TABLE data"
    )
    compile(setup_src, "<setup>", "exec")

    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a + 1 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r
    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_ds_data_0 = '{expected}'" in setup_src


def test_emitted_notebook_dataset_guard_fires_when_source_csv_is_mutated(
    tmp_path, call_tool
) -> None:
    """No model involved: a plain loaded dataset whose CSV is edited between
    emit and replay must fail the setup cell loudly (spec acceptance
    criterion, plain file-backed case)."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]})
    csv = tmp_path / "plain.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "plain"})
    assert r["ok"], r
    r = call_tool("describe_column", {"name": "plain", "column": "a"})
    assert r["ok"], r

    nb_path = tmp_path / "plain_drift.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    with open(csv, "a") as fh:
        fh.write("99\n")

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "AssertionError" in combined or "changed since the session was recorded" in combined


def test_emitted_notebook_overwrite_chain_guard_fires_when_base_csv_is_mutated(
    tmp_path, call_tool
) -> None:
    """Drift on a file reachable only via base_loader (after a
    transform-in-place overwrite) must also fail replay loudly (spec
    acceptance criterion, overwrite-chain case)."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]})
    csv = tmp_path / "base.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "data"})
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT a * 10 AS a FROM data", "name": "data", "overwrite": True},
    )
    assert r["ok"], r

    nb_path = tmp_path / "chain_drift.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    with open(csv, "a") as fh:
        fh.write("99\n")

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "AssertionError" in combined or "changed since the session was recorded" in combined


def test_emitted_notebook_model_guard_fires_when_dataset_reloaded_after_fit(
    tmp_path, call_tool
) -> None:
    """Same-name reload after a fit: the dataset guard passes (the current
    file matches the reloaded entry's hash) but the model assert must fail
    loudly — the model trained on the pre-reload data. This is the guard
    full unification would have lost (spec acceptance criterion)."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [1, 2, 3, 4, 5, 6]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    # Rewrite the file and reload under the same name — the dataset entry's
    # hash refreshes, the model's captured hash goes (correctly) stale.
    df2 = pd.DataFrame({"y": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "x": [1, 2, 3, 4, 5, 6]})
    df2.to_csv(csv, index=False)
    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r

    nb_path = tmp_path / "reload_after_fit.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    # Specifically the *model* guard message — the dataset guard passes here.
    assert "Training data for 'm' changed since the session was recorded" in combined


def test_setup_cell_refits_overwritten_training_dataset_from_base_loader(
    call_tool, tmp_path
) -> None:
    """A model whose training dataset is later overwritten by
    materialize_query must guard against the original file (not the
    '(query)' placeholder, which crashes the hash recompute) and re-fit on
    a dedicated train frame loaded from that file — not on the
    post-transform table."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    # Guard recomputes against the original file, not the placeholder.
    assert f"expected_hash_m = '{expected}'" in setup_src
    assert "open('(query)'" not in setup_src
    # Re-fit uses a dedicated frame loaded from the original file...
    assert "m_train_df = con.sql(" in setup_src
    assert "data=m_train_df" in setup_src
    assert "was overwritten by materialize_query" in setup_src
    # ...while train_df stays the (post-transform) current table for
    # predict/evaluate scoring cells.
    assert 'train_df = con.sql("SELECT * FROM train").df()' in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_fallback_guard_targets_base_path_after_overwrite(
    call_tool, tmp_path, monkeypatch
) -> None:
    """Above-ceiling training file + overwrite: the fallback stat/recompute
    must target the original path, not the '(query)' placeholder."""
    from data_analyst_mcp import provenance
    from data_analyst_mcp.recorder import get_recorder

    monkeypatch.setattr(provenance, "HASH_CONTENT_CEILING_BYTES", 4)
    csv = tmp_path / "big.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "expected_hash_m = 'fallback:" in setup_src
    assert f"_st = _os.stat({str(csv)!r})" in setup_src
    assert "_os.stat('(query)')" not in setup_src
    assert "data=m_train_df" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_sentinel_base_after_overwrite_refits_unguarded(tmp_path) -> None:
    """An s3-backed training dataset overwritten after fit: no hash assert
    is possible (sentinel), but the re-fit still loads from the original
    s3 loader rather than the post-transform table."""
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import NotebookRecorder
    from data_analyst_mcp.tools import models as _models

    session.reset()
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    session.register(
        name="train",
        path="s3://bucket/train.csv",
        read_options={},
        format="csv",
        rows=5,
        columns=[{"name": "y", "dtype": "DOUBLE"}, {"name": "x", "dtype": "DOUBLE"}],
    )
    con = session.get_connection()
    con.register("__train_df", df)
    con.execute("CREATE OR REPLACE TABLE train AS SELECT * FROM __train_df")
    con.unregister("__train_df")
    assert session.get_datasets()["train"].source_hash.startswith("sentinel:")

    _models.fit_model(
        _models.FitModelInput(
            name="train", formula="y ~ x", kind="ols", robust=False, model_name="m"
        )
    )
    # Overwrite via a direct derived registration mirroring materialize_query
    # (an s3 read through the tool would need network; the registry state is
    # what the recorder consumes).
    entry = session.get_datasets()["train"]
    session.register(
        name="train",
        path="(query)",
        read_options={"sql": "SELECT y * 10 AS y, x FROM train"},
        format="derived",
        rows=5,
        columns=entry.columns,
        base_loader={
            "path": entry.path,
            "format": entry.format,
            "read_options": dict(entry.read_options),
            "source_hash": entry.source_hash,
            "revision": entry.revision,
        },
    )
    con.execute("CREATE OR REPLACE TABLE train AS SELECT y * 10 AS y, x FROM train")

    setup_src = NotebookRecorder().to_notebook(include_setup=True).cells[0].source
    assert "assert actual_hash_m" not in setup_src
    assert "m_train_df = con.sql(" in setup_src
    assert "s3://bucket/train.csv" in setup_src
    assert "data=m_train_df" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_model_fit_after_overwrite_keeps_refitting_current_table(
    call_tool, tmp_path
) -> None:
    """A model fit on the already-derived table re-fits on <dataset>_df —
    the post-transform table is exactly what it trained on. No train frame,
    no hash assert (the derived entry's hash is a sentinel)."""
    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")
    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "m_train_df" not in setup_src
    assert "data=train_df" in setup_src
    assert "assert actual_hash_m" not in setup_src
    assert "non-file dataset" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_emitted_notebook_replays_fit_then_overwrite_end_to_end(tmp_path, call_tool) -> None:
    """fit_model → materialize_query overwrite → predict → emit → nbconvert
    executes cleanly: the model re-fits from the original file while the
    scoring cell sees the post-transform table, matching the live session."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r
    r = call_tool("predict", {"model_name": "m", "dataset": "train", "limit": 5})
    assert r["ok"], r

    nb_path = tmp_path / "fit_then_overwrite.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


def test_emitted_notebook_fit_then_overwrite_guard_fires_on_mutated_source(
    tmp_path, call_tool
) -> None:
    """Same session as the clean test, but the source CSV is edited between
    emit and replay: the setup cell must die with a loud AssertionError
    instead of silently re-fitting on different data."""
    import os
    import subprocess

    import pandas as pd

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)

    r = call_tool("load_dataset", {"path": str(csv), "name": "train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )
    assert r["ok"], r

    nb_path = tmp_path / "fit_then_overwrite_drift.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"], r

    csv.write_text("y,x\n9.0,0.0\n8.0,1.0\n7.0,2.0\n6.0,3.0\n5.0,4.0\n4.0,5.0\n")

    result = subprocess.run(
        [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0, "replay should fail after source mutation"
    assert "AssertionError" in result.stderr
    assert "changed since the session was recorded" in result.stderr


def test_setup_cell_train_frame_never_clobbers_dataset_scoring_frame(call_tool, tmp_path) -> None:
    """A dataset named <model>_train makes the naive train-frame variable
    collide with the dataset's <name>_df scoring frame; the emitted train
    frame must pick a distinct name so the post-transform scoring frame
    survives the setup cell."""
    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "m_train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")
    r = call_tool("load_dataset", {"path": str(csv), "name": "m_train"})
    assert r["ok"], r
    r = call_tool(
        "fit_model",
        {"name": "m_train", "formula": "y ~ x", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM m_train", "name": "m_train", "overwrite": True},
    )
    assert r["ok"], r

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    # The scoring frame keeps the post-transform table...
    assert '\nm_train_df = con.sql("SELECT * FROM m_train").df()' in setup_src
    # ...and the train frame picked a non-colliding name.
    assert "\n_m_train_df = con.sql(" in setup_src
    assert "data=_m_train_df" in setup_src
    # The scoring frame is assigned exactly once — never reassigned.
    assert setup_src.count('m_train_df = con.sql("SELECT * FROM m_train").df()') == 1
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_raises_for_model_fit_on_pure_query_overwritten_dataset(
    call_tool, tmp_path
) -> None:
    """S4 (ROADMAP gap): a model fit on a pure-query derived dataset that is
    later overwritten shares the constant '(query)' sentinel hash with the
    replacement, so hash comparison cannot see the overwrite. The revision
    can: the setup cell must emit a loud raise, never a silent re-fit on the
    post-transform table."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT y, x FROM base", "name": "d"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM base", "name": "d", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "'m'" in setup_src
    assert "'d'" in setup_src
    assert "was later replaced" in setup_src
    compile(setup_src, "<setup>", "exec")


def test_emitted_notebook_pure_query_fit_overwrite_fails_loudly_end_to_end(
    tmp_path, call_tool
) -> None:
    """S4 end-to-end: emit the fit-then-overwrite pure-query session and run
    it via nbconvert — replay must fail with the explanatory AssertionError
    (today it silently succeeds on the wrong table)."""
    import os
    import subprocess

    import pandas as pd

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT y, x FROM base", "name": "d"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM base", "name": "d", "overwrite": True},
    )["ok"] is True

    nb_path = tmp_path / "s4.ipynb"
    assert call_tool("emit_notebook", {"path": str(nb_path)})["ok"] is True

    result = subprocess.run(
        [
            "uv", "run", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--inplace", str(nb_path),
            "--ExecutePreprocessor.timeout=120",
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0, "replay must not silently re-fit on the replaced table"
    combined = result.stderr + result.stdout
    assert "cannot be replayed faithfully" in combined


def test_setup_cell_raises_for_model_fit_on_base_carrying_derived_then_overwritten(
    call_tool, tmp_path
) -> None:
    """S5: model fit on the middle base-carrying derived state, then another
    overwrite. Neither the current revision nor the carried base revision
    matches the fit — silently re-fitting from the base FILE would be as
    wrong as the post-transform table. Loud raise, no m_train_df frame."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "data"})["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 2 AS y, x FROM data", "name": "data", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "data", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y + 1 AS y, x FROM data", "name": "data", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "was later replaced" in setup_src
    assert "m_train_df" not in setup_src
    compile(setup_src, "<setup>", "exec")


def test_setup_cell_raises_for_model_fit_on_middle_materialization(
    call_tool, tmp_path
) -> None:
    """S10: fresh derived name (base_loader stays None throughout),
    materialized twice with a fit in between — the middle state is gone;
    replay must raise instead of silently re-fitting on the latest table."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "base.csv"
    pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
    ).to_csv(csv, index=False)

    assert call_tool("load_dataset", {"path": str(csv), "name": "base"})["ok"] is True
    assert call_tool(
        "materialize_query", {"sql": "SELECT y, x FROM base WHERE x < 5", "name": "d"}
    )["ok"] is True
    assert call_tool(
        "fit_model", {"name": "d", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y, x FROM base WHERE x < 4", "name": "d", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert "raise AssertionError" in setup_src
    assert "was later replaced" in setup_src


def test_setup_cell_still_refits_from_base_file_after_chained_overwrites(
    call_tool, tmp_path
) -> None:
    """S2 regression pin: a model fit ON the file-backed state, followed by
    two chained overwrites, still re-fits from the carried base file behind
    the fit-time hash guard — identified by base_loader['revision'] == R0."""
    import hashlib

    from data_analyst_mcp.recorder import get_recorder

    csv = tmp_path / "train.csv"
    csv.write_text("y,x\n1.0,0.0\n2.0,1.0\n3.0,2.0\n4.0,3.0\n5.0,4.0\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    assert call_tool("load_dataset", {"path": str(csv), "name": "train"})["ok"] is True
    assert call_tool(
        "fit_model", {"name": "train", "formula": "y ~ x", "kind": "ols", "model_name": "m"}
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y * 10 AS y, x FROM train", "name": "train", "overwrite": True},
    )["ok"] is True
    assert call_tool(
        "materialize_query",
        {"sql": "SELECT y + 1 AS y, x FROM train", "name": "train", "overwrite": True},
    )["ok"] is True

    setup_src = get_recorder().to_notebook(include_setup=True).cells[0].source
    assert f"expected_hash_m = '{expected}'" in setup_src
    assert "m_train_df = con.sql(" in setup_src
    assert "data=m_train_df" in setup_src
    assert "raise AssertionError" not in setup_src
    compile(setup_src, "<setup>", "exec")
