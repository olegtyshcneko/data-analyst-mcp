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
    con = session.get_connection()
    con.execute(f"CREATE OR REPLACE TABLE train AS SELECT * FROM read_csv_auto('{csv}')")

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
