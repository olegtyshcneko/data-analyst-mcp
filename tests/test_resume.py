"""load_session_from_notebook — happy-path round trips (spec v4 §4)."""

from __future__ import annotations

from typing import Any


def _fresh(call_tool: Any) -> None:
    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import get_recorder

    session.reset()
    get_recorder().reset()


def _emit(call_tool: Any, tmp_path: Any) -> str:
    target = str(tmp_path / "session.ipynb")
    result = call_tool("emit_notebook", {"path": target})
    assert result["ok"] is True
    return result["path"]


def test_round_trip_load_materialize_split(call_tool: Any, tmp_path: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.recorder import get_recorder

    rng = np.random.RandomState(0)
    csv = tmp_path / "rt.csv"
    pd.DataFrame({"x": rng.normal(size=40), "g": ["a", "b"] * 20}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "rt"})["ok"] is True
    assert (
        call_tool("materialize_query", {"sql": "SELECT x FROM rt WHERE x > 0", "name": "pos"})["ok"]
        is True
    )
    assert call_tool("split_dataset", {"name": "rt", "seed": 7})["ok"] is True
    n_cells_before = len(get_recorder().cells)
    journal_before = [dict(e) for e in session.get_journal()]
    datasets_before = {n: (e.rows, e.format, e.revision) for n, e in session.get_datasets().items()}
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is True, result
    assert result["n_cells_imported"] == n_cells_before
    assert result["n_journal_ops"] == len(journal_before)
    assert result["warnings"] == []
    assert {d["name"] for d in result["datasets"]} == {"rt", "pos", "rt_train", "rt_test"}

    after = {n: (e.rows, e.format, e.revision) for n, e in session.get_datasets().items()}
    assert after == datasets_before
    assert [e["op_id"] for e in session.get_journal()] == [e["op_id"] for e in journal_before]
    assert len(get_recorder().cells) == n_cells_before

    con = session.get_connection()
    assert con.execute('SELECT COUNT(*) FROM "rt_train"').fetchone()[0] == 30


def test_round_trip_overwrite_chain_restores_live_values(call_tool: Any, tmp_path: Any) -> None:
    """The 13-vs-7 case: snapshot reconstruction gives 7; journal replay must give 13."""
    import pandas as pd

    from data_analyst_mcp import session

    csv = tmp_path / "y.csv"
    pd.DataFrame({"y": [6]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "y"})["ok"] is True
    for sql in ("SELECT y * 2 AS y FROM y", "SELECT y + 1 AS y FROM y"):
        assert (
            call_tool("materialize_query", {"sql": sql, "name": "y", "overwrite": True})["ok"]
            is True
        )
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    assert call_tool("load_session_from_notebook", {"path": path})["ok"] is True
    con = session.get_connection()
    assert con.execute('SELECT y FROM "y"').fetchone()[0] == 13


def test_resume_then_continue_then_emit_again(call_tool: Any, tmp_path: Any) -> None:
    """Post-resume the session continues: new ops append, next emit validates."""
    import pandas as pd

    from data_analyst_mcp import session
    from data_analyst_mcp.manifest import validate_manifest

    csv = tmp_path / "c.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "c"})["ok"] is True
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    assert call_tool("load_session_from_notebook", {"path": path})["ok"] is True
    assert (
        call_tool("materialize_query", {"sql": "SELECT a + 1 AS a FROM c", "name": "c2"})["ok"]
        is True
    )
    # next_revision continued — no revision collision with the imported entry.
    assert session.get_datasets()["c2"].revision > session.get_datasets()["c"].revision

    import nbformat

    path2 = str(tmp_path / "session2.ipynb")
    assert call_tool("emit_notebook", {"path": path2})["ok"] is True
    nb2 = nbformat.read(path2, as_version=4)
    validate_manifest(dict(nb2.metadata["data_analyst_mcp"]))
    assert len(nb2.metadata["data_analyst_mcp"]["journal"]) == 2


def test_resume_restores_registered_model(call_tool: Any, tmp_path: Any) -> None:
    import numpy as np
    import pandas as pd

    from data_analyst_mcp import session

    rng = np.random.RandomState(2)
    x = rng.normal(size=50)
    csv = tmp_path / "mod.csv"
    pd.DataFrame({"x": x, "y": 3.0 * x + rng.normal(size=50)}).to_csv(csv, index=False)
    assert call_tool("load_dataset", {"path": str(csv), "name": "mod"})["ok"] is True
    assert (
        call_tool(
            "fit_model",
            {"name": "mod", "formula": "y ~ x", "kind": "ols", "robust": True, "model_name": "rm"},
        )["ok"]
        is True
    )
    params_before = dict(session.get_models()["rm"]._result.params.items())
    bse_before = dict(session.get_models()["rm"]._result.bse.items())
    path = _emit(call_tool, tmp_path)

    _fresh(call_tool)
    result = call_tool("load_session_from_notebook", {"path": path})
    assert result["ok"] is True, result
    assert result["models"] == ["rm"]
    entry = session.get_models()["rm"]
    assert entry.fit_options == {"robust": True}
    for k, v in params_before.items():
        assert abs(float(entry._result.params[k]) - float(v)) < 1e-9
    for k, v in bse_before.items():
        assert abs(float(entry._result.bse[k]) - float(v)) < 1e-9
    # predict works against the restored registry (live Results object).
    assert call_tool("predict", {"model_name": "rm", "dataset": "mod", "limit": 5})["ok"] is True
