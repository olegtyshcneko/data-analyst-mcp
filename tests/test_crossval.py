"""Tests for the ``cross_validate`` tool."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _load_linear(load_df_into_session: Any, n: int = 40) -> None:
    """y = 2x + noise — deterministic fixture."""
    import pandas as pd

    rng = np.random.RandomState(0)
    x = np.arange(n, dtype=float)
    y = 2.0 * x + rng.normal(0, 1.0, size=n)
    load_df_into_session("lin", pd.DataFrame({"x": x, "y": y}))


def test_cross_validate_ols_shape(call_tool: Any, load_df_into_session: Any) -> None:
    _load_linear(load_df_into_session)

    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})

    assert result["ok"] is True
    assert result["kind"] == "ols"
    assert result["k"] == 5
    assert result["seed"] == 42
    assert result["stratified"] is False
    assert result["n_obs"] == 40
    assert result["dropped_rows"] == 0
    assert result["fold_failures"] == []
    assert sorted(result["fold_sizes"]) == [8, 8, 8, 8, 8]
    metrics = result["metrics"]
    assert set(metrics.keys()) == {"rmse", "mae", "r_squared"}
    for m in metrics.values():
        assert set(m.keys()) == {"mean", "std", "per_fold"}
        assert len(m["per_fold"]) == 5
        assert all(isinstance(v, float) for v in m["per_fold"])
    assert isinstance(result["interpretation"], str) and result["interpretation"]
    assert "model_name" not in result  # fits are ephemeral


def test_cross_validate_never_touches_model_registry(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp import session as _session

    _load_linear(load_df_into_session)
    call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})
    assert _session.get_models() == {}


def test_cross_validate_ols_matches_manual_recompute(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Independent recompute: same RandomState fold assignment, statsmodels
    array fits, numpy metrics. Aggregates must match to 1e-10."""
    import statsmodels.api as sm

    _load_linear(load_df_into_session)
    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x", "k": 4, "seed": 7})
    assert result["ok"] is True

    from data_analyst_mcp import session as _session

    df = _session.get_connection().execute('SELECT * FROM "lin"').df()
    import statsmodels.formula.api as smf

    full = smf.ols("y ~ x", data=df).fit()
    y = np.asarray(full.model.endog)
    X = np.asarray(full.model.exog)
    n = len(y)
    rng = np.random.RandomState(7)
    fold = np.empty(n, dtype=int)
    perm = rng.permutation(n)
    fold[perm] = np.arange(n) % 4
    rmses = []
    for i in range(4):
        tr = fold != i
        res = sm.OLS(y[tr], X[tr]).fit()
        pred = np.asarray(res.predict(X[~tr]))
        rmses.append(float(np.sqrt(np.mean((y[~tr] - pred) ** 2))))
    assert abs(result["metrics"]["rmse"]["mean"] - float(np.mean(rmses))) < 1e-10
    assert abs(result["metrics"]["rmse"]["std"] - float(np.std(rmses, ddof=1))) < 1e-10


def test_cross_validate_same_seed_is_deterministic(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_linear(load_df_into_session)
    a = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})
    b = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x"})
    assert a["metrics"] == b["metrics"]


def _load_binary(load_df_into_session: Any, n: int = 60) -> None:
    """Noisy logistic fixture — not separable, both classes populated."""
    import pandas as pd

    rng = np.random.RandomState(1)
    x = rng.normal(0, 1, size=n)
    p = 1.0 / (1.0 + np.exp(-1.5 * x))
    y = (rng.uniform(size=n) < p).astype(int)
    # Force both classes present regardless of draw.
    y[0], y[1] = 0, 1
    load_df_into_session("bin", pd.DataFrame({"x": x, "y": y}))


def test_cross_validate_logistic_is_stratified_with_full_metrics(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_binary(load_df_into_session)

    result = call_tool(
        "cross_validate", {"name": "bin", "formula": "y ~ x", "kind": "logistic", "k": 3}
    )

    assert result["ok"] is True
    assert result["stratified"] is True
    assert set(result["metrics"].keys()) == {
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }
    for m in result["metrics"].values():
        assert len(m["per_fold"]) == 3


def test_cross_validate_logistic_folds_keep_class_balance(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Every fold's minority-class count differs from the ideal share by
    at most 1 — the structural guarantee that makes single-class folds
    impossible."""
    import statsmodels.formula.api as smf

    from data_analyst_mcp import session as _session
    from data_analyst_mcp.tools.crossval import _fold_ids

    _load_binary(load_df_into_session)
    df = _session.get_connection().execute('SELECT * FROM "bin"').df()
    full = smf.logit("y ~ x", data=df).fit(disp=0)
    y = np.asarray(full.model.endog, dtype=float)
    fold = _fold_ids(y, 3, 42, stratified=True)
    n_pos = int(np.sum(y == 1))
    for i in range(3):
        pos_in_fold = int(np.sum(y[fold == i] == 1))
        assert abs(pos_in_fold - n_pos / 3) <= 1
        assert pos_in_fold >= 1


def test_cross_validate_minority_class_smaller_than_k(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    rng = np.random.RandomState(2)
    load_df_into_session(
        "rare",
        pd.DataFrame({"x": rng.normal(size=30), "y": [1, 1] + [0] * 28}),
    )

    result = call_tool(
        "cross_validate", {"name": "rare", "formula": "y ~ x", "kind": "logistic", "k": 5}
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "outcome_class_too_small"


def test_cross_validate_nonbinary_outcome_rejected(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    load_df_into_session("multi", pd.DataFrame({"x": range(20), "y": [0, 1, 2, 3] * 5}))
    result = call_tool("cross_validate", {"name": "multi", "formula": "y ~ x", "kind": "logistic"})
    assert result["ok"] is False
    assert result["error"]["type"] == "outcome_dtype_mismatch"


def test_cross_validate_unknown_dataset(call_tool: Any) -> None:
    result = call_tool("cross_validate", {"name": "ghost", "formula": "y ~ x"})
    assert result["ok"] is False
    assert result["error"]["type"] == "dataset_not_found"


@pytest.mark.parametrize("k", [1, 0, 21, -3])
def test_cross_validate_k_static_range(call_tool: Any, load_df_into_session: Any, k: int) -> None:
    _load_linear(load_df_into_session)
    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x", "k": k})
    assert result["ok"] is False
    assert result["error"]["type"] == "k_out_of_range"


def test_cross_validate_k_exceeds_post_drop_rows(call_tool: Any, load_df_into_session: Any) -> None:
    import pandas as pd

    load_df_into_session(
        "holey",
        pd.DataFrame({"x": [1.0, 2.0, 3.0, None, None, None], "y": [1.0] * 6}),
    )
    result = call_tool("cross_validate", {"name": "holey", "formula": "y ~ x", "k": 5})
    assert result["ok"] is False
    assert result["error"]["type"] == "k_out_of_range"
    assert "after NaN drops" in result["error"]["message"]


def test_cross_validate_fold_too_small(call_tool: Any, load_df_into_session: Any) -> None:
    """n=8, k=4 → train slices of 6 rows vs 7 design params."""
    import pandas as pd

    rng = np.random.RandomState(3)
    cols = {f"x{i}": rng.normal(size=8) for i in range(6)}
    load_df_into_session("wide", pd.DataFrame({**cols, "y": rng.normal(size=8)}))
    formula = "y ~ x0 + x1 + x2 + x3 + x4 + x5"
    result = call_tool("cross_validate", {"name": "wide", "formula": formula, "k": 4})
    assert result["ok"] is False
    assert result["error"]["type"] == "fold_too_small"


def test_cross_validate_robust_negbin_rejected(call_tool: Any, load_df_into_session: Any) -> None:
    _load_linear(load_df_into_session)
    result = call_tool(
        "cross_validate",
        {"name": "lin", "formula": "y ~ x", "kind": "negbin", "robust": True},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "robust_not_supported"


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_cross_validate_threshold_endpoints_rejected(
    call_tool: Any, load_df_into_session: Any, threshold: float
) -> None:
    _load_linear(load_df_into_session)
    result = call_tool(
        "cross_validate", {"name": "lin", "formula": "y ~ x", "threshold": threshold}
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "threshold_out_of_range"


def test_cross_validate_formula_error_from_preflight(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_linear(load_df_into_session)
    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ ghost_col"})
    assert result["ok"] is False
    assert result["error"]["type"] == "formula_error"


def test_cross_validate_perfect_separation_from_preflight(
    call_tool: Any, load_df_into_session: Any
) -> None:
    import pandas as pd

    x = np.arange(20, dtype=float)
    y = (x >= 10).astype(int)
    load_df_into_session("sep", pd.DataFrame({"x": x, "y": y}))
    result = call_tool("cross_validate", {"name": "sep", "formula": "y ~ x", "kind": "logistic"})
    assert result["ok"] is False
    assert result["error"]["type"] == "perfect_separation"


def test_cross_validate_poisson_metric_keys(call_tool: Any, load_df_into_session: Any) -> None:
    import pandas as pd

    rng = np.random.RandomState(4)
    x = rng.normal(size=50)
    y = rng.poisson(np.exp(0.5 + 0.3 * x))
    load_df_into_session("counts", pd.DataFrame({"x": x, "y": y}))

    result = call_tool(
        "cross_validate", {"name": "counts", "formula": "y ~ x", "kind": "poisson", "k": 3}
    )

    assert result["ok"] is True
    assert set(result["metrics"].keys()) == {"rmse", "mae", "pearson_chi2", "deviance"}


def test_cross_validate_unknown_kind_is_structured(
    call_tool: Any, load_df_into_session: Any
) -> None:
    _load_linear(load_df_into_session)
    result = call_tool("cross_validate", {"name": "lin", "formula": "y ~ x", "kind": "quantile"})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_kind"


def test_classify_fold_failure_maps_exceptions() -> None:
    from numpy.linalg import LinAlgError
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    from data_analyst_mcp.tools.crossval import _classify_fold_failure

    assert (
        _classify_fold_failure("logistic", PerfectSeparationError("sep"), None)
        == "perfect_separation"
    )
    assert _classify_fold_failure("logistic", LinAlgError("singular"), None) == "perfect_separation"
    assert _classify_fold_failure("ols", ValueError("boom"), None) == "convergence_failed"
    assert _classify_fold_failure("poisson", RuntimeError("nan"), None) == "convergence_failed"


def test_cross_validate_records_replayable_cell(call_tool: Any, load_df_into_session: Any) -> None:
    from data_analyst_mcp.recorder import get_recorder

    _load_linear(load_df_into_session)
    call_tool("cross_validate", {"name": "lin", "formula": "y ~ x", "k": 4, "seed": 7})

    code_cells = [
        c
        for c in get_recorder().cells
        if c["cell_type"] == "code" and c["metadata"]["tool_name"] == "cross_validate"
    ]
    assert len(code_cells) == 1
    src = code_cells[0]["source"]
    assert "RandomState(7)" in src
    assert "smf.ols('y ~ x'" in src  # formulas render via !r — repr quoting
    assert "sm.OLS" in src
    assert "% 4" in src
    assert "pd.DataFrame(_cv_rows)" in src


def test_cross_validate_logistic_cell_is_stratified(
    call_tool: Any, load_df_into_session: Any
) -> None:
    from data_analyst_mcp.recorder import get_recorder

    _load_binary(load_df_into_session)
    call_tool("cross_validate", {"name": "bin", "formula": "y ~ x", "kind": "logistic", "k": 3})
    src = [  # noqa: RUF015
        c
        for c in get_recorder().cells
        if c["cell_type"] == "code" and c["metadata"]["tool_name"] == "cross_validate"
    ][0]["source"]
    assert "for _cls in (0, 1):" in src
    assert "sm.Logit" in src
    assert "roc_auc_score" in src


def test_cv_cell_on_dataframe_dataset_gets_raise_prefix(
    call_tool: Any, load_df_into_session: Any
) -> None:
    """Spec: prefix replay guards, mechanism 2. In-memory datasets are never
    recreated at replay (setup emits only a comment), so the CV cell must
    open with an explanatory raise; the computation stays below as the
    audit trail."""
    import pandas as pd

    from data_analyst_mcp.recorder import get_recorder

    df = pd.DataFrame({"y": [float(i % 7) for i in range(30)], "x": [float(i) for i in range(30)]})
    load_df_into_session("mem", df)

    result = call_tool("cross_validate", {"name": "mem", "formula": "y ~ x", "kind": "ols", "k": 3})
    assert result["ok"] is True

    src = get_recorder().cells[-1]["source"]
    first_line = src.splitlines()[0]
    assert first_line.startswith("raise AssertionError(")
    assert "cross_validate" in first_line
    assert "'mem'" in first_line or '"mem"' in first_line
    assert "in-memory" in first_line
    # Original computation retained below the raise.
    assert "_cv_df = con.sql(" in src


def test_cv_cell_on_file_dataset_has_no_raise_prefix(call_tool: Any, tmp_path: Any) -> None:
    """File-backed sources replay via guarded load cells — no prefix."""
    from data_analyst_mcp.recorder import get_recorder

    csv_path = tmp_path / "file_backed.csv"
    csv_path.write_text("y,x\n" + "\n".join(f"{(i * 7) % 13}.0,{i}.0" for i in range(30)) + "\n")
    call_tool("load_dataset", {"path": str(csv_path), "name": "fb"})
    result = call_tool("cross_validate", {"name": "fb", "formula": "y ~ x", "kind": "ols", "k": 3})
    assert result["ok"] is True
    assert not get_recorder().cells[-1]["source"].startswith("raise AssertionError(")
