"""Tests for the ``cross_validate`` tool."""

from __future__ import annotations

from typing import Any

import numpy as np


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
