"""Tests for the ``predict`` tool.

Slices 1–12 from proposal §TDD slices. Fixtures are inlined synthetic
generators with ``seed=0`` per proposal §Locked test fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

# ---- Synthetic fixtures (proposal §Locked test fixtures) ----------------


def _logistic_fixture_train_test(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture #1: synthetic logistic, n=10000, 80/20 train/test."""
    rng = np.random.default_rng(seed)
    n = 10_000
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    eta = 0.5 * x1 - 0.3 * x2 + 0.8 * x3
    y = rng.binomial(1, expit(eta))
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    train = df.iloc[: int(0.8 * n)].reset_index(drop=True)
    test = df.iloc[int(0.8 * n) :].reset_index(drop=True)
    return train, test


def _ols_fixture_train_test(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture #3: synthetic OLS, n=2000, β_true=[1, 2, -1]."""
    rng = np.random.default_rng(seed)
    n = 2000
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    y = 1.0 * x1 + 2.0 * x2 - 1.0 * x3 + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    train = df.iloc[: int(0.8 * n)].reset_index(drop=True)
    test = df.iloc[int(0.8 * n) :].reset_index(drop=True)
    return train, test


def _poisson_fixture(seed: int = 0) -> pd.DataFrame:
    """Fixture #4: μ=exp(0.5 + 0.3·x), x~Uniform(-1,1), n=2000."""
    rng = np.random.default_rng(seed)
    n = 2000
    x = rng.uniform(-1, 1, size=n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x": x})


# ---- 12 slices from proposal §predict TDD slices ------------------------


def test_predict_response_on_logistic_returns_probabilities(call_tool, load_df_into_session):
    """Slice 1: response on logistic → probabilities in [0, 1]."""
    train, test = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    load_df_into_session("test", test)
    r = call_tool(
        "fit_model",
        {
            "name": "train",
            "formula": "y ~ x1 + x2 + x3",
            "kind": "logistic",
            "model_name": "m",
        },
    )
    assert r["ok"], r
    r = call_tool("predict", {"model_name": "m", "dataset": "test", "limit": 100})
    assert r["ok"] is True
    assert r["output_mode"] == "response"
    for row in r["predictions"]:
        assert 0.0 <= row["y_pred"] <= 1.0


def test_predict_response_on_poisson_returns_nonneg(call_tool, load_df_into_session):
    """Slice 2: response on Poisson → strictly non-negative."""
    df = _poisson_fixture()
    load_df_into_session("counts", df)
    r = call_tool(
        "fit_model",
        {"name": "counts", "formula": "y ~ x", "kind": "poisson", "model_name": "m"},
    )
    assert r["ok"], r
    r = call_tool("predict", {"model_name": "m", "dataset": "counts", "limit": 100})
    assert r["ok"]
    for row in r["predictions"]:
        assert row["y_pred"] >= 0.0


def test_predict_link_on_logistic_returns_real_line(call_tool, load_df_into_session):
    """Slice 3: link output on logistic → η on real line (can be < 0 and > 1)."""
    train, test = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    load_df_into_session("test", test)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    r = call_tool(
        "predict",
        {"model_name": "m", "dataset": "test", "output": "link", "limit": 200},
    )
    assert r["ok"]
    ys = [row["y_pred"] for row in r["predictions"]]
    assert min(ys) < 0.0  # eta is real-valued — should span negative values
    assert max(ys) > 0.0


def test_predict_class_threshold_05_matches_response_threshold(call_tool, load_df_into_session):
    """Slice 4: class on logistic with threshold=0.5 → y_class == (response>=0.5)."""
    train, test = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    load_df_into_session("test", test)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    r = call_tool(
        "predict",
        {"model_name": "m", "dataset": "test", "output": "class", "limit": 200},
    )
    assert r["ok"]
    for row in r["predictions"]:
        # y_pred for class == 0/1 cast of (probability >= threshold)
        # but we cross-check against a separate "response" call.
        assert row["y_class"] in (0, 1)
    r_resp = call_tool(
        "predict",
        {"model_name": "m", "dataset": "test", "output": "response", "limit": 200},
    )
    pairs = list(zip(r["predictions"], r_resp["predictions"], strict=True))
    for cls_row, resp_row in pairs:
        expected = 1 if resp_row["y_pred"] >= 0.5 else 0
        assert cls_row["y_class"] == expected


def test_predict_class_on_ols_returns_class_output_requires_logistic(
    call_tool, load_df_into_session
):
    """Slice 5: class on OLS → class_output_requires_logistic."""
    train, _ = _ols_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "ols", "model_name": "m"},
    )
    r = call_tool("predict", {"model_name": "m", "dataset": "train", "output": "class"})
    assert r["ok"] is False
    assert r["error"]["type"] == "class_output_requires_logistic"


def test_predict_include_se_on_logistic_returns_include_se_requires_ols(
    call_tool, load_df_into_session
):
    """Slice 6: include_se on logistic → include_se_requires_ols."""
    train, _ = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    r = call_tool("predict", {"model_name": "m", "dataset": "train", "include_se": True})
    assert r["ok"] is False
    assert r["error"]["type"] == "include_se_requires_ols"


def test_predict_missing_predictor_returns_missing_predictors(call_tool, load_df_into_session):
    """Slice 7: dataset missing a predictor column → missing_predictors."""
    train, _ = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    # Build a scoring dataset that drops x3.
    bad = train[["y", "x1", "x2"]].copy()
    load_df_into_session("bad", bad)
    r = call_tool("predict", {"model_name": "m", "dataset": "bad"})
    assert r["ok"] is False
    assert r["error"]["type"] == "missing_predictors"
    assert "x3" in r["error"]["message"]


def test_predict_handles_q_quoted_predictors_with_special_chars(call_tool, load_df_into_session):
    """Regression: Q("col with spaces / slashes") must round-trip through
    the predictor-presence check without splitting the inner name into
    multiple bogus tokens."""
    train, test = _logistic_fixture_train_test()
    # Rename one predictor to a name that needs Q() quoting.
    train = train.rename(columns={"x1": "x one/two"})
    test = test.rename(columns={"x1": "x one/two"})
    load_df_into_session("train", train)
    load_df_into_session("test", test)
    r = call_tool(
        "fit_model",
        {
            "name": "train",
            "formula": 'y ~ Q("x one/two") + x2 + x3',
            "kind": "logistic",
            "model_name": "m",
        },
    )
    assert r["ok"], r
    r = call_tool("predict", {"model_name": "m", "dataset": "test", "limit": 5})
    assert r["ok"] is True, r
    for row in r["predictions"]:
        assert 0.0 <= row["y_pred"] <= 1.0


def test_predict_drops_rows_with_nan_predictors_and_skips_their_row_index(
    call_tool, load_df_into_session
):
    """Slice 8: patsy drops NaN-predictor rows; dropped_rows>0; row_index non-contiguous."""
    train, _ = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    # Create a scoring dataset with planted NaN in rows 5 and 10.
    test = train.iloc[:20].copy().reset_index(drop=True)
    test.loc[5, "x1"] = np.nan
    test.loc[10, "x2"] = np.nan
    load_df_into_session("test", test)
    r = call_tool("predict", {"model_name": "m", "dataset": "test", "limit": 100})
    assert r["ok"]
    assert r["dropped_rows"] == 2
    indices = [row["row_index"] for row in r["predictions"]]
    assert 5 not in indices
    assert 10 not in indices
    # Source-dataset row_index is non-contiguous after drops.
    assert indices == [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_predict_total_rows_excludes_dropped_rows(call_tool, load_df_into_session):
    """Slice 9: total_rows == len(dataset) - dropped_rows."""
    train, _ = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    test = train.iloc[:50].copy().reset_index(drop=True)
    test.loc[[3, 7, 15], "x1"] = np.nan
    load_df_into_session("test", test)
    r = call_tool("predict", {"model_name": "m", "dataset": "test", "limit": 1000})
    assert r["ok"]
    assert r["dropped_rows"] == 3
    assert r["total_rows"] == len(test) - 3


def test_predict_pagination_with_limit_and_cursor(call_tool, load_df_into_session):
    """Slice 10: limit=10, cursor='20' → rows starting at source row_index 20."""
    train, test = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    load_df_into_session("test", test)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    r = call_tool(
        "predict",
        {"model_name": "m", "dataset": "test", "limit": 10, "cursor": "20"},
    )
    assert r["ok"]
    # No drops planted → row_index is contiguous; first page row_index >= 20.
    assert r["predictions"][0]["row_index"] == 20
    assert len(r["predictions"]) == 10


def test_predict_threshold_zero_or_one_returns_threshold_out_of_range(
    call_tool, load_df_into_session
):
    """Slice 11: threshold=0 or threshold=1 → threshold_out_of_range."""
    train, _ = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    for bad in (0.0, 1.0):
        r = call_tool(
            "predict",
            {"model_name": "m", "dataset": "train", "output": "class", "threshold": bad},
        )
        assert r["ok"] is False
        assert r["error"]["type"] == "threshold_out_of_range"


def test_predict_include_se_on_ols_returns_se_block(call_tool, load_df_into_session):
    """Slice 12: OLS include_se=True → se_mean / mean_ci_lower / mean_ci_upper per row."""
    train, test = _ols_fixture_train_test()
    load_df_into_session("train", train)
    load_df_into_session("test", test)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "ols", "model_name": "m"},
    )
    r = call_tool(
        "predict",
        {"model_name": "m", "dataset": "test", "include_se": True, "limit": 50},
    )
    assert r["ok"]
    for row in r["predictions"]:
        assert "se_mean" in row
        assert row["se_mean"] > 0.0
        assert row["mean_ci_lower"] < row["y_pred"] < row["mean_ci_upper"]


# ---- Additional sanity checks for error paths ---------------------------


def test_predict_unknown_model_returns_model_not_found(call_tool):
    r = call_tool("predict", {"model_name": "nope", "dataset": "whatever"})
    assert r["ok"] is False
    assert r["error"]["type"] == "model_not_found"


def test_predict_unknown_dataset_returns_dataset_not_found(call_tool, load_df_into_session):
    train, _ = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    r = call_tool("predict", {"model_name": "m", "dataset": "nope"})
    assert r["ok"] is False
    assert r["error"]["type"] == "dataset_not_found"


def test_predict_emits_recorder_cell(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    train, _ = _logistic_fixture_train_test()
    load_df_into_session("train", train)
    call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "logistic", "model_name": "m"},
    )
    before = len(get_recorder().cells)
    call_tool("predict", {"model_name": "m", "dataset": "train", "limit": 5})
    after = len(get_recorder().cells)
    # One markdown + one code cell.
    assert after - before == 2


@pytest.mark.parametrize("output", ["response", "link"])
def test_predict_on_training_dataset_matches_fit_in_sample(call_tool, load_df_into_session, output):
    """Open question 4: in-sample predict matches the model's own predict path."""
    train, _ = _ols_fixture_train_test()
    load_df_into_session("train", train)
    r = call_tool(
        "fit_model",
        {"name": "train", "formula": "y ~ x1 + x2 + x3", "kind": "ols", "model_name": "m"},
    )
    assert r["ok"]
    r = call_tool(
        "predict",
        {"model_name": "m", "dataset": "train", "output": output, "limit": 5},
    )
    assert r["ok"]
    # For OLS, link == response — both pass through identity.
    assert len(r["predictions"]) == 5
