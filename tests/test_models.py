"""Tests for the modeling tool (fit_model).

Every assertion against a numeric coefficient / std-error / fit-statistic is
hard-pinned to a value computed independently in statsmodels with a fixed
random seed (or against a published statsmodels demo dataset), at the
tolerance noted next to the assertion. See the comment above each
assertion for the source of the expected number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


def test_fit_model_unknown_dataset_returns_not_found(call_tool):
    result = call_tool("fit_model", {"name": "nope", "formula": "y ~ x", "kind": "ols"})
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_fit_model_unknown_kind_returns_invalid_kind(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}))
    result = call_tool("fit_model", {"name": "tiny", "formula": "y ~ x", "kind": "probit"})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_kind"


def test_fit_model_missing_column_returns_formula_error(call_tool, load_df_into_session):
    load_df_into_session("tiny", pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}))
    result = call_tool("fit_model", {"name": "tiny", "formula": "y ~ nope", "kind": "ols"})
    assert result["ok"] is False
    assert result["error"]["type"] == "formula_error"


# === OLS — known-answer via statsmodels Duncan dataset ===
#
# All OLS expected values below are pinned from a one-time REPL run of
#   statsmodels.api.datasets.get_rdataset("Duncan", "carData").data
# fitted with smf.ols("prestige ~ income + education", data=df).fit().
# Source comment on each assertion cites the exact attribute used.


def _duncan_df() -> pd.DataFrame:
    """Load the Duncan occupational prestige dataset as a DataFrame.

    Cached in-memory once per test session via statsmodels' get_rdataset.
    """
    import statsmodels.api as sm

    ds = sm.datasets.get_rdataset("Duncan", "carData", cache=True)
    return ds.data


def test_fit_model_ols_returns_pinned_coefficients(call_tool, load_df_into_session):
    load_df_into_session("duncan", _duncan_df())
    result = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    assert result["ok"] is True
    # smf.ols("prestige ~ income + education", data=duncan).fit().params:
    #   Intercept = -6.064662922103344
    #   income    =  0.5987328215294951
    #   education =  0.5458339094008795
    by_name = {c["name"]: c for c in result["coefficients"]}
    assert by_name["Intercept"]["estimate"] == pytest.approx(-6.064662922, abs=1e-3)
    assert by_name["income"]["estimate"] == pytest.approx(0.5987328215, abs=1e-3)
    assert by_name["education"]["estimate"] == pytest.approx(0.5458339094, abs=1e-3)
    # m.bse — non-robust:
    #   Intercept = 4.271941174529124
    #   income    = 0.11966734981235409
    #   education = 0.09825264133039983
    assert by_name["income"]["std_err"] == pytest.approx(0.1196673498, abs=1e-3)
    # m.tvalues:
    #   income    = 5.003309778885767
    assert by_name["income"]["t"] == pytest.approx(5.003309779, abs=1e-3)
    # m.pvalues:
    #   income    = 1.0531839714905279e-05
    assert by_name["income"]["p_value"] == pytest.approx(1.0531839715e-05, abs=1e-7)
    # m.conf_int():
    #   income → (0.3572343324484092, 0.8402313106105811)
    assert by_name["income"]["ci_low"] == pytest.approx(0.3572343324, abs=1e-3)
    assert by_name["income"]["ci_high"] == pytest.approx(0.8402313106, abs=1e-3)


def test_fit_model_ols_returns_pinned_fit_block(call_tool, load_df_into_session):
    load_df_into_session("duncan", _duncan_df())
    result = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    fit = result["fit"]
    # m.rsquared       = 0.8281734172543813
    # m.rsquared_adj   = 0.8199911990283995
    # m.aic            = 363.9644534078201
    # m.bic            = 369.38444087713106
    # m.nobs           = 45.0
    # m.df_resid       = 42.0
    assert fit["r_squared"] == pytest.approx(0.82817341725, abs=1e-3)
    assert fit["adj_r_squared"] == pytest.approx(0.81999119903, abs=1e-3)
    assert fit["aic"] == pytest.approx(363.96445341, abs=1e-3)
    assert fit["bic"] == pytest.approx(369.38444088, abs=1e-3)
    assert fit["n_obs"] == 45
    assert fit["df_resid"] == 42


def test_fit_model_ols_returns_pinned_core_diagnostics(call_tool, load_df_into_session):
    load_df_into_session("duncan", _duncan_df())
    result = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    diag = result["diagnostics"]
    # statsmodels.stats.diagnostic.het_breuschpagan(m.resid, m.model.exog)[1]
    #   = 0.7500543806429694
    assert diag["breusch_pagan_p"] == pytest.approx(0.7500543806, abs=1e-4)
    # statsmodels.stats.stattools.durbin_watson(m.resid) = 1.4583328622352028
    assert diag["durbin_watson"] == pytest.approx(1.4583328622, abs=1e-4)
    # statsmodels.stats.stattools.jarque_bera(m.resid)[1] = 0.771233639090612
    assert diag["jarque_bera_p"] == pytest.approx(0.771233639, abs=1e-4)
    # m.condition_number = 162.8971473807718
    assert diag["condition_number"] == pytest.approx(162.8971474, abs=1e-3)


def test_fit_model_ols_returns_pinned_vif_per_coefficient(call_tool, load_df_into_session):
    load_df_into_session("duncan", _duncan_df())
    result = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    vif = result["diagnostics"]["vif"]
    # variance_inflation_factor on Duncan design (intercept excluded):
    #   income    = 2.1049004710947665
    #   education = 2.1049004710947674
    assert vif["income"] == pytest.approx(2.10490047, abs=1e-3)
    assert vif["education"] == pytest.approx(2.10490047, abs=1e-3)
    # Intercept must NOT appear in the VIF dict.
    assert "Intercept" not in vif


def _collinear_df() -> pd.DataFrame:
    """Build a deliberately collinear OLS fixture (VIF >> 10)."""
    rng = np.random.default_rng(20260511)
    n = 50
    x1 = rng.standard_normal(n)
    x2 = x1 + rng.standard_normal(n) * 0.01
    y = rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_fit_model_ols_emits_high_multicollinearity_warning(call_tool, load_df_into_session):
    load_df_into_session("collinear", _collinear_df())
    result = call_tool(
        "fit_model",
        {"name": "collinear", "formula": "y ~ x1 + x2", "kind": "ols"},
    )
    # variance_inflation_factor on this fixture (seed 20260511):
    #   vif x1 ≈ 6672.05, vif x2 ≈ 6672.05  → high_multicollinearity must fire
    vif = result["diagnostics"]["vif"]
    assert vif["x1"] > 10
    assert vif["x2"] > 10
    assert "high_multicollinearity" in result["warnings"]


def _heteroskedastic_df() -> pd.DataFrame:
    """Build a deliberately heteroskedastic OLS fixture (BP p << 0.05)."""
    rng = np.random.default_rng(20260511)
    n = 200
    x = rng.standard_normal(n)
    y = x + rng.normal(0, np.abs(x) + 0.1)
    return pd.DataFrame({"y": y, "x": x})


def test_fit_model_ols_emits_heteroskedasticity_warning(call_tool, load_df_into_session):
    load_df_into_session("hetero", _heteroskedastic_df())
    result = call_tool("fit_model", {"name": "hetero", "formula": "y ~ x", "kind": "ols"})
    # het_breuschpagan p on this fixture (seed 20260511) = 0.00014260162314631662
    assert result["diagnostics"]["breusch_pagan_p"] < 0.05
    assert "heteroskedasticity" in result["warnings"]


def _heavy_tail_df() -> pd.DataFrame:
    """Build an OLS fixture with t(3) heavy-tailed noise (JB p << 0.05)."""
    rng = np.random.default_rng(20260511)
    n = 200
    x = rng.standard_normal(n)
    noise = rng.standard_t(df=3, size=n) * 2
    y = x + noise
    return pd.DataFrame({"y": y, "x": x})


def test_fit_model_ols_emits_non_normal_residuals_warning(call_tool, load_df_into_session):
    load_df_into_session("heavy", _heavy_tail_df())
    result = call_tool("fit_model", {"name": "heavy", "formula": "y ~ x", "kind": "ols"})
    # jarque_bera p on this fixture (seed 20260511) = 0.0023420079865034194
    assert result["diagnostics"]["jarque_bera_p"] < 0.05
    assert "non_normal_residuals" in result["warnings"]


def test_fit_model_ols_robust_switches_to_hc3_standard_errors(call_tool, load_df_into_session):
    load_df_into_session("duncan", _duncan_df())
    plain = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    robust = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
            "robust": True,
        },
    )
    pl = {c["name"]: c for c in plain["coefficients"]}
    rb = {c["name"]: c for c in robust["coefficients"]}
    # Coefficients are unchanged across robust/non-robust.
    assert rb["income"]["estimate"] == pytest.approx(pl["income"]["estimate"], abs=1e-9)
    # HC3 std errors on Duncan (pinned via .fit(cov_type='HC3')):
    #   income    = 0.18346773162473987
    #   education = 0.14946918649417798
    assert rb["income"]["std_err"] == pytest.approx(0.18346773162, abs=1e-3)
    assert rb["education"]["std_err"] == pytest.approx(0.14946918649, abs=1e-3)
    # And they MUST differ from the non-robust std errors.
    assert rb["income"]["std_err"] != pytest.approx(pl["income"]["std_err"], abs=1e-6)


def test_fit_model_ols_returns_plain_english_interpretation(call_tool, load_df_into_session):
    load_df_into_session("duncan", _duncan_df())
    result = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    interp = result["interpretation"]
    assert isinstance(interp, str)
    assert len(interp) > 30
    # Strongest signal on Duncan = `education` (smallest p, p ≈ 1.73e-6),
    # and its coefficient is positive, so the interpretation must mention it.
    assert "education" in interp


def test_fit_model_ols_records_markdown_and_code_cells(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("duncan", _duncan_df())
    call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    cells = get_recorder().cells
    assert len(cells) == 2
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert cells[0]["metadata"]["tool_name"] == "fit_model"
    assert cells[1]["metadata"]["tool_name"] == "fit_model"
    assert "prestige ~ income + education" in cells[1]["source"]
    assert "smf.ols" in cells[1]["source"]


# === Logistic — known-answer on a seeded synthetic dataset ===
#
# Generated with numpy.random.default_rng(20260511), n=500:
#   x, z ~ N(0,1); logit = -0.5 + 1.2 x - 0.8 z; p = sigmoid(logit);
#   y = (uniform < p).
# Fitted once via smf.logit("y ~ x + z").fit(disp=0); pinned params below
# are the LITERAL output (not the data-generating parameters) so the
# assertion is fully reproducible.


def _logistic_df() -> pd.DataFrame:
    rng = np.random.default_rng(20260511)
    n = 500
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    logit = -0.5 + 1.2 * x - 0.8 * z
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_model_logistic_returns_pinned_coefficients(call_tool, load_df_into_session):
    load_df_into_session("logi", _logistic_df())
    result = call_tool("fit_model", {"name": "logi", "formula": "y ~ x + z", "kind": "logistic"})
    assert result["ok"] is True
    # smf.logit("y ~ x + z", data=logi).fit(disp=0).params:
    #   Intercept = -0.2936185311262021
    #   x         =  1.1464343597441933
    #   z         = -0.7152252266279717
    by_name = {c["name"]: c for c in result["coefficients"]}
    assert by_name["Intercept"]["estimate"] == pytest.approx(-0.2936185311, abs=1e-3)
    assert by_name["x"]["estimate"] == pytest.approx(1.1464343597, abs=1e-3)
    assert by_name["z"]["estimate"] == pytest.approx(-0.7152252266, abs=1e-3)
    # m.bse:  x = 0.12914932615314562
    assert by_name["x"]["std_err"] == pytest.approx(0.12914932615, abs=1e-3)


def test_fit_model_logistic_returns_pseudo_r_squared_fit_block(call_tool, load_df_into_session):
    load_df_into_session("logi", _logistic_df())
    result = call_tool("fit_model", {"name": "logi", "formula": "y ~ x + z", "kind": "logistic"})
    fit = result["fit"]
    # Logistic does NOT report r_squared / adj_r_squared.
    assert "r_squared" not in fit
    assert "adj_r_squared" not in fit
    # Pinned from m.aic / m.bic / m.nobs / m.df_resid / m.prsquared on the
    # seeded fixture:
    #   aic            = 552.8780654766584
    #   bic            = 565.521889771925
    #   nobs           = 500
    #   df_resid       = 497
    #   prsquared      = 0.19966906245268023
    assert fit["aic"] == pytest.approx(552.8780655, abs=1e-2)
    assert fit["bic"] == pytest.approx(565.5218898, abs=1e-2)
    assert fit["n_obs"] == 500
    assert fit["df_resid"] == 497
    assert fit["pseudo_r_squared"] == pytest.approx(0.1996690625, abs=1e-3)


def test_fit_model_logistic_diagnostics_omit_ols_only_fields(call_tool, load_df_into_session):
    load_df_into_session("logi", _logistic_df())
    result = call_tool("fit_model", {"name": "logi", "formula": "y ~ x + z", "kind": "logistic"})
    diag = result["diagnostics"]
    # condition_number is always present (numpy.linalg.cond fallback).
    assert isinstance(diag["condition_number"], float)
    # OLS-specific residual diagnostics MUST be null for logistic.
    assert diag["breusch_pagan_p"] is None
    assert diag["durbin_watson"] is None
    assert diag["jarque_bera_p"] is None
    # VIF is OLS-only per spec §5.9.
    assert diag["vif"] is None


def _logistic_collinear_df() -> pd.DataFrame:
    """Build a deliberately collinear logistic fixture (VIF >> 10)."""
    rng = np.random.default_rng(20260511)
    n = 500
    x = rng.standard_normal(n)
    xx = x + rng.standard_normal(n) * 0.01
    y = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame({"y": y, "x": x, "xx": xx})


def test_fit_model_logistic_emits_high_multicollinearity_warning(call_tool, load_df_into_session):
    load_df_into_session("logi_col", _logistic_collinear_df())
    result = call_tool(
        "fit_model",
        {"name": "logi_col", "formula": "y ~ x + xx", "kind": "logistic"},
    )
    # variance_inflation_factor on this fixture (seed 20260511, logistic exog):
    #   vif x  ≈ 10530.5, vif xx ≈ 10530.5  → high_multicollinearity must fire
    assert "high_multicollinearity" in result["warnings"]


def test_fit_model_logistic_interpretation_mentions_odds(call_tool, load_df_into_session):
    load_df_into_session("logi", _logistic_df())
    result = call_tool("fit_model", {"name": "logi", "formula": "y ~ x + z", "kind": "logistic"})
    interp = result["interpretation"]
    assert isinstance(interp, str)
    # On this seeded fixture, `x` has the smallest p-value (≈ 0), so the
    # interpretation MUST name it and report an odds-ratio change.
    assert "x" in interp
    assert "odds" in interp.lower()


def test_fit_model_logistic_records_markdown_and_code_cells(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("logi", _logistic_df())
    call_tool("fit_model", {"name": "logi", "formula": "y ~ x + z", "kind": "logistic"})
    cells = get_recorder().cells
    assert len(cells) == 2
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert "smf.logit" in cells[1]["source"]


def test_fit_model_logistic_coerces_bool_endog_to_numeric(call_tool, load_df_into_session):
    """Real-world boolean targets (e.g. `won`) must fit without manual casting.

    statsmodels' patsy refuses to dummy-code a single-column boolean endog,
    so the tool has to coerce it itself before handing off. Without this,
    every CSV that stores a 0/1 outcome as a Python bool (DuckDB infers
    BOOLEAN for `true`/`false`) breaks the logistic path with an opaque
    "endog has evaluated to ... multiple columns" error.
    """
    df = pd.DataFrame(
        {
            "y": [True, False, True, False, True, False, True, False, True, False],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    load_df_into_session("bool_logi", df)
    result = call_tool(
        "fit_model",
        {"name": "bool_logi", "formula": "y ~ x", "kind": "logistic"},
    )
    assert result["ok"] is True, result
    # Coefficient for `x` must be present and a finite number.
    by_name = {c["name"]: c for c in result["coefficients"]}
    assert "x" in by_name
    assert by_name["x"]["estimate"] is not None


def test_fit_model_logistic_coerces_nullable_boolean_endog(call_tool, load_df_into_session):
    """Nullable pandas BooleanDtype endogs must also be coerced.

    DuckDB returns a pandas extension ``BooleanDtype`` (not numpy ``bool``)
    whenever a CSV boolean column contains NULL — which is the exact shape
    of ``synthetic_crm/opportunities.csv``'s ``won`` column (NULL while the
    opportunity is still open). statsmodels rejects that dtype with
    ``Cannot interpret 'BooleanDtype' as a data type``.
    """
    df = pd.DataFrame(
        {
            "y": pd.array(
                [True, False, True, False, True, None, True, False, True, False],
                dtype="boolean",
            ),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    load_df_into_session("ext_logi", df)
    result = call_tool(
        "fit_model",
        {"name": "ext_logi", "formula": "y ~ x", "kind": "logistic"},
    )
    assert result["ok"] is True, result


# === Poisson — known-answer on a seeded synthetic dataset ===
#
# Generated with numpy.random.default_rng(20260511), n=400:
#   x ~ N(0,1); lambda = exp(0.5 + 0.7 x); y ~ Poisson(lambda).
# Fitted via smf.poisson("y ~ x").fit(disp=0). Pinned params are the LITERAL
# REPL output (≤1e-2 tolerance — MLE recovery on Poisson is noisier).


def _poisson_df() -> pd.DataFrame:
    rng = np.random.default_rng(20260511)
    n = 400
    x = rng.standard_normal(n)
    lam = np.exp(0.5 + 0.7 * x)
    y = rng.poisson(lam)
    return pd.DataFrame({"y": y, "x": x})


def test_fit_model_poisson_returns_pinned_coefficients(call_tool, load_df_into_session):
    load_df_into_session("pois", _poisson_df())
    result = call_tool("fit_model", {"name": "pois", "formula": "y ~ x", "kind": "poisson"})
    assert result["ok"] is True
    # smf.poisson("y ~ x", data=pois).fit(disp=0).params:
    #   Intercept = 0.4955550345349605
    #   x         = 0.7087011623187054
    by_name = {c["name"]: c for c in result["coefficients"]}
    assert by_name["Intercept"]["estimate"] == pytest.approx(0.4955550345, abs=1e-2)
    assert by_name["x"]["estimate"] == pytest.approx(0.7087011623, abs=1e-2)
    # m.bse:  x = 0.03469217336544108
    assert by_name["x"]["std_err"] == pytest.approx(0.0346921734, abs=1e-2)


def test_fit_model_poisson_returns_pseudo_r_squared_fit_block(call_tool, load_df_into_session):
    load_df_into_session("pois", _poisson_df())
    result = call_tool("fit_model", {"name": "pois", "formula": "y ~ x", "kind": "poisson"})
    fit = result["fit"]
    # m.aic        = 1261.6421455062296
    # m.bic        = 1269.6250746004455
    # m.nobs       = 400
    # m.df_resid   = 398
    # m.prsquared  = 0.2467755286394614
    assert "r_squared" not in fit
    assert fit["aic"] == pytest.approx(1261.6421455, abs=1e-2)
    assert fit["bic"] == pytest.approx(1269.6250746, abs=1e-2)
    assert fit["n_obs"] == 400
    assert fit["df_resid"] == 398
    assert fit["pseudo_r_squared"] == pytest.approx(0.2467755286, abs=1e-3)


def _overdispersed_df() -> pd.DataFrame:
    """Build an overdispersed Poisson fixture (NegBinom counts)."""
    rng = np.random.default_rng(20260511)
    n = 400
    x = rng.standard_normal(n)
    lam = np.exp(0.5 + 0.7 * x)
    # NegBinom with mean=lam but variance > mean → overdispersion vs Poisson.
    y = rng.negative_binomial(2, 2 / (2 + lam))
    return pd.DataFrame({"y": y, "x": x})


def test_fit_model_poisson_emits_overdispersion_warning(call_tool, load_df_into_session):
    load_df_into_session("overdisp", _overdispersed_df())
    result = call_tool("fit_model", {"name": "overdisp", "formula": "y ~ x", "kind": "poisson"})
    # Pearson dispersion on this fixture (seed 20260511) ≈ 1.89 >> 1.5.
    assert "overdispersion" in result["warnings"]


def test_fit_model_poisson_no_overdispersion_warning_on_clean_fixture(
    call_tool, load_df_into_session
):
    load_df_into_session("pois", _poisson_df())
    result = call_tool("fit_model", {"name": "pois", "formula": "y ~ x", "kind": "poisson"})
    # Pearson dispersion ≈ 0.95 on the clean fixture → no warning.
    assert "overdispersion" not in result["warnings"]


def test_fit_model_poisson_records_markdown_and_code_cells(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("pois", _poisson_df())
    call_tool("fit_model", {"name": "pois", "formula": "y ~ x", "kind": "poisson"})
    cells = get_recorder().cells
    assert len(cells) == 2
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert "smf.poisson" in cells[1]["source"]


# === MCP wiring — exercise the full FastMCP call_tool path per kind ===


def test_call_tool_fit_model_ols_returns_full_envelope(call_tool, load_df_into_session):
    load_df_into_session("duncan", _duncan_df())
    result = call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
        },
    )
    assert result["ok"] is True
    assert {"coefficients", "fit", "diagnostics", "warnings", "interpretation"} <= set(result)


def test_call_tool_fit_model_logistic_returns_full_envelope(call_tool, load_df_into_session):
    load_df_into_session("logi", _logistic_df())
    result = call_tool("fit_model", {"name": "logi", "formula": "y ~ x + z", "kind": "logistic"})
    assert result["ok"] is True
    assert {"coefficients", "fit", "diagnostics", "warnings", "interpretation"} <= set(result)


def test_call_tool_fit_model_poisson_returns_full_envelope(call_tool, load_df_into_session):
    load_df_into_session("pois", _poisson_df())
    result = call_tool("fit_model", {"name": "pois", "formula": "y ~ x", "kind": "poisson"})
    assert result["ok"] is True
    assert {"coefficients", "fit", "diagnostics", "warnings", "interpretation"} <= set(result)


# === Recorder fidelity — emitted code cell matches the runtime template ===


def test_fit_model_emitted_code_cell_matches_runtime_template(call_tool, load_df_into_session):
    """The recorded code cell must (a) reload the dataset via DuckDB, and
    (b) fit the same statsmodels model with the same formula + cov_type.

    Re-execution against the same data is a Phase 6 integration test; here
    we just verify the cell text is syntactically equivalent to the template
    we would run by hand.
    """
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("duncan", _duncan_df())
    call_tool(
        "fit_model",
        {
            "name": "duncan",
            "formula": "prestige ~ income + education",
            "kind": "ols",
            "robust": True,
        },
    )
    code = get_recorder().cells[1]["source"]
    expected = (
        "import statsmodels.formula.api as smf\n"
        'df = con.sql("SELECT * FROM duncan").df()\n'
        'model = smf.ols("prestige ~ income + education", data=df).fit(cov_type="HC3")\n'
        "model.summary()"
    )
    assert code == expected


# === Negative binomial (kind="negbin") — 15 TDD slices per
#     docs/proposals/negbin_kind.md ===
#
# Synthetic NB fixtures use numpy.random.default_rng(seed=0); the NB2
# variance parameterization (var = mu + alpha*mu^2) is sampled via
# numpy.random.Generator.negative_binomial with r = 1/alpha and
# p = r / (r + mu).


def _negbin_known_answer_df() -> pd.DataFrame:
    """Fixture #1 — NB2 with beta_true=[0.5, 0.3], alpha_true=1.5, n=2000."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(-1, 1, n)
    mu = np.exp(0.5 + 0.3 * x)
    alpha_true = 1.5
    r = 1.0 / alpha_true
    p = r / (r + mu)
    y = rng.negative_binomial(r, p)
    return pd.DataFrame({"y": y, "x": x})


def _negbin_overdispersed_df() -> pd.DataFrame:
    """Fixture #2 — Poisson(exp(0.5+0.3x)) with 10% inflated rows from Poisson(20)."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu)
    mask = rng.random(n) < 0.1
    y[mask] = rng.poisson(20, mask.sum())
    return pd.DataFrame({"y": y, "x": x})


def _negbin_true_poisson_df() -> pd.DataFrame:
    """Fixture #3 — true Poisson with mu=5 everywhere."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.poisson(5, n)
    x = rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x": x})


def _negbin_unstable_alpha_df() -> pd.DataFrame:
    """Fixture #4 — small n=25, true alpha=3, mu=1.5 — pins SE/alpha > 0.6.

    The proposal's recipe (n=80, NB(mu=2, alpha=2)) yields SE/alpha ≈ 0.23
    on this statsmodels release, which does not trigger the warning. After
    grid-searching seed/n/alpha combinations, (n=25, seed=1, alpha_true=3)
    pins SE/alpha ≈ 0.73 — well above the 0.5 threshold with margin.
    """
    rng = np.random.default_rng(1)
    n = 25
    x = rng.uniform(-1, 1, n)
    mu = np.full(n, 1.5)
    alpha_true = 3.0
    r = 1.0 / alpha_true
    p = r / (r + mu)
    y = rng.negative_binomial(r, p)
    return pd.DataFrame({"y": y, "x": x})


def _negbin_separated_df() -> pd.DataFrame:
    """Fixture #5 — perfectly separated counts; NB MLE does not converge."""
    return pd.DataFrame(
        {
            "y": [0, 0, 0, 0, 100, 100, 100, 100],
            "x": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


# --- Slice 1 -----------------------------------------------------------
def test_fit_model_negbin_rejects_negative_endog(call_tool, load_df_into_session):
    df = pd.DataFrame({"y": [1, 2, -1, 3, 4], "x": [0.1, 0.2, 0.3, 0.4, 0.5]})
    load_df_into_session("neg_y", df)
    result = call_tool("fit_model", {"name": "neg_y", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is False
    assert result["error"]["type"] == "negbin_requires_nonneg_int"
    # Hint must name the offending column so the agent can locate it.
    assert "y" in result["error"]["message"]


# --- Slice 2 -----------------------------------------------------------
def test_fit_model_negbin_rejects_non_integer_floats(call_tool, load_df_into_session):
    df = pd.DataFrame({"y": [1.5, 2.0, 3.0], "x": [0.1, 0.2, 0.3]})
    load_df_into_session("frac_y", df)
    result = call_tool("fit_model", {"name": "frac_y", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is False
    assert result["error"]["type"] == "negbin_requires_nonneg_int"


# --- Slice 3 -----------------------------------------------------------
def test_fit_model_negbin_accepts_integer_valued_floats(call_tool, load_df_into_session):
    """Integer-valued floats (e.g. [1.0, 2.0]) are accepted with a
    ``coerced_float_to_int`` warning so silent unit errors are still visible
    in the response envelope."""
    df = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 6, "x": list(range(60))}
    )
    load_df_into_session("intfloat_y", df)
    result = call_tool("fit_model", {"name": "intfloat_y", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is True, result
    assert "coerced_float_to_int" in result["warnings"]


# --- Slice 4 -----------------------------------------------------------
def test_fit_model_negbin_rejects_robust_true(call_tool, load_df_into_session):
    load_df_into_session("nb", _negbin_known_answer_df())
    result = call_tool(
        "fit_model",
        {"name": "nb", "formula": "y ~ x", "kind": "negbin", "robust": True},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "robust_not_supported"


# --- Slice 5 — known-answer (load-bearing correctness) -----------------
def test_fit_model_negbin_recovers_synthetic_truth(call_tool, load_df_into_session):
    """Slope β within ±0.05, intercept within ±0.1, α within ±0.1.

    Proposal §Fixtures #1 lists "β within ±0.05" globally, but on the
    seed=0 fixture the intercept lands at 0.434 vs truth 0.5 — a ~0.066
    offset that is ≈2·SE on this n=2000 NB sample (SE on Intercept ≈
    0.032, computed empirically). The proposal's "±0.05 (≈3·SE)" estimate
    was an underestimate of the intercept SE; we widen the intercept
    tolerance to ±0.1 (still tight enough to fail an NB1 vs NB2 mixup by
    orders of magnitude) and keep the slope ±0.05 which the seed-0 fit
    comfortably satisfies (β_x = 0.298, off truth by 0.002).
    """
    load_df_into_session("nb", _negbin_known_answer_df())
    result = call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is True, result
    by_name = {c["name"]: c for c in result["coefficients"]}
    # Truth: Intercept=0.5, x=0.3, alpha=1.5
    assert by_name["Intercept"]["estimate"] == pytest.approx(0.5, abs=0.1)
    assert by_name["x"]["estimate"] == pytest.approx(0.3, abs=0.05)
    assert result["fit"]["dispersion_alpha"] == pytest.approx(1.5, abs=0.1)


# --- Slice 6 -----------------------------------------------------------
def test_fit_model_negbin_exposes_dispersion_alpha(call_tool, load_df_into_session):
    """``fit.dispersion_alpha`` must equal ``model.params['alpha']`` exactly."""
    import statsmodels.formula.api as smf

    df = _negbin_known_answer_df()
    load_df_into_session("nb", df)
    result = call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    m = smf.negativebinomial("y ~ x", data=df).fit(disp=False)
    assert result["fit"]["dispersion_alpha"] == pytest.approx(float(m.params["alpha"]), abs=1e-9)


# --- Slice 7 -----------------------------------------------------------
def test_fit_model_negbin_exposes_dispersion_alpha_se(call_tool, load_df_into_session):
    """``fit.dispersion_alpha_se`` must equal ``model.bse['alpha']`` exactly."""
    import statsmodels.formula.api as smf

    df = _negbin_known_answer_df()
    load_df_into_session("nb", df)
    result = call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    m = smf.negativebinomial("y ~ x", data=df).fit(disp=False)
    assert result["fit"]["dispersion_alpha_se"] == pytest.approx(float(m.bse["alpha"]), abs=1e-9)


# --- Slice 8 -----------------------------------------------------------
def test_fit_model_negbin_pearson_chi2_over_df_matches_formula(call_tool, load_df_into_session):
    """``fit.pearson_chi2_over_df`` matches ``sum(resid_pearson^2) / df_resid``."""
    import statsmodels.formula.api as smf

    df = _negbin_known_answer_df()
    load_df_into_session("nb", df)
    result = call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    m = smf.negativebinomial("y ~ x", data=df).fit(disp=False)
    resid = np.asarray(m.resid_pearson)
    expected = float(np.sum(resid * resid) / m.df_resid)
    assert result["fit"]["pearson_chi2_over_df"] == pytest.approx(expected, abs=1e-9)


# --- Slice 9 -----------------------------------------------------------
def test_fit_model_negbin_beats_poisson_aic_on_overdispersed_data(call_tool, load_df_into_session):
    """NB AIC must be ≥50 points below Poisson AIC on the overdispersed fixture."""
    load_df_into_session("od", _negbin_overdispersed_df())
    pois = call_tool("fit_model", {"name": "od", "formula": "y ~ x", "kind": "poisson"})
    nb = call_tool("fit_model", {"name": "od", "formula": "y ~ x", "kind": "negbin"})
    assert pois["ok"] is True and nb["ok"] is True
    assert pois["fit"]["aic"] - nb["fit"]["aic"] >= 50.0


# --- Slice 10 ----------------------------------------------------------
def test_fit_model_negbin_emits_underdispersion_warning_on_true_poisson(
    call_tool, load_df_into_session
):
    load_df_into_session("tp", _negbin_true_poisson_df())
    result = call_tool("fit_model", {"name": "tp", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is True, result
    fit = result["fit"]
    # On truly Poisson data the NB collapses: alpha shrinks to ~0 and is
    # statistically indistinguishable from zero. Both must hold for the
    # warning to fire (gates against false positives on healthy NB fits
    # where pearson_chi2/df is also ~1 but alpha is large).
    assert fit["dispersion_alpha"] < 0.05
    assert fit["dispersion_alpha"] / fit["dispersion_alpha_se"] < 2.0
    assert "underdispersion_vs_negbin" in result["warnings"]


def test_fit_model_negbin_does_not_emit_underdispersion_on_genuine_nb(
    call_tool, load_df_into_session
):
    """Regression: a well-fit NB2 on genuinely overdispersed data must NOT
    be flagged as underdispersed just because pearson_chi2/df is near 1.0
    (that's the *desired* outcome, not a warning sign)."""
    load_df_into_session("nb", _negbin_known_answer_df())
    result = call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is True, result
    # Fixture has alpha_true=1.5; statsmodels recovers alpha ≈ 1.44.
    assert result["fit"]["dispersion_alpha"] > 1.0
    assert "underdispersion_vs_negbin" not in result["warnings"]


# --- Slice 11 ----------------------------------------------------------
def test_fit_model_negbin_emits_unstable_dispersion_warning(call_tool, load_df_into_session):
    load_df_into_session("uns", _negbin_unstable_alpha_df())
    result = call_tool("fit_model", {"name": "uns", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is True, result
    fit = result["fit"]
    assert fit["dispersion_alpha"] > 0
    ratio = fit["dispersion_alpha_se"] / fit["dispersion_alpha"]
    assert ratio > 0.5
    assert "unstable_dispersion" in result["warnings"]


# --- Slice 12 ----------------------------------------------------------
def test_fit_model_negbin_reports_convergence_failure(call_tool, load_df_into_session):
    """Perfectly separated counts → NB MLE does not converge; return a
    structured error and NO partial result."""
    load_df_into_session("sep", _negbin_separated_df())
    result = call_tool("fit_model", {"name": "sep", "formula": "y ~ x", "kind": "negbin"})
    assert result["ok"] is False
    assert result["error"]["type"] == "convergence_failed"
    # Hint must point at the canonical remedies.
    assert "Poisson" in result["error"]["hint"]
    # No partial fit/coefficients leak through on a failed convergence.
    assert "coefficients" not in result
    assert "fit" not in result


# --- Slice 13 ----------------------------------------------------------
def test_fit_model_negbin_interpretation_mentions_irr(call_tool, load_df_into_session):
    load_df_into_session("nb", _negbin_known_answer_df())
    result = call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    interp = result["interpretation"]
    # Proposal Acceptance: text always contains "IRR" or "incidence rate ratio".
    assert ("IRR" in interp) or ("incidence rate ratio" in interp)
    assert "x" in interp


# --- Slice 14 ----------------------------------------------------------
def test_fit_model_negbin_records_markdown_and_code_cells(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    load_df_into_session("nb", _negbin_known_answer_df())
    call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    cells = get_recorder().cells
    assert len(cells) == 2
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["cell_type"] == "code"
    assert "smf.negativebinomial" in cells[1]["source"]
    # Recorder code cell must use disp=0 (matches poisson/logistic pattern)
    # and the same formula passed in.
    assert "disp=0" in cells[1]["source"]
    assert "y ~ x" in cells[1]["source"]


# --- Slice 15 — recorder code roundtrip --------------------------------
def test_fit_model_negbin_recorder_code_reproduces_alpha_and_beta(call_tool, load_df_into_session):
    """The recorded code cell, when executed against the same DataFrame,
    must reproduce α and β to 1e-6 — proves the recorder template is
    faithful to the runtime fit."""
    from data_analyst_mcp import session as _session
    from data_analyst_mcp.recorder import get_recorder

    df = _negbin_known_answer_df()
    load_df_into_session("nb", df)
    result = call_tool("fit_model", {"name": "nb", "formula": "y ~ x", "kind": "negbin"})
    code = get_recorder().cells[1]["source"]
    # Execute the recorder cell against the live session connection.
    con = _session.get_connection()
    ns: dict[str, Any] = {"con": con}  # type: ignore[name-defined]
    exec(code, ns)
    m = ns["model"]
    by_name = {c["name"]: c for c in result["coefficients"]}
    assert float(m.params["x"]) == pytest.approx(by_name["x"]["estimate"], abs=1e-6)
    assert float(m.params["alpha"]) == pytest.approx(result["fit"]["dispersion_alpha"], abs=1e-6)


# --- Reference suite regression test (catches statsmodels-version drift) ---
def test_fit_model_negbin_matches_reference_suite(call_tool, load_df_into_session):
    """The live fit must match the pinned `fixtures/_negbin_reference/` outputs
    to ≤1e-3 on coefficients/SEs/dispersion. See the README in that directory
    for the rationale (statsmodels-pinned today, R-cross-validated later)."""
    import json
    from pathlib import Path

    ref_dir = Path(__file__).parent.parent / "fixtures" / "_negbin_reference"
    expected = json.loads((ref_dir / "expected.json").read_text())
    for key, ref in expected.items():
        df = pd.read_csv(ref_dir / f"{key}.csv")
        load_df_into_session(key, df)
        result = call_tool("fit_model", {"name": key, "formula": ref["formula"], "kind": "negbin"})
        assert result["ok"] is True, (key, result)
        assert result["fit"]["dispersion_alpha"] == pytest.approx(ref["dispersion_alpha"], abs=1e-3)
        assert result["fit"]["dispersion_alpha_se"] == pytest.approx(
            ref["dispersion_alpha_se"], abs=1e-3
        )
        by_name = {c["name"]: c for c in result["coefficients"]}
        ref_by_name = {c["name"]: c for c in ref["coefficients"]}
        assert set(by_name) == set(ref_by_name), (key, by_name.keys(), ref_by_name.keys())
        for name, rc in ref_by_name.items():
            assert by_name[name]["estimate"] == pytest.approx(rc["estimate"], abs=1e-3), (
                key,
                name,
            )
            assert by_name[name]["std_err"] == pytest.approx(rc["std_err"], abs=1e-3), (
                key,
                name,
            )
