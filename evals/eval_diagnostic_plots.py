"""End-to-end evals for ``regression_line`` + ``residual_diagnostic``.

Drives both tools through the live MCP stdio transport:

1. Load ``opportunities.csv`` (synthetic CRM fixture).
2. Materialize a numeric-only view so the OLS fit has clean predictors.
3. Fit an OLS model and register it.
4. Call ``regression_line`` and verify a PNG header + positive dimensions
   come back through the protocol.
5. Call ``residual_diagnostic`` for every supported ``kind`` and verify
   the same PNG-validity envelope (plus that ``kind="all"`` returns
   ``plot_kind="all"``).
"""

from __future__ import annotations

import base64

import pytest
from conftest import CRM_DIR, call, mcp_session

OPPS = str(CRM_DIR / "opportunities.csv")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png_envelope(result: dict, expected_kind: str, expected_model: str) -> None:
    """PNG header + positive dims + echoed model_name/plot_kind."""
    assert result["ok"], result
    raw = base64.b64decode(result["png_base64"])
    assert raw[:8] == _PNG_MAGIC, "expected PNG magic bytes"
    assert len(raw) >= 5000, "expected non-trivial PNG payload"
    assert isinstance(result["width"], int) and result["width"] > 0
    assert isinstance(result["height"], int) and result["height"] > 0
    assert result["plot_kind"] == expected_kind
    assert result["model_name"] == expected_model


async def _fit_opps_ols(s) -> None:
    """Load opportunities, materialize numeric view, fit OLS on amount."""
    r = await call(s, "load_dataset", {"path": OPPS, "name": "opps"})
    assert r["ok"], r
    # Drop NULLs in the numeric columns we'll regress against, and add a
    # synthetic numeric predictor (closed-vs-open as 0/1) so the formula
    # has more than one term — exercises the "hold others at mean" path
    # in the regression-line helper.
    r = await call(
        s,
        "materialize_query",
        {
            "sql": (
                "SELECT amount, "
                "CASE WHEN closed_at IS NULL THEN 0 ELSE 1 END AS is_closed "
                "FROM opps WHERE amount IS NOT NULL"
            ),
            "name": "opps_num",
        },
    )
    assert r["ok"], r
    r = await call(
        s,
        "fit_model",
        {
            "name": "opps_num",
            "formula": "amount ~ is_closed",
            "kind": "ols",
            "model_name": "opps_ols",
        },
    )
    assert r["ok"], r


@pytest.mark.eval
async def eval_regression_line_returns_valid_png():
    """regression_line on the OLS model returns a non-trivial PNG."""
    async with mcp_session() as s:
        await _fit_opps_ols(s)
        r = await call(
            s,
            "regression_line",
            {"model_name": "opps_ols", "predictor": "is_closed"},
        )
        _assert_valid_png_envelope(r, expected_kind="regression_line", expected_model="opps_ols")


@pytest.mark.eval
async def eval_residual_diagnostic_each_kind_returns_valid_png():
    """residual_diagnostic returns a valid PNG for every supported kind."""
    async with mcp_session() as s:
        await _fit_opps_ols(s)
        for kind in ("resid_vs_fitted", "qq", "scale_location", "all"):
            r = await call(
                s,
                "residual_diagnostic",
                {"model_name": "opps_ols", "kind": kind},
            )
            _assert_valid_png_envelope(r, expected_kind=kind, expected_model="opps_ols")
