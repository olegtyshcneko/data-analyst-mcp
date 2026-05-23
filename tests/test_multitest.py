"""Tests for ``adjust_pvalues`` — multiple-testing correction.

The 17 TDD slices in ``docs/proposals/adjust_pvalues.md`` map one-to-one to
the test functions below. Numeric expected values for slices #7, #8, #10,
#11 are pinned to hand-computed or published references at ≤1e-4 tolerance
per SPEC §3.
"""

from __future__ import annotations

import pytest

# === slice 1: empty input → ok, no statsmodels call ===


def test_slice01_empty_p_values_returns_ok_with_zero_counts(call_tool, monkeypatch):
    """An empty ``p_values`` list must short-circuit before statsmodels."""
    import data_analyst_mcp.tools.multitest as _mt

    called: list[bool] = []

    class _BoomModule:
        @staticmethod
        def multipletests(*_args, **_kwargs):  # pragma: no cover - asserts a non-call
            called.append(True)
            raise RuntimeError(
                "statsmodels.multipletests should not have been called for empty input"
            )

    monkeypatch.setattr(_mt, "_sm_multitest", lambda: _BoomModule)

    result = call_tool("adjust_pvalues", {"p_values": []})

    assert result["ok"] is True
    assert result["results"] == []
    assert result["n_tests"] == 0
    assert result["n_rejected"] == 0
    assert called == []


# === slice 2: out-of-range p_values → invalid_p_value, hint names index 0 ===


def test_slice02_p_value_above_one_returns_invalid_p_value(call_tool):
    result = call_tool("adjust_pvalues", {"p_values": [1.5]})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_p_value"
    assert "p_values[0]" in result["error"]["message"]


# === slice 3: NaN → invalid_p_value, hint names index 0 ===


def test_slice03_nan_p_value_returns_invalid_p_value(call_tool):
    result = call_tool("adjust_pvalues", {"p_values": [float("nan")]})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_p_value"
    assert "p_values[0]" in result["error"]["message"]


# === slice 4: negative p_values → invalid_p_value (covered by range check) ===


def test_slice04_negative_p_value_returns_invalid_p_value(call_tool):
    result = call_tool("adjust_pvalues", {"p_values": [-0.1]})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_p_value"
    assert "p_values[0]" in result["error"]["message"]


# === slice 5: alpha boundary → invalid_alpha ===


def test_slice05_alpha_zero_returns_invalid_alpha(call_tool):
    result = call_tool("adjust_pvalues", {"p_values": [0.01], "alpha": 0.0})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_alpha"


def test_slice05_alpha_one_returns_invalid_alpha(call_tool):
    result = call_tool("adjust_pvalues", {"p_values": [0.01], "alpha": 1.0})
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_alpha"


# === slice 6: labels length mismatch → length_mismatch ===


def test_slice06_labels_longer_than_p_values_returns_length_mismatch(call_tool):
    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.01], "labels": ["a", "b"]},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "length_mismatch"


# === slice 7: Bonferroni hand-compute on [0.01, 0.04, 0.03] ===


def test_slice07_bonferroni_hand_computed(call_tool):
    # Bonferroni: p_adj = min(p * m, 1), rejection at alpha=0.05.
    # m=3, [0.01, 0.04, 0.03] -> [0.03, 0.12, 0.09]; rejected = [True, False, False].
    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.01, 0.04, 0.03], "method": "bonferroni", "alpha": 0.05},
    )
    assert result["ok"] is True
    p_adj = [r["p_adj"] for r in result["results"]]
    rejected = [r["rejected"] for r in result["results"]]
    assert p_adj[0] == pytest.approx(0.03, abs=1e-4)
    assert p_adj[1] == pytest.approx(0.12, abs=1e-4)
    assert p_adj[2] == pytest.approx(0.09, abs=1e-4)
    assert rejected == [True, False, False]


# === slice 8: BH 1995 worked example, 15-element series ===


def test_slice08_bh_published_worked_example(call_tool):
    """Benjamini & Hochberg (1995), Table 1 — 4 rejections at α=0.05."""
    p_raw = [
        0.0001,
        0.0004,
        0.0019,
        0.0095,
        0.0201,
        0.0278,
        0.0298,
        0.0344,
        0.0459,
        0.3240,
        0.4262,
        0.5719,
        0.6528,
        0.7590,
        1.0000,
    ]
    expected_p_adj = [
        0.0015,
        0.003,
        0.0095,
        0.035625,
        0.0603,
        0.06385714285714286,
        0.06385714285714286,
        0.0645,
        0.07650000000000001,
        0.48600000000000004,
        0.5811818181818182,
        0.7148749999999999,
        0.7532307692307693,
        0.8132142857142857,
        1.0,
    ]
    expected_rejected = [True, True, True, True] + [False] * 11

    result = call_tool(
        "adjust_pvalues",
        {"p_values": p_raw, "method": "bh", "alpha": 0.05},
    )
    assert result["ok"] is True
    assert result["n_tests"] == 15
    assert result["n_rejected"] == 4

    rows = result["results"]
    for i, (exp_adj, exp_rej) in enumerate(zip(expected_p_adj, expected_rejected, strict=True)):
        assert rows[i]["p_adj"] == pytest.approx(exp_adj, abs=1e-4), f"row {i}"
        assert rows[i]["rejected"] is exp_rej, f"row {i}"


# === slice 9: Holm matches statsmodels reference to 1e-10 ===


def test_slice09_holm_matches_statsmodels_reference(call_tool):
    import numpy as np
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(42)
    p_raw = rng.uniform(0.001, 0.5, size=10).tolist()
    ref_rejected, ref_p_adj, _, _ = multipletests(p_raw, alpha=0.05, method="holm")

    result = call_tool(
        "adjust_pvalues",
        {"p_values": p_raw, "method": "holm", "alpha": 0.05},
    )
    assert result["ok"] is True
    rows = result["results"]
    for i, (ref_adj, ref_rej) in enumerate(zip(ref_p_adj.tolist(), ref_rejected.tolist())):
        assert rows[i]["p_adj"] == pytest.approx(ref_adj, abs=1e-10), f"row {i}"
        assert rows[i]["rejected"] is bool(ref_rej), f"row {i}"


# === slice 10: Sidak hand-compute on [0.01, 0.01, 0.01] ===


def test_slice10_sidak_hand_computed(call_tool):
    # Šidák single-step: p_adj = 1 - (1 - p)^m
    # m=3, p=0.01 → 1 - 0.99**3 = 0.029701
    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.01, 0.01, 0.01], "method": "sidak", "alpha": 0.05},
    )
    assert result["ok"] is True
    p_adj = [r["p_adj"] for r in result["results"]]
    for v in p_adj:
        assert v == pytest.approx(0.029701, abs=1e-9)


# === slice 11: BH on tied p-values gives tied adjusted values ===


def test_slice11_bh_ties_get_tied_adjusted_values(call_tool):
    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.02, 0.02, 0.02, 0.04], "method": "bh", "alpha": 0.05},
    )
    assert result["ok"] is True
    rows = result["results"]
    # Tied raw p-values must receive identical adjusted p-values.
    assert rows[0]["p_adj"] == rows[1]["p_adj"] == rows[2]["p_adj"]


# === slice 12: single p-value under every method → p_adj == p_raw, rejected ===


@pytest.mark.parametrize("method", ["bonferroni", "sidak", "holm", "bh", "by"])
def test_slice12_single_p_value_round_trips_under_every_method(call_tool, method):
    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.04], "method": method, "alpha": 0.05},
    )
    assert result["ok"] is True
    row = result["results"][0]
    assert row["p_raw"] == pytest.approx(0.04, abs=1e-12)
    assert row["p_adj"] == pytest.approx(0.04, abs=1e-9)
    assert row["rejected"] is True


# === slice 13: input order preserved under every method ===


@pytest.mark.parametrize("method", ["bonferroni", "sidak", "holm", "bh", "by"])
def test_slice13_input_order_preserved_under_every_method(call_tool, method):
    # Deliberately *not* sorted on input.
    p_raw = [0.30, 0.001, 0.20, 0.04, 0.002, 0.50]
    labels = ["a", "b", "c", "d", "e", "f"]

    result = call_tool(
        "adjust_pvalues",
        {"p_values": p_raw, "labels": labels, "method": method, "alpha": 0.05},
    )
    assert result["ok"] is True
    rows = result["results"]
    # Labels (and therefore rows) must appear in the original input order.
    assert [r["label"] for r in rows] == labels
    # p_raw must echo the input verbatim, in order.
    assert [r["p_raw"] for r in rows] == pytest.approx(p_raw, abs=1e-12)


# === slice 14: labels=None → every label field is JSON null ===


def test_slice14_labels_none_gives_null_label_per_row(call_tool):
    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.01, 0.5], "method": "bh"},
    )
    assert result["ok"] is True
    rows = result["results"]
    for row in rows:
        # JSON null round-trips to Python None.
        assert row["label"] is None
        assert "label" in row  # field present, not omitted


# === slice 15: counts match ===


def test_slice15_n_rejected_and_n_tests_match_row_data(call_tool):
    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.001, 0.5, 0.0005, 0.6], "method": "bonferroni", "alpha": 0.05},
    )
    assert result["ok"] is True
    rows = result["results"]
    assert result["n_tests"] == len(rows) == 4
    assert result["n_rejected"] == sum(1 for r in rows if r["rejected"])


# === slice 16: recorder emits markdown+code pair; markdown contains rejected ratio ===


def test_slice16_recorder_emits_cell_pair_with_rejected_ratio(call_tool):
    from data_analyst_mcp.recorder import get_recorder

    result = call_tool(
        "adjust_pvalues",
        {
            "p_values": [1.4e-58, 1.1e-22, 8.6e-22, 0.3677],
            "labels": [
                "sex_vs_survived",
                "pclass_vs_survived",
                "fare_vs_survived",
                "age_vs_survived",
            ],
            "method": "bh",
            "alpha": 0.05,
        },
    )
    assert result["ok"] is True
    cells = get_recorder().cells
    assert len(cells) == 2
    md, code = cells
    assert md["cell_type"] == "markdown"
    assert code["cell_type"] == "code"
    assert md["metadata"]["tool_name"] == "adjust_pvalues"
    assert "3 / 4 hypotheses rejected" in md["source"]
    # Code cell must compile as Python.
    compile(code["source"], "<adjust_pvalues_cell>", "exec")


# === slice 17: when n_rejected == 0, "Largest adjusted-p" line is omitted ===


def test_slice17_recorder_omits_largest_line_when_nothing_rejected(call_tool):
    from data_analyst_mcp.recorder import get_recorder

    result = call_tool(
        "adjust_pvalues",
        {"p_values": [0.6, 0.7, 0.8], "method": "bh", "alpha": 0.05},
    )
    assert result["ok"] is True
    assert result["n_rejected"] == 0
    cells = get_recorder().cells
    md = cells[0]
    assert "0 / 3 hypotheses rejected" in md["source"]
    assert "Largest adjusted-p" not in md["source"]


# === extra: round-trip recorded notebook executes cleanly ===


def test_recorded_notebook_round_trip_executes(call_tool, tmp_path):
    """End-to-end: adjust_pvalues + emit_notebook → ``jupyter nbconvert --execute`` is clean."""
    import os
    import subprocess

    r = call_tool(
        "adjust_pvalues",
        {
            "p_values": [1.4e-58, 1.1e-22, 8.6e-22, 0.3677],
            "labels": [
                "sex_vs_survived",
                "pclass_vs_survived",
                "fare_vs_survived",
                "age_vs_survived",
            ],
            "method": "bh",
            "alpha": 0.05,
        },
    )
    assert r["ok"]

    nb_path = tmp_path / "adjust_pvalues_round_trip.ipynb"
    r = call_tool("emit_notebook", {"path": str(nb_path)})
    assert r["ok"]

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
