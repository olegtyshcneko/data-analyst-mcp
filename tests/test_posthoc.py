"""Tests for ``pairwise_comparisons`` — post-hoc pairwise comparisons.

The TDD slices in ``docs/proposals/pairwise_comparisons.md`` map one-to-one
to the test functions below. This module covers slices 1–9, the validation
surface (proposal "Behavior" steps 1–8 and the "Errors" table): dataset /
column / dtype / alpha checks, label resolution (duplicate + missing-label
rejection, the 3–20 group bounds), and the ``method="tukey"`` +
``p_adjust`` conflict. The Tukey / Dunn engines (slices 10+) land in a
later task; until then a passing validation path returns an ``internal``
stub, so error-type assertions here never rely on ``ok is True``.
"""

from __future__ import annotations

import pytest

# === slice 1: pairwise_comparisons returns not_found for unregistered dataset ===


def test_slice01_not_found_for_unregistered_dataset(call_tool):
    result = call_tool(
        "pairwise_comparisons",
        {"name": "nope", "group_column": "grp", "metric_column": "val"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


# === slice 2: pairwise_comparisons returns column_not_found for missing group or metric column ===


def test_slice02_column_not_found_for_missing_column(call_tool, load_df_into_session):
    import pandas as pd

    df = pd.DataFrame({"grp": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
    load_df_into_session("ds", df)

    missing_group = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "nope", "metric_column": "val"},
    )
    assert missing_group["ok"] is False
    assert missing_group["error"]["type"] == "column_not_found"

    missing_metric = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "nope"},
    )
    assert missing_metric["ok"] is False
    assert missing_metric["error"]["type"] == "column_not_found"


# === slice 3: pairwise_comparisons returns metric_not_numeric for a VARCHAR metric column ===


def test_slice03_metric_not_numeric_for_varchar_metric(call_tool, load_df_into_session):
    import pandas as pd

    df = pd.DataFrame({"grp": ["A", "B", "C"], "val": ["x", "y", "z"]})
    load_df_into_session("ds", df)

    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "metric_not_numeric"


# === slice 4: pairwise_comparisons returns too_few_groups below three groups and hints at compare_groups ===


def test_slice04_too_few_groups_hints_compare_groups(call_tool, load_df_into_session):
    import pandas as pd

    df = pd.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0]})
    load_df_into_session("ds", df)

    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "too_few_groups"
    assert "compare_groups" in result["error"]["hint"]


# === slice 5: pairwise_comparisons returns invalid_alpha outside the open unit interval ===


def _three_group_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B", "C", "C"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_slice05_alpha_zero_returns_invalid_alpha(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())
    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "alpha": 0.0},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_alpha"


def test_slice05_alpha_one_returns_invalid_alpha(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())
    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "alpha": 1.0},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_alpha"


# === slice 6: pairwise_comparisons returns duplicate_groups for repeated labels in groups ===


def test_slice06_duplicate_groups_for_repeated_labels(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())
    result = call_tool(
        "pairwise_comparisons",
        {
            "name": "ds",
            "group_column": "grp",
            "metric_column": "val",
            "groups": ["A", "A", "B"],
        },
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "duplicate_groups"


# === slice 7: pairwise_comparisons returns group_not_found for a label with no rows ===


def test_slice07_group_not_found_for_label_with_no_rows(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())
    result = call_tool(
        "pairwise_comparisons",
        {
            "name": "ds",
            "group_column": "grp",
            "metric_column": "val",
            "groups": ["A", "B", "Z"],
        },
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "group_not_found"
    assert "Z" in result["error"]["message"]


# === slice 8: pairwise_comparisons returns too_many_groups above twenty labels ===


def test_slice08_too_many_groups_above_twenty(call_tool, load_df_into_session):
    import pandas as pd

    # 21 distinct labels (> the 20-group cap), one row each.
    labels = [f"g{i:02d}" for i in range(21)]
    df = pd.DataFrame({"grp": labels, "val": [float(i) for i in range(21)]})
    load_df_into_session("ds", df)

    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "too_many_groups"
    assert "groups" in result["error"]["hint"]


# === slice 9: pairwise_comparisons rejects explicit p_adjust with method tukey as p_adjust_not_applicable ===


def test_slice09_tukey_with_explicit_p_adjust_is_not_applicable(call_tool, load_df_into_session):
    load_df_into_session("ds", _three_group_frame())

    # method="tukey" + an explicit p_adjust is the only path that errors:
    # Tukey controls FWER internally via the studentized-range distribution.
    rejected = call_tool(
        "pairwise_comparisons",
        {
            "name": "ds",
            "group_column": "grp",
            "metric_column": "val",
            "method": "tukey",
            "p_adjust": "holm",
        },
    )
    assert rejected["ok"] is False
    assert rejected["error"]["type"] == "p_adjust_not_applicable"

    # Under auto, an explicit p_adjust is never an error. The engines land in
    # T3, so the request still returns the internal stub — assert only that
    # it is NOT rejected as p_adjust_not_applicable.
    allowed = call_tool(
        "pairwise_comparisons",
        {
            "name": "ds",
            "group_column": "grp",
            "metric_column": "val",
            "method": "auto",
            "p_adjust": "holm",
        },
    )
    assert allowed["error"]["type"] != "p_adjust_not_applicable"


# === slice 10: pairwise_comparisons tukey matches statsmodels known answer with confidence intervals ===


def _tukey_frame():
    import pandas as pd

    # A=[1..5], B=[3..7], C=[5..9] — the pinned Tukey fixture.
    return pd.DataFrame(
        {
            "grp": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )


def test_slice10_tukey_matches_statsmodels(call_tool, load_df_into_session):
    load_df_into_session("ds", _tukey_frame())
    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "method": "tukey"},
    )
    assert result["ok"] is True
    assert result["method"] == "tukey"
    assert result["method_requested"] == "tukey"
    # Tukey controls FWER internally, so p_adjust is echoed null.
    assert result["p_adjust"] is None
    assert result["estimate_name"] == "mean_diff"
    assert result["n_comparisons"] == 3  # C(3, 2)
    assert result["n_rejected"] == 1

    by = {(c["group_a"], c["group_b"]): c for c in result["comparisons"]}
    ab, ac, bc = by[("A", "B")], by[("A", "C")], by[("B", "C")]

    # Reference: statsmodels.stats.multicomp.pairwise_tukeyhsd(vals, grps, alpha=0.05)
    #   meandiffs [2.0, 4.0, 2.0]  (group_b - group_a orientation)
    #   pvalues   [0.1545799684, 0.0046340806, 0.1545799684]
    #   confint   [[-0.6678636566, 4.6678636566],
    #              [ 1.3321363434, 6.6678636566],
    #              [-0.6678636566, 4.6678636566]]
    #   reject    [False, True, False]
    assert ab["n_a"] == 5 and ab["n_b"] == 5
    assert ab["estimate"] == pytest.approx(2.0, abs=1e-4)
    assert ab["p_adj"] == pytest.approx(0.1545799684, abs=1e-4)
    assert ab["ci_low"] == pytest.approx(-0.6678636566, abs=1e-4)
    assert ab["ci_high"] == pytest.approx(4.6678636566, abs=1e-4)
    assert ab["reject"] is False
    # Tukey does not produce a per-row z or raw p-value.
    assert ab["statistic"] is None
    assert ab["p_raw"] is None

    assert ac["estimate"] == pytest.approx(4.0, abs=1e-4)
    assert ac["p_adj"] == pytest.approx(0.0046340806, abs=1e-4)
    assert ac["ci_low"] == pytest.approx(1.3321363434, abs=1e-4)
    assert ac["ci_high"] == pytest.approx(6.6678636566, abs=1e-4)
    assert ac["reject"] is True

    assert bc["estimate"] == pytest.approx(2.0, abs=1e-4)
    assert bc["p_adj"] == pytest.approx(0.1545799684, abs=1e-4)
    assert bc["ci_low"] == pytest.approx(-0.6678636566, abs=1e-4)
    assert bc["ci_high"] == pytest.approx(4.6678636566, abs=1e-4)
    assert bc["reject"] is False

    # Omnibus: scipy.stats.f_oneway([1..5], [3..7], [5..9]) -> F=8.0, p=0.0061963978
    assert result["omnibus"]["test"] == "anova"
    assert result["omnibus"]["statistic"] == pytest.approx(8.0, abs=1e-4)
    assert result["omnibus"]["p_value"] == pytest.approx(0.0061963978, abs=1e-4)
    assert result["omnibus"]["significant"] is True


# === extra: pairwise_comparisons returns insufficient_group_size for a group with one row ===


def test_extra_insufficient_group_size_for_single_row_group(call_tool, load_df_into_session):
    import pandas as pd

    # Group C has a single non-null row. Tukey needs within-group variance, so
    # the n<2 guard must fire BEFORE any statsmodels/scipy call and name C.
    df = pd.DataFrame(
        {
            "grp": ["A", "A", "B", "B", "C"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    load_df_into_session("ds", df)

    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "method": "tukey"},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "insufficient_group_size"
    assert "C" in result["error"]["message"]


# === slice 11: pairwise_comparisons dunn matches hand-computed ranks on untied data ===


def _dunn_untied_frame():
    import pandas as pd

    # A=[1,2,3], B=[4,5,6], C=[7,8,9] — no ties, so the tie term T=0.
    return pd.DataFrame(
        {
            "grp": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )


def test_slice11_dunn_untied_hand_computed_ranks(call_tool, load_df_into_session):
    load_df_into_session("ds", _dunn_untied_frame())
    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "method": "dunn"},
    )
    assert result["ok"] is True
    assert result["method"] == "dunn"
    assert result["method_requested"] == "dunn"
    # Dunn with p_adjust omitted resolves to Holm.
    assert result["p_adjust"] == "holm"
    assert result["estimate_name"] == "mean_rank_diff"
    assert result["n_comparisons"] == 3

    by = {(c["group_a"], c["group_b"]): c for c in result["comparisons"]}
    ab, ac, bc = by[("A", "B")], by[("A", "C")], by[("B", "C")]

    # Vendored Dunn on pooled scipy.stats.rankdata: mean ranks A=2, B=5, C=8;
    # T=0; var=N(N+1)/12=7.5; SE=sqrt(7.5*(1/3+1/3))=sqrt(5)=2.2360679775.
    # z is SIGNED (b - a): z(A,B)=(5-2)/sqrt(5)=+1.3416407865.  A flipped
    # orientation (a - b) would give -1.3416... and fail this assertion.
    assert ab["statistic"] == pytest.approx(1.3416407865, abs=1e-4)
    assert ab["estimate"] == pytest.approx(3.0, abs=1e-4)  # mean_rank_diff = 5 - 2
    # p_raw = 2*norm.sf(|z|) = 0.1797124949
    assert ab["p_raw"] == pytest.approx(0.1797124949, abs=1e-4)
    # z(A,C)=(8-2)/sqrt(5)=+2.6832815730; p_raw=0.0072903581
    assert ac["statistic"] == pytest.approx(2.6832815730, abs=1e-4)
    assert ac["estimate"] == pytest.approx(6.0, abs=1e-4)
    assert ac["p_raw"] == pytest.approx(0.0072903581, abs=1e-4)
    assert bc["statistic"] == pytest.approx(1.3416407865, abs=1e-4)
    assert bc["p_raw"] == pytest.approx(0.1797124949, abs=1e-4)

    # Holm on [0.1797124949, 0.0072903581, 0.1797124949] via statsmodels
    # multipletests(method="holm"): p_adj=[0.3594249898, 0.0218710743,
    # 0.3594249898], reject=[False, True, False].
    assert ab["p_adj"] == pytest.approx(0.3594249898, abs=1e-4)
    assert ac["p_adj"] == pytest.approx(0.0218710743, abs=1e-4)
    assert bc["p_adj"] == pytest.approx(0.3594249898, abs=1e-4)
    assert [ab["reject"], ac["reject"], bc["reject"]] == [False, True, False]
    assert result["n_rejected"] == 1

    # Tukey-only fields are null on the Dunn engine.
    assert ab["ci_low"] is None and ab["ci_high"] is None

    # Omnibus: scipy.stats.kruskal([1,2,3],[4,5,6],[7,8,9]) -> H=7.2, p=0.0273237224
    assert result["omnibus"]["test"] == "kruskal_wallis"
    assert result["omnibus"]["statistic"] == pytest.approx(7.2, abs=1e-4)
    assert result["omnibus"]["p_value"] == pytest.approx(0.0273237224, abs=1e-4)
    assert result["omnibus"]["significant"] is True


# === slice 12: pairwise_comparisons dunn applies the tie correction ===


def _dunn_tied_frame():
    import pandas as pd

    # A=[1,2,2], B=[2,3,4], C=[5,5,6] — pooled ties drive the T tie term.
    return pd.DataFrame(
        {
            "grp": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "val": [1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0],
        }
    )


def test_slice12_dunn_tie_correction(call_tool, load_df_into_session):
    load_df_into_session("ds", _dunn_tied_frame())
    result = call_tool(
        "pairwise_comparisons",
        {"name": "ds", "group_column": "grp", "metric_column": "val", "method": "dunn"},
    )
    assert result["ok"] is True
    # Default correction when p_adjust is omitted under method="dunn".
    assert result["p_adjust"] == "holm"

    by = {(c["group_a"], c["group_b"]): c for c in result["comparisons"]}
    ab, ac, bc = by[("A", "B")], by[("A", "C")], by[("B", "C")]

    # Tie-corrected vendored Dunn: pooled counts give T=Σ(t³−t)=30, so
    # var=N(N+1)/12 − T/(12(N−1)) = 7.5 − 30/96 = 7.1875 and
    # SE=sqrt(7.1875*(1/3+1/3))=2.1889875894. Mean ranks A=7/3, B=14/3, C=8.
    # Without the tie term SE would be sqrt(5)=2.236 and these z's would be
    # smaller — this asserts the correction is applied.
    # z = [1.0659417827, 2.5887157579, 1.5227739753]
    assert ab["statistic"] == pytest.approx(1.0659417827, abs=1e-4)
    assert ac["statistic"] == pytest.approx(2.5887157579, abs=1e-4)
    assert bc["statistic"] == pytest.approx(1.5227739753, abs=1e-4)
    # p_raw = [0.2864499597, 0.0096334576, 0.1278152631]
    assert ab["p_raw"] == pytest.approx(0.2864499597, abs=1e-4)
    assert ac["p_raw"] == pytest.approx(0.0096334576, abs=1e-4)
    assert bc["p_raw"] == pytest.approx(0.1278152631, abs=1e-4)
    # Holm p_adj = [0.2864499597, 0.0289003729, 0.2556305262]; reject [F, T, F]
    assert ab["p_adj"] == pytest.approx(0.2864499597, abs=1e-4)
    assert ac["p_adj"] == pytest.approx(0.0289003729, abs=1e-4)
    assert bc["p_adj"] == pytest.approx(0.2556305262, abs=1e-4)
    assert [ab["reject"], ac["reject"], bc["reject"]] == [False, True, False]
