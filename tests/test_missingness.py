"""Tests for ``analyze_missingness`` (v1 descriptive + v1.1 Little's MCAR).

Slices 1–18 cover the v1 descriptive surface; M1–M7 cover the v1.1
MCAR-test slice. Fixtures: #1 Titanic-style independent plant, #2 joined
alignment, #3 MCAR-style independent, #4 MAR plant (inline), plus the
``fixtures/_mcar_reference/`` external suite for M5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---- shared inline fixtures (proposal §"Locked test fixtures" #1, #2, #3)


def _titanic_like_fixture() -> pd.DataFrame:
    """Fixture #1: n=891, plant nulls in age (177), cabin (687), embarked (2).

    All three columns null independently (Bernoulli draws with a fixed
    seed); used as the "structurally similar to real data" walk-through.
    """
    rng = np.random.default_rng(0)
    n = 891
    age = rng.normal(30, 12, size=n).round(1)
    fare = rng.lognormal(2.5, 1.0, size=n).round(2)
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])
    cabin = rng.choice(["A1", "B2", "C3", "D4", "E5"], size=n, p=[0.25, 0.25, 0.2, 0.2, 0.1])

    df = pd.DataFrame(
        {
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "fare": fare,
            "embarked": embarked,
            "cabin": cabin,
        }
    )

    # Plant nulls independently (separate Bernoulli masks per column).
    df.loc[rng.choice(n, size=177, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, size=687, replace=False), "cabin"] = None
    df.loc[rng.choice(n, size=2, replace=False), "embarked"] = None
    return df


def _join_alignment_fixture() -> pd.DataFrame:
    """Fixture #2: ``email`` missing exactly for rows where ``region == "EMEA"``.

    Used to drive ``null_grouping.aligned_with`` tests (slices #8, #10).
    Inline construction is simpler than fixture-on-disk and matches the
    rest of the test suite's style.
    """
    rng = np.random.default_rng(1)
    n = 400
    # 3 regions; "EMEA" is exactly one of them.
    region = np.array(["AMER"] * 150 + ["APAC"] * 150 + ["EMEA"] * 100)
    rng.shuffle(region)
    df = pd.DataFrame(
        {
            "account_id": np.arange(n),
            "region": region,
            "industry": rng.choice(["tech", "fin", "health"], size=n),
            "email": [f"user{i}@example.com" for i in range(n)],
        }
    )
    df.loc[df["region"] == "EMEA", "email"] = None
    return df


def _mcar_independent_fixture() -> pd.DataFrame:
    """Fixture #3: n=10000, 5 cols, each independently null at p=0.1."""
    rng = np.random.default_rng(0)
    n = 10000
    cols = {}
    for k in range(5):
        vals = rng.normal(0, 1, size=n)
        mask = rng.random(n) < 0.1
        arr = vals.copy()
        arr[mask] = np.nan
        cols[f"x{k}"] = arr
    return pd.DataFrame(cols)


# === slice 1: zero-null dataset → ok, per_column empty, suggestion text ===


def test_slice01_zero_null_dataset_returns_no_missingness(call_tool, load_df_into_session):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    load_df_into_session("clean", df)

    result = call_tool("analyze_missingness", {"name": "clean", "run_mcar_test": False})

    assert result["ok"] is True
    assert result["per_column"] == []
    assert result["mcar_test"] is None
    assert "No missingness detected." in result["suggestions"]


# === slice 2: per-column null_count matches hand-computed planted value ===


def test_slice02_per_column_null_count_matches_planted_value(call_tool, load_df_into_session):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"age": rng.normal(30, 5, size=100).round(1)})
    # Plant exactly 13 nulls at deterministic indices.
    df.loc[list(range(13)), "age"] = np.nan
    load_df_into_session("planted", df)

    result = call_tool("analyze_missingness", {"name": "planted", "run_mcar_test": False})

    assert result["ok"] is True
    age_row = next(r for r in result["per_column"] if r["column"] == "age")
    assert age_row["null_count"] == 13
    assert age_row["null_pct"] == pytest.approx(13.0, abs=1e-6)


# === slice 3: unknown_columns error ===


def test_slice03_unknown_columns_returns_unknown_columns(call_tool, load_df_into_session):
    load_df_into_session("d", pd.DataFrame({"a": [1, 2, 3]}))
    result = call_tool(
        "analyze_missingness",
        {"name": "d", "columns": ["typo"], "run_mcar_test": False},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "unknown_columns"


# === slice 4: pattern_top_k=0 → pattern_top_k_out_of_range ===


def test_slice04_pattern_top_k_zero_returns_out_of_range(call_tool, load_df_into_session):
    load_df_into_session("d", pd.DataFrame({"a": [1, 2, 3]}))
    result = call_tool(
        "analyze_missingness",
        {"name": "d", "pattern_top_k": 0, "run_mcar_test": False},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "pattern_top_k_out_of_range"


# === slice 5: pairwise_corr_threshold=1.5 → out_of_range ===


def test_slice05_pairwise_corr_threshold_above_one_returns_out_of_range(
    call_tool, load_df_into_session
):
    load_df_into_session("d", pd.DataFrame({"a": [1, 2, 3]}))
    result = call_tool(
        "analyze_missingness",
        {"name": "d", "pairwise_corr_threshold": 1.5, "run_mcar_test": False},
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "pairwise_corr_threshold_out_of_range"


# === slice 6: run_mcar_test=True executes the routine in v1.1 ===


def test_slice06_run_mcar_test_true_executes_routine(call_tool, load_df_into_session):
    # Trivially-numeric dataset with no missing patterns to test → routine
    # short-circuits with skipped: true, not the old defer-gate error.
    load_df_into_session("d", pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    result = call_tool("analyze_missingness", {"name": "d"})
    assert result["ok"] is True
    # With no nulls there is only one pattern (all-present); the routine
    # cannot form a chi-square and must mark itself skipped.
    assert result["mcar_test"]["skipped"] is True
    assert result["mcar_test"]["reason"] == "insufficient_patterns"


# === slice 7: top-K patterns descending, all-present pattern included ===


def test_slice07_patterns_descending_and_include_all_present(call_tool, load_df_into_session):
    # 3 cols, mostly complete with a single (a-missing only) pattern.
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 3, 4, 5, 6], "c": [1, 2, 3, 4, 5, 6]})
    # Plant nulls so we get two distinct patterns and the no-null pattern wins.
    df.loc[0, "a"] = np.nan
    df.loc[1, "a"] = np.nan
    # rows 0..1: (a=True, b=False, c=False)
    # rows 2..5: (all False)
    load_df_into_session("d", df)

    result = call_tool(
        "analyze_missingness", {"name": "d", "pattern_top_k": 5, "run_mcar_test": False}
    )
    assert result["ok"] is True
    patterns = result["patterns"]
    counts = [p["count"] for p in patterns]
    # Descending by count.
    assert counts == sorted(counts, reverse=True)
    # The all-present pattern (everything False) is in top-K and is the
    # most common pattern.
    top = patterns[0]
    assert top["count"] == 4
    assert all(v is False for v in top["pattern"].values())


# === slice 8: null_grouping.all_or_nothing=True on joined fixture ===


def test_slice08_null_grouping_aligned_with_region(call_tool, load_df_into_session):
    df = _join_alignment_fixture()
    load_df_into_session("joined", df)

    result = call_tool("analyze_missingness", {"name": "joined", "run_mcar_test": False})
    assert result["ok"] is True
    email_row = next(r for r in result["per_column"] if r["column"] == "email")
    assert email_row["null_grouping"]["all_or_nothing"] is True
    assert email_row["null_grouping"]["aligned_with"] == "region"
    assert email_row["null_grouping"]["group_count"] == 3


# === slice 9: null_grouping false when missingness uniform across groups ===


def test_slice09_null_grouping_false_when_uniform(call_tool, load_df_into_session):
    rng = np.random.default_rng(2)
    n = 600
    region = rng.choice(["AMER", "EMEA", "APAC"], size=n)
    # Uniform 25% null across every region — alignment should NOT fire.
    email = np.array([f"u{i}@x.com" for i in range(n)], dtype=object)
    null_mask = rng.random(n) < 0.25
    email[null_mask] = None
    df = pd.DataFrame({"region": region, "email": email})
    load_df_into_session("uniform", df)

    result = call_tool("analyze_missingness", {"name": "uniform", "run_mcar_test": False})
    assert result["ok"] is True
    email_row = next(r for r in result["per_column"] if r["column"] == "email")
    assert email_row["null_grouping"]["all_or_nothing"] is False
    assert email_row["null_grouping"]["aligned_with"] is None


# === slice 10: tiebreaker — pick fewest-groups categorical ===


def test_slice10_null_grouping_picks_fewest_groups(call_tool, load_df_into_session):
    # Two perfectly-aligned categoricals: "region" (3 groups) and
    # "block" (5 groups). The proposal mandates fewest-groups wins.
    n = 300
    region = np.array(["A"] * 100 + ["B"] * 100 + ["EMEA"] * 100)
    block = np.array(["b1"] * 50 + ["b2"] * 50 + ["b3"] * 50 + ["b4"] * 50 + ["EMEA_block"] * 100)
    email = np.array([f"u{i}@x.com" for i in range(n)], dtype=object)
    # Null whenever region == "EMEA" (which is also exactly the block == "EMEA_block" rows).
    email[region == "EMEA"] = None
    df = pd.DataFrame({"region": region, "block": block, "email": email})
    load_df_into_session("tiebreak", df)

    result = call_tool("analyze_missingness", {"name": "tiebreak", "run_mcar_test": False})
    assert result["ok"] is True
    email_row = next(r for r in result["per_column"] if r["column"] == "email")
    # "region" has 3 distinct vs "block"'s 5 — fewest wins.
    assert email_row["null_grouping"]["aligned_with"] == "region"
    assert email_row["null_grouping"]["group_count"] == 3


# === slice 11: perfectly co-missing pair → φ = 1.0 ===


def test_slice11_perfectly_comissing_pair_has_phi_one(call_tool, load_df_into_session):
    n = 100
    a = np.arange(n, dtype=float)
    b = np.arange(n, dtype=float)
    # Identical null mask on both — φ must be 1.0.
    mask_indices = list(range(30))
    for idx in mask_indices:
        a[idx] = np.nan
        b[idx] = np.nan
    df = pd.DataFrame({"a": a, "b": b})
    load_df_into_session("comissing", df)

    result = call_tool(
        "analyze_missingness",
        {"name": "comissing", "pairwise_corr_threshold": 0.0, "run_mcar_test": False},
    )
    assert result["ok"] is True
    pair = next(p for p in result["pairwise_correlation"] if {p["col_a"], p["col_b"]} == {"a", "b"})
    assert float(pair["phi"]) == pytest.approx(1.0, abs=1e-9)


# === slice 12: independently-missing columns on n=10000 → |φ| < 0.05 ===


def test_slice12_independent_missingness_phi_under_005(call_tool, load_df_into_session):
    df = _mcar_independent_fixture()
    load_df_into_session("indep", df)
    result = call_tool(
        "analyze_missingness",
        {"name": "indep", "pairwise_corr_threshold": 0.0, "run_mcar_test": False},
    )
    assert result["ok"] is True
    for pair in result["pairwise_correlation"]:
        assert abs(float(pair["phi"])) < 0.05, pair


# === slice 13: pair below threshold is dropped from output ===


def test_slice13_pair_below_threshold_dropped(call_tool, load_df_into_session):
    df = _mcar_independent_fixture()
    load_df_into_session("indep", df)
    # With a high threshold (0.5) and independent missingness, nothing
    # should survive.
    result = call_tool(
        "analyze_missingness",
        {"name": "indep", "pairwise_corr_threshold": 0.5, "run_mcar_test": False},
    )
    assert result["ok"] is True
    assert result["pairwise_correlation"] == []


# === slice 14: constant (all-null) column → variance_zero, no φ pairs ===


def test_slice14_constant_column_is_variance_zero(call_tool, load_df_into_session):
    n = 50
    a = np.full(n, np.nan, dtype=float)  # all-null → variance_zero
    b = np.arange(n, dtype=float)
    b[:20] = np.nan  # b has nulls but variance > 0
    df = pd.DataFrame({"a": a, "b": b})
    load_df_into_session("const", df)

    result = call_tool(
        "analyze_missingness",
        {"name": "const", "pairwise_corr_threshold": 0.0, "run_mcar_test": False},
    )
    assert result["ok"] is True
    a_row = next(r for r in result["per_column"] if r["column"] == "a")
    assert a_row["variance_zero"] is True
    # No pair containing "a" should appear (variance is zero, φ undefined).
    for pair in result["pairwise_correlation"]:
        assert "a" not in (pair["col_a"], pair["col_b"]), pair


# === slice 15: >50% null triggers drop-suggestion; 49% does not ===


def test_slice15_high_null_threshold_suggestion(call_tool, load_df_into_session):
    n = 100
    # 51% null → triggers.
    high = np.arange(n, dtype=float)
    high[:51] = np.nan
    df_high = pd.DataFrame({"col": high})
    load_df_into_session("hi", df_high)
    r_hi = call_tool("analyze_missingness", {"name": "hi", "run_mcar_test": False})
    assert any("consider dropping or recoding" in s for s in r_hi["suggestions"])

    # 49% null → does NOT trigger.
    low = np.arange(n, dtype=float)
    low[:49] = np.nan
    df_low = pd.DataFrame({"col": low})
    load_df_into_session("lo", df_low)
    r_lo = call_tool("analyze_missingness", {"name": "lo", "run_mcar_test": False})
    assert not any("consider dropping or recoding" in s for s in r_lo["suggestions"])


# === slice 16: |φ| > 0.5 triggers co-missing; |φ|=0.49 does not ===


def test_slice16_comissing_threshold_suggestion(call_tool, load_df_into_session):
    # Strongly co-missing: identical masks → φ = 1.0
    n = 200
    a = np.arange(n, dtype=float)
    b = np.arange(n, dtype=float)
    for idx in range(50):
        a[idx] = np.nan
        b[idx] = np.nan
    df_co = pd.DataFrame({"a": a, "b": b})
    load_df_into_session("co", df_co)
    r_co = call_tool(
        "analyze_missingness",
        {"name": "co", "pairwise_corr_threshold": 0.0, "run_mcar_test": False},
    )
    assert any("co-missing" in s for s in r_co["suggestions"])

    # Independent missingness should NOT fire the |φ|>0.5 rule.
    indep = _mcar_independent_fixture()
    load_df_into_session("indep2", indep)
    r_ind = call_tool(
        "analyze_missingness",
        {"name": "indep2", "pairwise_corr_threshold": 0.0, "run_mcar_test": False},
    )
    assert not any("co-missing" in s for s in r_ind["suggestions"])


# === slice 17: more than 6 suggestions → return top 6 by severity ===


def test_slice17_suggestions_capped_at_six_severity_sorted(call_tool, load_df_into_session):
    # Build a dataset that fires 1 structural alignment + many >50%-null
    # columns + several co-missing pairs.
    n = 200
    region = np.array(["EMEA"] * 100 + ["AMER"] * 100)
    df = pd.DataFrame({"region": region})
    # Six high-null columns; each one >50% null AND co-missing pairwise.
    for k in range(6):
        col = np.arange(n, dtype=float)
        col[:120] = np.nan  # 60% null
        df[f"hi{k}"] = col
    # Add an alignment column: missing exactly for EMEA region.
    aligned = np.arange(n, dtype=float)
    aligned[region == "EMEA"] = np.nan
    df["aligned_col"] = aligned

    load_df_into_session("many", df)
    result = call_tool(
        "analyze_missingness",
        {"name": "many", "pairwise_corr_threshold": 0.0, "run_mcar_test": False},
    )
    assert result["ok"] is True
    suggestions = result["suggestions"]
    assert len(suggestions) <= 6
    # Structural alert must appear first (highest severity).
    assert "structural / join issue" in suggestions[0]


# === slice 18: recorder emits markdown + code; markdown lists top pattern ===


def test_slice18_recorder_emits_markdown_and_code(call_tool, load_df_into_session):
    from data_analyst_mcp.recorder import get_recorder

    df = _titanic_like_fixture()
    load_df_into_session("titanic_like", df)
    result = call_tool("analyze_missingness", {"name": "titanic_like", "run_mcar_test": False})
    assert result["ok"] is True

    cells = get_recorder().cells
    assert len(cells) == 2
    md, code = cells
    assert md["cell_type"] == "markdown"
    assert code["cell_type"] == "code"
    assert md["metadata"]["tool_name"] == "analyze_missingness"
    assert code["metadata"]["tool_name"] == "analyze_missingness"
    # Markdown summarizes the top pattern with its row count.
    top_count = result["patterns"][0]["count"]
    assert f"({top_count} rows)" in md["source"]
    # Code cell must compile as Python.
    compile(code["source"], "<analyze_missingness_cell>", "exec")
    # Code cell uses the proposal's `nulls = raw_df.isna()` shape.
    assert "nulls = raw_df.isna()" in code["source"]


# ======================================================================
# v1.1 - Little's MCAR test (slices M1-M7)
# ======================================================================


def _mcar_plant_fixture() -> pd.DataFrame:
    """Fixture #3 specialised for MCAR tests: n=1000, 5 cols, ~10% MCAR."""
    rng = np.random.default_rng(0)
    n = 1000
    cols: dict[str, np.ndarray] = {}
    for k in range(5):
        vals = rng.normal(0.0, 1.0, size=n)
        mask = rng.random(n) < 0.10
        arr = vals.copy()
        arr[mask] = np.nan
        cols[f"x{k}"] = arr
    return pd.DataFrame(cols)


def _mar_plant_fixture() -> pd.DataFrame:
    """Fixture #4: n=1000, 5 cols; col1 null with p = expit(0.5·col2).

    Strong MAR signal — column 1's missingness is fully explained by
    column 2. Little's test should reject MCAR sharply (p < 0.01).
    """
    rng = np.random.default_rng(0)
    n = 1000
    data = {f"x{k}": rng.normal(0.0, 1.0, size=n) for k in range(5)}
    df = pd.DataFrame(data)
    p_miss = 1.0 / (1.0 + np.exp(-0.5 * df["x2"].to_numpy()))
    mask = rng.random(n) < p_miss
    df.loc[mask, "x1"] = np.nan
    return df


# === M1: MCAR plant → p > 0.05 ===========================================


def test_M1_mcar_plant_fails_to_reject(call_tool, load_df_into_session):
    df = _mcar_plant_fixture()
    load_df_into_session("mcar_plant", df)
    result = call_tool("analyze_missingness", {"name": "mcar_plant"})
    assert result["ok"] is True
    mcar = result["mcar_test"]
    assert mcar is not None
    assert "p_value" in mcar
    # Truly MCAR data should not be flagged as violated.
    assert mcar["p_value"] > 0.05, mcar
    assert mcar["violated"] is False


# === M2: MAR plant → p < 0.01 ============================================


def test_M2_mar_plant_rejects_mcar(call_tool, load_df_into_session):
    df = _mar_plant_fixture()
    load_df_into_session("mar_plant", df)
    result = call_tool("analyze_missingness", {"name": "mar_plant"})
    assert result["ok"] is True
    mcar = result["mcar_test"]
    assert mcar is not None
    assert mcar["p_value"] < 0.01, mcar
    assert mcar["violated"] is True
    assert mcar["name"] == "little"


# === M3: consequence text matches rule table =============================


def test_M3_consequence_text_matches_rule_table(call_tool, load_df_into_session):
    # Rejected branch — MAR fixture guarantees a small p-value.
    load_df_into_session("mar", _mar_plant_fixture())
    rejected = call_tool("analyze_missingness", {"name": "mar"})["mcar_test"]
    assert (
        rejected["consequence"]
        == "Reject MCAR — missingness depends on observed data; mean-imputation will bias."
    )

    # Not-rejected branch — MCAR plant fixture.
    load_df_into_session("mcar", _mcar_plant_fixture())
    not_rejected = call_tool("analyze_missingness", {"name": "mcar"})["mcar_test"]
    assert (
        not_rejected["consequence"]
        == "Fail to reject MCAR; missingness is consistent with random absence."
    )


# === M3b: MCAR suggestion fires across the full [1, 50] null_pct band ====


def test_M3b_mcar_suggestion_fires_between_30_and_50_pct_null(
    call_tool, load_df_into_session
):
    """Regression: a column with null_pct in (30%, 50%] used to fall in a
    gap — MCAR-violation suggestion suppressed (cap was 30%), high-null
    suggestion not yet active (kicks in at 50%) — so `suggestions: []`
    even though MCAR was rejected. The MCAR band now extends to 50%."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x": x, "y": y})
    # Plant MAR on y conditional on x — high-x rows preferentially missing.
    # ~40% null places the column squarely in the formerly-empty band.
    p_miss = 1.0 / (1.0 + np.exp(-(1.5 * x - 0.4)))
    mask = rng.random(n) < p_miss
    df.loc[mask, "y"] = np.nan
    null_pct = float(df["y"].isna().mean() * 100)
    assert 30.0 < null_pct <= 50.0, null_pct
    load_df_into_session("mar40", df)
    result = call_tool("analyze_missingness", {"name": "mar40"})
    assert result["mcar_test"]["violated"] is True
    assert any(
        "mean-imputation will bias `y`" in s for s in result["suggestions"]
    ), result["suggestions"]


# === M4: run_mcar_test=False → mcar_test: null ===========================


def test_M4_run_mcar_test_false_skips_block(call_tool, load_df_into_session):
    df = _mar_plant_fixture()
    load_df_into_session("mar", df)
    result = call_tool("analyze_missingness", {"name": "mar", "run_mcar_test": False})
    assert result["ok"] is True
    assert result["mcar_test"] is None
    # And the MCAR-dependent suggestions must not fire.
    assert not any("MCAR" in s for s in result["suggestions"])


# === M5: agreement with the pinned reference suite =======================


_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "_mcar_reference"


@pytest.mark.parametrize("name", ["ozone", "iris_mar", "nhanes"])
def test_M5_matches_reference_suite(call_tool, load_df_into_session, name):
    csv_path = _REFERENCE_DIR / f"{name}.csv"
    with (_REFERENCE_DIR / "expected.json").open() as f:
        expected = json.load(f)
    df = pd.read_csv(csv_path)
    load_df_into_session(f"ref_{name}", df)
    result = call_tool("analyze_missingness", {"name": f"ref_{name}"})
    mcar = result["mcar_test"]
    assert mcar is not None, mcar
    assert "p_value" in mcar, mcar
    exp = expected[name]
    # ±1e-3 tolerance per the proposal's M5 contract.
    assert mcar["p_value"] == pytest.approx(exp["p_value"], abs=1e-3)
    assert mcar["df"] == exp["df"]
    assert mcar["statistic"] == pytest.approx(exp["statistic"], abs=1e-3)


# === M6: fewer than 2 valid patterns → skipped ===========================


def test_M6_insufficient_patterns_marks_skipped(call_tool, load_df_into_session):
    # Two numeric cols, but every row has the same (no-null) pattern → only
    # one pattern survives the n_j >= 2 filter, so the routine cannot form
    # a chi-square.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    load_df_into_session("one_pat", df)
    result = call_tool("analyze_missingness", {"name": "one_pat"})
    mcar = result["mcar_test"]
    assert mcar is not None
    assert mcar["skipped"] is True
    assert mcar["reason"] == "insufficient_patterns"


# === M7: categorical columns excluded from the test ======================


def test_M7_categorical_columns_excluded(call_tool, load_df_into_session):
    # Build a dataset where the *only* missingness is on a categorical
    # column. The numeric columns are fully observed, so once we exclude
    # the categorical the routine must report skipped (single pattern).
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0.0, 1.0, size=n),
            "num2": rng.normal(0.0, 1.0, size=n),
            "cat": rng.choice(["x", "y", "z"], size=n).astype(object),
        }
    )
    # Plant missingness only on the categorical.
    df.loc[rng.choice(n, size=50, replace=False), "cat"] = None
    load_df_into_session("cat_only", df)
    result = call_tool("analyze_missingness", {"name": "cat_only"})
    mcar = result["mcar_test"]
    assert mcar is not None
    # Since num1/num2 are fully observed, after numeric-only filtering
    # there is exactly one missingness pattern — routine must skip.
    assert mcar["skipped"] is True
    assert mcar["reason"] == "insufficient_patterns"
