"""End-to-end evals for ``pairwise_comparisons`` — post-hoc pairwise tests.

These drive the follow-up to ``compare_groups`` through the live stdio
server: the Shapiro-gated ``auto`` engine pick (Dunn on skewed CRM amounts),
an explicit Tukey run that recovers a planted mean shift, the
``method='tukey'`` + ``p_adjust`` conflict as a protocol error envelope, and
the recorded-notebook nbconvert round-trip that keeps the reproducibility
promise honest.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import CRM_DIR, PROJECT_ROOT, call, load_crm, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"


@pytest.mark.eval
async def eval_auto_picks_dunn_on_skewed_crm_amounts():
    """Right-skewed CRM amounts fail Shapiro, so auto resolves to Dunn's test."""
    async with mcp_session() as s:
        await load_crm(s)
        r = await call(
            s,
            "pairwise_comparisons",
            {
                "name": "opportunities",
                "group_column": "stage",
                "metric_column": "amount",
            },
        )
        assert r["ok"], r
        # Amount is heavily right-skewed (lognormal-ish), so per-group Shapiro
        # rejects and the auto gate switches ANOVA/Tukey -> Kruskal/Dunn.
        assert r["method"] == "dunn", r["method"]
        assert r["method_requested"] == "auto", r["method_requested"]
        # 6 stages -> C(6, 2) = 15 pairwise comparisons.
        assert r["n_comparisons"] == 15, r["n_comparisons"]
        assert r["n_rejected"] == 0, r["n_rejected"]

        omnibus = r["omnibus"]
        assert omnibus["test"] == "kruskal_wallis", omnibus
        assert omnibus["statistic"] == pytest.approx(8.9686, abs=1e-3), omnibus
        assert omnibus["p_value"] == pytest.approx(0.1103, abs=1e-3), omnibus
        # p=0.1103 >= alpha, so the omnibus is not significant and the
        # interpretation must caveat the pairwise findings (see posthoc.py
        # _interpretation).
        assert omnibus["significant"] is False, omnibus
        assert "so treat the pairwise findings cautiously" in r["interpretation"], r[
            "interpretation"
        ]


@pytest.mark.eval
async def eval_explicit_tukey_recovers_planted_shift(tmp_path_factory):
    """Explicit Tukey recovers a planted C-vs-{A,B} mean shift; A vs B stays null."""
    tmp_path: Path = tmp_path_factory.mktemp("pairwise_eval")
    # Three groups n=40: A/B ~ N(0, 1), C ~ N(1, 1), one shared seeded RNG so
    # the draw order (A, then B, then C) is reproducible. Verified locally with
    # statsmodels.pairwise_tukeyhsd on this exact data: A-C and B-C reject, A-B
    # does not; A-C meandiff≈1.0849 ci≈[0.5733, 1.5966].
    rng = np.random.default_rng(12345)
    group_a = rng.normal(loc=0.0, scale=1.0, size=40)
    group_b = rng.normal(loc=0.0, scale=1.0, size=40)
    group_c = rng.normal(loc=1.0, scale=1.0, size=40)
    csv_path = tmp_path / "planted_shift.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "metric"])
        for label, values in (("A", group_a), ("B", group_b), ("C", group_c)):
            for v in values:
                writer.writerow([label, float(v)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "planted"})
        assert r["ok"], r
        r = await call(
            s,
            "pairwise_comparisons",
            {
                "name": "planted",
                "group_column": "group",
                "metric_column": "metric",
                "method": "tukey",
            },
        )
        assert r["ok"], r
        assert r["method"] == "tukey", r["method"]
        # Tukey controls FWER internally, so no separate correction is echoed.
        assert r["p_adjust"] is None, r["p_adjust"]
        assert r["estimate_name"] == "mean_diff", r["estimate_name"]

        by = {(c["group_a"], c["group_b"]): c for c in r["comparisons"]}
        ab, ac, bc = by[("A", "B")], by[("A", "C")], by[("B", "C")]
        # The planted +1 shift on C is recovered against both A and B; the two
        # zero-mean groups are indistinguishable.
        assert ab["reject"] is False, ab
        assert ac["reject"] is True, ac
        assert bc["reject"] is True, bc

        # A rejected pair carries a confidence interval bracketing its estimate.
        assert ac["ci_low"] < ac["estimate"] < ac["ci_high"], ac


@pytest.mark.eval
async def eval_tukey_with_p_adjust_returns_structured_error():
    """method='tukey' + explicit p_adjust is rejected as a live error envelope."""
    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(CRM_DIR / "opportunities.csv"), "name": "opportunities"},
        )
        assert r["ok"], r
        r = await call(
            s,
            "pairwise_comparisons",
            {
                "name": "opportunities",
                "group_column": "stage",
                "metric_column": "amount",
                "method": "tukey",
                "p_adjust": "holm",
            },
        )
        assert r["ok"] is False, r
        assert r["error"]["type"] == "p_adjust_not_applicable", r["error"]
        assert r["error"]["hint"], r["error"]


@pytest.mark.eval
async def eval_pairwise_notebook_round_trips():
    """CRM Dunn pairwise → emit → jupyter nbconvert --execute exits 0."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_path = ARTIFACTS / "pairwise_round_trip.ipynb"
    if nb_path.exists():
        nb_path.unlink()

    async with mcp_session() as s:
        await load_crm(s)
        r = await call(
            s,
            "pairwise_comparisons",
            {
                "name": "opportunities",
                "group_column": "stage",
                "metric_column": "amount",
            },
        )
        assert r["ok"], r
        assert r["method"] == "dunn", r["method"]
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"], r

    assert nb_path.exists()
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
            "--ExecutePreprocessor.timeout=180",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )
