"""End-to-end evals for the model workflow bundle: split + CV + replay.

The full session — load titanic → split_dataset → fit on train →
evaluate on test → cross_validate — is emitted as a notebook and
re-executed via ``jupyter nbconvert --execute``. The membership-checksum
and source SHA-256 asserts inside the notebook are the drift guards under
test: a clean session must round-trip to exit 0, and a source CSV mutated
between the recording session and replay must fail loudly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import FIXTURES_DIR, PROJECT_ROOT, call, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"


def _nbconvert(nb_path: Path) -> subprocess.CompletedProcess[str]:
    """Re-execute the notebook in-place via the project's jupyter."""
    return subprocess.run(
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


@pytest.mark.eval
async def eval_split_fit_evaluate_cv_replays_via_nbconvert():
    """split → fit(train) → evaluate(test) → cross_validate → emit →
    nbconvert --execute exits 0 (checksum assert passes)."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_path = ARTIFACTS / "eval_split_cv.ipynb"
    if nb_path.exists():
        nb_path.unlink()

    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "titanic.csv"), "name": "titanic"},
        )
        assert r["ok"], r
        r = await call(s, "split_dataset", {"name": "titanic", "seed": 7})
        assert r["ok"], r
        assert r["train"]["rows"] + r["test"]["rows"] == 887
        r = await call(
            s,
            "fit_model",
            {
                "name": "titanic_train",
                "formula": "Survived ~ C(Sex) + C(Pclass)",
                "kind": "logistic",
                "model_name": "surv",
            },
        )
        assert r["ok"], r
        r = await call(s, "evaluate_model", {"model_name": "surv", "dataset": "titanic_test"})
        assert r["ok"], r
        cv = await call(
            s,
            "cross_validate",
            {
                "name": "titanic_train",
                "formula": "Survived ~ C(Sex) + C(Pclass)",
                "kind": "logistic",
                "k": 3,
            },
        )
        assert cv["ok"], cv
        assert cv["stratified"] is True

        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"], r

    assert nb_path.exists()
    result = _nbconvert(nb_path)
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


@pytest.mark.eval
async def eval_split_of_derived_source_replays():
    """materialize_query (filter) → split_dataset on the derived table →
    emit → replay. Covers the derived-source checksum path."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_path = ARTIFACTS / "eval_split_derived.ipynb"
    if nb_path.exists():
        nb_path.unlink()

    async with mcp_session() as s:
        r = await call(
            s,
            "load_dataset",
            {"path": str(FIXTURES_DIR / "titanic.csv"), "name": "titanic"},
        )
        assert r["ok"], r
        r = await call(
            s,
            "materialize_query",
            {
                "sql": 'SELECT * FROM "titanic" WHERE "Age" IS NOT NULL',
                "name": "adults",
            },
        )
        assert r["ok"], r
        r = await call(s, "split_dataset", {"name": "adults"})
        assert r["ok"], r

        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"], r

    assert nb_path.exists()
    result = _nbconvert(nb_path)
    assert result.returncode == 0, (
        f"nbconvert failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
    )


@pytest.mark.eval
async def eval_split_source_drift_fails_replay(tmp_path: Path):
    """Edit the source CSV between session and replay → the setup cell's
    SHA-256 assert must fail before any split runs."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    nb_path = ARTIFACTS / "eval_split_drift.ipynb"
    if nb_path.exists():
        nb_path.unlink()

    src = tmp_path / "drifting.csv"
    src.write_text("x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(20)) + "\n")

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(src), "name": "drift"})
        assert r["ok"], r
        r = await call(s, "split_dataset", {"name": "drift"})
        assert r["ok"], r

        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"], r

    # Mutate the source after the session — the recorded SHA-256 no longer matches.
    src.write_text("x,y\n" + "\n".join(f"{i},{i * 3}" for i in range(20)) + "\n")

    assert nb_path.exists()
    result = _nbconvert(nb_path)
    assert result.returncode != 0, "replay should have failed on the drift assert"
    assert "AssertionError" in (result.stderr + result.stdout)
