"""End-to-end evals for prefix replay guards (spec: prefix-guard realization).

The ROADMAP failure class: load → cross_validate/fit → mutate the source
CSV → reload → emit. The setup cell's latest-registration assert passes,
but the FIRST load cell's own hash assert must fail replay loudly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import PROJECT_ROOT, call, mcp_session

ARTIFACTS = PROJECT_ROOT / "evals" / "_artifacts"


def _nbconvert(nb_path: Path) -> subprocess.CompletedProcess[str]:
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


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("y,x\n" + "\n".join(rows) + "\n")


@pytest.mark.eval
async def eval_mutated_and_reloaded_source_fails_replay_at_first_load_cell():
    """load → cross_validate → edit CSV → reload → emit → nbconvert must
    fail with the load-cell drift message, not silently recompute."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_source.csv"
    nb_path = ARTIFACTS / "eval_replay_guards.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(
            s, "cross_validate", {"name": "g", "formula": "y ~ x", "kind": "ols", "k": 4}
        )
        assert r["ok"] is True
        # Mutate the source, then reload — setup will assert the NEW hash.
        _write_csv(csv_path, [f"{(i * 5) % 11}.0,{i}.0" for i in range(40)])
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode != 0, "replay must fail loudly, not recompute silently"
    combined = proc.stdout + proc.stderr
    assert "changed since the session was recorded" in combined
    assert "'g'" in combined or '"g"' in combined


@pytest.mark.eval
async def eval_mutated_and_reloaded_source_fails_replay_for_ephemeral_fit():
    """Sibling case: ephemeral fit_model instead of cross_validate."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_source_fit.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_fit.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "fit_model", {"name": "g", "formula": "y ~ x", "kind": "ols"})
        assert r["ok"] is True
        _write_csv(csv_path, [f"{(i * 5) % 11}.0,{i}.0" for i in range(40)])
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode != 0
    assert "changed since the session was recorded" in (proc.stdout + proc.stderr)


@pytest.mark.eval
async def eval_stable_source_ephemeral_fit_replays_cleanly():
    """Success sibling (spec slice 9): an untouched source with an ephemeral
    fit must round-trip through nbconvert with exit 0 — the guards must not
    fire on a faithful session."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_stable.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_stable.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "g"})
        assert r["ok"] is True
        r = await call(s, "fit_model", {"name": "g", "formula": "y ~ x", "kind": "ols"})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.eval
async def eval_rematerialize_after_cv_replays_cleanly():
    """materialize S1 → CV(d) → materialize S2 (overwrite) is faithful at
    replay (the prefix recreates S1 before the CV cell) and must exit 0 —
    the first-draft dispatch wrongly raised here; pin the fix.

    Constraint (spec slice 10): BOTH recipes read the stable surviving
    file-backed `base` — a self-referential S2 (SELECT ... FROM d) would
    hit the parked S4b setup failure instead and invalidate the pin."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_base.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_remat.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(40)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "base"})
        assert r["ok"] is True
        r = await call(
            s,
            "materialize_query",
            {"sql": "SELECT y, x FROM base WHERE x < 30", "name": "d"},
        )
        assert r["ok"] is True
        r = await call(
            s, "cross_validate", {"name": "d", "formula": "y ~ x", "kind": "ols", "k": 3}
        )
        assert r["ok"] is True
        r = await call(
            s,
            "materialize_query",
            {"sql": "SELECT y, x FROM base", "name": "d", "overwrite": True},
        )
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.eval
async def eval_resplit_after_cv_replays_cleanly():
    """split(seed=1) → CV(train) → split(seed=2, overwrite) under the same
    names: the prefix recreates the seed-1 sides (own checksums) before the
    CV cell, then the seed-2 sides after. Must exit 0."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "guards_split.csv"
    nb_path = ARTIFACTS / "eval_replay_guards_resplit.ipynb"
    if nb_path.exists():
        nb_path.unlink()
    _write_csv(csv_path, [f"{(i * 7) % 13}.0,{i}.0" for i in range(60)])

    async with mcp_session() as s:
        r = await call(s, "load_dataset", {"path": str(csv_path), "name": "src"})
        assert r["ok"] is True
        r = await call(s, "split_dataset", {"name": "src", "seed": 1})
        assert r["ok"] is True
        r = await call(
            s,
            "cross_validate",
            {"name": "src_train", "formula": "y ~ x", "kind": "ols", "k": 3},
        )
        assert r["ok"] is True
        r = await call(s, "split_dataset", {"name": "src", "seed": 2, "overwrite": True})
        assert r["ok"] is True
        r = await call(s, "emit_notebook", {"path": str(nb_path)})
        assert r["ok"] is True

    proc = _nbconvert(nb_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
