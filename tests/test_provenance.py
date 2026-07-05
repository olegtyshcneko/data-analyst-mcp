"""Tests for provenance.compute_source_hash — the shared drift-guard hash."""

from __future__ import annotations

import hashlib
from pathlib import Path


def test_compute_source_hash_matches_sha256(tmp_path: Path) -> None:
    from data_analyst_mcp.provenance import compute_source_hash

    csv = tmp_path / "tiny.csv"
    csv.write_bytes(b"a,b\n1,2\n3,4\n")
    expected = hashlib.sha256(csv.read_bytes()).hexdigest()

    assert compute_source_hash(str(csv)) == expected


def test_compute_source_hash_handles_in_memory_dataset() -> None:
    """In-memory datasets (no file path) get a deterministic sentinel
    rather than throwing — the recorder cell uses this to skip the
    hash assert without silently mismatching."""
    from data_analyst_mcp.provenance import compute_source_hash

    h = compute_source_hash("(dataframe)")
    assert h.startswith("sentinel:")


def test_compute_source_hash_returns_read_failed_sentinel_on_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    """A file that stats fine but fails to open/read (vanished mid-hash,
    permissions) must yield a sentinel, not an exception — hashing runs
    inside load_dataset's success path and must never add a failure mode."""
    import builtins

    from data_analyst_mcp.provenance import compute_source_hash

    csv = tmp_path / "vanish.csv"
    csv.write_bytes(b"a\n1\n")
    real_open = builtins.open

    def _raising_open(file: object, *args: object, **kwargs: object) -> object:
        if str(file) == str(csv):
            raise OSError("disappeared mid-hash")
        return real_open(file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", _raising_open)

    assert compute_source_hash(str(csv)) == f"sentinel:read-failed:{csv}"
