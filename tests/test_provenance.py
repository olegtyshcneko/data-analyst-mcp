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
