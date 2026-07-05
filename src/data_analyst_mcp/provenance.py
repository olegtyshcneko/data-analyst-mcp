"""Source-file provenance hashing — the drift guard shared by datasets and models."""

from __future__ import annotations

import hashlib
import os

# Above this file size (bytes) we skip content-hashing in favour of a
# cheap ``(path, mtime, size)`` tuple — content-hash on a 5 GB CSV is
# slow enough that the pause is user-visible. Documented as a weaker
# drift guarantee in the provenance-hashes design spec.
HASH_CONTENT_CEILING_BYTES = 100 * 1024 * 1024


def compute_source_hash(path: str) -> str:
    """Hash a source file for the recorder's drift guard.

    Files up to ``HASH_CONTENT_CEILING_BYTES`` are content-hashed
    (SHA-256 of bytes). Larger files fall back to a cheap
    ``(path, mtime, size)`` tuple — a weaker guarantee. In-memory
    datasets (``path == "(dataframe)"``), derived datasets
    (``path == "(query)"``), and any other non-file path are tagged with
    a stable sentinel so the recorder can detect them and skip the hash
    assert without silently mismatching.
    """
    if not os.path.isfile(path):
        return f"sentinel:no-file:{path}"
    try:
        size = os.path.getsize(path)
    except OSError:
        return f"sentinel:stat-failed:{path}"
    if size <= HASH_CONTENT_CEILING_BYTES:
        h = hashlib.sha256()
        # Stream in 1 MB chunks; SHA-256 of a 100 MB file at ~500 MB/s is
        # under a quarter second on commodity hardware.
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    # Above the ceiling: fall back to (path, mtime, size). Weaker guarantee
    # — a careful edit that preserves mtime + size will not trigger the
    # drift assert — but content-hashing 5 GB is too slow for an
    # interactive session.
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    fallback_hash = hashlib.sha256(f"{path}|{mtime}|{size}".encode()).hexdigest()
    return f"fallback:{fallback_hash}"
