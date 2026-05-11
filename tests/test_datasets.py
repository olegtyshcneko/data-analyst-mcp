"""Tests for the dataset tools (load_dataset, list_datasets, profile, describe)."""

from __future__ import annotations

import os
from typing import Any

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
MESSY_CSV = os.path.join(FIXTURE_DIR, "messy.csv")


def test_load_dataset_rejects_unsupported_extension(call_tool: Any) -> None:
    result = call_tool("load_dataset", {"path": "/tmp/nope.xyz"})

    assert result["ok"] is False
    assert result["error"]["type"] == "unsupported_format"
    assert ".xyz" in result["error"]["message"] or "xyz" in result["error"]["message"]
