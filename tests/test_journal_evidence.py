"""Unit tests for nonfinite tagging and evidence comparison."""

from __future__ import annotations

import math


def test_tag_and_untag_round_trip() -> None:
    from data_analyst_mcp.journal_evidence import tag_nonfinite, untag

    assert tag_nonfinite(1.5) == 1.5
    assert tag_nonfinite(float("nan")) == "NaN"
    assert tag_nonfinite(float("inf")) == "Infinity"
    assert tag_nonfinite(float("-inf")) == "-Infinity"
    assert math.isnan(untag("NaN"))
    assert untag("Infinity") == float("inf")
    assert untag(2.0) == 2.0


def test_evidence_equal_exact_key_set() -> None:
    from data_analyst_mcp.journal_evidence import evidence_equal

    assert evidence_equal({"a": 1.0}, {"a": 1.0 + 1e-12})
    assert not evidence_equal({"a": 1.0}, {"a": 1.0, "b": 2.0})  # extra key
    assert not evidence_equal({"a": 1.0, "b": 2.0}, {"a": 1.0})  # missing key
    assert not evidence_equal({"a": 1.0}, {"a": 1.001})  # outside rtol=1e-7


def test_evidence_equal_nonfinite_semantics() -> None:
    from data_analyst_mcp.journal_evidence import evidence_equal

    assert evidence_equal({"a": "NaN"}, {"a": "NaN"})
    assert not evidence_equal({"a": "Infinity"}, {"a": "-Infinity"})
    assert not evidence_equal({"a": "NaN"}, {"a": 0.0})
