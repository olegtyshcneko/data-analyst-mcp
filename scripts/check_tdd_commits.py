#!/usr/bin/env python3
"""TDD discipline auditor.

Walks `git log --reverse --pretty=%H%x09%s` and verifies that every commit
whose subject starts with ``green:`` is immediately preceded (on the same
linear branch) by a commit whose subject starts with ``red:`` and shares the
same behavior suffix.

Exits non-zero with a per-mismatch report if the discipline is broken.

Commits whose subject starts with ``refactor:`` are ignored — they may appear
anywhere after a ``green:``. Commits with any other prefix are also ignored
(e.g. ``chore:``, ``docs:``, ``ci:``).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str

    @property
    def prefix(self) -> str | None:
        for marker in ("red:", "green:", "refactor:"):
            if self.subject.startswith(marker):
                return marker[:-1]
        return None

    @property
    def behavior(self) -> str:
        prefix = self.prefix
        if prefix is None:
            return self.subject
        return self.subject[len(prefix) + 1 :].strip()


def load_commits() -> list[Commit]:
    """Return commits in chronological order (oldest first)."""
    result = subprocess.run(
        ["git", "log", "--reverse", "--pretty=%H%x09%s"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "does not have any commits yet" in stderr or "bad default revision" in stderr:
            return []
        sys.stderr.write(f"git log failed: {result.stderr}\n")
        sys.exit(2)

    commits: list[Commit] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        sha, _, subject = line.partition("\t")
        commits.append(Commit(sha=sha, subject=subject))
    return commits


def audit(commits: list[Commit]) -> list[str]:
    """Return a list of human-readable mismatches; empty list = pass."""
    mismatches: list[str] = []
    prev_red: Commit | None = None

    for commit in commits:
        prefix = commit.prefix
        if prefix == "red":
            prev_red = commit
            continue
        if prefix == "green":
            if prev_red is None:
                mismatches.append(
                    f"{commit.sha[:8]} green: with no preceding red: — {commit.subject!r}"
                )
            elif prev_red.behavior != commit.behavior:
                mismatches.append(
                    f"{commit.sha[:8]} green: behavior {commit.behavior!r} does not "
                    f"match preceding red: behavior {prev_red.behavior!r} "
                    f"(red sha {prev_red.sha[:8]})"
                )
            prev_red = None
            continue
        if prefix == "refactor":
            continue

    return mismatches


def main() -> int:
    commits = load_commits()
    mismatches = audit(commits)

    if mismatches:
        sys.stderr.write("TDD discipline check FAILED:\n")
        for m in mismatches:
            sys.stderr.write(f"  - {m}\n")
        sys.stderr.write(f"\n{len(mismatches)} mismatch(es) found across {len(commits)} commits.\n")
        return 1

    red_count = sum(1 for c in commits if c.prefix == "red")
    green_count = sum(1 for c in commits if c.prefix == "green")
    refactor_count = sum(1 for c in commits if c.prefix == "refactor")
    sys.stderr.write(
        f"TDD discipline OK: {red_count} red, {green_count} green, "
        f"{refactor_count} refactor across {len(commits)} commits.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
