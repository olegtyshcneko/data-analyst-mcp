"""Shared SQL safety helpers used by the read-only tools (``query``,
``materialize_query``).

The threat model is narrow: the tools' allowlist on the *leading* keyword
already blocks single-statement DDL/DML. The remaining gap is a payload
like ``"SELECT 1; DROP TABLE base"`` — a SELECT that splices a second
destructive statement after the first ``;``. DuckDB happily executes
multi-statement strings, so we must reject these at the tool layer.

A real SQL parser is overkill for this single check. The hand-rolled
scanner below walks the string character-by-character, tracking whether
we're inside a quoted literal or a comment, and reports True iff it
encounters a ``;`` that is *not* part of trailing whitespace and *not*
inside a literal/comment. That is the only thing we need to reject.
"""

from __future__ import annotations


def contains_unsafe_semicolon(sql: str) -> bool:
    """Return True if ``sql`` contains a ``;`` that would terminate the
    first statement and let a second statement follow.

    The scanner ignores:

    - ``;`` inside ``'...'`` single-quoted string literals (with ``''``
      escape), and ``"..."`` double-quoted identifiers/strings.
    - ``;`` inside ``-- line comments`` (until end-of-line).
    - ``;`` inside ``/* block comments */``.
    - A trailing ``;`` followed only by whitespace / comments — that is a
      single statement with a terminator, not a multi-statement payload.

    Everything else is flagged.
    """
    n = len(sql)
    i = 0
    # Track positions of every "real" semicolon (outside literals/comments).
    semicolons: list[int] = []
    while i < n:
        ch = sql[i]
        # ---- single-quoted SQL string literal ('...' with '' escape) ----
        if ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    # ''-escape: stay inside the literal.
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        # ---- double-quoted identifier / string ("..." with "" escape) ----
        if ch == '"':
            i += 1
            while i < n:
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        # ---- line comment: -- ... end-of-line ----
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            continue
        # ---- block comment: /* ... */ (no nesting per SQL spec) ----
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            i += 2
            while i < n:
                if sql[i] == "*" and i + 1 < n and sql[i + 1] == "/":
                    i += 2
                    break
                i += 1
            continue
        if ch == ";":
            semicolons.append(i)
        i += 1

    if not semicolons:
        return False
    # A trailing ``;`` with nothing but whitespace / comments after it is
    # benign — that's the single-statement terminator case. Scan the tail
    # from the last semicolon; if no executable content follows, allow it.
    last = semicolons[-1]
    tail = sql[last + 1 :]
    if not _has_executable_content(tail):
        # And every *earlier* semicolon would still be a multi-statement
        # marker — but in practice that's already a multistatement payload.
        return len(semicolons) > 1
    return True


def _has_executable_content(fragment: str) -> bool:
    """True if ``fragment`` contains anything other than whitespace and comments."""
    n = len(fragment)
    i = 0
    while i < n:
        ch = fragment[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "-" and i + 1 < n and fragment[i + 1] == "-":
            i += 2
            while i < n and fragment[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and fragment[i + 1] == "*":
            i += 2
            while i < n:
                if fragment[i] == "*" and i + 1 < n and fragment[i + 1] == "/":
                    i += 2
                    break
                i += 1
            continue
        return True
    return False
