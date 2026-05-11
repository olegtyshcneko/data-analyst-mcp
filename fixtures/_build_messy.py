"""Generate `fixtures/messy.csv` — a 5000-row hostile CSV.

This script is a fixture generator, not library code. It deliberately plants
the nine failure modes enumerated in `docs/SPEC.md` §7 so that downstream
tools (profile_dataset, describe_column, etc.) have a non-trivial target to
exercise. The output must be byte-identical on every run from a clean shell.

Planted issues (see `fixtures/README.md` for the mapping table):

1. UTF-8 BOM prefixing the first header.
2. Trailing whitespace in exactly 2 of the 12 column headers.
3. `signup_date` column mixes three formats: ISO, DD/MM/YYYY, "Mon DD, YYYY".
4. `revenue` column: "N/A" in 4% of rows, empty in another 2%.
5. `notes` column: exactly one row contains a comma inside its (quoted) value.
6. Exactly two byte-identical duplicate rows.
7. `email` column: exactly 78% empty.
8. `score` column: 20 IQR outliers, of which 5 are clear data-entry errors.
9. `country` column: 95% canonical {PL, US, DE, UA}, 5% dirty variants.

Run:
    uv run python fixtures/_build_messy.py

Re-running must produce a byte-identical file (asserted in the verifier).
"""

from __future__ import annotations

import csv
import io
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 20260511
N_ROWS = 5000

# The 12 columns. `revenue` and `last_login` carry trailing whitespace in
# their header strings (planted issue #2 — recorded in fixtures/README.md).
COLUMNS = [
    "customer_id",
    "signup_date",
    "country",
    "revenue ",  # trailing space — planted issue #2
    "score",
    "email",
    "notes",
    "plan",
    "age",
    "last_login ",  # trailing space — planted issue #2
    "channel",
    "is_active",
]

CANONICAL_COUNTRIES = ["PL", "US", "DE", "UA"]
DIRTY_COUNTRIES = ["poland", "Poland", "POL", "pl "]
PLANS = ["free", "pro", "enterprise", "trial"]
CHANNELS = ["organic", "paid", "referral", "email", "social"]


def _fmt_date(d: date, style: int) -> str:
    """Format a date in one of three planted styles (issue #3)."""
    if style == 0:
        return d.strftime("%Y-%m-%d")  # 2024-01-15
    if style == 1:
        return d.strftime("%d/%m/%Y")  # 15/01/2024
    return d.strftime("%b %d, %Y")  # Jan 15, 2024


def _customer_id(rng: random.Random, i: int) -> str:
    return f"CUST-{i:06d}-{rng.randint(1000, 9999)}"


def _email(rng: random.Random, i: int) -> str:
    handles = ["alex", "kasia", "ivan", "maria", "tom", "anna", "piotr", "lena"]
    domains = ["example.com", "mail.test", "corp.io", "biz.dev"]
    return f"{rng.choice(handles)}{i}@{rng.choice(domains)}"


def _last_login_iso(rng: random.Random, signup: date) -> str:
    delta_days = rng.randint(0, 600)
    ts = datetime.combine(signup, datetime.min.time()) + timedelta(
        days=delta_days, hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
    )
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _build_score_pool(
    rng: random.Random, n_rows: int, outlier_positions: list[int]
) -> list[float]:
    """Build a score column with exactly 20 IQR outliers at known positions.

    Strategy: draw n_rows-20 "clean" scores from a tight roughly-Gaussian
    distribution so the empirical IQR is well-defined and bounded, then
    place 20 deliberate outliers at the supplied row indices — 15
    mild-but-out-of-bounds and 5 obvious data-entry errors (e.g. 99999).

    We use a tight inlier band (mean=70, sd=8, clamped to [50, 90]) so the
    empirical IQR sits in roughly [64, 76] and the 1.5·IQR fences land
    near [46, 94]. Mild outliers placed at 20–35 and 105–130 are then
    comfortably outside; the 99999-class entries are wildly outside.
    """
    assert len(outlier_positions) == 20
    outlier_set = set(outlier_positions)

    # Inliers fill every non-outlier row in iteration order.
    pool: list[float] = [0.0] * n_rows
    for i in range(n_rows):
        if i in outlier_set:
            continue
        while True:
            v = rng.gauss(70.0, 8.0)
            if 50.0 <= v <= 90.0:
                pool[i] = round(v, 1)
                break

    # 15 mild outliers + 5 data-entry errors. Deterministic order so the
    # output is byte-stable across runs.
    mild_low = [round(rng.uniform(20.0, 35.0), 1) for _ in range(8)]
    mild_high = [round(rng.uniform(105.0, 130.0), 1) for _ in range(7)]
    data_entry_errors = [99999.0, 88888.0, 99999.0, 77777.0, 123456.0]
    outliers = mild_low + mild_high + data_entry_errors
    for pos, val in zip(outlier_positions, outliers, strict=True):
        pool[pos] = val
    return pool


def _build_rows(rng: random.Random) -> list[list[str]]:
    """Construct the 5000 raw data rows.

    All randomness flows from a single seeded `random.Random` so output is
    fully deterministic given SEED. Returns a list of row lists matching
    the COLUMNS order.

    Position picking is performed in one pass at the top so the duplicate
    injection (issue #6) can deliberately pick two rows that are NOT
    members of any other planted-issue set — guaranteeing the duplicate
    overwrite does not perturb any of the eight count-based assertions.
    """
    # Pre-compute the country dirty-positions: 5% of rows take dirty variants.
    n_dirty = N_ROWS * 5 // 100  # 250
    dirty_idx = set(rng.sample(range(N_ROWS), n_dirty))

    # Pre-compute revenue null/NA positions:
    #   issue #4: "N/A" in 4% of rows (200) and empty string in another 2% (100).
    n_na = N_ROWS * 4 // 100  # 200
    n_empty = N_ROWS * 2 // 100  # 100
    all_rev_special = set(rng.sample(range(N_ROWS), n_na + n_empty))
    na_idx = set(rng.sample(sorted(all_rev_special), n_na))
    empty_rev_idx = all_rev_special - na_idx

    # Pre-compute email-null positions:
    #   issue #7: 78% empty → 3900 empty cells.
    n_empty_email = N_ROWS * 78 // 100  # 3900
    empty_email_idx = set(rng.sample(range(N_ROWS), n_empty_email))

    # Pre-compute the 20 outlier-score positions (issue #8).
    outlier_positions = rng.sample(range(N_ROWS), 20)

    # Build the score pool with outliers at those exact positions.
    scores = _build_score_pool(rng, N_ROWS, outlier_positions)

    # Pre-compute signup_date format mix: roughly even thirds across rows.
    # Using rng so order is deterministic.
    date_styles = [rng.randint(0, 2) for _ in range(N_ROWS)]

    # Pick exactly one row to carry the comma-bearing `notes` value
    # (issue #5). CSV writer quotes it so DuckDB parses cleanly.
    notes_comma_row = rng.randrange(N_ROWS)

    # Pre-compute the duplicate-row pair (issue #6) BEFORE constructing
    # rows, choosing two indices that are NOT members of any other planted
    # issue set. This way overwriting row B with a copy of row A perturbs
    # none of the other counts. We also avoid the commaful-notes row.
    excluded = (
        dirty_idx
        | na_idx
        | empty_rev_idx
        | empty_email_idx
        | set(outlier_positions)
        | {notes_comma_row}
    )
    candidates = [i for i in range(N_ROWS) if i not in excluded]
    assert len(candidates) >= 2, "Not enough untouched rows to place the duplicate pair"
    dup_a, dup_b = sorted(rng.sample(candidates, 2))

    # Anchor for signup_date generation.
    epoch = date(2023, 1, 1)

    rows: list[list[str]] = []
    for i in range(N_ROWS):
        signup = epoch + timedelta(days=rng.randint(0, 730))
        signup_str = _fmt_date(signup, date_styles[i])

        country = (
            rng.choice(DIRTY_COUNTRIES) if i in dirty_idx else rng.choice(CANONICAL_COUNTRIES)
        )

        if i in na_idx:
            revenue = "N/A"
        elif i in empty_rev_idx:
            revenue = ""
        else:
            revenue = f"{rng.uniform(10.0, 5000.0):.2f}"

        score = scores[i]
        email = "" if i in empty_email_idx else _email(rng, i)

        if i == notes_comma_row:
            # The lone commaful note (issue #5). CSV writer will quote it.
            notes = "Customer reported issue with shipping, refund issued."
        else:
            note_phrases = [
                "follow up next quarter",
                "high-value lead",
                "needs onboarding call",
                "renewal at risk",
                "expansion opportunity",
                "champion in finance team",
            ]
            notes = rng.choice(note_phrases)

        plan = rng.choice(PLANS)
        age = rng.randint(18, 75)
        last_login = _last_login_iso(rng, signup)
        channel = rng.choice(CHANNELS)
        is_active = "true" if rng.random() < 0.7 else "false"

        rows.append(
            [
                _customer_id(rng, i),
                signup_str,
                country,
                revenue,
                f"{score}",
                email,
                notes,
                plan,
                f"{age}",
                last_login,
                channel,
                is_active,
            ]
        )

    # Inject exactly two byte-identical duplicate rows (issue #6) by
    # overwriting row dup_b with a copy of row dup_a. We pre-screened both
    # indices to fall outside every other planted-issue set, so the copy
    # leaves all other counts undisturbed.
    rows[dup_b] = list(rows[dup_a])

    return rows


def build() -> bytes:
    """Render the messy CSV to bytes (BOM + header + rows).

    We render via `csv.writer` into a text buffer first (so quoting rules
    are applied correctly for the commaful notes value), then prepend the
    UTF-8 BOM and encode as bytes. The BOM is part of the file, not part of
    any column name — DuckDB's CSV reader handles it but it is a legitimate
    planted issue for downstream profilers.
    """
    rng = random.Random(SEED)
    rows = _build_rows(rng)

    buf = io.StringIO()
    # newline="" semantics: csv.writer emits \r\n by default. We force \n
    # for portability and reproducibility.
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(COLUMNS)
    writer.writerows(rows)

    text = buf.getvalue()
    # UTF-8 BOM as bytes — issue #1.
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def main() -> None:
    out_path = Path(__file__).parent / "messy.csv"
    out_path.write_bytes(build())


if __name__ == "__main__":
    main()
