"""Generate `fixtures/synthetic_crm/{accounts,contacts,opportunities}.csv`.

A small, realistic CRM-shaped dataset for exercising the analytical tools
(joins, ANOVA across stages, logistic regression on `won ~ amount + industry`,
chi-square on industry × won, etc.). Deterministic given the fixed seeds.

Realism properties exercised (see `fixtures/README.md`):

- 2000 accounts, 20000 contacts, 8000 opportunities.
- ARR log-normally distributed (so describe_column has skew to report).
- Opportunity `amount` correlates with parent account `arr` (Pearson ≥ 0.3).
- ~70% of opportunities closed; of closed, ~28% won. Open opps have
  NULL `closed_at` and NULL `won`.
- ~10% of accounts flagged `churned=true` (extension to the spec — see README).
- Every account has ≥1 contact; ~70% have ≥3; exactly one contact per
  account has `is_primary=true`.
- All FKs resolve: `contacts.account_id` and `opportunities.account_id`
  are guaranteed to be members of `accounts.account_id`.

Run:
    uv run python fixtures/generate_synthetic_crm.py

Re-running must produce byte-identical CSVs.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

SEED = 20260511
N_ACCOUNTS = 2000
N_CONTACTS = 20000
N_OPPS = 8000

INDUSTRIES = [
    "Software",
    "Manufacturing",
    "Healthcare",
    "Retail",
    "Finance",
    "Education",
    "Logistics",
    "Energy",
]
COUNTRIES = [
    "US",
    "PL",
    "DE",
    "UA",
    "GB",
    "FR",
    "ES",
    "IT",
    "NL",
    "SE",
    "NO",
    "FI",
]
STAGES = [
    "Prospecting",
    "Qualification",
    "Proposal",
    "Negotiation",
    "Closed Won",
    "Closed Lost",
]

CRM_DIR = Path(__file__).parent / "synthetic_crm"


def _seed_everything() -> tuple[random.Random, np.random.Generator, Faker]:
    """Reset every randomness source we use so output is byte-stable."""
    random.seed(SEED)
    np.random.seed(SEED)
    Faker.seed(SEED)
    fake = Faker("en_US")
    py_rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    return py_rng, np_rng, fake


def _account_uuid(py_rng: random.Random) -> str:
    """UUID-shaped string (deterministic via py_rng — uuid.uuid4 is not).

    8-4-4-4-12 hex, matching the visual shape of a UUID without claiming
    actual RFC 4122 conformance. Determinism matters more here than
    standards-compliance.
    """
    hexchars = "0123456789abcdef"
    parts = [8, 4, 4, 4, 12]
    return "-".join("".join(py_rng.choice(hexchars) for _ in range(n)) for n in parts)


def _build_accounts(
    py_rng: random.Random, np_rng: np.random.Generator, fake: Faker
) -> list[dict]:
    """2000 accounts with log-normal ARR and ~10% churn flag."""
    # Log-normal ARR. mean of log = 11 → median ARR ~ 60k; sd 1.2 → fat tail.
    arrs = np.exp(np_rng.normal(11.0, 1.2, N_ACCOUNTS))
    arrs = np.round(arrs, 2)

    # 10% churn — deterministic boolean array.
    churn_mask = np_rng.random(N_ACCOUNTS) < 0.10

    # Created-at distributed across the last 5 years.
    today = date(2026, 5, 11)
    earliest = today - timedelta(days=5 * 365)

    accounts: list[dict] = []
    for i in range(N_ACCOUNTS):
        acc_id = _account_uuid(py_rng)
        # Random offset (deterministic via py_rng).
        days_old = py_rng.randint(0, (today - earliest).days)
        created = earliest + timedelta(days=days_old)
        accounts.append(
            {
                "account_id": acc_id,
                "account_name": fake.company(),
                "industry": py_rng.choice(INDUSTRIES),
                "country": py_rng.choice(COUNTRIES),
                "employees": int(np.clip(np_rng.lognormal(4.5, 1.0), 1, 200_000)),
                "arr": f"{arrs[i]:.2f}",
                "created_at": created.isoformat(),
                "churned": "true" if churn_mask[i] else "false",
            }
        )
    return accounts


def _build_contacts(
    py_rng: random.Random, np_rng: np.random.Generator, fake: Faker, accounts: list[dict]
) -> list[dict]:
    """20000 contacts. Every account has >=1; ~70% have >=3; exactly one
    `is_primary=true` per account.

    Allocation strategy: start by giving every account 1 contact, then
    distribute the remaining 18000 across accounts in a way that pushes
    >=70% of accounts to have >=3 total. Concretely:
      - Step 1: 1 per account → 2000 placed, 18000 remaining.
      - Step 2: give 2 more contacts to ~80% of accounts (so they have 3
        each), consuming 2 * 1600 = 3200 → 14800 remaining, ~80% accounts
        at >=3.
      - Step 3: distribute the remaining 14800 randomly over all accounts.
    Final guarantee: every account has >=1; >=80% have >=3 (>=70% spec).
    """
    counts = [1] * N_ACCOUNTS

    # Push ~80% of accounts to >=3.
    boost_target_frac = 0.80
    n_boost = int(N_ACCOUNTS * boost_target_frac)
    boost_indices = py_rng.sample(range(N_ACCOUNTS), n_boost)
    for idx in boost_indices:
        counts[idx] += 2  # 1 -> 3
    placed = sum(counts)
    remaining = N_CONTACTS - placed
    assert remaining >= 0, "Over-allocated contacts; tighten boost params"

    # Distribute remainder uniformly at random.
    for _ in range(remaining):
        counts[py_rng.randrange(N_ACCOUNTS)] += 1

    assert sum(counts) == N_CONTACTS
    assert min(counts) >= 1
    pct_ge3 = sum(1 for c in counts if c >= 3) / N_ACCOUNTS
    assert pct_ge3 >= 0.70, f"Only {pct_ge3:.2%} of accounts have >=3 contacts"

    contacts: list[dict] = []
    next_id = 1
    for acc_idx, acc in enumerate(accounts):
        n = counts[acc_idx]
        # Pick the primary contact slot (deterministic).
        primary_slot = py_rng.randrange(n)
        for slot in range(n):
            first = fake.first_name()
            last = fake.last_name()
            email_domain = (
                acc["account_name"]
                .lower()
                .replace(" ", "")
                .replace(",", "")
                .replace(".", "")
                .replace("'", "")
            )[:30]
            email = f"{first.lower()}.{last.lower()}@{email_domain}.test"
            contacts.append(
                {
                    "contact_id": f"CT-{next_id:07d}",
                    "account_id": acc["account_id"],
                    "first_name": first,
                    "last_name": last,
                    "email": email,
                    "title": fake.job(),
                    "is_primary": "true" if slot == primary_slot else "false",
                }
            )
            next_id += 1
    return contacts


def _build_opportunities(
    py_rng: random.Random, np_rng: np.random.Generator, fake: Faker, accounts: list[dict]
) -> list[dict]:
    """8000 opportunities.

    - Parent account chosen uniformly (so popular accounts get multiple).
    - `amount = 0.05 * arr * exp(N(0, 0.5))` — yields strong positive
      correlation with parent ARR (verified ≥ 0.3 in the verifier).
    - ~70% closed; of closed, ~28% won. Open opps: NULL closed_at + won.
    """
    arrs = {acc["account_id"]: float(acc["arr"]) for acc in accounts}
    account_ids = [acc["account_id"] for acc in accounts]

    # Pre-draw the noise multiplier so it stays correlated with the ARR-only
    # source of variance.
    noise = np_rng.normal(0.0, 0.5, N_OPPS)

    today = date(2026, 5, 11)
    earliest = today - timedelta(days=3 * 365)

    opps: list[dict] = []
    for i in range(N_OPPS):
        acc_id = py_rng.choice(account_ids)
        arr = arrs[acc_id]
        amount = round(0.05 * arr * float(np.exp(noise[i])), 2)

        days_old = py_rng.randint(0, (today - earliest).days)
        created = earliest + timedelta(days=days_old)

        is_closed = py_rng.random() < 0.70
        if is_closed:
            # Closed N days after creation, before today.
            closed_offset = py_rng.randint(7, max(8, (today - created).days))
            closed = created + timedelta(days=closed_offset)
            if closed > today:
                closed = today
            won = py_rng.random() < 0.28
            stage = "Closed Won" if won else "Closed Lost"
            closed_at = closed.isoformat()
            won_field = "true" if won else "false"
        else:
            # Open — pick a non-terminal stage.
            stage = py_rng.choice(
                ["Prospecting", "Qualification", "Proposal", "Negotiation"]
            )
            closed_at = ""  # NULL in CSV
            won_field = ""  # NULL in CSV

        opps.append(
            {
                "opp_id": f"OP-{i + 1:07d}",
                "account_id": acc_id,
                "stage": stage,
                "amount": f"{amount:.2f}",
                "created_at": created.isoformat(),
                "closed_at": closed_at,
                "won": won_field,
            }
        )
    return opps


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write a deterministic CSV: \\n line endings, QUOTE_MINIMAL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    py_rng, np_rng, fake = _seed_everything()
    accounts = _build_accounts(py_rng, np_rng, fake)
    contacts = _build_contacts(py_rng, np_rng, fake, accounts)
    opportunities = _build_opportunities(py_rng, np_rng, fake, accounts)

    _write_csv(
        CRM_DIR / "accounts.csv",
        accounts,
        [
            "account_id",
            "account_name",
            "industry",
            "country",
            "employees",
            "arr",
            "created_at",
            "churned",
        ],
    )
    _write_csv(
        CRM_DIR / "contacts.csv",
        contacts,
        [
            "contact_id",
            "account_id",
            "first_name",
            "last_name",
            "email",
            "title",
            "is_primary",
        ],
    )
    _write_csv(
        CRM_DIR / "opportunities.csv",
        opportunities,
        [
            "opp_id",
            "account_id",
            "stage",
            "amount",
            "created_at",
            "closed_at",
            "won",
        ],
    )

if __name__ == "__main__":
    main()
