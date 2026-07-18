"""damcp-digest-v1 — order-sensitive table digest (spec v4 §3).

The digest is the resume feature's primary state-equality algorithm. The
byte layout is FROZEN under this algorithm id — any change requires a new
id. Temporal columns are projected to integer epochs in SQL because
DuckDB's Python fetch truncates TIMESTAMP_NS to microseconds via datetime.
Nested temporal values (inside LIST/STRUCT/MAP) encode at fetched
resolution — a documented limitation of v1.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Generator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, cast

DIGEST_ALGORITHM = "damcp-digest-v1"
CHUNK_ROWS = 8192

_SCHEMA_PART = b"\x01"
_VALUE_PART = b"\x02"

_TAG_NULL = 0x00
_TAG_BOOL = 0x10
_TAG_INT = 0x11
_TAG_FLOAT32 = 0x12
_TAG_FLOAT64 = 0x13
_TAG_DECIMAL = 0x14
_TAG_VARCHAR = 0x20
_TAG_BLOB = 0x21
_TAG_TEXTUAL = 0x22  # BIT / TIME / TIME_TZ / INTERVAL / UUID canonical text share
_TAG_DATE = 0x30
_TAG_TS_S = 0x32
_TAG_TS_MS = 0x33
_TAG_TS_US = 0x34
_TAG_TS_NS = 0x35
_TAG_TS_TZ = 0x36
_TAG_ENUM = 0x41
_TAG_LIST = 0x50
_TAG_STRUCT = 0x51
_TAG_MAP = 0x52

_INT_BASES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "BIGNUM",
}
_TEXTUAL_BASES = {
    "BIT",
    "TIME",
    "TIMETZ",
    "TIME_TZ",
    "TIME WITH TIME ZONE",
    "INTERVAL",
    "UUID",
}
_UNDIGESTABLE_BASES = {"UNION", "VARIANT"}


def _u64(n: int) -> bytes:
    return n.to_bytes(8, "little")


def _lp(payload: bytes) -> bytes:
    return _u64(len(payload)) + payload


class _Undigestable(Exception):
    """Internal: table contains a type outside the v1 encoding table."""


def _enc_int(v: Any) -> bytes:
    n = int(v)
    length = max(1, (n.bit_length() + 8) // 8)
    return n.to_bytes(length, "little", signed=True)


def _enc_bool(v: Any) -> bytes:
    return b"\x01" if v else b"\x00"


def _enc_f32(v: Any) -> bytes:
    return struct.pack("<f", float(v))


def _enc_f64(v: Any) -> bytes:
    return struct.pack("<d", float(v))


def _enc_decimal(v: Any) -> bytes:
    d = v if isinstance(v, Decimal) else Decimal(str(v))
    sign, digits, _exponent = d.as_tuple()
    unscaled = int("".join(str(x) for x in digits)) * (-1 if sign else 1)
    return _enc_int(unscaled)


def _enc_text(v: Any) -> bytes:
    return str(v).encode("utf-8")


def _enc_blob(v: Any) -> bytes:
    return bytes(v)


def _enc_nested(v: Any) -> bytes:
    """Recursive encoding for fetched LIST/STRUCT/MAP Python values."""
    out = bytearray()
    if isinstance(v, dict):
        mapping = cast("dict[Any, Any]", v)
        out += bytes([_TAG_STRUCT]) + _u64(len(mapping))
        for key, item in mapping.items():
            out += _lp(str(key).encode("utf-8"))
            out += _tagged_nested(item)
    elif isinstance(v, (list, tuple)):
        seq = tuple(cast("Any", v))
        out += bytes([_TAG_LIST]) + _u64(len(seq))
        for item in seq:
            out += _tagged_nested(item)
    else:
        out += _tagged_nested(v)
    return bytes(out)


def _tagged_nested(v: Any) -> bytes:
    if v is None:
        return bytes([_TAG_NULL]) + _u64(0)
    if isinstance(v, bool):
        return bytes([_TAG_BOOL]) + _lp(_enc_bool(v))
    if isinstance(v, int):
        return bytes([_TAG_INT]) + _lp(_enc_int(v))
    if isinstance(v, float):
        return bytes([_TAG_FLOAT64]) + _lp(_enc_f64(v))
    if isinstance(v, Decimal):
        return bytes([_TAG_DECIMAL]) + _lp(_enc_decimal(v))
    if isinstance(v, (bytes, bytearray)):
        return bytes([_TAG_BLOB]) + _lp(bytes(v))
    if isinstance(v, (dict, list, tuple)):
        return _enc_nested(v)
    # datetimes, UUIDs, everything else: canonical text at fetched resolution.
    return bytes([_TAG_TEXTUAL]) + _lp(_enc_text(v))


class _Column:
    """One column's SQL projection + value encoder."""

    name: str
    dtype: str
    select: str
    tag: int
    enc: Callable[[Any], bytes]

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype
        base = dtype.split("(")[0].strip().upper()
        q = '"' + name.replace('"', '""') + '"'
        self.select = q
        if base in _UNDIGESTABLE_BASES:
            raise _Undigestable(dtype)
        if base in _INT_BASES:
            self.tag, self.enc = _TAG_INT, _enc_int
        elif base == "BOOLEAN":
            self.tag, self.enc = _TAG_BOOL, _enc_bool
        elif base == "FLOAT" or base == "REAL":
            self.tag, self.enc = _TAG_FLOAT32, _enc_f32
        elif base == "DOUBLE":
            self.tag, self.enc = _TAG_FLOAT64, _enc_f64
        elif base == "DECIMAL":
            self.tag, self.enc = _TAG_DECIMAL, _enc_decimal
        elif base in {"VARCHAR", "CHAR", "TEXT", "STRING"}:
            self.tag, self.enc = _TAG_VARCHAR, _enc_text
        elif base == "BLOB":
            self.tag, self.enc = _TAG_BLOB, _enc_blob
        elif base == "DATE":
            self.tag, self.enc = _TAG_DATE, _enc_int
            self.select = f"date_diff('day', DATE '1970-01-01', {q})"
        elif base == "TIMESTAMP_S":
            self.tag, self.enc = _TAG_TS_S, _enc_int
            self.select = f"CAST(epoch({q}) AS BIGINT)"
        elif base == "TIMESTAMP_MS":
            self.tag, self.enc = _TAG_TS_MS, _enc_int
            self.select = f"epoch_ms({q})"
        elif base in {"TIMESTAMP", "DATETIME"}:
            self.tag, self.enc = _TAG_TS_US, _enc_int
            self.select = f"epoch_us({q})"
        elif base == "TIMESTAMP_NS":
            self.tag, self.enc = _TAG_TS_NS, _enc_int
            self.select = f"epoch_ns({q})"
        elif base in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
            self.tag, self.enc = _TAG_TS_TZ, _enc_int
            self.select = f"epoch_us({q})"
        elif base in _TEXTUAL_BASES:
            self.tag, self.enc = _TAG_TEXTUAL, _enc_text
            self.select = f"CAST({q} AS VARCHAR)"
        elif base == "ENUM":
            self.tag, self.enc = _TAG_ENUM, _enc_text
            self.select = f"CAST({q} AS VARCHAR)"
        elif (
            base in {"LIST", "STRUCT", "MAP"}
            or dtype.endswith("[]")
            or base.startswith(("STRUCT", "MAP"))
        ):
            self.tag, self.enc = _TAG_LIST, _enc_nested
        else:
            raise _Undigestable(dtype)


@contextmanager
def single_thread_scan(con: Any) -> Generator[None]:
    """Pin threads=1 for a deterministic scan; ALWAYS restore.

    PRAGMAs are not transactional (survive ROLLBACK), so restoration in
    ``finally`` is the only thing standing between a failed digest/replay
    and a permanently single-threaded live connection.
    """
    row = con.execute("SELECT current_setting('threads')").fetchone()
    old = int(row[0])
    con.execute("SET threads=1")
    try:
        yield
    finally:
        con.execute(f"SET threads={old}")


def digest_table(con: Any, table: str) -> str | None:
    """Order-sensitive damcp-digest-v1 of a live table, or None if undigestable."""
    q = '"' + table.replace('"', '""') + '"'
    describe = con.execute(f"DESCRIBE {q}").fetchall()
    try:
        columns = [_Column(str(r[0]), str(r[1])) for r in describe]
    except _Undigestable:
        return None
    h = hashlib.sha256()
    h.update(_SCHEMA_PART)
    for pos, col in enumerate(columns):
        h.update(_u64(pos))
        h.update(_lp(col.name.encode("utf-8")))
        h.update(_lp(col.dtype.encode("utf-8")))
    h.update(_VALUE_PART)
    select = ", ".join(c.select for c in columns)
    with single_thread_scan(con):
        cur = con.execute(f"SELECT {select} FROM {q}")
        while True:
            rows = cur.fetchmany(CHUNK_ROWS)
            if not rows:
                break
            for row in rows:
                for col, value in zip(columns, row, strict=True):
                    if value is None:
                        h.update(bytes([_TAG_NULL]) + _u64(0))
                    else:
                        h.update(bytes([col.tag]) + _lp(col.enc(value)))
    return h.hexdigest()
