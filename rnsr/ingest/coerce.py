"""Conservative numeric coercion for extracted table columns (spec §3.2).

Per column: attempt numeric coercion; if >= coerce_threshold (default 95%)
of non-null cells coerce, the column is stored as REAL/INTEGER with the raw
strings preserved in a ``{col}__raw`` shadow column; otherwise TEXT.
Currency symbols, thousands separators, parenthesized negatives, and
percentage signs are normalized, and the applied rule is recorded in table
metadata so the checksum pass (§3.3) can roll back any coercion that
corrupts meaning (§9 — e.g. "1,234" as a European decimal).

Everything here is pure and deterministic; no I/O, no LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Cell contents that mean "no value" in financial tables. A bare dash is
# conventionally an empty cell, not zero.
NULL_TOKENS = frozenset({"", "-", "–", "—", "n/a", "na", "n.a.", "nm", "n.m.", "none", "nil", "*"})

_CURRENCY = "$€£¥₹"
_MINUS_CHARS = "−–—"  # unicode minus/dashes used as negative signs

# Unambiguous separator-style evidence.
_US_PATTERN = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")           # 1,234.56
_EU_PATTERN = re.compile(r"^(\d{1,3}(\.\d{3})+(,\d+)?|\d+,\d{1,2})$")  # 1.234,56 / 12,5
_EU_STRONG = re.compile(r"^\d{1,3}(\.\d{3})+(,\d+)?$")            # dot-grouped: EU for sure


@dataclass(frozen=True)
class CoercionRule:
    """Normalization applied to a coerced column; recorded in schema_json (§3.2)."""

    style: str                       # 'us' | 'eu' decimal/grouping convention
    features: tuple[str, ...]        # subset of: currency, percent, parens_negative

    def to_dict(self) -> dict:
        return {"style": self.style, "features": list(self.features)}


@dataclass
class CoercedColumn:
    sql_type: str                    # 'INTEGER' | 'REAL' | 'TEXT'
    values: list[float | int | str | None]
    rule: CoercionRule | None        # None for TEXT columns
    coverage: float                  # fraction of non-null cells that coerced
    features_seen: set[str] = field(default_factory=set)

    @property
    def is_numeric(self) -> bool:
        return self.sql_type in ("INTEGER", "REAL")


def is_null_cell(raw: str | None) -> bool:
    return raw is None or raw.strip().lower() in NULL_TOKENS


def _clean(raw: str) -> tuple[str, set[str], bool]:
    """Strip decoration, returning (bare numeric text, features, negative)."""
    s = raw.strip()
    features: set[str] = set()
    negative = False

    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
        features.add("parens_negative")
        negative = True
    if s.endswith("%"):
        s = s[:-1].strip()
        features.add("percent")
    for ch in _MINUS_CHARS:
        if s.startswith(ch):
            s = "-" + s[1:]
            break
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()
    stripped = s.strip(_CURRENCY + " ")
    if stripped != s:
        features.add("currency")
        s = stripped.strip()
        if s.startswith("-"):  # currency before sign, e.g. "$-5"
            negative = True
            s = s[1:].strip()
    return s, features, negative


def detect_style(raw_values: list[str | None]) -> str:
    """Vote on decimal convention from unambiguous cells; default 'us' on tie.

    The default is deliberately conservative — the checksum pass catches a
    wrong call and triggers per-column rollback (§9).
    """
    us = eu = 0
    for raw in raw_values:
        if is_null_cell(raw):
            continue
        s, _, _ = _clean(raw)  # type: ignore[arg-type]
        if _EU_STRONG.match(s) or (re.match(r"^\d+,\d{1,2}$", s) and not _US_PATTERN.match(s)):
            eu += 1
        elif _US_PATTERN.match(s) or re.match(r"^\d+\.\d+$", s):
            us += 1
    return "eu" if eu > us else "us"


def coerce_cell(raw: str, style: str = "us") -> float | None:
    """Coerce one cell to a float, or None if it does not parse as a number."""
    s, _, negative = _clean(raw)
    s = s.replace(".", "").replace(",", ".") if style == "eu" else s.replace(",", "")
    if not s or not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    value = float(s)
    return -value if negative else value


def coerce_column(
    raw_values: list[str | None],
    threshold: float = 0.95,
    style: str | None = None,
) -> CoercedColumn:
    """Apply the >=threshold rule to a whole column (§3.2).

    Returns TEXT (values passed through raw) when too few cells coerce or
    when the column has no non-null cells at all — an all-null column gives
    no evidence it is numeric, and TEXT is the conservative type.
    """
    style = style or detect_style(raw_values)
    non_null = [v for v in raw_values if not is_null_cell(v)]
    if not non_null:
        return CoercedColumn("TEXT", list(raw_values), None, 0.0)

    features: set[str] = set()
    coerced: dict[int, float] = {}
    for i, raw in enumerate(raw_values):
        if is_null_cell(raw):
            continue
        _, feats, _ = _clean(raw)  # type: ignore[arg-type]
        value = coerce_cell(raw, style)  # type: ignore[arg-type]
        if value is not None:
            coerced[i] = value
            features |= feats

    coverage = len(coerced) / len(non_null)
    if coverage < threshold:
        return CoercedColumn("TEXT", list(raw_values), None, coverage)

    sql_type = "INTEGER" if all(v == int(v) for v in coerced.values()) else "REAL"

    values: list[float | int | str | None] = []
    for i, _raw in enumerate(raw_values):
        if i in coerced:
            values.append(int(coerced[i]) if sql_type == "INTEGER" else coerced[i])
        else:
            values.append(None)  # null cell or (rare) uncoercible within tolerance

    rule = CoercionRule(style=style, features=tuple(sorted(features)))
    return CoercedColumn(sql_type, values, rule, coverage, features)
