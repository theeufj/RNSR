"""Small caller-declared formula contracts, never inferred by the model.

A contract authorizes exact source cells and a named operation graph. It does
not establish that a non-GAAP metric has a universal definition, that a source
span implies a formula, or that the caller chose the right economic convention.
"""
from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal

from rnsr.db.metadata import decode_table_schema
from rnsr.db.schema import quote_ident

FORMULA_ROLES = {
    "direct_ratio": ("numerator", "denominator"),
    "ebitdar_to_ebit_coverage": ("ebitdar", "depreciation_amortization", "rent", "interest"),
}
_KEYS = {"contract_id", "metric", "formula", "basis", "unit", "period", "sources"}
_OPTIONAL_KEYS = {"nonpositive_numerator"}
_SOURCE_KEYS = {"table", "rowid", "column", "label_column", "label", "unit_span", "period_span"}


def _text(value, name, limit=256):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"metric contract {name} must be nonempty bounded text")


class MetricContract:
    """Immutable-by-copy parent policy. A null formula authorizes abstention only."""

    def __init__(self, specification: dict):
        if (not isinstance(specification, dict) or not set(specification) >= _KEYS
                or set(specification) - _KEYS - _OPTIONAL_KEYS):
            raise ValueError("invalid metric contract fields")
        spec = copy.deepcopy(specification)
        for name in ("contract_id", "metric", "basis"):
            _text(spec[name], name, 2000 if name == "basis" else 256)
        formula, sources = spec["formula"], spec["sources"]
        if formula is not None and (not isinstance(formula, str) or formula not in FORMULA_ROLES):
            raise ValueError("unknown metric contract formula")
        spec.setdefault('nonpositive_numerator', 'preserve')
        if spec['nonpositive_numerator'] not in ('preserve', 'zero'):
            raise ValueError("metric contract nonpositive_numerator must be preserve or zero")
        if spec['nonpositive_numerator'] == 'zero' and formula != 'ebitdar_to_ebit_coverage':
            raise ValueError("zero-coverage convention requires the declared coverage bridge")
        if not isinstance(sources, dict) or set(sources) != set(FORMULA_ROLES.get(formula, ())):
            raise ValueError("metric contract requires exactly the formula's source roles")
        for name in ("unit", "period"):
            if formula is not None or spec[name] is not None:
                _text(spec[name], name)
        for source in sources.values():
            if not isinstance(source, dict) or set(source) != _SOURCE_KEYS:
                raise ValueError("invalid metric contract source selector")
            for name in ("table", "column", "label_column", "label"):
                _text(source[name], name, 1000 if name == "label" else 256)
            if (isinstance(source["rowid"], bool) or not isinstance(source["rowid"], int)
                    or source["rowid"] < 1):
                raise ValueError("metric contract source rowid must be positive")
            for name in ("unit_span", "period_span"):
                span = source[name]
                if (not isinstance(span, dict) or set(span) != {"char_start", "char_end"}
                        or any(isinstance(v, bool) or not isinstance(v, int) for v in span.values())
                        or not 0 <= span["char_start"] < span["char_end"]
                        or span["char_end"] - span["char_start"] > 256):
                    raise ValueError("metric contract metadata requires a bounded source span")
        self._spec = spec
        self.digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def describe(self) -> dict:
        return {**copy.deepcopy(self._spec), "sha256": self.digest,
                "authority": "caller_declared",
                "scope": "Exact source identities and declared formula; economic meaning not inferred"}

    @property
    def nonpositive_numerator(self) -> str:
        return self._spec['nonpositive_numerator']

    def _source(self, registry, selector) -> dict:
        table = selector["table"]
        meta = registry.conn.execute(
            "SELECT schema_json FROM manifest_tables WHERE table_name=?", (table,)).fetchone()
        if meta is None:
            raise ValueError("metric contract source table does not exist")
        schema = decode_table_schema(meta[0])
        label_column = selector["label_column"]
        col = next((c for c in schema.columns if c.name == label_column), None)
        if col is None or col.annotation or col.type != "TEXT":
            raise ValueError("metric contract label must be an original text column")
        row = registry.conn.execute(
            f"SELECT {quote_ident(label_column)} FROM {quote_ident(table)} WHERE rowid=?",
            (selector["rowid"],)).fetchone()
        if row is None or row[0] != selector["label"]:
            raise ValueError("metric contract source label differs from caller policy")
        fact = registry.source_number(table, selector["rowid"], selector["column"],
                                      unit_span=selector["unit_span"],
                                      period_span=selector["period_span"])
        if fact["unit"] != self._spec["unit"] or fact["period"] != self._spec["period"]:
            raise ValueError("metric contract unit or period evidence differs from caller policy")
        return fact

    def compute(self, registry) -> tuple[dict, dict[str, str], str]:
        formula = self._spec["formula"]
        if formula is None:
            raise ValueError("requested metric has no caller-approved formula; return NOT_FOUND")
        sources = {role: self._source(registry, selector)
                   for role, selector in self._spec["sources"].items()}
        if formula == "direct_ratio":
            numerator, denominator = sources['numerator'], sources['denominator']
        else:
            net = registry.calculate("subtract", [sources["ebitdar"]["id"],
                                                   sources["depreciation_amortization"]["id"]])
            numerator = registry.calculate("subtract", [net["id"], sources["rent"]["id"]])
            denominator = sources['interest']
        if self.nonpositive_numerator == 'zero' and Decimal(denominator['value']) <= 0:
            raise ValueError("zero-coverage convention requires a positive denominator")
        result = registry.calculate('divide', [numerator['id'], denominator['id']])
        return result, {role: source["id"] for role, source in sources.items()}, numerator['id']

    def validate_sources(self, registry) -> None:
        """Recheck selected labels/metadata too, including changes within a generation."""
        for selector in self._spec["sources"].values():
            # Avoid issuing fresh IDs while validating an existing final result.
            table, label_column = selector["table"], selector["label_column"]
            row = registry.conn.execute(
                f"SELECT {quote_ident(label_column)} FROM {quote_ident(table)} WHERE rowid=?",
                (selector["rowid"],)).fetchone()
            if row is None or row[0] != selector["label"]:
                raise ValueError("metric contract source label changed")
