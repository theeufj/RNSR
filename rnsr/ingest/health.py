"""Corpus health: persist ingest quality and gate answering on it.

Ingest used to print counters and discard them. A corpus with untranscribed
scans or a low table-validation rate then answered as confidently as a
clean one. Health is written into ``manifest.health`` at ingest time and
re-checked at every open-before-answer choke point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rnsr.config import Settings
from rnsr.errors import CorpusHealthError

Grade = str  # 'ok' | 'degraded' | 'blocked'


@dataclass
class Finding:
    code: str
    severity: str  # 'info' | 'warn' | 'error'
    detail: str


@dataclass
class CorpusHealth:
    n_documents: int = 0
    parse_failed: int = 0
    scanned_pages_total: int = 0
    scanned_pages_untranscribed: int = 0
    tables_total: int = 0
    tables_untrusted: int = 0
    tables_unchecked: int = 0
    validation_pass_rate: float = 1.0
    aggregate_rows_flagged: int = 0
    findings: list[Finding] = field(default_factory=list)
    grade: Grade = "ok"
    source: str = "ingest"  # 'ingest' | 'derived'

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "findings"},
            "findings": [asdict(f) for f in self.findings],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CorpusHealth:
        raw = dict(d)
        findings = [Finding(**f) if not isinstance(f, Finding) else f
                    for f in raw.pop("findings", [])]
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(findings=findings, **{k: v for k, v in raw.items() if k in known})

    def as_counters(self) -> dict[str, int | float]:
        return {
            "n_documents": self.n_documents,
            "parse_failed": self.parse_failed,
            "scanned_pages_total": self.scanned_pages_total,
            "scanned_pages_untranscribed": self.scanned_pages_untranscribed,
            "tables_total": self.tables_total,
            "tables_untrusted": self.tables_untrusted,
            "tables_unchecked": self.tables_unchecked,
            "aggregate_rows_flagged": self.aggregate_rows_flagged,
        }


def validation_pass_rate(tables_total: int, tables_untrusted: int,
                         tables_unchecked: int) -> float:
    """Trusted+reextracted over checked tables. Unchecked are excluded."""
    checked = tables_total - tables_unchecked
    if checked <= 0:
        return 1.0
    passed = checked - tables_untrusted
    return max(0.0, passed / checked)


def evaluate(counters: dict[str, int | float],
             settings: Settings | None = None,
             *,
             source: str = "ingest") -> CorpusHealth:
    """Grade ingest/derived counters against Settings thresholds."""
    settings = settings or Settings()
    n_docs = int(counters.get("n_documents") or 0)
    parse_failed = int(counters.get("parse_failed") or 0)
    scanned_total = int(counters.get("scanned_pages_total") or 0)
    scanned_gap = int(counters.get("scanned_pages_untranscribed") or 0)
    tables_total = int(counters.get("tables_total") or 0)
    tables_untrusted = int(counters.get("tables_untrusted") or 0)
    tables_unchecked = int(counters.get("tables_unchecked") or 0)
    flagged = int(counters.get("aggregate_rows_flagged") or 0)
    rate = validation_pass_rate(tables_total, tables_untrusted, tables_unchecked)

    findings: list[Finding] = []
    attempted = n_docs + parse_failed
    parse_rate = (parse_failed / attempted) if attempted else 0.0

    if scanned_gap > settings.health_max_untranscribed_pages:
        findings.append(Finding(
            "untranscribed_scans", "error",
            f"{scanned_gap} scanned page(s) have no text "
            f"(max {settings.health_max_untranscribed_pages})"))
    elif scanned_gap:
        findings.append(Finding(
            "untranscribed_scans", "warn",
            f"{scanned_gap} scanned page(s) have no text"))

    if parse_rate > settings.health_max_parse_failed_rate:
        findings.append(Finding(
            "parse_failed", "error",
            f"{parse_failed}/{attempted} documents failed to parse "
            f"({parse_rate:.1%} > {settings.health_max_parse_failed_rate:.1%})"))
    elif parse_failed:
        findings.append(Finding(
            "parse_failed", "warn",
            f"{parse_failed} document(s) failed to parse"))

    checked = tables_total - tables_unchecked
    if checked and rate < settings.health_min_validation_rate:
        findings.append(Finding(
            "validation_rate", "error",
            f"table validation pass rate {rate:.1%} is below "
            f"{settings.health_min_validation_rate:.1%} "
            f"({tables_untrusted} untrusted of {checked} checked)"))
    elif tables_untrusted:
        findings.append(Finding(
            "untrusted_tables", "warn",
            f"{tables_untrusted} table(s) flagged untrusted"))

    if tables_unchecked:
        findings.append(Finding(
            "unchecked_tables", "warn",
            f"{tables_unchecked} table(s) had no arithmetic/prose evidence "
            "and were not counted in the pass rate"))

    if flagged:
        findings.append(Finding(
            "aggregate_rows", "info",
            f"{flagged} aggregate row(s) flagged (_row_kind total/subtotal)"))

    if any(f.severity == "error" for f in findings):
        grade: Grade = "blocked"
    elif any(f.severity == "warn" for f in findings):
        grade = "degraded"
    else:
        grade = "ok"

    return CorpusHealth(
        n_documents=n_docs,
        parse_failed=parse_failed,
        scanned_pages_total=scanned_total,
        scanned_pages_untranscribed=scanned_gap,
        tables_total=tables_total,
        tables_untrusted=tables_untrusted,
        tables_unchecked=tables_unchecked,
        validation_pass_rate=round(rate, 4),
        aggregate_rows_flagged=flagged,
        findings=findings,
        grade=grade,
        source=source,
    )


def counters_from_corpus(corpus, extra: dict[str, int] | None = None) -> dict[str, int]:
    """Count tables/docs from an open CorpusDB; merge ingest-only extras."""
    conn = corpus.conn
    n_docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    rows = list(conn.execute("SELECT status FROM manifest_tables"))
    tables_total = len(rows)
    tables_untrusted = sum(1 for (s,) in rows if s == "untrusted")
    tables_unchecked = sum(1 for (s,) in rows if s == "unchecked")
    flagged = int((extra or {}).get("aggregate_rows_flagged") or 0)
    if not flagged:
        try:
            import json
            for (raw,) in conn.execute("SELECT schema_json FROM manifest_tables"):
                schema = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(schema, dict):
                    flagged += int(schema.get("n_total_rows") or 0)
                elif isinstance(schema, list) and schema and isinstance(schema[0], dict):
                    flagged += int(schema[0].get("n_total_rows") or 0)
        except Exception:
            pass
    extra = extra or {}
    return {
        "n_documents": n_docs,
        "parse_failed": int(extra.get("parse_failed") or 0),
        "scanned_pages_total": int(extra.get("scanned_pages_total") or 0),
        "scanned_pages_untranscribed": int(extra.get("scanned_pages_untranscribed") or 0),
        "tables_total": tables_total,
        "tables_untrusted": tables_untrusted,
        "tables_unchecked": tables_unchecked,
        "aggregate_rows_flagged": int(extra.get("aggregate_rows_flagged") or flagged),
    }


def health_from_corpus(corpus, settings: Settings | None = None,
                       extra: dict[str, int] | None = None,
                       *, source: str = "ingest") -> CorpusHealth:
    return evaluate(counters_from_corpus(corpus, extra), settings, source=source)


def load_health(corpus, settings: Settings | None = None) -> CorpusHealth:
    """Stored health, re-graded against current settings; else derived."""
    settings = settings or Settings()
    stored = corpus.manifest_get("health")
    if stored:
        existing = CorpusHealth.from_dict(stored)
        return evaluate(existing.as_counters(), settings, source=existing.source)
    return health_from_corpus(corpus, settings, source="derived")


def enforce_health(health: CorpusHealth, settings: Settings | None = None) -> CorpusHealth:
    """Raise CorpusHealthError when the corpus is blocked and not allowed."""
    settings = settings or Settings()
    if health.grade == "blocked" and not settings.allow_degraded:
        raise CorpusHealthError(health)
    return health


def persist_health(corpus, settings: Settings | None = None,
                   extra: dict[str, int] | None = None) -> CorpusHealth:
    health = health_from_corpus(corpus, settings, extra, source="ingest")
    corpus.manifest_set("health", health.to_dict())
    return health
