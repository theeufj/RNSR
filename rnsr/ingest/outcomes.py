"""Durable ingestion coverage, an optional backwards-compatible v3 extension.

Outcomes are ingestion metadata, not writable source text. Document and failed
attachment records cascade with their owning document, so replacement removes
obsolete gaps. Older artifacts retain their aggregate health as a conservative
baseline: missing historical per-document coverage cannot be reconstructed.
"""
from __future__ import annotations

import hashlib
import json

COUNTERS = ("parse_failed", "scanned_pages_total", "scanned_pages_untranscribed")
LEGACY_KEY = "ingest_health_legacy"


def has_outcomes(conn) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ingest_outcomes'"
    ).fetchone() is not None


def ensure_outcomes(conn) -> None:
    if has_outcomes(conn):
        return
    # Creating this optional metadata table never changes the artifact version
    # or unfreezes its source tables. Preserve historical aggregate counters.
    stored = conn.execute("SELECT value FROM manifest WHERE key='health'").fetchone()
    legacy = json.loads(stored[0]) if stored else {}
    conn.execute(
        "INSERT INTO manifest(key,value) VALUES (?,?) ON CONFLICT(key) DO NOTHING",
        (LEGACY_KEY, json.dumps({key: int(legacy.get(key) or 0) for key in COUNTERS})))
    conn.execute("""
        CREATE TABLE ingest_outcomes (
            outcome_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
            parent_doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('parsed','parse_failed')),
            scanned_pages_total INTEGER NOT NULL DEFAULT 0 CHECK (scanned_pages_total >= 0),
            scanned_pages_untranscribed INTEGER NOT NULL DEFAULT 0
                CHECK (scanned_pages_untranscribed BETWEEN 0 AND scanned_pages_total),
            error TEXT,
            CHECK ((status='parsed' AND doc_id IS NOT NULL AND error IS NULL)
                OR (status='parse_failed' AND doc_id IS NULL AND error IS NOT NULL))
        )
    """)
    conn.execute("CREATE INDEX ingest_outcomes_source ON ingest_outcomes(source_path)")


def record_failure(conn, failure: dict) -> None:
    ensure_outcomes(conn)
    source = str(failure["source"])
    parent = failure.get("parent_doc_id")
    identity = hashlib.sha256(json.dumps([source, parent]).encode()).hexdigest()
    conn.execute(
        "INSERT INTO ingest_outcomes(outcome_id,source_path,parent_doc_id,status,error) "
        "VALUES (?,?,?,'parse_failed',?) ON CONFLICT(outcome_id) DO UPDATE SET error=excluded.error",
        ("failure:" + identity, source, parent, str(failure["error"])))


def record_document(conn, parsed) -> None:
    ensure_outcomes(conn)
    # A successful retry resolves its earlier failure; identical text in a
    # different source remains a separate document/coverage record.
    conn.execute(
        "DELETE FROM ingest_outcomes WHERE status='parse_failed' AND source_path=? "
        "AND parent_doc_id IS ?", (parsed.source_path, parsed.parent_doc_id))
    gap = parsed.untranscribed_pages
    gap = parsed.scanned_pages if gap is None else gap
    conn.execute(
        "INSERT INTO ingest_outcomes(outcome_id,source_path,doc_id,parent_doc_id,status,"
        "scanned_pages_total,scanned_pages_untranscribed) VALUES (?,?,?,?,'parsed',?,?)",
        ("document:" + parsed.doc_id, parsed.source_path, parsed.doc_id, parsed.parent_doc_id,
         len(set(parsed.scanned_pages)), len(set(gap))))
    for failure in parsed.attachment_failures:
        record_failure(conn, failure)


def counters(conn) -> dict[str, int] | None:
    if not has_outcomes(conn):
        return None
    row = conn.execute(
        "SELECT coalesce(sum(status='parse_failed'),0),"
        "coalesce(sum(scanned_pages_total),0),coalesce(sum(scanned_pages_untranscribed),0) "
        "FROM ingest_outcomes").fetchone()
    legacy = conn.execute("SELECT value FROM manifest WHERE key=?", (LEGACY_KEY,)).fetchone()
    baseline = json.loads(legacy[0]) if legacy else {}
    return {key: int(value) + int(baseline.get(key) or 0)
            for key, value in zip(COUNTERS, row, strict=True)}


def failures(conn) -> list[dict]:
    if not has_outcomes(conn):
        return []
    return [{"source": source, "parent_doc_id": parent, "error": error}
            for source, parent, error in conn.execute(
                "SELECT source_path,parent_doc_id,error FROM ingest_outcomes "
                "WHERE status='parse_failed' ORDER BY source_path,parent_doc_id")]
