"""Atomic append/replace of frozen artifacts, including transactional DDL.

BEGIN IMMEDIATE precedes unfreezing. SQLite rolls the trigger changes back
along with all data on any exception, process termination, or power loss.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from rnsr.config import Settings
from rnsr.db import fts, schema
from rnsr.db.artifact import CorpusDB
from rnsr.ingest.dispatch import parse_any
from rnsr.ingest.expand import expand_document
from rnsr.ingest.manifest import write_corpus_manifest
from rnsr.ingest.parse import PARSER_NAME
from rnsr.ingest.transcription import merge_transcriptions
from rnsr.ingest.writer import write_document


def _write_docs(conn, parsed_ok, config, *, prose_checker=None) -> int:
    for src, parsed in parsed_ok:
        write_document(conn, src, parsed, config, prose_checker=prose_checker)
    return len(parsed_ok)


def _parse_new(sources, parse, seen_ids, existing_shas, existing_content,
               transcriber=None, *, strict=False):
    parsed_ok = []
    failures = []
    scanned_total = scanned_gap = 0
    for src in sources:
        src = Path(src)
        try:
            parsed = parse(src)
            if (parsed.sha256 in existing_shas or
                    (parsed.content_sha256 or parsed.sha256) in existing_content):
                continue
            children = expand_document(parsed, parse, seen_ids, strict=strict)
        except Exception as exc:
            if strict:
                raise
            failures.append({"source": str(src), "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        for child in children:
            parsed_ok.append((src, child))
            scanned_total += len(child.scanned_pages)
            if child.scanned_pages:
                if transcriber is None:
                    scanned_gap += len(child.scanned_pages)
                else:
                    scanned_gap += len(merge_transcriptions(
                        child, transcriber(src, child.scanned_pages)))
    return parsed_ok, {"parse_failed": len(failures),
                       "scanned_pages_total": scanned_total,
                       "scanned_pages_untranscribed": scanned_gap}, failures


def _finish(corpus, config, kind, sources, count, extra_health):
    fts.populate_fts(corpus.conn)
    schema.record_ingest_batch(corpus.conn, kind, sources, n_docs=count)
    stored = corpus.manifest_get("health", {})
    extras = {key: int(stored.get(key) or 0) + value
              for key, value in extra_health.items()}
    write_corpus_manifest(corpus, PARSER_NAME, config=config, extra_health=extras)
    schema.validate_integrity(corpus.conn)
    schema.finalize_corpus(corpus.conn)
    corpus.conn.commit()


def append(sources, corpus_db, *, config: Settings | None = None,
           parse=parse_any, transcriber=None, kind: str = "append") -> dict:
    """Append whole documents and freeze changes in a single durable commit."""
    from rnsr.ingest.pipeline import ingest as run_ingest

    sources = [Path(s) for s in (sources if isinstance(sources, (list, tuple)) else [sources])]
    corpus_db = Path(corpus_db)
    config = config or Settings()
    if not corpus_db.exists():
        report = run_ingest(sources, corpus_db, config=config, parse=parse,
                            transcriber=transcriber)
        return {"kind": "create", "new_docs": len(report.documents),
                "out_db": str(corpus_db), "parse_failed": report.parse_failed}
    with CorpusDB(corpus_db, mode="rw") as corpus:
        conn = corpus.conn
        schema.validate_frozen(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing_shas = {r[0] for r in conn.execute("SELECT sha256 FROM documents") if r[0]}
            existing_content = {r[0] for r in conn.execute(
                "SELECT content_sha256 FROM documents") if r[0]}
            seen_ids = {r[0] for r in conn.execute("SELECT doc_id FROM documents")}
            parsed, extras, failures = _parse_new(
                sources, parse, seen_ids, existing_shas, existing_content, transcriber)
            if failures and not parsed:
                raise RuntimeError(f"all new documents failed to parse: {failures}")
            schema.unfreeze_corpus(conn)
            n_new = _write_docs(conn, parsed, config)
            _finish(corpus, config, kind, sources, n_new, extras)
        except BaseException:
            conn.rollback()
            raise
    return {"kind": kind, "new_docs": n_new, "out_db": str(corpus_db),
            "parse_failed": failures}


def _descendants(conn, doc_id):
    # UNION (not UNION ALL) terminates even on corrupt legacy parent cycles.
    return [r[0] for r in conn.execute(
        "WITH RECURSIVE family(id) AS (SELECT doc_id FROM documents WHERE doc_id=? "
        "UNION SELECT d.doc_id FROM documents d JOIN family f ON d.parent_doc_id=f.id) "
        "SELECT id FROM family", (doc_id,))]


def delete_document(conn, doc_id: str) -> None:
    """Drop dynamic tables; FK cascades own deletion of relational children."""
    for child in _descendants(conn, doc_id):
        for (table,) in conn.execute(
                "SELECT table_name FROM manifest_tables WHERE doc_id=?", (child,)).fetchall():
            schema.unfreeze_table(conn, table)
            conn.execute(f"DROP TABLE {schema.quote_ident(table)}")
            conn.execute("DELETE FROM annotation_log WHERE table_name=?", (table,))
    conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))


def replace_document(corpus_db, doc_id: str, source, *,
                     config: Settings | None = None,
                     parse=parse_any, transcriber=None) -> dict:
    """Replace only after parsing succeeds; failures preserve the entire old document."""
    corpus_db, source = Path(corpus_db), Path(source)
    config = config or Settings()
    with CorpusDB(corpus_db, mode="rw") as corpus:
        conn = corpus.conn
        schema.validate_frozen(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            removed = set(_descendants(conn, doc_id))
            if not removed:
                raise KeyError(f"unknown doc_id: {doc_id}")
            seen = {r[0] for r in conn.execute("SELECT doc_id FROM documents")} - removed
            parsed, extras, _ = _parse_new(
                [source], parse, seen, set(), set(), transcriber, strict=True)
            if not parsed:
                raise RuntimeError("replacement did not produce a document")
            schema.unfreeze_corpus(conn)
            delete_document(conn, doc_id)
            count = _write_docs(conn, parsed, config)
            _finish(corpus, config, "replace", [source], count, extras)
        except BaseException:
            conn.rollback()
            raise
    return {"kind": "replace", "new_docs": count, "out_db": str(corpus_db)}


def file_index(corpus_db: str | Path) -> dict[str, dict]:
    """Exact source paths plus only unambiguous basename aliases."""
    with CorpusDB(corpus_db) as corpus:
        rows = [dict(row) for row in corpus.conn.execute(
            "SELECT doc_id,source_path,sha256,content_sha256,source_identity FROM documents "
            "ORDER BY doc_id") if "::" not in row["source_path"]]
    counts = Counter(Path(row["source_path"]).name for row in rows)
    out = {}
    for row in rows:
        path = Path(row["source_path"])
        out[str(path.resolve())] = row
        if counts[path.name] == 1:
            out[path.name] = row
    return out
