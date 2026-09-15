"""Append and replace documents on a finalized corpus.db.

Freeze triggers are dropped inside one transaction, new rows are written,
FTS/manifest/health are rebuilt, then the substrate is refrozen. The
resulting artifact is the same shape as a fresh full ingest.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rnsr.config import Settings
from rnsr.db import fts, schema
from rnsr.db.artifact import CorpusDB
from rnsr.ingest.chunk import chunk_document
from rnsr.ingest.dispatch import parse_any
from rnsr.ingest.expand import expand_document
from rnsr.ingest.manifest import write_corpus_manifest, write_table_manifest
from rnsr.ingest.parse import PARSER_NAME
from rnsr.ingest.pipeline import _merge_transcriptions
from rnsr.ingest.tables import build_data_table, merge_multipage
from rnsr.ingest.validate import assign_table_status, validate_table


def _write_docs(conn, parsed_ok, config, *, prose_checker=None) -> int:
    n = 0
    for src, parsed in parsed_ok:
        pages, chunks = chunk_document(
            parsed, chunk_chars=config.chunk_chars, overlap=config.chunk_overlap)
        page_texts = {p.page: p.text for p in pages}
        modified = parsed.modified_at
        if not modified:
            try:
                modified = datetime.fromtimestamp(
                    Path(src).stat().st_mtime, UTC).isoformat()
            except OSError:
                modified = None
        schema.insert_document(
            conn,
            doc_id=parsed.doc_id,
            source_path=parsed.source_path,
            sha256=parsed.sha256,
            n_pages=parsed.n_pages,
            parser=parsed.parser,
            ingested_at=datetime.now(UTC).isoformat(),
            title=parsed.title,
            doc_date=parsed.doc_date,
            author=parsed.author,
            modified_at=modified,
            content_sha256=parsed.content_sha256 or parsed.sha256,
            parent_doc_id=parsed.parent_doc_id,
        )
        conn.executemany(
            "INSERT INTO doc_text VALUES (?,?,?,?,?)",
            [(parsed.doc_id, p.page, p.char_start, p.char_end, p.text)
             for p in pages],
        )
        conn.executemany(
            "INSERT INTO chunks (doc_id, page, char_start, char_end, "
            "heading_path, text) VALUES (?,?,?,?,?,?)",
            [(parsed.doc_id, c.page, c.char_start, c.char_end,
              c.heading_path, c.text) for c in chunks],
        )
        for seq, raw in enumerate(merge_multipage(parsed.tables), start=1):
            validation = validate_table(
                raw, coerce_threshold=config.coerce_threshold,
                rel_tol=config.arithmetic_rel_tol,
                abs_tol=config.arithmetic_abs_tol,
                page_texts=page_texts,
                prose_checker=prose_checker)
            status = assign_table_status(
                validation, config.table_confidence_threshold)
            built = build_data_table(
                conn, parsed.doc_id, seq, raw,
                coerce_threshold=config.coerce_threshold,
                style_overrides=validation.style_overrides,
                cells=config.cells_index)
            write_table_manifest(conn, built, validation, status)
        n += 1
    return n


def _parse_new(sources, parse, seen_ids, existing_shas, existing_content,
               transcriber=None):
    parsed_ok = []
    for src in sources:
        src = Path(src)
        try:
            parsed = parse(src)
        except Exception:
            continue
        for child in expand_document(parsed, parse, seen_ids):
            sha = child.sha256
            csha = child.content_sha256 or sha
            if sha in existing_shas or csha in existing_content:
                continue
            parsed_ok.append((src, child))
    if transcriber is not None:
        for src, parsed in parsed_ok:
            if parsed.scanned_pages:
                merged = transcriber(src, parsed.scanned_pages)
                _merge_transcriptions(parsed, merged)
    return parsed_ok


def append(sources, corpus_db, *, config: Settings | None = None,
           parse=parse_any, transcriber=None, kind: str = "append") -> dict:
    """Add new documents to an existing artifact. Creates it if missing."""
    from rnsr.ingest.pipeline import ingest as run_ingest

    sources = [Path(s) for s in (sources if isinstance(sources, list) else [sources])]
    corpus_db = Path(corpus_db)
    config = config or Settings()
    if not corpus_db.exists():
        report = run_ingest(sources, corpus_db, config=config, parse=parse,
                            transcriber=transcriber)
        return {"kind": "create", "new_docs": len(report.documents),
                "out_db": str(corpus_db)}

    corpus = CorpusDB(corpus_db, mode="rw")
    conn = corpus.conn
    try:
        schema.unfreeze_corpus(conn)
        existing_shas = {r[0] for r in conn.execute("SELECT sha256 FROM documents")}
        content_cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
        existing_content: set[str] = set()
        if "content_sha256" in content_cols:
            existing_content = {
                r[0] for r in conn.execute(
                    "SELECT content_sha256 FROM documents")
                if r[0]
            }
        seen_ids = {r[0] for r in conn.execute("SELECT doc_id FROM documents")}
        parsed_ok = _parse_new(
            sources, parse, seen_ids, existing_shas, existing_content,
            transcriber=transcriber)
        n_new = _write_docs(conn, parsed_ok, config)
        fts.populate_fts(conn)
        schema.record_ingest_batch(conn, kind, sources, n_docs=n_new)
        write_corpus_manifest(corpus, PARSER_NAME, config=config)
        schema.finalize_corpus(conn)
        conn.commit()
    finally:
        corpus.close()
    return {"kind": kind, "new_docs": n_new, "out_db": str(corpus_db)}


def delete_document(conn, doc_id: str) -> None:
    """Remove one document and its derived rows/tables (triggers must be down)."""
    children = [
        r[0] for r in conn.execute(
            "SELECT doc_id FROM documents WHERE parent_doc_id = ?", (doc_id,))
    ]
    for child in children:
        delete_document(conn, child)
    tables = [
        r[0] for r in conn.execute(
            "SELECT table_name FROM manifest_tables WHERE doc_id = ?", (doc_id,))
    ]
    for table in tables:
        schema.unfreeze_table(conn, table)
        conn.execute(f"DROP TABLE IF EXISTS {schema.quote_ident(table)}")
    conn.execute("DELETE FROM manifest_tables WHERE doc_id = ?", (doc_id,))
    cells = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cells'")}
    if cells:
        conn.execute("DELETE FROM cells WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM doc_text WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def replace_document(corpus_db, doc_id: str, source, *,
                     config: Settings | None = None,
                     parse=parse_any, transcriber=None) -> dict:
    """Delete ``doc_id`` (and children) then append ``source``."""
    corpus_db = Path(corpus_db)
    config = config or Settings()
    source = Path(source)
    corpus = CorpusDB(corpus_db, mode="rw")
    try:
        schema.unfreeze_corpus(corpus.conn)
        delete_document(corpus.conn, doc_id)
        corpus.conn.commit()
    finally:
        corpus.close()
    return append([source], corpus_db, config=config, parse=parse,
                  transcriber=transcriber, kind="replace")


def file_index(corpus_db: str | Path) -> dict[str, dict]:
    """Map resolved source_path -> {doc_id, sha256, content_sha256}."""
    out: dict[str, dict] = {}
    with CorpusDB(corpus_db) as c:
        cols = {r[1] for r in c.conn.execute("PRAGMA table_info(documents)")}
        extra = ", content_sha256" if "content_sha256" in cols else ""
        for row in c.conn.execute(
                f"SELECT doc_id, source_path, sha256{extra} FROM documents"):
            rec = dict(row)
            path = rec.get("source_path") or ""
            if "::" in path:
                continue
            try:
                key = str(Path(path).resolve())
            except OSError:
                key = path
            out[key] = rec
            out[Path(path).name] = rec
    return out
