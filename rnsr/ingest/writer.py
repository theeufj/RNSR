"""The single document-to-artifact writer shared by every ingestion path."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rnsr.db import schema
from rnsr.ingest.chunk import chunk_document
from rnsr.ingest.manifest import write_table_manifest
from rnsr.ingest.tables import build_data_table, merge_multipage
from rnsr.ingest.validate import assign_table_status, validate_table


@dataclass
class TableReport:
    name: str
    doc_id: str
    status: Literal["trusted", "reextracted", "untrusted", "unchecked"]
    confidence: float
    extractor: str               # rung that produced the stored table
    attempts: list[dict]         # every (extractor, confidence) tried
    style_overrides: dict[str, str]
    n_rows: int
    n_cols: int



@dataclass
class WrittenDocument:
    document: dict
    tables: list[TableReport]


def write_document(conn, source, parsed, config, *, prose_checker=None,
                   select_table=None) -> WrittenDocument:
    """Write one complete document inside the caller's transaction.

    A savepoint prevents a failed table write leaving half a document in
    resumable bulk builds. The caller owns commits and corpus finalization.
    """
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    conn.execute("SAVEPOINT write_document")
    try:
        pages, chunks = chunk_document(
            parsed, chunk_chars=config.chunk_chars, overlap=config.chunk_overlap)
        page_texts = {p.page: p.text for p in pages}
        modified = parsed.modified_at
        if not modified:
            with contextlib.suppress(OSError):
                modified = datetime.fromtimestamp(Path(source).stat().st_mtime, UTC).isoformat()
        from rnsr.ingest.fast_parse import stat_identity

        identity = parsed.source_identity
        if identity is None and "::" not in parsed.source_path:
            with contextlib.suppress(OSError):
                identity = stat_identity(Path(source))
        schema.insert_document(
            conn, doc_id=parsed.doc_id, source_path=parsed.source_path,
            sha256=parsed.sha256, n_pages=parsed.n_pages, parser=parsed.parser,
            ingested_at=datetime.now(UTC).isoformat(), title=parsed.title,
            doc_date=parsed.doc_date, author=parsed.author, modified_at=modified,
            content_sha256=parsed.content_sha256 or parsed.sha256,
            parent_doc_id=parsed.parent_doc_id, source_identity=identity)
        conn.executemany("INSERT INTO doc_text VALUES (?,?,?,?,?)", [
            (parsed.doc_id, p.page, p.char_start, p.char_end, p.text) for p in pages])
        conn.executemany(
            "INSERT INTO chunks (doc_id,page,char_start,char_end,heading_path,text) "
            "VALUES (?,?,?,?,?,?)", [(parsed.doc_id, c.page, c.char_start, c.char_end,
                                     c.heading_path, c.text) for c in chunks])
        reports = []
        for seq, raw in enumerate(merge_multipage(parsed.tables), 1):
            if select_table is not None:
                chosen, validation, status, attempts = select_table(raw, page_texts)
            else:
                chosen = raw
                validation = validate_table(
                    raw, coerce_threshold=config.coerce_threshold,
                    rel_tol=config.arithmetic_rel_tol, abs_tol=config.arithmetic_abs_tol,
                    page_texts=page_texts, prose_checker=prose_checker,
                    prose_cells=config.prose_check_cells, seed=config.llm_seed)
                status = assign_table_status(validation, config.table_confidence_threshold)
                attempts = [{"extractor": chosen.extractor,
                             "confidence": round(validation.confidence, 4)}]
            built = build_data_table(
                conn, parsed.doc_id, seq, chosen, coerce_threshold=config.coerce_threshold,
                style_overrides=validation.style_overrides, cells=config.cells_index)
            write_table_manifest(conn, built, validation, status)
            reports.append(TableReport(
                built.name, parsed.doc_id, status, round(validation.confidence, 4),
                chosen.extractor, attempts, validation.style_overrides,
                built.n_rows, built.n_cols))
        result = WrittenDocument(
            {"doc_id": parsed.doc_id, "source": str(source), "n_pages": parsed.n_pages,
             "n_tables_detected": len(parsed.tables), "n_chunks": len(chunks)}, reports)
        conn.execute("RELEASE SAVEPOINT write_document")
        return result
    except BaseException:
        conn.execute("ROLLBACK TO SAVEPOINT write_document")
        conn.execute("RELEASE SAVEPOINT write_document")
        raise
