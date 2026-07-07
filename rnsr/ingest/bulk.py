"""Resumable corpus-scale ingest (thousands of documents).

The single-shot pipeline builds atomically — right for one filing, wrong
for 18,000 files where a crash at file 17,999 must not lose four days of
work. Bulk ingest checkpoints per document into the .ingesting artifact:
already-ingested files (matched by sha256) are skipped on resume, commits
happen every `commit_every` docs, and the FTS build / manifest / freeze /
atomic rename run once at the end. The final artifact is identical in
shape to a single-shot one.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from rnsr.config import Settings
from rnsr.db import fts, schema
from rnsr.ingest.chunk import chunk_document
from rnsr.ingest.fast_parse import parse_pdf_fast
from rnsr.ingest.fast_parse import stat_identity as _file_sha
from rnsr.ingest.manifest import write_corpus_manifest, write_table_manifest
from rnsr.ingest.model import ParsedDocument
from rnsr.ingest.pipeline import _merge_transcriptions
from rnsr.ingest.tables import build_data_table, merge_multipage
from rnsr.ingest.validate import validate_table

Progress = Callable[[str], None]


def _is_finalized(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' "
        "AND name='chunks__no_insert'").fetchone())


def ingest_bulk(
    sources: list[Path],
    out_db: str | Path,
    *,
    config: Settings | None = None,
    parse=parse_pdf_fast,
    transcriber=None,
    commit_every: int = 50,
    workers: int | None = None,
    progress: Progress = lambda s: None,
) -> dict:
    """Ingest many documents with per-document resume. Returns summary stats.

    Parsing parallelizes across a process pool (auto: cores-2) when using
    the default picklable parser; a custom `parse` callable runs serially
    unless `workers` is set explicitly."""
    config = config or Settings()
    out_db = Path(out_db)
    if out_db.exists():
        return {"resumed": False, "already_complete": True, "out_db": str(out_db)}

    tmp = out_db.with_suffix(out_db.suffix + ".ingesting")
    fresh = not tmp.exists()
    conn = sqlite3.connect(tmp)
    try:
        if fresh:
            schema.create_corpus_db(conn)
        finalized = _is_finalized(conn)

        done_shas: set[str] = set()
        seen_ids: set[str] = set()
        if not fresh:
            for sha, did in conn.execute("SELECT sha256, doc_id FROM documents"):
                done_shas.add(sha)
                seen_ids.add(did)
            progress(f"resuming: {len(done_shas)} documents already ingested")

        n_new = n_skipped = n_scanned_gap = n_failed = 0
        table_seq: dict[str, int] = {}

        def write_parsed(src: Path, parsed: ParsedDocument) -> None:
            nonlocal n_new, n_scanned_gap
            if parsed.doc_id in seen_ids:
                parsed.doc_id = f"{parsed.doc_id}_{len(seen_ids)}"
            seen_ids.add(parsed.doc_id)

            if parsed.scanned_pages:
                if transcriber is not None:
                    merged = transcriber(src, parsed.scanned_pages)
                    _merge_transcriptions(parsed, merged)
                else:
                    n_scanned_gap += len(parsed.scanned_pages)

            pages, chunks = chunk_document(
                parsed, chunk_chars=config.chunk_chars,
                overlap=config.chunk_overlap)
            page_texts = {p.page: p.text for p in pages}
            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                (parsed.doc_id, parsed.source_path, parsed.sha256,
                 parsed.n_pages, parsed.parser,
                 datetime.now(UTC).isoformat()))
            conn.executemany(
                "INSERT INTO doc_text VALUES (?,?,?,?,?)",
                [(parsed.doc_id, p.page, p.char_start, p.char_end, p.text)
                 for p in pages])
            conn.executemany(
                "INSERT INTO chunks (doc_id, page, char_start, char_end, "
                "heading_path, text) VALUES (?,?,?,?,?,?)",
                [(parsed.doc_id, c.page, c.char_start, c.char_end,
                  c.heading_path, c.text) for c in chunks])
            for raw in merge_multipage(parsed.tables):
                seq = table_seq.get(parsed.doc_id, 0) + 1
                table_seq[parsed.doc_id] = seq
                validation = validate_table(
                    raw, coerce_threshold=config.coerce_threshold,
                    rel_tol=config.arithmetic_rel_tol,
                    abs_tol=config.arithmetic_abs_tol,
                    page_texts=page_texts)
                status = ("trusted" if validation.confidence
                          >= config.table_confidence_threshold else "untrusted")
                built = build_data_table(
                    conn, parsed.doc_id, seq, raw,
                    coerce_threshold=config.coerce_threshold,
                    style_overrides=validation.style_overrides)
                write_table_manifest(conn, built, validation, status)
            n_new += 1
            if n_new % commit_every == 0:
                conn.commit()
                progress(f"ingested {n_new} new documents")

        if not finalized:
            pending: list[Path] = []
            for src in sources:
                if _file_sha(src) in done_shas:
                    n_skipped += 1
                else:
                    pending.append(src)
            if n_skipped:
                progress(f"skipping {n_skipped} already-ingested files")

            # Parsing is embarrassingly parallel; writing stays single-threaded
            # in this process. Custom (possibly unpicklable) parsers run
            # serially unless workers is set explicitly.
            if workers is None:
                workers = (max(1, (os.cpu_count() or 2) - 2)
                           if parse is parse_pdf_fast else 1)
            if workers > 1 and pending:
                progress(f"parsing with {workers} workers")
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(parse, src): src for src in pending}
                    for fut in as_completed(futures):
                        src = futures[fut]
                        try:
                            write_parsed(src, fut.result())
                        except Exception as e:
                            n_failed += 1
                            progress(f"PARSE FAILED {src.name}: "
                                     f"{type(e).__name__}: {e}")
            else:
                for src in pending:
                    try:
                        parsed: ParsedDocument = parse(src)
                    except Exception as e:
                        n_failed += 1
                        progress(f"PARSE FAILED {src.name}: {type(e).__name__}: {e}")
                        continue
                    write_parsed(src, parsed)
            conn.commit()
            if pending and n_new == 0 and n_failed == len(pending):
                # every parse failed (broken pool, wrong file type, …):
                # finalizing would produce a plausible-looking empty artifact.
                raise RuntimeError(
                    f"all {n_failed} parses failed — not finalizing; "
                    "resume state kept at " + str(tmp))

            progress("building FTS index…")
            fts.populate_fts(conn)

            from rnsr.db.artifact import CorpusDB

            conn.commit()
            conn.close()
            corpus = CorpusDB(tmp, mode="rw")
            from rnsr.ingest.fast_parse import FAST_PARSER_NAME

            write_corpus_manifest(corpus, FAST_PARSER_NAME)
            schema.finalize_corpus(corpus.conn)
            corpus.conn.commit()
            corpus.close()
        else:
            conn.close()

        tmp.rename(out_db)
        return {"resumed": not fresh, "new_docs": n_new, "skipped": n_skipped,
                "parse_failed": n_failed,
                "scanned_pages_untranscribed": n_scanned_gap,
                "out_db": str(out_db)}
    except BaseException:
        # leave the .ingesting artifact in place: that IS the resume state
        import contextlib

        with contextlib.suppress(Exception):
            conn.close()
        raise
