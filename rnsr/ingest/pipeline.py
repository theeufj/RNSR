"""Phase A entry point: ingest(sources) -> corpus.db + validation report (§3, §10).

Fully deterministic and LLM-free by default: the vision re-extraction rung
and the prose cross-check only run when their hooks are injected (Phase C
wires them to the sub-LM). Skipped stages are recorded in the report —
no silent failures (§3.3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rnsr.config import Settings
from rnsr.db import fts, schema
from rnsr.db.artifact import CorpusDB
from rnsr.ingest.chunk import chunk_document
from rnsr.ingest.fallback import VisionExtractor, reextract
from rnsr.ingest.manifest import write_corpus_manifest, write_table_manifest
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.parse import PARSER_NAME, parse_pdf
from rnsr.ingest.tables import build_data_table, merge_multipage
from rnsr.ingest.validate import ProseChecker, TableValidation, validate_table


@dataclass
class TableReport:
    name: str
    doc_id: str
    status: str                  # trusted | reextracted | untrusted
    confidence: float
    extractor: str               # rung that produced the stored table
    attempts: list[dict]         # every (extractor, confidence) tried
    style_overrides: dict[str, str]
    n_rows: int
    n_cols: int


@dataclass
class IngestReport:
    out_db: str
    documents: list[dict] = field(default_factory=list)
    tables: list[TableReport] = field(default_factory=list)
    n_chunks: int = 0
    skipped_stages: list[str] = field(default_factory=list)
    scanned_pages_transcribed: int = 0
    scanned_pages_untranscribed: list[dict] = field(default_factory=list)  # visible gaps

    @property
    def validation_pass_rate(self) -> float:
        """Fraction of tables not untrusted — the §9 early health metric."""
        if not self.tables:
            return 1.0
        ok = sum(t.status in ("trusted", "reextracted") for t in self.tables)
        return ok / len(self.tables)

    def to_json(self) -> str:
        return json.dumps(
            {
                "out_db": self.out_db,
                "documents": self.documents,
                "tables": [asdict(t) for t in self.tables],
                "n_chunks": self.n_chunks,
                "validation_pass_rate": round(self.validation_pass_rate, 4),
                "skipped_stages": self.skipped_stages,
                "scanned_pages_transcribed": self.scanned_pages_transcribed,
                "scanned_pages_untranscribed": self.scanned_pages_untranscribed,
            },
            indent=2,
        )


def _merge_transcriptions(parsed: ParsedDocument,
                          transcriptions: dict[int, dict | None]) -> list[int]:
    """Fold VLM page transcriptions into the parsed document; returns pages
    that failed to transcribe. Elements/tables are stamped extractor=vision
    and flow through the normal checksum-validation path (§3.3)."""
    failed: list[int] = []
    for page in sorted(transcriptions):
        t = transcriptions[page]
        if t is None:
            failed.append(page)
            continue
        for block in t.get("blocks", []):
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            kind = "heading" if block.get("kind") == "heading" else "text"
            level = 1 if kind == "heading" else None
            parsed.elements.append(Element(kind, text, page, heading_level=level))
        for grid in t.get("tables", []):
            header = [str(h) if h is not None else "" for h in grid.get("header", [])]
            rows = [[None if c is None else str(c) for c in row]
                    for row in grid.get("rows", [])]
            if header and rows:
                parsed.tables.append(RawTable(page=page, header=header, rows=rows,
                                              extractor="vision"))
    return failed


def _validate(raw: RawTable, config: Settings, prose_checker: ProseChecker | None,
              page_texts: dict[int, str]) -> TableValidation:
    return validate_table(
        raw,
        coerce_threshold=config.coerce_threshold,
        rel_tol=config.arithmetic_rel_tol,
        abs_tol=config.arithmetic_abs_tol,
        prose_checker=prose_checker,
        page_texts=page_texts,
        prose_cells=config.prose_check_cells,
        seed=config.llm_seed,
    )


def _extract_best_table(
    pdf_path: Path,
    raw: RawTable,
    config: Settings,
    prose_checker: ProseChecker | None,
    vision: VisionExtractor | None,
    page_texts: dict[int, str],
) -> tuple[RawTable, TableValidation, str, list[dict]]:
    """Validate, re-extracting down the fallback chain while below threshold.

    Returns the best-scoring variant (§3.3: trusted, retried, or flagged).
    """
    attempts: list[dict] = []
    best: tuple[RawTable, TableValidation] | None = None
    current: RawTable | None = raw
    while current is not None:
        validation = _validate(current, config, prose_checker, page_texts)
        attempts.append({"extractor": current.extractor,
                         "confidence": round(validation.confidence, 4)})
        if best is None or validation.confidence > best[1].confidence:
            best = (current, validation)
        if validation.confidence >= config.table_confidence_threshold:
            break
        current = reextract(pdf_path, current, vision=vision)

    assert best is not None
    chosen, validation = best
    if validation.confidence < config.table_confidence_threshold:
        status = "untrusted"
    elif chosen.extractor == raw.extractor and len(attempts) == 1:
        status = "trusted"
    else:
        status = "reextracted"
    return chosen, validation, status, attempts


def ingest_text(
    named_texts: dict[str, str],
    out_db: str | Path,
    *,
    config: Settings | None = None,
) -> IngestReport:
    """Ingest raw text strings (doc_id -> text) into a corpus.db.

    Flat-text benchmarks (OOLONG) and any no-PDF corpus go through the same
    pipeline — chunks, FTS, manifest, freeze, atomic build — with one
    element per non-empty line. Lines also become a `lines`-shaped table
    (line_no, text) so semantic_annotate + exact SQL aggregation work over
    them: the §4.1 pattern needs a table to write its column back to.
    Fully deterministic.
    """
    import hashlib
    import re

    def parse(src) -> ParsedDocument:
        key = str(src)
        text = named_texts[key]
        doc_id = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")[:48] or "doc"
        lines = [line for line in text.split("\n") if line.strip()]
        elements = [Element("text", line, 1) for line in lines]
        tables = []
        if len(lines) > 1:
            tables.append(RawTable(
                page=1,
                header=["line_no", "text"],
                rows=[[str(i), line] for i, line in enumerate(lines, 1)],
                extractor="text",
                caption=f"lines of {doc_id}",
            ))
        return ParsedDocument(
            doc_id=doc_id,
            source_path=f"text:{key}",
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            n_pages=1,
            parser="text",
            elements=elements,
            tables=tables,
        )

    return ingest(list(named_texts), out_db, config=config, parse=parse)


def ingest(
    sources: list[str | Path] | str | Path,
    out_db: str | Path,
    *,
    config: Settings | None = None,
    prose_checker: ProseChecker | None = None,
    vision: VisionExtractor | None = None,
    transcriber=None,
    parse=parse_pdf,
) -> IngestReport:
    """Ingest documents into a single self-contained corpus.db artifact.

    `parse` is injectable for tests (any callable path -> ParsedDocument).
    `transcriber` (llm_hooks.make_page_transcriber) turns scanned pages into
    elements/tables via the VLM; without it, scanned pages are reported as
    untranscribed — visible, never silent.
    """
    config = config or Settings()
    if isinstance(sources, (str, Path)):
        sources = [sources]
    out_db = Path(out_db)

    report = IngestReport(out_db=str(out_db))
    if prose_checker is None:
        report.skipped_stages.append("prose_cross_check (no LLM client)")
    if vision is None:
        report.skipped_stages.append("vision_reextraction (no LLM client)")
    if transcriber is None:
        report.skipped_stages.append("scanned_page_transcription (no LLM client)")

    # Atomic artifact creation: build under a temp name, rename on success.
    # An interruption mid-ingest must never leave a partial corpus.db that a
    # cache later mistakes for a complete one (seen live: empty JPM corpus).
    tmp_db = out_db.with_suffix(out_db.suffix + ".ingesting")
    tmp_db.unlink(missing_ok=True)
    corpus = CorpusDB.create(tmp_db)
    conn = corpus.conn
    try:
        seen_ids: set[str] = set()
        for src in sources:
            src = Path(src)
            parsed: ParsedDocument = parse(src)
            if parsed.doc_id in seen_ids:
                parsed.doc_id = f"{parsed.doc_id}_{len(seen_ids)}"
            seen_ids.add(parsed.doc_id)

            if parsed.scanned_pages:
                if transcriber is not None:
                    transcriptions = transcriber(src, parsed.scanned_pages)
                    failed = _merge_transcriptions(parsed, transcriptions)
                    report.scanned_pages_transcribed += (
                        len(parsed.scanned_pages) - len(failed))
                    if failed:
                        report.scanned_pages_untranscribed.append(
                            {"doc_id": parsed.doc_id, "pages": failed,
                             "reason": "transcription failed"})
                else:
                    report.scanned_pages_untranscribed.append(
                        {"doc_id": parsed.doc_id, "pages": parsed.scanned_pages,
                         "reason": "no transcriber (run with --llm)"})

            pages, chunks = chunk_document(
                parsed, chunk_chars=config.chunk_chars, overlap=config.chunk_overlap
            )
            page_texts = {p.page: p.text for p in pages}

            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                (parsed.doc_id, parsed.source_path, parsed.sha256, parsed.n_pages,
                 parsed.parser, datetime.now(UTC).isoformat()),
            )
            conn.executemany(
                "INSERT INTO doc_text VALUES (?,?,?,?,?)",
                [(parsed.doc_id, p.page, p.char_start, p.char_end, p.text) for p in pages],
            )
            conn.executemany(
                "INSERT INTO chunks (doc_id, page, char_start, char_end, heading_path, text) "
                "VALUES (?,?,?,?,?,?)",
                [(parsed.doc_id, c.page, c.char_start, c.char_end, c.heading_path, c.text)
                 for c in chunks],
            )
            report.documents.append(
                {"doc_id": parsed.doc_id, "source": str(src), "n_pages": parsed.n_pages,
                 "n_tables_detected": len(parsed.tables), "n_chunks": len(chunks)}
            )

            for seq, raw in enumerate(merge_multipage(parsed.tables), start=1):
                chosen, validation, status, attempts = _extract_best_table(
                    src, raw, config, prose_checker, vision, page_texts
                )
                built = build_data_table(
                    conn, parsed.doc_id, seq, chosen,
                    coerce_threshold=config.coerce_threshold,
                    style_overrides=validation.style_overrides,
                )
                write_table_manifest(conn, built, validation, status)
                report.tables.append(TableReport(
                    name=built.name, doc_id=parsed.doc_id, status=status,
                    confidence=round(validation.confidence, 4),
                    extractor=chosen.extractor, attempts=attempts,
                    style_overrides=validation.style_overrides,
                    n_rows=built.n_rows, n_cols=built.n_cols,
                ))

        report.n_chunks = fts.populate_fts(conn)
        write_corpus_manifest(corpus, PARSER_NAME)
        schema.finalize_corpus(conn)
        conn.commit()
        corpus.close()
        tmp_db.rename(out_db)
        report.out_db = str(out_db)
    except BaseException:
        corpus.close()
        tmp_db.unlink(missing_ok=True)
        raise
    return report
