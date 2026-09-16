"""Phase A entry point: ingest(sources) -> corpus.db + validation report (§3, §10).

Fully deterministic and LLM-free by default: the vision re-extraction rung
and the prose cross-check only run when their hooks are injected (Phase C
wires them to the sub-LM). Skipped stages are recorded in the report —
no silent failures (§3.3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rnsr.config import Settings
from rnsr.db import fts, schema
from rnsr.db.artifact import CorpusDB
from rnsr.ingest.dispatch import parse_any
from rnsr.ingest.expand import expand_document
from rnsr.ingest.fallback import VisionExtractor, reextract
from rnsr.ingest.manifest import write_corpus_manifest
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.parse import PARSER_NAME
from rnsr.ingest.transcription import merge_transcriptions
from rnsr.ingest.validate import (
    ProseChecker,
    TableValidation,
    assign_table_status,
    validate_table,
)
from rnsr.ingest.writer import TableReport, write_document


@dataclass
class IngestReport:
    out_db: str
    documents: list[dict] = field(default_factory=list)
    tables: list[TableReport] = field(default_factory=list)
    n_chunks: int = 0
    skipped_stages: list[str] = field(default_factory=list)
    scanned_pages_transcribed: int = 0
    scanned_pages_untranscribed: list[dict] = field(default_factory=list)  # visible gaps
    parse_failed: list[dict] = field(default_factory=list)

    @property
    def validation_pass_rate(self) -> float:
        """Trusted+reextracted over checked tables; unchecked are excluded."""
        from rnsr.ingest.health import validation_pass_rate

        return validation_pass_rate(
            len(self.tables), sum(t.status == "untrusted" for t in self.tables),
            sum(t.status == "unchecked" for t in self.tables))

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
                "parse_failed": self.parse_failed,
            },
            indent=2,
        )



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
    reextracted = not (chosen.extractor == raw.extractor and len(attempts) == 1)
    status = assign_table_status(
        validation, config.table_confidence_threshold,
        first_attempt=not reextracted, reextracted=reextracted)
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
    parse=parse_any,
) -> IngestReport:
    """Ingest documents into a single self-contained corpus.db artifact.

    Formats are dispatched by extension: PDFs through Docling, office
    formats (Word/Excel/PowerPoint/OpenDocument/RTF/EPUB/CSV) through
    anydoc, and .md/.txt/.eml through the built-in text-like parsers.
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

    # Parse first so scanned-page cost is known before any VLM spend.
    parsed_ok: list[tuple[Path, ParsedDocument]] = []
    seen_ids: set[str] = set()
    for src in sources:
        src = Path(src)
        try:
            parsed = parse(src)
        except Exception as e:
            report.parse_failed.append(
                {"source": str(src), "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        for child in expand_document(parsed, parse, seen_ids):
            parsed_ok.append((src, child))

    n_scanned = sum(len(p.scanned_pages) for _, p in parsed_ok)
    if n_scanned and transcriber is not None:
        from rnsr.ingest.cost_estimate import estimate_transcription_usd

        model = getattr(transcriber, "model", "") or config.vision_model or "vision"
        est = estimate_transcription_usd(n_scanned, model)
        # estimate is informational; spend is still governed
        _ = est
        for src, parsed in parsed_ok:
            if not parsed.scanned_pages:
                continue
            transcriptions = transcriber(src, parsed.scanned_pages)
            failed = merge_transcriptions(parsed, transcriptions)
            report.scanned_pages_transcribed += (
                len(parsed.scanned_pages) - len(failed))
            if failed:
                report.scanned_pages_untranscribed.append(
                    {"doc_id": parsed.doc_id, "pages": failed,
                     "reason": "transcription failed"})
    else:
        for _src, parsed in parsed_ok:
            if parsed.scanned_pages:
                report.scanned_pages_untranscribed.append(
                    {"doc_id": parsed.doc_id, "pages": parsed.scanned_pages,
                     "reason": "no transcriber"})

    if not parsed_ok:
        raise RuntimeError(
            f"all {len(report.parse_failed)} parses failed — not writing artifact")

    # Atomic artifact creation: build under a temp name, rename on success.
    # An interruption mid-ingest must never leave a partial corpus.db that a
    # cache later mistakes for a complete one (seen live: empty JPM corpus).
    tmp_db = out_db.with_suffix(out_db.suffix + ".ingesting")
    tmp_db.unlink(missing_ok=True)
    corpus = CorpusDB.create(tmp_db)
    conn = corpus.conn
    try:
        for src, parsed in parsed_ok:
            written = write_document(
                conn, src, parsed, config, prose_checker=prose_checker,
                select_table=lambda raw, page_texts, src=src: _extract_best_table(
                    src, raw, config, prose_checker, vision, page_texts),
            )
            report.documents.append(written.document)
            report.tables.extend(written.tables)

        report.n_chunks = fts.populate_fts(conn)
        schema.record_ingest_batch(
            conn, "create", sources, n_docs=len(parsed_ok))
        write_corpus_manifest(
            corpus, PARSER_NAME, config=config,
            extra_health={
                "parse_failed": len(report.parse_failed),
                "scanned_pages_total": n_scanned,
                "scanned_pages_untranscribed": sum(
                    len(x["pages"]) for x in report.scanned_pages_untranscribed),
            })
        schema.validate_integrity(conn)
        schema.finalize_corpus(conn)
        conn.commit()
        schema.checkpoint_for_publish(conn)
        corpus.close()
        tmp_db.rename(out_db)
        report.out_db = str(out_db)
    except BaseException:
        corpus.close()
        tmp_db.unlink(missing_ok=True)
        raise
    return report
