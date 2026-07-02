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
from rnsr.ingest.model import ParsedDocument, RawTable
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
    if validation.confidence < config.table_confidence_threshold:
        status = "untrusted"
    elif chosen.extractor == raw.extractor and len(attempts) == 1:
        status = "trusted"
    else:
        status = "reextracted"
    return chosen, validation, status, attempts


def ingest(
    sources: list[str | Path] | str | Path,
    out_db: str | Path,
    *,
    config: Settings | None = None,
    prose_checker: ProseChecker | None = None,
    vision: VisionExtractor | None = None,
    parse=parse_pdf,
) -> IngestReport:
    """Ingest documents into a single self-contained corpus.db artifact.

    `parse` is injectable for tests (any callable path -> ParsedDocument).
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

    corpus = CorpusDB.create(out_db)
    conn = corpus.conn
    try:
        seen_ids: set[str] = set()
        for src in sources:
            src = Path(src)
            parsed: ParsedDocument = parse(src)
            if parsed.doc_id in seen_ids:
                parsed.doc_id = f"{parsed.doc_id}_{len(seen_ids)}"
            seen_ids.add(parsed.doc_id)

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
    finally:
        corpus.close()
    return report
