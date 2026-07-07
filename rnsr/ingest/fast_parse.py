"""Fast text-tier parser for corpus-scale ingest (the 18k-file regime).

pdfium text extraction: ~two orders of magnitude faster than Docling's
layout ML, no external binaries (pypdfium2 is already a dependency). Full
text retained per page — grep/FTS/verify/annotate all work. What is NOT
extracted: typed tables and heading structure. Structure stays an
accelerant (§1.3): Docling can be escalated per-document later for the
files the questions actually touch.
"""

from __future__ import annotations

from pathlib import Path

from rnsr.ingest.model import Element, ParsedDocument
from rnsr.ingest.parse import make_doc_id


def stat_identity(path: Path) -> str:
    """Identity from path+size+mtime — no byte reads (fable-replicate's own
    manifest scheme). Content-exact dedupe is traded for speed at scale."""
    import hashlib

    st = path.stat()
    return hashlib.sha256(
        f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()


FAST_PARSER_NAME = "pdfium-fast"


def parse_pdf_fast(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    import pypdfium2 as pdfium

    path = Path(path)
    pdf = pdfium.PdfDocument(str(path))
    try:
        elements: list[Element] = []
        scanned: list[int] = []
        n_pages = len(pdf)
        for pno in range(n_pages):
            page = pdf[pno]
            tp = page.get_textpage()
            try:
                text = tp.get_text_bounded() or ""
            finally:
                tp.close()
            if len(text.strip()) < 50:
                scanned.append(pno + 1)
            if text.strip():
                elements.append(Element("text", text, pno + 1))
        return ParsedDocument(
            doc_id=doc_id or make_doc_id(path),
            source_path=str(path),
            sha256=stat_identity(path),
            n_pages=n_pages or 1,
            parser=FAST_PARSER_NAME,
            elements=elements,
            scanned_pages=scanned,
        )
    finally:
        pdf.close()
