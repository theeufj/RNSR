"""Extension-based parser dispatch for multi-format ingest.

Two tiers mirroring the PDF split: `parse_any` (quality: Docling for
PDFs) and `parse_any_fast` (scale: pdfium for PDFs, stat-based identity
for resume matching). Non-PDF formats use the same parser in both tiers —
office parsing is milliseconds per document, so there is nothing to trade
away at scale. Both entry points are module-level functions, picklable
for bulk ingest's process pool.
"""

from __future__ import annotations

from pathlib import Path

from rnsr.ingest.fast_parse import parse_pdf_fast, stat_identity
from rnsr.ingest.model import ParsedDocument
from rnsr.ingest.office import OFFICE_EXTENSIONS, parse_office
from rnsr.ingest.textlike import parse_eml, parse_markdown, parse_text

PDF_EXTENSIONS = frozenset({".pdf"})
MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
TEXT_EXTENSIONS = frozenset({".txt"})
EML_EXTENSIONS = frozenset({".eml"})

SUPPORTED_EXTENSIONS = (PDF_EXTENSIONS | OFFICE_EXTENSIONS | MARKDOWN_EXTENSIONS
                        | TEXT_EXTENSIONS | EML_EXTENSIONS)


def is_ingestable(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def _parse_non_pdf(path: Path) -> ParsedDocument:
    ext = path.suffix.lower()
    if ext in OFFICE_EXTENSIONS:
        return parse_office(path)
    if ext in MARKDOWN_EXTENSIONS:
        return parse_markdown(path)
    if ext in EML_EXTENSIONS:
        return parse_eml(path)
    if ext in TEXT_EXTENSIONS:
        return parse_text(path)
    raise ValueError(
        f"unsupported document type {ext!r}: {path} "
        f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")


def parse_any(path: str | Path) -> ParsedDocument:
    """Quality tier: Docling for PDFs, format-appropriate parser otherwise."""
    from rnsr.ingest.parse import parse_pdf

    path = Path(path)
    if path.suffix.lower() in PDF_EXTENSIONS:
        return parse_pdf(path)
    return _parse_non_pdf(path)


def parse_any_fast(path: str | Path) -> ParsedDocument:
    """Scale tier: pdfium for PDFs; identity is stat-based to match bulk
    ingest's resume check (no byte reads over the corpus)."""
    path = Path(path)
    if path.suffix.lower() in PDF_EXTENSIONS:
        return parse_pdf_fast(path)
    parsed = _parse_non_pdf(path)
    parsed.sha256 = stat_identity(path)
    return parsed
