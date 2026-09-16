"""Extension-based parser dispatch for multi-format ingest.

Two tiers mirroring the PDF split: `parse_any` (quality: Docling for
PDFs) and `parse_any_fast` (scale: pdfium for PDFs). Non-PDF formats use the same parser in both tiers —
office parsing is milliseconds per document, so there is nothing to trade
away at scale. Both entry points are module-level functions, picklable
for bulk ingest's process pool.
"""

from __future__ import annotations

from pathlib import Path

from rnsr.ingest.fast_parse import parse_pdf_fast, stat_identity
from rnsr.ingest.model import ParsedDocument
from rnsr.ingest.office import OFFICE_EXTENSIONS, parse_office
from rnsr.ingest.textlike import (
    parse_eml,
    parse_html,
    parse_image,
    parse_markdown,
    parse_msg,
    parse_text,
    parse_zip,
)

PDF_EXTENSIONS = frozenset({".pdf"})
MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
TEXT_EXTENSIONS = frozenset({".txt"})
EML_EXTENSIONS = frozenset({".eml"})
MSG_EXTENSIONS = frozenset({".msg"})
HTML_EXTENSIONS = frozenset({".html", ".htm"})
ZIP_EXTENSIONS = frozenset({".zip"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})

SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS | OFFICE_EXTENSIONS | MARKDOWN_EXTENSIONS
    | TEXT_EXTENSIONS | EML_EXTENSIONS | MSG_EXTENSIONS | HTML_EXTENSIONS
    | ZIP_EXTENSIONS | IMAGE_EXTENSIONS
)


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
    if ext in MSG_EXTENSIONS:
        return parse_msg(path)
    if ext in HTML_EXTENSIONS:
        return parse_html(path)
    if ext in ZIP_EXTENSIONS:
        return parse_zip(path)
    if ext in IMAGE_EXTENSIONS:
        return parse_image(path)
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
    """Scale tier: fast PDF extraction with the same content identity contract."""
    path = Path(path)
    parsed = (parse_pdf_fast(path) if path.suffix.lower() in PDF_EXTENSIONS
              else _parse_non_pdf(path))
    parsed.content_sha256 = parsed.content_sha256 or parsed.sha256
    parsed.source_identity = stat_identity(path)
    return parsed
