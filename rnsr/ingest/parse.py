"""Primary parser: Docling -> ParsedDocument (spec §3.1).

Layout-aware extraction of text blocks and table candidates, retaining
page numbers and bounding boxes for every element. Table content is also
rendered into the element stream (kind='table') so the raw text remains
part of the retained canonical string — tables in SQLite are an
*additional* view, never a replacement (§1 commitment 4).

Docling is imported lazily so the core package works without the heavy
[ingest] extra installed.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path

from rnsr.ingest.model import BBox, Element, ParsedDocument, RawTable

PARSER_NAME = "docling"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_doc_id(path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or "doc"
    return stem[:48]


def _prov(item) -> tuple[int, BBox | None]:
    if getattr(item, "prov", None):
        p = item.prov[0]
        bbox = p.bbox
        return p.page_no, (bbox.l, bbox.t, bbox.r, bbox.b) if bbox else None
    return 1, None


def _grid_from_table(item) -> tuple[list[str], list[list[str | None]]]:
    """TableItem.data -> (header, rows) string grid.

    Uses the column_header flags when present; otherwise the first row is
    taken as the header (conservative — validate.py checks the result).
    """
    data = item.data
    ncols = data.num_cols
    matrix: list[list[str | None]] = [[None] * ncols for _ in range(data.num_rows)]
    header_rows: set[int] = set()
    for cell in data.table_cells:
        for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
            for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                if 0 <= r < data.num_rows and 0 <= c < ncols and matrix[r][c] is None:
                    matrix[r][c] = cell.text
        if cell.column_header:
            header_rows.update(
                range(cell.start_row_offset_idx, cell.end_row_offset_idx)
            )

    n_header = (max(header_rows) + 1) if header_rows else 1
    n_header = min(n_header, max(len(matrix) - 1, 1))
    header_matrix = matrix[:n_header]
    header = [
        " ".join(filter(None, (header_matrix[r][c] for r in range(n_header)))).strip()
        for c in range(ncols)
    ]
    return header, matrix[n_header:]


def render_table_text(header: list[str], rows: list[list[str | None]]) -> str:
    lines = [" | ".join(header)]
    lines += [" | ".join("" if c is None else str(c) for c in row) for row in rows]
    return "\n".join(lines)


def parse_pdf(path: str | Path, doc_id: str | None = None, *, ocr: bool = False) -> ParsedDocument:
    """Parse one document with Docling into the shared IR.

    OCR is off by default: target corpora have text layers, and the bundled
    rapidocr engine is broken in this environment. Scanned documents need
    ``ocr=True`` with a working engine configured.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import (
        ListItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    path = Path(path)
    pipeline = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    result = converter.convert(path)
    doc = result.document

    parsed = ParsedDocument(
        doc_id=doc_id or make_doc_id(path),
        source_path=str(path),
        sha256=_sha256(path),
        n_pages=len(doc.pages) or 1,
        parser=PARSER_NAME,
    )

    for item, _level in doc.iterate_items():
        page, bbox = _prov(item)
        if isinstance(item, TableItem):
            header, rows = _grid_from_table(item)
            caption = None
            with contextlib.suppress(Exception):
                caption = item.caption_text(doc) or None
            if rows:  # header-only grids carry no data
                parsed.tables.append(
                    RawTable(page=page, header=header, rows=rows, bbox=bbox,
                             extractor=PARSER_NAME, caption=caption)
                )
            parsed.elements.append(
                Element("table", render_table_text(header, rows), page, bbox)
            )
        elif isinstance(item, TitleItem):
            parsed.elements.append(Element("heading", item.text, page, bbox, heading_level=1))
        elif isinstance(item, SectionHeaderItem):
            level = max(int(getattr(item, "level", 1)), 1) + 1  # below any title
            parsed.elements.append(Element("heading", item.text, page, bbox, heading_level=level))
        elif isinstance(item, ListItem):
            parsed.elements.append(Element("list", item.text, page, bbox))
        elif isinstance(item, TextItem):
            if item.text and item.text.strip():
                parsed.elements.append(Element("text", item.text, page, bbox))

    # Pages with (almost) no extractable text have no text layer — scanned.
    # OCR engines are deliberately not used; the VLM transcribes these pages
    # when an LLM client is provided at ingest (pipeline.transcriber).
    chars_by_page: dict[int, int] = {}
    for e in parsed.elements:
        chars_by_page[e.page] = chars_by_page.get(e.page, 0) + len(e.text)
    parsed.scanned_pages = [
        p for p in range(1, parsed.n_pages + 1) if chars_by_page.get(p, 0) < 50
    ]
    return parsed
