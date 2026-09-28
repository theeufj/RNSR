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
import logging
import re
from pathlib import Path

from rnsr.ingest.coerce import coerce_column, is_null_cell
from rnsr.ingest.model import BBox, Element, ParsedDocument, RawTable

PARSER_NAME = "docling"
_LOG = logging.getLogger(__name__)
_MEASUREMENT_HEADER = re.compile(
    r"^(?:(?:net|gross|total|operating|adjusted|average)\s+)*"
    r"(?:amount|value|revenue|sales|income|profit|loss|cost|expense|balance|"
    r"count|quantity|qty|headcount|votes|shares|price|weight|duration|distance)"
    r"s?(?:\s*(?:\([^)]*\)|\[[^]]*\]|[%$€£¥]))?$", re.IGNORECASE,
)


def content_sha256(path: Path) -> str:
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

    Prefer a leading run of explicit column_header rows. Without flags,
    infer a header only from measurement headings above numeric data. An
    ambiguous first observation is retained under generated column names.
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

    n_header = 0
    while n_header < len(matrix) and n_header in header_rows:
        n_header += 1
    if not n_header and len(matrix) > 1:
        first = matrix[0]
        # Do not infer a header from a blank/missing observation or a row of
        # label/value data. Numeric-looking years may be explicit headers,
        # but absent parser metadata they must not cost us a source row.
        first_has_values = any(coerce_column([v]).is_numeric for v in first)
        type_contrast = any(
            not is_null_cell(first[c])
            and _MEASUREMENT_HEADER.fullmatch(str(first[c]).strip())
            and coerce_column([row[c] for row in matrix[1:]]).is_numeric
            for c in range(ncols)
        )
        if not first_has_values and type_contrast:
            n_header = 1
    if not n_header:
        return [f"column_{c + 1}" for c in range(ncols)], matrix
    header_matrix = matrix[:n_header]
    header = [
        " ".join(filter(None, (header_matrix[r][c] for r in range(n_header)))).strip()
        for c in range(ncols)
    ]
    return header, matrix[n_header:]


def render_table_text(header: list[str], rows: list[list[str | None]]) -> str:
    lines = [" | ".join(header)] if header else []
    lines += [" | ".join("" if c is None else str(c) for c in row) for row in rows]
    return "\n".join(lines)


def _has_unretained_marks(page, textpage) -> bool:
    """Require transcription unless every rendered mark is already accounted for.

    Native glyphs are retained separately. Only long, thin horizontal rules in
    the header/footer margin (at most 10% or 72 points) count as decoration;
    charts, pictures, raster text,
    and even a small unexplained mark remain transcription candidates. This is
    deliberately not an ink-percentage heuristic, which loses sparse content.
    """
    from PIL import ImageDraw
    from pypdfium2 import raw

    with contextlib.closing(page.render(scale=2, grayscale=True)) as bitmap:
        remaining = bitmap.to_pil().copy()
        draw = ImageDraw.Draw(remaining)
        coordinates = bitmap.get_posconv(page)

        def cover(bounds, padding=1):
            left, bottom, right, top = bounds
            points = [coordinates.to_bitmap(x, y)
                      for x in (left, right) for y in (bottom, top)]
            xs, ys = zip(*points, strict=True)
            draw.rectangle((min(xs) - padding, min(ys) - padding,
                            max(xs) + padding, max(ys) + padding), fill=255)

        for index in range(textpage.count_chars()):
            char = textpage.get_text_range(index, 1)
            if char.strip() and char.isprintable() and "\ufffd" not in char:
                cover(textpage.get_charbox(index))

        page_left, page_bottom, page_right, page_top = page.get_bbox()
        width, height = page_right - page_left, page_top - page_bottom
        margin = min(height * 0.1, 72)
        objects = list(page.get_objects())
        # A raster/shading layer could contain tiny text near a separator.
        # In that case do not erase any rule-sized band over that layer.
        rules = ([] if any(obj.type not in {raw.FPDF_PAGEOBJ_TEXT, raw.FPDF_PAGEOBJ_PATH,
                                           raw.FPDF_PAGEOBJ_FORM} for obj in objects)
                 else [obj for obj in objects if obj.type == raw.FPDF_PAGEOBJ_PATH])
        for obj in rules:
            left, bottom, right, top = obj.get_bounds()
            segments = raw.FPDFPath_CountSegments(obj)
            simple = 1 <= segments <= 5 and all(
                raw.FPDFPathSegment_GetType(raw.FPDFPath_GetPathSegment(obj, i))
                in {raw.FPDF_SEGMENT_MOVETO, raw.FPDF_SEGMENT_LINETO}
                for i in range(segments))
            # Include tiny end caps attached to these separator rules, but no
            # other vector objects. A two-point band covers their stroke bounds.
            if (simple and right - left >= width * 0.5 and top - bottom <= 2
                    and (bottom >= page_top - margin or top <= page_bottom + margin)):
                cover((left - 2, bottom - 2, right + 2, top + 2))
        return remaining.getextrema()[0] < 254


def _resolve_sparse_pdf_pages(parsed: ParsedDocument, path: Path) -> None:
    """Recover native text and classify sparse pages against the original PDF.

    Docling's layout model can omit a short or unusually positioned text block.
    PDFium provides a conservative additional text view, with its original page
    and bounding box; existing elements are never replaced. Failure to inspect
    a page preserves its scan candidate, rather than concealing a source gap.
    """
    chars_by_page: dict[int, int] = {}
    for element in parsed.elements:
        chars_by_page[element.page] = chars_by_page.get(element.page, 0) + len(element.text.strip())
    candidates = [p for p in range(1, parsed.n_pages + 1) if chars_by_page.get(p, 0) < 50]
    parsed.scanned_pages = list(candidates)
    if not candidates:
        return
    try:
        import pypdfium2 as pdfium

        with contextlib.closing(pdfium.PdfDocument(path)) as source:
            for number in candidates:
                try:
                    with (contextlib.closing(source[number - 1]) as page,
                          contextlib.closing(page.get_textpage()) as textpage):
                        native = textpage.get_text_bounded().strip()
                        existing = re.sub(r"\s+", "", parsed.page_text(number))
                        if native and re.sub(r"\s+", "", native) not in existing:
                            rects = [textpage.get_rect(i)
                                     for i in range(textpage.count_rects())]
                            bbox = ((min(r[0] for r in rects), max(r[3] for r in rects),
                                     max(r[2] for r in rects), min(r[1] for r in rects))
                                    if rects else None)
                            parsed.elements.append(Element("text", native, number, bbox))
                            parsed.parser = f"{PARSER_NAME}+pdfium"
                        if not _has_unretained_marks(page, textpage):
                            parsed.scanned_pages.remove(number)
                except Exception:
                    _LOG.warning("Could not inspect sparse PDF page %s; retaining transcription requirement",
                                 number, exc_info=True)
    except Exception:
        _LOG.warning("Could not inspect sparse PDF pages; retaining transcription requirements",
                     exc_info=True)


def parse_pdf(path: str | Path, doc_id: str | None = None, *, ocr: bool = False) -> ParsedDocument:
    """Parse one document with Docling into the shared IR.

    OCR is off by default. Sparse layout output is checked against the native
    text and visible source page; unexplained content remains a candidate for
    the ingestion pipeline's page transcriber. ``ocr=True`` enables Docling's
    OCR engine when one is configured.
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

    digest = content_sha256(path)
    parsed = ParsedDocument(
        doc_id=doc_id or make_doc_id(path),
        source_path=str(path),
        sha256=digest,
        content_sha256=digest,
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
            # Generated column names belong only to the typed view; they
            # are not source text and must not become quotable evidence.
            source_header = header if len(rows) < item.data.num_rows else []
            parsed.elements.append(
                Element("table", render_table_text(source_header, rows), page, bbox)
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

    _resolve_sparse_pdf_pages(parsed, path)
    return parsed
