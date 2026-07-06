"""Shared intermediate representation between parsers and the SQLite writer.

Every parser rung (§3.1: docling → pdfplumber/camelot → vision) normalizes
to these types, so tables.py/chunk.py/validate.py are parser-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in page coordinates


@dataclass
class Element:
    """One layout element from the parsed document, in reading order."""

    kind: str                      # 'text' | 'heading' | 'caption' | 'list' | 'other'
    text: str
    page: int                      # 1-based
    bbox: BBox | None = None
    heading_level: int | None = None   # for kind == 'heading'


@dataclass
class RawTable:
    """A detected table as a raw string grid, before typing/coercion."""

    page: int                      # first page the table appears on
    header: list[str]
    rows: list[list[str | None]]
    bbox: BBox | None = None
    row_pages: list[int] | None = None    # per-row page when known (multi-page)
    row_bboxes: list[BBox | None] | None = None
    extractor: str = "docling"     # which fallback rung produced it (§3.1)
    caption: str | None = None     # nearest caption/heading text, machine-extracted

    @property
    def n_cols(self) -> int:
        return len(self.header)

    def row_page(self, i: int) -> int:
        return self.row_pages[i] if self.row_pages else self.page

    def row_bbox(self, i: int) -> BBox | None:
        if self.row_bboxes and self.row_bboxes[i] is not None:
            return self.row_bboxes[i]
        return self.bbox


@dataclass
class ParsedDocument:
    """Normalized parser output for one source document."""

    doc_id: str
    source_path: str
    sha256: str
    n_pages: int
    parser: str
    elements: list[Element] = field(default_factory=list)
    tables: list[RawTable] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)  # no text layer; VLM candidates

    def page_text(self, page: int) -> str:
        return "\n".join(e.text for e in self.elements if e.page == page and e.text)
