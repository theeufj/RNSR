"""anydoc-backed parser: office formats -> ParsedDocument.

Word (incl. legacy .doc), Excel, PowerPoint, OpenDocument, RTF, EPUB, and
CSV parse through firecrawl-anydoc's document model (pure Rust, no ML, no
external services). Tables arrive as canonical grids with header-row
counts and merged-cell spans, so they flow into the same RawTable ->
typed-SQL-table -> checksum-validation path as PDF tables (§3.3); layout
tables (positioning scaffolding) are kept as text only. Office formats
have no pages, so every element lands on page 1 — provenance stays exact
through char offsets into the canonical string (§1 commitment 4).

PDFs deliberately stay with Docling (quality) / pdfium (scale): anydoc's
PDF path is markdown-only, with no document-model tables, no bboxes, and
no layout ML — see `anydoc.to_document`, which refuses the format.

anydoc is imported lazily so the core package works without the [ingest]
extra installed.
"""

from __future__ import annotations

from pathlib import Path

from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.parse import _sha256, make_doc_id, render_table_text

OFFICE_PARSER_NAME = "anydoc"

# Everything anydoc converts, minus PDF (which keeps its own two tiers).
OFFICE_EXTENSIONS = frozenset({
    ".doc", ".docx", ".docm",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub", ".csv",
})


def _inline_text(inlines) -> str:
    parts: list[str] = []
    for i in inlines or []:
        if i.kind == "text":
            parts.append(i.text or "")
        elif i.kind == "link":
            parts.append(_inline_text(i.content))
        elif i.kind == "image":
            if i.alt:
                parts.append(i.alt)
        elif i.kind == "line_break":
            parts.append("\n")
        # anchor / note_ref: zero-width markers; note bodies are appended
        # as trailing elements by parse_office.
    return "".join(parts)


def _block_text(block) -> str:
    """Flatten any block to plain text (used inside cells, quotes, lists)."""
    if block.kind in ("heading", "paragraph"):
        return _inline_text(block.content)
    if block.kind == "list":
        return "\n".join(_item_text(it) for it in block.list.items)
    if block.kind == "table":
        header, rows = _grid(block.table)
        return render_table_text(header, rows)
    if block.kind == "block_quote":
        return "\n".join(_block_text(b) for b in block.blocks or [])
    if block.kind == "code_block":
        return block.text or ""
    return ""  # rule


def _item_text(item) -> str:
    text = "\n".join(filter(None, (_block_text(b) for b in item.blocks)))
    return f"{item.marker_label} {text}" if item.marker_label else text


def _cell_text(cell) -> str:
    if cell is None:
        return ""
    # Space-joined: cell text must stay single-line for the pipe rendering.
    return " ".join(filter(None, (_block_text(b).replace("\n", " ") for b in cell.blocks)))


def _grid(table) -> tuple[list[str], list[list[str | None]]]:
    """anydoc Table -> (header, rows) string grid.

    Covered slots (merged-cell spans) copy the origin's text into every
    position, matching the Docling grid convention in parse.py. Without
    header-row info the first row is taken as header (conservative —
    validate.py checks the result).
    """
    txt: list[list[str | None]] = []
    for row in table.grid:
        out: list[str | None] = []
        for slot in row:
            if slot.kind == "origin":
                out.append(_cell_text(slot.cell))
            else:
                origin = table.grid[slot.origin_row][slot.origin_col]
                out.append(_cell_text(origin.cell))
        txt.append(out)
    if not txt:
        return [], []
    ncols = len(txt[0])
    n_header = max(table.header_rows, 1)
    n_header = min(n_header, max(len(txt) - 1, 1))
    header = [
        " ".join(filter(None, (txt[r][c] for r in range(n_header)))).strip()
        for c in range(ncols)
    ]
    return header, txt[n_header:]


def parse_office(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    """Parse one office document with anydoc into the shared IR."""
    import anydoc

    path = Path(path)
    data = path.read_bytes()
    # Content detection first (mislabeled files still convert); the
    # extension names signature-less formats (CSV).
    fmt = anydoc.format_from_bytes(data) or anydoc.format_from_path(str(path))
    doc = anydoc.to_document(data, fmt)

    parsed = ParsedDocument(
        doc_id=doc_id or make_doc_id(path),
        source_path=str(path),
        sha256=_sha256(path),
        n_pages=1,
        parser=OFFICE_PARSER_NAME,
    )

    for block in doc.blocks:
        if block.kind == "heading":
            text = _inline_text(block.content)
            if text.strip():
                parsed.elements.append(
                    Element("heading", text, 1, heading_level=block.level or 1))
        elif block.kind == "paragraph":
            text = _inline_text(block.content)
            if text.strip():
                parsed.elements.append(Element("text", text, 1))
        elif block.kind == "list":
            for item in block.list.items:
                text = _item_text(item)
                if text.strip():
                    parsed.elements.append(Element("list", text, 1))
        elif block.kind == "table":
            header, rows = _grid(block.table)
            if not header:
                continue
            parsed.elements.append(
                Element("table", render_table_text(header, rows), 1))
            if block.table.kind == "data" and rows:
                parsed.tables.append(RawTable(
                    page=1, header=header, rows=rows,
                    extractor=OFFICE_PARSER_NAME))
        elif block.kind in ("block_quote", "code_block"):
            text = _block_text(block)
            if text.strip():
                parsed.elements.append(Element("text", text, 1))

    # Footnote/endnote bodies stay retained text (§1.4), tagged by id.
    for note in doc.notes:
        text = "\n".join(filter(None, (_block_text(b) for b in note.blocks)))
        if text.strip():
            parsed.elements.append(Element("text", f"[{note.kind} {note.id}] {text}", 1))

    return parsed
