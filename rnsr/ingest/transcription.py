"""Merge page transcriptions into the shared parser representation."""

from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.parse import render_table_text


def merge_transcriptions(parsed: ParsedDocument,
                          transcriptions: dict[int, dict | None]) -> list[int]:
    """Fold VLM page transcriptions into the parsed document; returns pages
    that failed to transcribe. Tables are stamped extractor=vision and flow
    through the normal checksum-validation path (§3.3); their text also stays
    in the canonical element stream, including table-only pages."""
    failed: list[int] = []
    touched: set[int] = set()
    for page in sorted(transcriptions):
        t = transcriptions[page]
        touched.add(page)
        if t is None:
            failed.append(page)
            continue
        before = sum(len((e.text or "").strip())
                     for e in parsed.elements if e.page == page)
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
            if header and any(str(c).strip() for c in header
                              + [c for row in rows for c in row if c is not None]):
                if rows:
                    parsed.tables.append(RawTable(page=page, header=header, rows=rows,
                                                  extractor="vision"))
                text = render_table_text(header, rows)
                if text not in parsed.page_text(page):
                    parsed.elements.append(Element("table", text, page))
        after = sum(len((e.text or "").strip())
                    for e in parsed.elements if e.page == page)
        # Empty / whitespace "success" is a silent gap — same as no transcriber.
        if after <= before:
            failed.append(page)
    for page in parsed.scanned_pages:
        if page not in touched and page not in failed:
            failed.append(page)
    return failed
