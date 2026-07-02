"""Table re-extraction fallback chain (spec §3.1, §3.3).

Rungs, in order: docling (primary parse, rung 0) → pdfplumber (rung 1) →
vision sub-LM on a rasterized page crop (rung 2, wired in Phase C when the
LLM layer exists). Every extracted table records which rung produced it in
``RawTable.extractor``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rnsr.ingest.model import RawTable

# vision_extract(pdf_path, page) -> RawTable | None; injected in Phase C
VisionExtractor = Callable[[Path, int], RawTable | None]

RUNG_ORDER = ("docling", "pdfplumber", "vision")


def next_rung(current: str) -> str | None:
    try:
        i = RUNG_ORDER.index(current)
    except ValueError:
        return None
    return RUNG_ORDER[i + 1] if i + 1 < len(RUNG_ORDER) else None


def reextract_pdfplumber(pdf_path: str | Path, page: int) -> RawTable | None:
    """Rung 1: pdfplumber table extraction on the given page (1-based).

    Docling bbox coordinates use a different origin than pdfplumber, so we
    extract all tables on the page and take the largest by cell count —
    conservative, and validation decides whether it's better.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if not 1 <= page <= len(pdf.pages):
            return None
        p = pdf.pages[page - 1]
        tables = p.extract_tables()
        if not tables:
            return None
        grid = max(tables, key=lambda t: sum(len(r) for r in t))
        if len(grid) < 2:
            return None
        header = [(c or "").strip() for c in grid[0]]
        rows = [[c.strip() if isinstance(c, str) else c for c in row] for row in grid[1:]]
        bbox = None
        found = p.find_tables()
        if found:
            largest = max(found, key=lambda t: len(t.cells))
            bbox = tuple(float(v) for v in largest.bbox)  # type: ignore[assignment]
        return RawTable(page=page, header=header, rows=rows, bbox=bbox,
                        extractor="pdfplumber")


def reextract(
    pdf_path: str | Path,
    raw: RawTable,
    *,
    vision: VisionExtractor | None = None,
) -> RawTable | None:
    """Produce the next-rung extraction of `raw`, or None if the chain is done.

    The vision rung is skipped when no vision extractor is injected —
    Phase A stays LLM-free (§10) and the table is flagged untrusted instead
    of silently failing (§3.3). A rung that errors (corrupt page, missing
    file) yields nothing and the chain advances to the next rung.
    """
    rung = next_rung(raw.extractor)
    while rung is not None:
        out: RawTable | None = None
        try:
            if rung == "pdfplumber":
                out = reextract_pdfplumber(pdf_path, raw.page)
            elif rung == "vision" and vision is not None:
                out = vision(Path(pdf_path), raw.page)
        except Exception:
            out = None
        if out is not None:
            return out
        rung = next_rung(rung)
    return None
