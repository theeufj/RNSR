"""Table re-extraction fallback chain (spec §3.1, §3.3).

Rungs, in order: docling (primary parse, rung 0) → pdfplumber (rung 1) →
vision sub-LM on a rasterized page (rung 2). Every replacement must still
identify the original table; extraction quality alone cannot establish
identity on a page containing several tables.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from rnsr.ingest.coerce import coerce_column
from rnsr.ingest.model import BBox, RawTable

# vision_extract(pdf_path, page) -> RawTable | None; injected in Phase C
VisionExtractor = Callable[[Path, int], RawTable | None]

RUNG_ORDER = ("docling", "pdfplumber", "vision")
_MIN_BOX_COVERAGE = 0.85
_MIN_CONTENT_COVERAGE = 0.70


def next_rung(current: str) -> str | None:
    try:
        i = RUNG_ORDER.index(current)
    except ValueError:
        return None
    return RUNG_ORDER[i + 1] if i + 1 < len(RUNG_ORDER) else None


def _single_page(table: RawTable) -> bool:
    return not table.row_pages or all(p == table.page for p in table.row_pages)


def _top_left_box(bbox: BBox | None, height: float) -> BBox | None:
    if bbox is None or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        return None
    x0, y0, x1, y1 = bbox
    # Docling uses (left, top, right, bottom) in bottom-left coordinates;
    # pdfplumber uses the same edges measured from the top of the page.
    if y0 > y1:
        y0, y1 = height - y0, height - y1
    if x0 >= x1 or y0 >= y1 or y0 < 0 or y1 > height:
        return None
    return x0, y0, x1, y1


def _box_coverage(a: BBox, b: BBox) -> tuple[float, float]:
    intersection = (max(0., min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0., min(a[3], b[3]) - max(a[1], b[1])))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / area_a, intersection / area_b


def _body(table: RawTable) -> list[tuple[str, ...]]:
    rows = []
    for row in table.rows:
        values = []
        for cell in row:
            text = " ".join(unicodedata.normalize("NFKC", cell or "").casefold().split())
            if text:
                values.append(text)
        rows.append(tuple(values))
    return rows


def _cells(table: RawTable) -> Counter[str]:
    return Counter(cell for row in _body(table) for cell in row
                   if any(c.isalnum() for c in cell))


def _header_tokens(table: RawTable) -> set[str]:
    return set(re.findall(r"[^\W_]+", unicodedata.normalize(
        "NFKC", " ".join(table.header)).casefold()))


def _same_content(target: RawTable, candidate: RawTable) -> bool:
    """Conservative identity check for extractors without table geometry.

    Count complete cells (including numeric values), not just row labels:
    financial tables on the same page often repeat exactly the same labels.
    """
    original, replacement = _cells(target), _cells(candidate)
    n_original, n_replacement = original.total(), replacement.total()
    shared = (original & replacement).total()
    if min(n_original, n_replacement) < 2 or shared < min(3, n_original):
        return False
    if min(shared / n_original, shared / n_replacement) < _MIN_CONTENT_COVERAGE:
        return False
    original_header, replacement_header = _header_tokens(target), _header_tokens(candidate)
    return bool(original_header & replacement_header) and (
        len(original_header & replacement_header)
        / min(len(original_header), len(replacement_header)) >= 0.5)


def reextract_pdfplumber(
    pdf_path: str | Path, page: int, *, target: RawTable | None = None,
) -> RawTable | None:
    """Return a unique extraction matching the target on this 1-based page.

    Geometry must cover both tables. Nested boxes can also match when every
    ordered body cell survives: a malformed source box may include a title
    and empty columns. Without usable geometry, require substantial shared
    cell content. A caller without a target needs an unambiguous one-table page.
    """
    if target is not None and (target.page != page or not _single_page(target)):
        return None
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if not 1 <= page <= len(pdf.pages):
            return None
        p = pdf.pages[page - 1]
        target_box = _top_left_box(target.bbox, p.height) if target else None
        matches = []
        for found in p.find_tables():
            grid = found.extract()
            if not grid:
                continue
            header = [(c or "").strip() for c in grid[0]]
            rows = [[c.strip() if isinstance(c, str) else c for c in row] for row in grid[1:]]
            # pdfplumber has no header flags. A numeric observation matching
            # the body is data, not a column name. Bare years remain valid
            # financial period headers. Target metadata can establish a
            # headerless grid even when it contains text-only observations.
            generated = [f"column_{i + 1}" for i in range(len(header))]
            headerless = target is not None and target.header == generated
            headerless = headerless or any(
                value and not re.fullmatch(r"(?:19|20)\d{2}", value)
                and coerce_column([value]).is_numeric
                and (not rows or coerce_column(
                    [row[i] if i < len(row) else None for row in rows]).is_numeric)
                for i, value in enumerate(header)
            )
            if headerless:
                rows.insert(0, list(header))
                header = generated
            elif not rows:
                continue
            bbox = tuple(float(v) for v in found.bbox)
            candidate = RawTable(page=page, header=header, rows=rows, bbox=bbox,
                                 extractor="pdfplumber")
            candidate_box = _top_left_box(candidate.bbox, p.height)
            if target is not None:
                if target_box is not None and candidate_box is not None:
                    coverage = _box_coverage(target_box, candidate_box)
                    # Never substitute a partial grid based only on overlap.
                    # Exact row contents permit trimming an oversized source
                    # bbox/empty phantom column, without discarding source cells.
                    same_body = (max(coverage) >= _MIN_BOX_COVERAGE
                                 and _body(target) == _body(candidate)
                                 and _same_content(target, candidate))
                    if min(coverage) < _MIN_BOX_COVERAGE and not same_body:
                        continue
                elif not _same_content(target, candidate):
                    continue
            if target is not None and candidate.caption is None:
                candidate = replace(candidate, caption=target.caption)
            matches.append(candidate)
        return matches[0] if len(matches) == 1 else None


def reextract(
    pdf_path: str | Path,
    raw: RawTable,
    *,
    vision: VisionExtractor | None = None,
    target: RawTable | None = None,
) -> RawTable | None:
    """Produce the next-rung extraction of `raw`, or None if the chain is done.

    The vision rung is skipped when no vision extractor is injected —
    Phase A stays LLM-free (§10) and the table is flagged untrusted instead
    of silently failing (§3.3). A rung that errors (corrupt page, missing
    file) yields nothing and the chain advances to the next rung.
    """
    target = target if target is not None else raw
    if not _single_page(target) or not _single_page(raw) or raw.page != target.page:
        return None
    rung = next_rung(raw.extractor)
    while rung is not None:
        out: RawTable | None = None
        try:
            if rung == "pdfplumber":
                out = reextract_pdfplumber(pdf_path, raw.page, target=target)
            elif rung == "vision" and vision is not None:
                out = vision(Path(pdf_path), raw.page)
                # The vision hook receives a whole page and supplies no reliable
                # table geometry. Its output must independently retain identity.
                if (out is not None and out.page == target.page and _single_page(out)
                        and _same_content(target, out)):
                    out = replace(out, extractor="vision", caption=out.caption or target.caption)
                else:
                    out = None
        except Exception:
            out = None
        if out is not None:
            return out
        rung = next_rung(rung)
    return None
