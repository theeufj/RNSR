"""Canonical text assembly + chunking (spec §3.4).

The canonical document text is built once from the parsed elements; every
offset in the system (doc_text pages, chunks, verify() hits) indexes into
that one string, which is what makes all compressed views resolvable back
to retained source text (§1 commitment 4).

Chunking is structure-aware where headings exist (sections bounded by
headings, heading_path recorded); otherwise fixed windows of
``chunk_chars`` with ``overlap``. Oversized sections fall back to windows
internally, keeping their heading_path.
"""

from __future__ import annotations

from dataclasses import dataclass

from rnsr.ingest.model import Element, ParsedDocument


@dataclass
class PageText:
    page: int
    char_start: int
    char_end: int
    text: str


@dataclass
class Chunk:
    page: int                    # page containing the chunk start
    char_start: int
    char_end: int
    heading_path: str | None
    text: str


@dataclass
class _Placed:
    element: Element
    char_start: int
    char_end: int


def _assemble(parsed: ParsedDocument) -> tuple[list[PageText], list[_Placed]]:
    """Concatenate element texts into the canonical string, tracking offsets.

    Elements are separated by a newline; each page's text ends with a
    newline so page boundaries are stable and full_text == concat(pages).
    """
    pages: list[PageText] = []
    placed: list[_Placed] = []
    offset = 0
    max_page = max((e.page for e in parsed.elements), default=0)
    for page_no in range(1, max_page + 1):
        page_start = offset
        parts: list[str] = []
        for e in parsed.elements:
            if e.page != page_no or not e.text:
                continue
            placed.append(_Placed(e, offset, offset + len(e.text)))
            parts.append(e.text)
            offset += len(e.text) + 1  # +1 for the joining newline
        page_text = "\n".join(parts) + "\n" if parts else "\n"
        if not parts:
            offset += 1
        pages.append(PageText(page_no, page_start, page_start + len(page_text), page_text))
    return pages, placed


def _window(text: str, base: int, page_of: list[PageText], heading: str | None,
            chunk_chars: int, overlap: int) -> list[Chunk]:
    """Fixed windows over `text` (positioned at absolute offset `base`)."""
    chunks: list[Chunk] = []
    step = max(chunk_chars - overlap, 1)
    i = 0
    while i < len(text):
        seg = text[i : i + chunk_chars]
        start = base + i
        chunks.append(Chunk(_page_at(page_of, start), start, start + len(seg), heading, seg))
        if i + chunk_chars >= len(text):
            break
        i += step
    return chunks


def _page_at(pages: list[PageText], offset: int) -> int:
    for p in pages:
        if p.char_start <= offset < p.char_end:
            return p.page
    return pages[-1].page if pages else 1


def chunk_document(
    parsed: ParsedDocument, chunk_chars: int = 1500, overlap: int = 200
) -> tuple[list[PageText], list[Chunk]]:
    """Build canonical page texts and chunks for one document."""
    pages, placed = _assemble(parsed)
    full_len = pages[-1].char_end if pages else 0

    headings = [p for p in placed if p.element.kind == "heading"]
    if not headings:
        full = "".join(p.text for p in pages)
        return pages, _window(full, 0, pages, None, chunk_chars, overlap)

    # Sections: [section start, next heading start), with a heading-stack path.
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    full = "".join(p.text for p in pages)

    boundaries = [(h.char_start, h) for h in headings]
    # Preamble before the first heading, if any.
    if boundaries[0][0] > 0:
        chunks += _window(full[: boundaries[0][0]], 0, pages, None, chunk_chars, overlap)

    for i, (start, h) in enumerate(boundaries):
        level = h.element.heading_level or 1
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, h.element.text.strip()))
        path = " > ".join(title for _, title in stack)
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else full_len
        section = full[start:end]
        if section.strip():
            chunks += _window(section, start, pages, path, chunk_chars, overlap)
    return pages, chunks
