"""chunk.py: canonical text assembly, offset integrity, heading-aware chunking."""

from rnsr.ingest.chunk import chunk_document
from rnsr.ingest.model import Element, ParsedDocument


def _doc(elements):
    return ParsedDocument(
        doc_id="d", source_path="x.pdf", sha256="0" * 64, n_pages=max(e.page for e in elements),
        parser="test", elements=elements,
    )


def test_offsets_resolve_into_canonical_text():
    parsed = _doc([
        Element("text", "First paragraph on page one.", 1),
        Element("text", "Second paragraph.", 1),
        Element("text", "Page two content here.", 2),
    ])
    pages, chunks = chunk_document(parsed)
    full = "".join(p.text for p in pages)
    for c in chunks:
        assert full[c.char_start : c.char_end] == c.text
    for p in pages:
        assert full[p.char_start : p.char_end] == p.text


def test_no_headings_fixed_windows_with_overlap():
    text = "abcdefghij" * 400  # 4000 chars
    parsed = _doc([Element("text", text, 1)])
    pages, chunks = chunk_document(parsed, chunk_chars=1500, overlap=200)
    assert len(chunks) == 3
    assert chunks[0].char_start == 0
    # consecutive windows overlap by 200
    assert chunks[1].char_start == 1300
    assert chunks[0].text[-200:] == chunks[1].text[:200]
    assert all(c.heading_path is None for c in chunks)


def test_heading_sections_and_paths():
    parsed = _doc([
        Element("heading", "Item 7", 1, heading_level=1),
        Element("text", "Liquidity intro.", 1),
        Element("heading", "Liquidity", 1, heading_level=2),
        Element("text", "Cash was strong.", 1),
        Element("heading", "Item 8", 2, heading_level=1),
        Element("text", "Financial statements.", 2),
    ])
    _, chunks = chunk_document(parsed)
    paths = [c.heading_path for c in chunks]
    assert "Item 7" in paths
    assert "Item 7 > Liquidity" in paths
    assert "Item 8" in paths  # stack popped back to level 1
    liq = next(c for c in chunks if c.heading_path == "Item 7 > Liquidity")
    assert "Cash was strong." in liq.text


def test_preamble_before_first_heading():
    parsed = _doc([
        Element("text", "Cover page text.", 1),
        Element("heading", "Introduction", 1, heading_level=1),
        Element("text", "Body.", 1),
    ])
    _, chunks = chunk_document(parsed)
    assert chunks[0].heading_path is None
    assert "Cover page text." in chunks[0].text


def test_oversized_section_windows_keep_path():
    parsed = _doc([
        Element("heading", "Big Section", 1, heading_level=1),
        Element("text", "x" * 4000, 1),
    ])
    _, chunks = chunk_document(parsed, chunk_chars=1500, overlap=200)
    assert len(chunks) >= 3
    assert all(c.heading_path == "Big Section" for c in chunks)


def test_chunk_page_assignment():
    parsed = _doc([
        Element("text", "Page one.", 1),
        Element("text", "Page two.", 2),
    ])
    pages, chunks = chunk_document(parsed)
    assert chunks[0].page == 1
    full = "".join(p.text for p in pages)
    assert full[pages[1].char_start : pages[1].char_end] == "Page two.\n"


def test_empty_page_preserved_in_offsets():
    parsed = _doc([
        Element("text", "Page one.", 1),
        Element("text", "Page three.", 3),  # page 2 has no elements
    ])
    pages, chunks = chunk_document(parsed)
    assert [p.page for p in pages] == [1, 2, 3]
    full = "".join(p.text for p in pages)
    assert full[pages[2].char_start : pages[2].char_end] == "Page three.\n"
    for c in chunks:
        assert full[c.char_start : c.char_end] == c.text
