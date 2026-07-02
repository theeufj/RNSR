"""parse.py against a synthetic PDF. Requires the [ingest] extra (skipped in CI).

First run downloads Docling's layout/TableFormer models.
"""

import pytest

pytest.importorskip("docling")

from rnsr.ingest.parse import make_doc_id, parse_pdf  # noqa: E402


@pytest.fixture(scope="session")
def parsed(fixture_pdf):
    return parse_pdf(fixture_pdf)


def test_doc_identity(parsed, fixture_pdf):
    assert parsed.doc_id == "acme_2023_report"
    assert parsed.parser == "docling"
    assert parsed.n_pages == 1
    assert len(parsed.sha256) == 64


def test_elements_extracted(parsed):
    text = " ".join(e.text for e in parsed.elements)
    assert "Net revenue for fiscal 2023" in text
    kinds = {e.kind for e in parsed.elements}
    assert "heading" in kinds
    assert all(e.page >= 1 for e in parsed.elements)


def test_table_detected_with_grid(parsed):
    assert parsed.tables, "docling should detect the revenue table"
    t = parsed.tables[0]
    norm_header = [h.lower() for h in t.header]
    assert any("segment" in h for h in norm_header)
    flat = [c for row in t.rows for c in row if c]
    assert any("1,234" in c for c in flat)
    assert t.extractor == "docling"


def test_table_text_retained_in_elements(parsed):
    # no-eviction: table content must exist in the canonical text stream too
    table_elements = [e for e in parsed.elements if e.kind == "table"]
    assert table_elements
    assert "1,234" in table_elements[0].text


def test_bboxes_present(parsed):
    with_bbox = [e for e in parsed.elements if e.bbox is not None]
    assert with_bbox, "docling should provide bounding boxes"


def test_make_doc_id():
    from pathlib import Path

    assert make_doc_id(Path("/x/3M 2018 10-K (final).pdf")) == "3m_2018_10_k_final"
