"""Fallback quality must never substitute a different table on the page."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from rnsr.ingest.fallback import reextract, reextract_pdfplumber
from rnsr.ingest.model import RawTable


def table(*, bbox=None, page=1, header=None, rows=None, **kwargs):
    return RawTable(page=page, header=header or ["Region", "Revenue"],
                    rows=rows or [["North", "15"], ["South", "19"]],
                    bbox=bbox, **kwargs)


def mock_page(monkeypatch, candidates, height=792):
    pdfplumber = pytest.importorskip("pdfplumber")
    found = [SimpleNamespace(bbox=t.bbox, extract=lambda t=t: [t.header, *t.rows])
             for t in candidates]
    page = SimpleNamespace(height=height, find_tables=lambda: found)
    monkeypatch.setattr(pdfplumber, "open", lambda _: nullcontext(SimpleNamespace(pages=[page])))


def test_selects_target_instead_of_larger_neighbor(monkeypatch):
    # Docling bottom-left coordinates identify the first table. The second
    # table repeats its region labels but contains different numeric measures.
    original = table(bbox=(50, 710, 560, 502), caption="Revenue (in millions)")
    revenue = table(bbox=(50, 82, 560, 290))
    profit = table(bbox=(50, 310, 560, 520), header=["Region", "Profit"],
                   rows=[["North", "23"], ["South", "24"], ["Total", "47"]])
    mock_page(monkeypatch, [profit, revenue])
    actual = reextract_pdfplumber("unused.pdf", 1, target=original)
    assert actual is not None
    assert actual.rows == revenue.rows
    assert actual.bbox == revenue.bbox
    assert actual.extractor == "pdfplumber"
    assert actual.caption == original.caption


def test_rejects_pepsico_style_fragments_and_wrong_neighbor(monkeypatch):
    original = table(bbox=(47.73, 710.76, 563.82, 501.92))
    fragment = table(bbox=(49.24, 83.13, 431.25, 288.63), header=["Region"],
                     rows=[["North"], ["South"]])
    right_fragment = table(bbox=(436.5, 83.13, 562.74, 288.63), header=["Revenue"],
                           rows=[["15"], ["19"]])
    neighbor = table(bbox=(49.24, 309.38, 562.74, 517.88))
    mock_page(monkeypatch, [fragment, right_fragment, neighbor])
    assert reextract_pdfplumber("unused.pdf", 1, target=original) is None


@pytest.mark.parametrize("target", [None, table(bbox=(0, 0, 100, 100))])
def test_rejects_ambiguous_candidates(monkeypatch, target):
    mock_page(monkeypatch, [table(bbox=(0, 0, 100, 100)),
                            table(bbox=(1, 1, 101, 101))])
    assert reextract_pdfplumber("unused.pdf", 1, target=target) is None


def test_no_target_accepts_only_unique_usable_table(monkeypatch):
    mock_page(monkeypatch, [table(bbox=(0, 0, 100, 100))])
    assert reextract_pdfplumber("unused.pdf", 1) is not None


def test_content_identity_without_geometry_includes_values(monkeypatch):
    wrong = table(bbox=(0, 100, 100, 200), header=["Region", "Profit"],
                  rows=[["North", "23"], ["South", "24"]])
    right = table(bbox=(0, 0, 100, 100))
    mock_page(monkeypatch, [wrong, right])
    result = reextract_pdfplumber("unused.pdf", 1, target=table())
    assert result is not None and result.rows == right.rows


def test_known_disjoint_geometry_cannot_be_overridden_by_duplicate_text(monkeypatch):
    mock_page(monkeypatch, [table(bbox=(0, 200, 100, 300))])
    assert reextract_pdfplumber("unused.pdf", 1, target=table(bbox=(0, 0, 100, 100))) is None


def test_nested_bbox_can_trim_title_and_empty_phantom_column(monkeypatch):
    rows = [["Miete", "1.234,50"], ["Strom", "800,00"], ["Total", "2.034,50"]]
    original = table(bbox=(160, 764, 434, 613),
                     header=["Nordic GmbH", "-Ledger Konto", "extract Betrag"],
                     rows=[[None, *row] for row in rows])
    clean = table(bbox=(246, 118, 350, 226), header=["Konto", "Betrag"], rows=rows)
    mock_page(monkeypatch, [clean], height=842)
    result = reextract_pdfplumber("unused.pdf", 1, target=original)
    assert result is not None and result.header == clean.header
    assert result.rows == rows


def test_nested_bbox_does_not_drop_even_one_populated_cell(monkeypatch):
    original = table(bbox=(0, 0, 300, 150), header=["Region", "Revenue", "Profit"],
                     rows=[["North", "15", "1"], ["South", "19", "2"]])
    fragment = table(bbox=(0, 0, 100, 150))
    mock_page(monkeypatch, [fragment])
    assert reextract_pdfplumber("unused.pdf", 1, target=original) is None


def test_vision_rejects_different_table_despite_repeated_row_labels():
    original = table(extractor="pdfplumber")
    wrong = table(header=["Region", "Profit"], rows=[["North", "23"], ["South", "24"]])
    assert reextract("unused.pdf", original, vision=lambda *_: wrong) is None


def test_vision_keeps_matching_content_and_sets_rung():
    original = table(extractor="pdfplumber", caption="Revenue (in millions)")
    result = reextract("unused.pdf", original, vision=lambda *_: table())
    assert result is not None and result.extractor == "vision"
    assert result.rows == original.rows
    assert result.caption == original.caption


def test_later_rung_remains_anchored_to_original_target():
    original = table()
    intermediate = table(extractor="pdfplumber", header=["Region", "Profit"],
                          rows=[["North", "23"], ["South", "24"]])
    assert reextract("unused.pdf", intermediate, target=original,
                     vision=lambda *_: intermediate) is None


def test_fallback_cannot_replace_multipage_aggregate(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("a page-local extractor was invoked for a multipage table")

    monkeypatch.setattr("rnsr.ingest.fallback.reextract_pdfplumber", fail)
    assert reextract("unused.pdf", table(row_pages=[1, 2]), vision=fail) is None


def test_vision_cannot_move_evidence_to_different_page():
    assert reextract("unused.pdf", table(extractor="pdfplumber"),
                     vision=lambda *_: table(page=2)) is None


def test_real_pdf_with_two_tables_retains_target_identity(tmp_path):
    pytest.importorskip("pdfplumber")
    pytest.importorskip("reportlab")
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Table, TableStyle

    path = tmp_path / "two-tables.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792))
    grids = [(["Region", "Revenue"], [["North", "15"], ["South", "19"]], 600),
             (["Region", "Profit"], [["North", "23"], ["South", "24"],
                                      ["Total", "47"]], 400)]
    for header, rows, bottom in grids:
        pdf_table = Table([header, *rows], colWidths=[100, 100], rowHeights=20)
        pdf_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, "black")]))
        pdf_table.wrapOn(canvas, 612, 792)
        pdf_table.drawOn(canvas, 50, bottom)
    canvas.save()
    result = reextract_pdfplumber(path, 1, target=table(bbox=(50, 660, 250, 600)))
    assert result is not None
    assert result.header == ["Region", "Revenue"]
    assert result.rows == [["North", "15"], ["South", "19"]]
