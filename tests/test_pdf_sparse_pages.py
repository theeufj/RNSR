"""Source-backed sparse-page detection, without layout models or provider calls."""

from pathlib import Path

import pytest

from rnsr.ingest.model import Element, ParsedDocument
from rnsr.ingest.parse import _resolve_sparse_pdf_pages, content_sha256


@pytest.fixture
def sparse_pdf(tmp_path):
    pytest.importorskip("pypdfium2")
    pytest.importorskip("reportlab")
    from PIL import Image, ImageDraw
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    path = tmp_path / "sparse.pdf"
    canvas = Canvas(str(path), pagesize=(600, 800))
    canvas.showPage()  # A genuinely empty page.
    canvas.line(20, 740, 580, 740)
    canvas.drawString(295, 785, "ii")
    canvas.showPage()  # A separator rule and retained page number.
    canvas.drawString(50, 400, "Amount due: $12.50")
    canvas.showPage()  # Short native content is still meaningful.
    image = Image.new("RGB", (110, 18), "white")
    ImageDraw.Draw(image).text((1, 1), "Debt: $901", fill="black")
    canvas.drawImage(ImageReader(image), 50, 400, width=110, height=18)
    canvas.drawString(295, 785, "4")
    canvas.showPage()  # A tiny genuine scan must remain a gap.
    canvas.circle(100, 400, 3, fill=1)
    canvas.showPage()  # Unexplained vector content must remain a gap too.
    canvas.drawString(50, 400, "Native contract clause omitted by layout extraction: fee is $7,391.")
    canvas.showPage()
    canvas.drawImage(ImageReader(Image.new("RGB", (100, 100), "white")),
                     0, 0, width=600, height=800)
    canvas.showPage()  # Even a raster-backed blank page has no missing text.
    canvas.save()
    return path


def empty_layout(path: Path, pages=7):
    return ParsedDocument("sparse", str(path), content_sha256(path), pages, "docling")


def test_sparse_classification_uses_source_content(sparse_pdf):
    parsed = empty_layout(sparse_pdf)
    parsed.elements.append(Element("text", "Existing fragment", 6))
    _resolve_sparse_pdf_pages(parsed, sparse_pdf)

    assert parsed.scanned_pages == [4, 5]
    assert parsed.page_text(1) == ""
    assert parsed.page_text(2) == "ii"
    assert parsed.page_text(3) == "Amount due: $12.50"
    assert "Native contract clause omitted" in parsed.page_text(6)
    assert "$7,391" in parsed.page_text(6)
    assert "Existing fragment" in parsed.page_text(6)
    assert parsed.parser == "docling+pdfium"
    recovered = [element for element in parsed.elements if "$7,391" in element.text]
    assert recovered[0].page == 6 and recovered[0].bbox is not None


def test_recovery_keeps_canonical_page_offsets_and_health(sparse_pdf, tmp_path):
    from rnsr.db.artifact import CorpusDB
    from rnsr.ingest.health import load_health
    from rnsr.ingest.pipeline import ingest

    def parse(path):
        parsed = empty_layout(path)
        _resolve_sparse_pdf_pages(parsed, path)
        return parsed

    database = tmp_path / "source.db"
    report = ingest([sparse_pdf], database, parse=parse)
    assert report.scanned_pages_untranscribed[0]["pages"] == [4, 5]
    with CorpusDB(database) as corpus:
        assert load_health(corpus).grade == "blocked"
        text = corpus.full_text(corpus.doc_ids()[0])
        assert "$12.50" in text and "$7,391" in text
        pages = corpus.conn.execute("SELECT page, text FROM doc_text ORDER BY page").fetchall()
        assert len(pages) == 7 and "$7,391" in pages[5][1]


def test_inspection_failure_never_clears_scan_candidates(sparse_pdf, monkeypatch):
    def broken(*args):
        raise RuntimeError("render unavailable")

    monkeypatch.setattr("rnsr.ingest.parse._has_unretained_marks", broken)
    parsed = empty_layout(sparse_pdf)
    _resolve_sparse_pdf_pages(parsed, sparse_pdf)
    assert parsed.scanned_pages == list(range(1, 8))
    assert "$7,391" in parsed.page_text(6)  # Native recovery is not discarded.


def test_existing_native_text_not_duplicated(sparse_pdf):
    parsed = empty_layout(sparse_pdf)
    parsed.elements.append(Element("text", "Amount due: $12.50", 3))
    _resolve_sparse_pdf_pages(parsed, sparse_pdf)
    assert parsed.page_text(3).count("$12.50") == 1


def test_parse_pdf_repairs_layout_omissions(sparse_pdf, monkeypatch):
    from types import SimpleNamespace

    converter = pytest.importorskip("docling.document_converter")
    from rnsr.ingest.parse import parse_pdf

    # Reproduce a layout model dropping native text, while inspecting the real
    # source PDF for every decision. No model inference/download is necessary.
    document = SimpleNamespace(pages=dict.fromkeys(range(1, 8)), iterate_items=lambda: iter(()))
    monkeypatch.setattr(converter.DocumentConverter, "convert",
                        lambda self, path: SimpleNamespace(document=document))
    parsed = parse_pdf(sparse_pdf)
    assert parsed.scanned_pages == [4, 5]
    assert "$7,391" in parsed.page_text(6)
    assert parsed.sha256 == parsed.content_sha256 == content_sha256(sparse_pdf)


def test_native_text_on_rotated_page_is_retained(tmp_path):
    pytest.importorskip("pypdfium2")
    pytest.importorskip("reportlab")
    from reportlab.pdfgen.canvas import Canvas

    path = tmp_path / "rotated.pdf"
    canvas = Canvas(str(path), pagesize=(600, 800))
    canvas.setPageRotation(90)
    canvas.drawString(40, 200, "Rotated source amount: $17.93")
    canvas.save()
    parsed = empty_layout(path, pages=1)
    _resolve_sparse_pdf_pages(parsed, path)
    assert not parsed.scanned_pages
    assert "$17.93" in parsed.page_text(1)
    assert parsed.elements[0].bbox is not None


def test_thin_header_graphic_is_not_assumed_to_be_a_separator(tmp_path):
    pytest.importorskip("pypdfium2")
    pytest.importorskip("reportlab")
    from reportlab.pdfgen.canvas import Canvas

    path = tmp_path / "thin-graphic.pdf"
    canvas = Canvas(str(path), pagesize=(600, 800))
    graphic = canvas.beginPath()
    graphic.moveTo(30, 760)
    graphic.curveTo(200, 761, 400, 759, 570, 760)
    canvas.drawPath(graphic)
    canvas.save()
    parsed = empty_layout(path, pages=1)
    _resolve_sparse_pdf_pages(parsed, path)
    assert parsed.scanned_pages == [1]
