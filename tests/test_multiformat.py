"""Multi-format ingest: office documents (anydoc), markdown, text, email.

Office tests skip without firecrawl-anydoc (the [ingest] extra); the
text-like parsers are stdlib-only and always run. The docx fixture is a
minimal hand-built OOXML package — no Word required.
"""

import sqlite3
import zipfile

import pytest

from rnsr.db import fts
from rnsr.ingest.dispatch import is_ingestable, parse_any, parse_any_fast
from rnsr.ingest.fast_parse import stat_identity
from rnsr.ingest.pipeline import ingest
from rnsr.ingest.textlike import parse_eml, parse_markdown, parse_text

# --- fixtures ----------------------------------------------------------------

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DOCX_PARTS = {
    "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>""",
    "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>""",
    "word/_rels/document.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
    "word/styles.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{_W}">
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
</w:styles>""",
    "word/document.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{_W}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Fiscal Results</w:t></w:r></w:p>
<w:p><w:r><w:t>Revenue was strong this quarter.</w:t></w:r></w:p>
<w:tbl>
<w:tr><w:tc><w:p><w:r><w:t>Segment</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Revenue</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Widgets</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>1234</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Gadgets</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>2000</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>
</w:body></w:document>""",
}


def make_docx(path):
    with zipfile.ZipFile(path, "w") as z:
        for name, content in _DOCX_PARTS.items():
            z.writestr(name, content)
    return path


def make_eml(path, *, html=False, attachment=False):
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Settlement offer"
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Date"] = "Tue, 04 Aug 2026 10:00:00 +1000"
    if html:
        msg.set_content("plain fallback")
        msg.add_alternative(
            "<html><head><style>p{color:red}</style></head>"
            "<body><p>We offer <b>$50,000</b> to settle.</p>"
            "<script>alert('x')</script></body></html>",
            subtype="html")
    else:
        msg.set_content("We offer $50,000 to settle.\n\nPlease respond by Friday.")
    if attachment:
        msg.add_attachment(b"%PDF-fake", maintype="application",
                           subtype="pdf", filename="deed.pdf")
    path.write_bytes(bytes(msg))
    return path


# --- office (anydoc) ---------------------------------------------------------


class TestOfficeParser:
    def test_csv_becomes_data_table(self, tmp_path):
        pytest.importorskip("anydoc")
        from rnsr.ingest.office import parse_office

        f = tmp_path / "sales.csv"
        f.write_text("segment,revenue\nwidgets,100\ngadgets,200\n")
        parsed = parse_office(f)
        assert parsed.parser == "anydoc"
        assert parsed.doc_id == "sales"
        assert len(parsed.tables) == 1
        t = parsed.tables[0]
        assert t.extractor == "anydoc"
        assert t.header == ["segment", "revenue"]
        assert t.rows == [["widgets", "100"], ["gadgets", "200"]]
        # the table text is retained in the element stream (§1.4)
        assert any(e.kind == "table" and "widgets" in e.text for e in parsed.elements)

    def test_docx_structure(self, tmp_path):
        pytest.importorskip("anydoc")
        from rnsr.ingest.office import parse_office

        parsed = parse_office(make_docx(tmp_path / "report.docx"))
        kinds = [(e.kind, e.text) for e in parsed.elements]
        assert ("heading", "Fiscal Results") in kinds
        heading = next(e for e in parsed.elements if e.kind == "heading")
        assert heading.heading_level == 1
        assert any(e.kind == "text" and "strong this quarter" in e.text
                   for e in parsed.elements)
        assert len(parsed.tables) == 1
        assert parsed.tables[0].header == ["Segment", "Revenue"]
        assert parsed.tables[0].rows == [["Widgets", "1234"], ["Gadgets", "2000"]]

    def test_rtf_text(self, tmp_path):
        pytest.importorskip("anydoc")
        from rnsr.ingest.office import parse_office

        f = tmp_path / "memo.rtf"
        f.write_bytes(rb"{\rtf1\ansi Quarterly revenue was \b 300\b0 .\par}")
        parsed = parse_office(f)
        assert any("Quarterly revenue was" in e.text for e in parsed.elements)


# --- text-like (stdlib) ------------------------------------------------------


class TestTextlikeParsers:
    def test_text_paragraphs(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("First paragraph\nstill first.\n\nSecond paragraph.\n")
        parsed = parse_text(f)
        assert parsed.parser == "text"
        assert [e.text for e in parsed.elements] == [
            "First paragraph\nstill first.", "Second paragraph."]

    def test_markdown_structure(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(
            "# Title\n\nIntro para.\n\n## Numbers\n\n"
            "| item | qty |\n| --- | --- |\n| apple | 3 |\n| pear | 5 |\n\n"
            "```python\nx = 1\n```\nTail para.\n")
        parsed = parse_markdown(f)
        headings = [(e.text, e.heading_level) for e in parsed.elements
                    if e.kind == "heading"]
        assert headings == [("Title", 1), ("Numbers", 2)]
        assert len(parsed.tables) == 1
        assert parsed.tables[0].header == ["item", "qty"]
        assert parsed.tables[0].rows == [["apple", "3"], ["pear", "5"]]
        texts = [e.text for e in parsed.elements if e.kind == "text"]
        assert "Intro para." in texts
        assert any("x = 1" in t for t in texts)          # fenced code retained
        assert "Tail para." in texts

    def test_eml_plain(self, tmp_path):
        parsed = parse_eml(make_eml(tmp_path / "mail.eml"))
        assert parsed.parser == "eml"
        assert parsed.elements[0].kind == "heading"
        assert parsed.elements[0].text == "Settlement offer"
        texts = [e.text for e in parsed.elements]
        assert "From: alice@example.com" in texts
        assert "We offer $50,000 to settle." in texts

    def test_eml_html_stripped_and_attachment_named(self, tmp_path):
        parsed = parse_eml(make_eml(tmp_path / "mail.eml", html=True,
                                    attachment=True))
        texts = [e.text for e in parsed.elements]
        # html body wins over the plain fallback only when preferred; plain
        # is preferred, so the fallback body is used
        assert any("plain fallback" in t for t in texts)
        assert "[attachment: deed.pdf]" in texts

    def test_eml_html_only_body(self, tmp_path):
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "s"
        msg.set_content(
            "<html><body><p>Hello <b>there</b>.</p>"
            "<script>bad()</script></body></html>", subtype="html")
        f = tmp_path / "h.eml"
        f.write_bytes(bytes(msg))
        parsed = parse_eml(f)
        texts = " ".join(e.text for e in parsed.elements)
        assert "Hello there" in texts
        assert "bad()" not in texts


# --- dispatch ----------------------------------------------------------------


class TestDispatch:
    def test_is_ingestable(self):
        for name in ("a.pdf", "a.docx", "a.XLSX", "a.md", "a.eml", "a.txt", "a.csv"):
            assert is_ingestable(name), name
        for name in ("a.exe", "a.png", "a.db", "a"):
            assert not is_ingestable(name), name

    def test_routes_by_extension(self, tmp_path):
        f = tmp_path / "n.txt"
        f.write_text("hello")
        assert parse_any(f).parser == "text"

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "n.xyz"
        f.write_text("hello")
        with pytest.raises(ValueError, match="unsupported document type"):
            parse_any(f)

    def test_fast_tier_uses_stat_identity(self, tmp_path):
        f = tmp_path / "n.txt"
        f.write_text("hello")
        assert parse_any_fast(f).sha256 == stat_identity(f)
        assert parse_any(f).sha256 != stat_identity(f)  # quality: content hash


# --- end-to-end --------------------------------------------------------------


def _mixed_corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "sales.csv").write_text("segment,revenue\nwidgets,100\ngadgets,200\n")
    (d / "notes.md").write_text("# Findings\n\nThe deadline is 30 June 2026.\n")
    (d / "memo.txt").write_text("The password is xylophone.\n")
    make_eml(d / "offer.eml")
    make_docx(d / "report.docx")
    return d


class TestMixedIngest:
    def test_pipeline_end_to_end(self, tmp_path):
        pytest.importorskip("anydoc")
        d = _mixed_corpus(tmp_path)
        out = tmp_path / "corpus.db"
        report = ingest(sorted(d.iterdir()), out)
        assert out.exists()
        assert len(report.documents) == 5

        conn = sqlite3.connect(out)
        try:
            # every format's content is reachable through FTS (rung 2)
            for token in ("widgets", "deadline", "xylophone", "settle", "fiscal"):
                assert fts.match(conn, token), f"FTS miss: {token}"
            # the CSV became a typed, queryable data table
            (name,) = conn.execute(
                "SELECT table_name FROM manifest_tables WHERE doc_id='sales'"
            ).fetchone()
            total = conn.execute(f'SELECT sum(revenue) FROM "{name}"').fetchone()[0]
            assert total == 300
        finally:
            conn.close()

    def test_bulk_mixed(self, tmp_path):
        pytest.importorskip("anydoc")
        from rnsr.ingest.bulk import ingest_bulk

        d = _mixed_corpus(tmp_path)
        out = tmp_path / "bulk.db"
        stats = ingest_bulk(sorted(d.iterdir()), out, workers=1)
        assert stats["new_docs"] == 5
        assert stats["parse_failed"] == 0

        conn = sqlite3.connect(out)
        try:
            n = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            assert n == 5
            # resume identity is stat-based for every format
            shas = {r[0] for r in conn.execute("SELECT sha256 FROM documents")}
            assert stat_identity(d / "memo.txt") in shas
        finally:
            conn.close()
