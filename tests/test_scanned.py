"""Scanned-PDF support: detection, VLM transcription, visible gaps.

No OCR engine — scanned pages are transcribed by the vision model
(spec §3.1's vision rung applied to whole pages).
"""

import json

import pytest

from rnsr.ingest.llm_hooks import _parse_transcription, make_page_transcriber
from rnsr.llm.mock import MockLLM

TRANSCRIPTION = json.dumps({
    "blocks": [
        {"kind": "heading", "text": "ACME Corporation Annual Report 2023"},
        {"kind": "text", "text": "Net revenue for fiscal 2023 was $3,234 million."},
    ],
    "tables": [{
        "header": ["Segment", "Revenue ($M)"],
        "rows": [["Widgets", "$1,234"], ["Gadgets", "$2,000"], ["Total", "$3,234"]],
    }],
})


class TestParseTranscription:
    def test_valid(self):
        t = _parse_transcription("```json\n" + TRANSCRIPTION + "\n```")
        assert len(t["blocks"]) == 2 and len(t["tables"]) == 1

    def test_blocks_required(self):
        assert _parse_transcription('{"tables": []}') is None
        assert _parse_transcription("no json here") is None

    def test_tables_optional(self):
        t = _parse_transcription('{"blocks": [{"kind": "text", "text": "x"}]}')
        assert t["tables"] == []


class TestTableTranscription:
    def test_table_only_page_retains_canonical_text(self, tmp_path):
        from rnsr.db.artifact import CorpusDB
        from rnsr.ingest.model import ParsedDocument
        from rnsr.ingest.pipeline import ingest

        source = tmp_path / "table-only.pdf"
        source.write_bytes(b"Source is supplied by the deterministic parser fixture")
        parsed = ParsedDocument("table_only", str(source), "a" * 64, 2, "fixture",
                                scanned_pages=[2])
        transcription = {"blocks": [], "tables": [{"header": ["Item", "Amount"],
                                                   "rows": [["Uncommon source label", "$731"]]}]}
        database = tmp_path / "table.db"
        report = ingest([source], database, parse=lambda path: parsed,
                        transcriber=lambda path, pages: {2: transcription})
        assert report.scanned_pages_transcribed == 1
        assert not report.scanned_pages_untranscribed
        with CorpusDB(database) as corpus:
            full_text = corpus.full_text("table_only")
            assert "Uncommon source label | $731" in full_text
            pages = corpus.conn.execute("SELECT page, text FROM doc_text ORDER BY page").fetchall()
            assert pages[0][1] == "\n" and "$731" in pages[1][1]
            assert any("$731" in row[0] for row in corpus.conn.execute("SELECT text FROM chunks"))

    def test_empty_table_is_not_a_successful_transcription(self):
        from rnsr.ingest.model import ParsedDocument
        from rnsr.ingest.transcription import merge_transcriptions

        parsed = ParsedDocument("empty", "empty.pdf", "a" * 64, 1, "fixture",
                                scanned_pages=[1])
        failed = merge_transcriptions(parsed, {1: {"blocks": [], "tables": [
            {"header": [" ", ""], "rows": [[None, " "]]}
        ]}})
        assert failed == [1] and not parsed.elements and not parsed.tables

    def test_table_already_in_blocks_is_not_duplicated(self):
        from rnsr.ingest.model import ParsedDocument
        from rnsr.ingest.transcription import merge_transcriptions

        parsed = ParsedDocument("table", "table.pdf", "a" * 64, 1, "fixture",
                                scanned_pages=[1])
        transcription = {"blocks": [{"text": "Item | Amount\nWidgets | 222"}],
                         "tables": [{"header": ["Item", "Amount"],
                                     "rows": [["Widgets", "222"]]}]}
        assert merge_transcriptions(parsed, {1: transcription}) == []
        assert parsed.page_text(1).count("Widgets | 222") == 1

    def test_header_only_table_text_survives_alongside_prose(self):
        from rnsr.ingest.chunk import chunk_document
        from rnsr.ingest.model import ParsedDocument
        from rnsr.ingest.transcription import merge_transcriptions

        parsed = ParsedDocument("header", "header.pdf", "a" * 64, 2, "fixture",
                                scanned_pages=[2])
        transcription = {"blocks": [{"text": "The signed notice states:"}],
                         "tables": [{"header": ["Amount payable", "$731"], "rows": []}]}
        assert merge_transcriptions(parsed, {2: transcription}) == []
        pages, chunks = chunk_document(parsed)
        assert "The signed notice states:" in pages[1].text
        assert "Amount payable | $731" in pages[1].text
        assert any("Amount payable | $731" in chunk.text for chunk in chunks)
        assert not parsed.tables  # Retained text does not fabricate a data row.


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestScannedIngest:
    def test_detection(self, scanned_pdf):
        pytest.importorskip("docling")
        from rnsr.ingest.parse import parse_pdf

        parsed = parse_pdf(scanned_pdf)
        assert parsed.scanned_pages == [1]

    def test_transcribed_ingest_end_to_end(self, scanned_pdf, tmp_path):
        pytest.importorskip("docling")
        from rnsr.db.artifact import CorpusDB
        from rnsr.ingest.pipeline import ingest

        mock = MockLLM(default=TRANSCRIPTION)
        transcriber = make_page_transcriber(mock, "mock-vision")
        report = ingest([scanned_pdf], tmp_path / "scan.db", transcriber=transcriber)

        assert report.scanned_pages_transcribed == 1
        assert report.scanned_pages_untranscribed == []
        assert report.tables and report.tables[0].extractor == "vision"
        assert report.tables[0].status == "trusted"  # checksum ran on VLM table
        with CorpusDB(tmp_path / "scan.db") as corpus:
            full = corpus.full_text(corpus.doc_ids()[0])
            assert "Net revenue for fiscal 2023" in full
            table = report.tables[0].name
            total = corpus.conn.execute(
                f'SELECT MAX(revenue_m) FROM "{table}"').fetchone()[0]
            assert total == 3234
        # a real page image reached the model
        assert mock.calls and mock.calls[0]["kind"] == "vision"

    def test_without_transcriber_gap_is_visible(self, scanned_pdf, tmp_path):
        pytest.importorskip("docling")
        from rnsr.ingest.pipeline import ingest

        report = ingest([scanned_pdf], tmp_path / "gap.db")
        assert report.scanned_pages_transcribed == 0
        assert report.scanned_pages_untranscribed[0]["pages"] == [1]
        assert "scanned_page_transcription (no LLM client)" in report.skipped_stages
        assert '"scanned_pages_untranscribed"' in report.to_json()

    def test_failed_transcription_recorded(self, scanned_pdf, tmp_path):
        pytest.importorskip("docling")
        from rnsr.ingest.pipeline import ingest

        mock = MockLLM(default="I cannot read this page.")
        transcriber = make_page_transcriber(mock, "mock-vision")
        report = ingest([scanned_pdf], tmp_path / "fail.db", transcriber=transcriber)
        assert report.scanned_pages_transcribed == 0
        assert report.scanned_pages_untranscribed[0]["reason"] == "transcription failed"

    def test_empty_transcription_is_a_visible_gap(self, scanned_pdf, tmp_path):
        pytest.importorskip("docling")
        from rnsr.ingest.pipeline import ingest

        mock = MockLLM(default='{"blocks": [{"kind": "text", "text": "   "}]}')
        transcriber = make_page_transcriber(mock, "mock-vision")
        report = ingest([scanned_pdf], tmp_path / "empty.db", transcriber=transcriber)
        assert report.scanned_pages_transcribed == 0
        assert report.scanned_pages_untranscribed[0]["reason"] == "transcription failed"
