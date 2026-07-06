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
