"""Phase A end-to-end: ingest -> corpus.db + validation report.

Two layers: a synthetic-parser test (LLM-free, docling-free, runs in CI)
and a real-Docling test on the fixture PDF (skipped without [ingest]).
"""

import sqlite3

import pytest

from rnsr.db.artifact import CorpusDB
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


def _fake_parse(path):
    """Deterministic stand-in for parse_pdf: prose + one checksum-valid table."""
    return ParsedDocument(
        doc_id="acme",
        source_path=str(path),
        sha256="a" * 64,
        n_pages=2,
        parser="fake",
        elements=[
            Element("heading", "Item 7. Management Discussion", 1, heading_level=1),
            Element("text", "Net revenue for fiscal 2023 was $3,234 million.", 1),
            Element("table", "Segment | Revenue\nWidgets | $1,234", 1),
            Element("text", "Outlook remains strong.", 2),
        ],
        tables=[RawTable(
            page=1,
            header=["Segment", "Revenue ($M)"],
            rows=[["Widgets", "$1,234"], ["Gadgets", "$2,000"], ["Total", "$3,234"]],
            bbox=(10.0, 10.0, 500.0, 200.0),
            extractor="docling",
        )],
    )


def _fake_parse_bad_table(path):
    p = _fake_parse(path)
    p.tables[0].rows[-1][1] = "$9,999"  # corrupt the total
    return p


@pytest.fixture
def artifact(tmp_path):
    out = tmp_path / "corpus.db"
    report = ingest([tmp_path / "acme.pdf"], out, parse=_fake_parse)
    return out, report


class TestIngestSynthetic:
    def test_report_shape(self, artifact):
        _, report = artifact
        assert report.validation_pass_rate == 1.0
        assert report.documents[0]["doc_id"] == "acme"
        assert report.tables[0].status == "trusted"
        assert report.tables[0].confidence >= 0.9
        assert "prose_cross_check (no LLM client)" in report.skipped_stages
        assert report.n_chunks >= 1
        assert '"validation_pass_rate"' in report.to_json()

    def test_artifact_contents(self, artifact):
        out, _ = artifact
        with CorpusDB(out) as corpus:
            assert corpus.doc_ids() == ["acme"]
            full = corpus.full_text("acme")
            assert "Net revenue for fiscal 2023" in full
            assert "$1,234" in full  # table text retained in canonical string
            m = corpus.manifest_dict()
            assert m["documents"][0]["doc_id"] == "acme"
            assert m["tables"][0]["table_name"] == "t_acme_001"
            assert m["tables"][0]["status"] == "trusted"
            assert m["untrusted_tables"] == []
            rows = corpus.conn.execute(
                "SELECT segment, revenue_m, _page FROM t_acme_001 ORDER BY rowid"
            ).fetchall()
            assert tuple(rows[0]) == ("Widgets", 1234, 1)

    def test_source_frozen_annotations_writable(self, artifact):
        out, _ = artifact
        with CorpusDB(out, mode="rw") as corpus:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                corpus.conn.execute("UPDATE t_acme_001 SET revenue_m = 0")
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                corpus.conn.execute("DELETE FROM chunks")
            corpus.conn.execute("ALTER TABLE t_acme_001 ADD COLUMN label")
            corpus.conn.execute("UPDATE t_acme_001 SET label = 'x'")

    def test_fts_queryable(self, artifact):
        out, _ = artifact
        from rnsr.db import fts

        with CorpusDB(out) as corpus:
            hits = fts.match(corpus.conn, "revenue")
            assert hits and hits[0]["doc_id"] == "acme"

    def test_bad_table_flagged_untrusted(self, tmp_path):
        out = tmp_path / "bad.db"
        report = ingest([tmp_path / "acme.pdf"], out, parse=_fake_parse_bad_table)
        t = report.tables[0]
        # pdfplumber re-extraction on a nonexistent PDF yields nothing, so the
        # chain exhausts and the table is flagged — never silently dropped.
        assert t.status == "untrusted"
        assert report.validation_pass_rate == 0.0
        assert len(t.attempts) >= 1
        with CorpusDB(out) as corpus:
            assert corpus.manifest_get("untrusted_tables") == ["t_acme_001"]
            # data still present and queryable despite the flag (§3.3)
            n = corpus.conn.execute("SELECT count(*) FROM t_acme_001").fetchone()[0]
            assert n == 3

    def test_duplicate_doc_ids_disambiguated(self, tmp_path):
        out = tmp_path / "dup.db"
        report = ingest([tmp_path / "a.pdf", tmp_path / "b.pdf"], out, parse=_fake_parse)
        ids = [d["doc_id"] for d in report.documents]
        assert len(set(ids)) == 2


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestIngestDocling:
    def test_real_pdf_end_to_end(self, fixture_pdf, tmp_path):
        pytest.importorskip("docling")
        out = tmp_path / "real.db"
        report = ingest([fixture_pdf], out)
        assert report.documents[0]["doc_id"] == "acme_2023_report"
        assert report.validation_pass_rate >= 0.99, report.to_json()
        assert report.tables, "revenue table should be extracted"
        with CorpusDB(out) as corpus:
            table = report.tables[0].name
            total = corpus.conn.execute(
                f'SELECT MAX(revenue_m) FROM "{table}"'
            ).fetchone()[0]
            assert total == 3234  # numeric needle: exact SQL, no LLM
