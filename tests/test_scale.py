"""Corpus-scale tier: fast parser, resumable bulk ingest, lazy docs,
manifest summarization."""

import sqlite3

import pytest

from rnsr.env.lazydoc import LazyDoc
from rnsr.harness.prompts.base import compact_manifest


@pytest.fixture(scope="module")
def many_pdfs(tmp_path_factory):
    """Twelve small PDFs with known distinct content."""
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    d = tmp_path_factory.mktemp("many")
    styles = getSampleStyleSheet()
    paths = []
    for i in range(12):
        p = d / f"doc_{i:02d}.pdf"
        SimpleDocTemplate(str(p), pagesize=LETTER).build(
            [Paragraph(f"Document number {i}. The marker value is {1000 + i}. "
                       + "Padding sentence for realistic length. " * 30,
                       styles["BodyText"])])
        paths.append(p)
    return paths


class TestFastParse:
    def test_extracts_text_fast(self, many_pdfs):
        from rnsr.ingest.fast_parse import parse_pdf_fast

        parsed = parse_pdf_fast(many_pdfs[3])
        assert parsed.parser == "pdfium-fast"
        text = " ".join(e.text for e in parsed.elements)
        assert "marker value is 1003" in text
        assert parsed.tables == []
        assert parsed.scanned_pages == []

    def test_detects_scanned(self, scanned_pdf):
        from rnsr.ingest.fast_parse import parse_pdf_fast

        parsed = parse_pdf_fast(scanned_pdf)
        assert parsed.scanned_pages == [1]


class TestBulkIngest:
    def test_full_build_and_query(self, many_pdfs, tmp_path):
        from rnsr.db import fts
        from rnsr.db.artifact import CorpusDB
        from rnsr.ingest.bulk import ingest_bulk

        out = tmp_path / "bulk.db"
        stats = ingest_bulk(many_pdfs, out, progress=lambda s: None)
        assert stats["new_docs"] == 12
        with CorpusDB(out) as corpus:
            assert len(corpus.doc_ids()) == 12
            assert fts.match(corpus.conn, "marker value")
        with CorpusDB(out, mode="rw") as corpus, \
                pytest.raises(sqlite3.IntegrityError, match="immutable"):
            # frozen like a single-shot artifact
            corpus.conn.execute("DELETE FROM chunks")

    def test_resume_skips_ingested(self, many_pdfs, tmp_path, monkeypatch):
        from rnsr.ingest import bulk as bulk_mod
        from rnsr.ingest.bulk import ingest_bulk

        out = tmp_path / "resume.db"
        # crash after 5 documents
        calls = {"n": 0}
        real_parse = bulk_mod.parse_pdf_fast

        def exploding(src):
            calls["n"] += 1
            if calls["n"] > 5:
                raise KeyboardInterrupt("simulated crash")
            return real_parse(src)

        with pytest.raises(KeyboardInterrupt):
            ingest_bulk(many_pdfs, out, parse=exploding, commit_every=2)
        assert not out.exists()
        assert out.with_suffix(".db.ingesting").exists()  # resume state kept

        stats = ingest_bulk(many_pdfs, out, progress=lambda s: None)
        assert stats["resumed"] is True
        assert stats["skipped"] >= 4          # committed docs not re-parsed
        assert out.exists()
        from rnsr.db.artifact import CorpusDB

        with CorpusDB(out) as corpus:
            assert len(corpus.doc_ids()) == 12

    def test_second_call_noop(self, many_pdfs, tmp_path):
        from rnsr.ingest.bulk import ingest_bulk

        out = tmp_path / "noop.db"
        ingest_bulk(many_pdfs, out)
        again = ingest_bulk(many_pdfs, out)
        assert again.get("already_complete") is True


class TestLazyDoc:
    def test_mapping_interface_and_lru(self, many_pdfs, tmp_path):
        from rnsr.ingest.bulk import ingest_bulk

        out = tmp_path / "lazy.db"
        ingest_bulk(many_pdfs, out)
        conn = sqlite3.connect(out)
        doc = LazyDoc(conn, cache_size=3)
        assert len(doc) == 12
        first = next(iter(doc))
        assert "marker value" in doc[first]
        # touch more docs than the cache holds; nothing breaks, memory bounded
        for did in doc:
            assert doc[did]
        assert len(doc._cache) <= 3
        assert "db-backed" in repr(doc)
        conn.close()

    def test_verify_works_over_lazydoc(self, many_pdfs, tmp_path):
        from rnsr.env.verify import Verifier
        from rnsr.ingest.bulk import ingest_bulk

        out = tmp_path / "v.db"
        ingest_bulk(many_pdfs, out)
        conn = sqlite3.connect(out)
        v = Verifier(LazyDoc(conn))
        assert v.verify("x", ["The marker value is 1007"])["passed"]
        assert not v.verify("x", ["The marker value is 9999"])["passed"]
        conn.close()


class TestManifestSummarization:
    def test_small_manifest_unchanged(self):
        m = {"documents": [{"doc_id": "a"}], "tables": [{"table_name": "t", "schema": []}]}
        out = compact_manifest(m)
        assert isinstance(out["documents"], list)

    def test_large_manifest_summarized(self):
        m = {"documents": [{"doc_id": f"d{i}"} for i in range(500)],
             "tables": [{"table_name": f"t{i}", "schema": [], "doc_id": f"d{i}"}
                        for i in range(500)]}
        out = compact_manifest(m)
        assert out["documents"]["n_documents"] == 500
        assert "SELECT" in out["documents"]["note"]
        assert out["tables_summary"]["n_tables"] == 500
        assert len(out["tables"]) == 100


class TestParallelIngest:
    def test_parallel_matches_serial(self, many_pdfs, tmp_path):
        from rnsr.db.artifact import CorpusDB
        from rnsr.ingest.bulk import ingest_bulk

        serial = tmp_path / "serial.db"
        parallel = tmp_path / "parallel.db"
        ingest_bulk(many_pdfs, serial, workers=1)
        stats = ingest_bulk(many_pdfs, parallel, workers=4)
        assert stats["new_docs"] == 12
        with CorpusDB(serial) as a, CorpusDB(parallel) as b:
            assert sorted(a.doc_ids()) == sorted(b.doc_ids())
            ta = a.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            tb = b.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            assert ta == tb

    def test_stat_identity_no_content_read(self, many_pdfs):
        from rnsr.ingest.fast_parse import stat_identity

        a = stat_identity(many_pdfs[0])
        b = stat_identity(many_pdfs[0])
        assert a == b and a != stat_identity(many_pdfs[1])
