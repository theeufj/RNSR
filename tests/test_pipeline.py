"""Phase A end-to-end: ingest -> corpus.db + validation report.

Two layers: a synthetic-parser test (LLM-free, docling-free, runs in CI)
and a real-Docling test on the fixture PDF (skipped without [ingest]).
"""

import sqlite3

import pytest

from rnsr.db.artifact import CorpusDB
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest, ingest_text


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


class TestAtomicity:
    def test_interrupted_ingest_leaves_no_artifact(self, tmp_path):
        def exploding_parse(path):
            raise RuntimeError("parser died mid-run")

        out = tmp_path / "corpus.db"
        with pytest.raises(RuntimeError):
            ingest([tmp_path / "x.pdf"], out, parse=exploding_parse)
        assert not out.exists()
        assert not out.with_suffix(".db.ingesting").exists()

    def test_successful_ingest_renames_into_place(self, tmp_path):
        out = tmp_path / "corpus.db"
        report = ingest([tmp_path / "acme.pdf"], out, parse=_fake_parse)
        assert out.exists()
        assert report.out_db == str(out)
        assert not out.with_suffix(".db.ingesting").exists()


def test_plain_text_total_mentions_preserve_all_lines_and_healthy_corpus(tmp_path):
    text = "First clause.\nSecond clause.\nThird clause.\nTotal payments are due monthly."
    out = tmp_path / "lines.db"
    ingest_text({"agreement": text}, out)
    with CorpusDB(out, mode="rw") as corpus:
        from rnsr.ingest.health import load_health
        assert load_health(corpus).grade == "ok"
        table = corpus.manifest_dict()["tables"][0]["table_name"]
        rows = corpus.conn.execute(f'SELECT line_no, text, _row_kind FROM "{table}"').fetchall()
        assert [tuple(row) for row in rows] == [(i, line, "data")
                                                for i, line in enumerate(text.splitlines(), 1)]
        status, checks = corpus.conn.execute(
            "SELECT status, checks_json FROM manifest_tables").fetchone()
        import json
        assert status == "unchecked"
        assert json.loads(checks)["arithmetic"]["applicable"] == 0
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            corpus.conn.execute(f'UPDATE "{table}" SET text="changed"')


def test_malformed_ledger_forces_fallback_even_when_prose_passes(tmp_path, monkeypatch):
    from rnsr.config import Settings
    from rnsr.ingest.pipeline import _extract_best_table

    clean = RawTable(page=1, header=["Konto", "Betrag"], extractor="pdfplumber",
                     rows=[["Rent", "1234.5"], ["Power", "800"], ["Subtotal", "2034.5"],
                           ["Adjustment", "-34.5"], ["Total", "2000"]])
    bad = RawTable(page=1, header=["NORDIC GmbH", "Ledger Konto", "Extract Betrag"],
                   rows=[[None, *row] for row in clean.rows])
    calls = []

    def alternate(path, current, *, target, vision):
        calls.append(target)
        return clean

    monkeypatch.setattr("rnsr.ingest.pipeline.reextract", alternate)
    chosen, validation, status, attempts = _extract_best_table(
        tmp_path / "ledger.pdf", bad, Settings(), lambda ps: [True] * len(ps), None, {})
    assert calls == [bad]
    assert chosen.header == ["Konto", "Betrag"]
    assert not validation.structural_errors and status == "reextracted"
    assert len(attempts) == 2


def test_multipage_failed_total_cannot_be_replaced_by_one_page(tmp_path, monkeypatch):
    from rnsr.config import Settings
    from rnsr.ingest.pipeline import _extract_best_table

    original = RawTable(page=1, header=["Item", "Amount"],
                        rows=[["A", "10"], ["B", "20"], ["Total", "999"]],
                        row_pages=[1, 2, 2])

    def forbidden(*args, **kwargs):
        raise AssertionError("A single-page fallback must not replace a multipage grid")

    monkeypatch.setattr("rnsr.ingest.pipeline.reextract", forbidden)
    chosen, _, status, attempts = _extract_best_table(
        tmp_path / "multi.pdf", original, Settings(), None, None, {})
    assert chosen is original and len(chosen.rows) == 3
    assert status == "untrusted" and len(attempts) == 1


def test_reextracted_grid_controls_stored_schema_coercion_and_provenance(tmp_path, monkeypatch):
    import json

    original = RawTable(page=1, header=["Phantom title", "Label", "Value"],
                        rows=[[None, "A", "1.234,50"], [None, "B", "800,00"],
                              [None, "Total", "2.034,50"]], bbox=(1, 2, 30, 40))
    replacement = RawTable(page=1, header=["Account name", "Amount (EUR)"],
                           rows=[row[1:] for row in original.rows],
                           bbox=(5, 6, 20, 30), extractor="pdfplumber")
    monkeypatch.setattr("rnsr.ingest.pipeline.reextract",
                        lambda *args, **kwargs: replacement)
    parsed = ParsedDocument(doc_id="ledger", source_path="ledger.pdf", sha256="a" * 64,
                            parser="test", n_pages=1, tables=[original],
                            elements=[Element("text", "Ledger", 1)])
    out = tmp_path / "ledger.db"
    report = ingest([tmp_path / "ledger.pdf"], out, parse=lambda _: parsed)
    assert report.tables[0].status == "reextracted"
    with CorpusDB(out) as corpus:
        meta = corpus.manifest_dict()["tables"][0]
        assert [c["name"] for c in meta["schema"]] == ["account_name", "amount_eur"]
        numeric = meta["schema"][1]
        assert numeric["raw_col"] == "amount_eur__raw"
        assert numeric["coercion_rule"]["style"] == "eu"
        rows = corpus.conn.execute(
            "SELECT amount_eur, amount_eur__raw, _page, _bbox, _extractor, _row_kind "
            "FROM t_ledger_001 ORDER BY rowid").fetchall()
        assert rows[0][0:3] == (1234.5, "1.234,50", 1)
        assert json.loads(rows[0][3]) == [5, 6, 20, 30]
        assert rows[0][4:] == ("pdfplumber", "data")
        assert rows[-1][-1] == "total"
