"""Corpus health: evaluate, persist, gate, unchecked tables."""

import pytest

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.errors import CorpusHealthError
from rnsr.ingest.cost_estimate import estimate_transcription_usd
from rnsr.ingest.health import (
    CorpusHealth,
    enforce_health,
    evaluate,
    load_health,
)
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest
from rnsr.ingest.validate import validate_table
from rnsr.sdk import corpus_env


def _parse_with_totals(path):
    return ParsedDocument(
        doc_id="acme",
        source_path=str(path),
        sha256="a" * 64,
        n_pages=1,
        parser="fake",
        elements=[Element("text", "Revenue was $3,234 million.", 1)],
        tables=[RawTable(
            page=1,
            header=["Segment", "Revenue ($M)"],
            rows=[["Widgets", "$1,234"], ["Gadgets", "$2,000"], ["Total", "$3,234"]],
            extractor="docling",
        )],
    )


def _parse_no_totals(path):
    p = _parse_with_totals(path)
    p.tables[0].rows = [["Widgets", "$1,234"], ["Gadgets", "$2,000"]]
    return p


def _parse_bad_total(path):
    p = _parse_with_totals(path)
    p.tables[0].rows[-1][1] = "$9,999"
    return p


class TestEvaluate:
    def test_ok_when_clean(self):
        h = evaluate({"n_documents": 3, "tables_total": 2})
        assert h.grade == "ok"
        assert h.validation_pass_rate == 1.0
        assert h.findings == []

    def test_untranscribed_blocks_by_default(self):
        h = evaluate({"n_documents": 1, "scanned_pages_untranscribed": 3,
                      "scanned_pages_total": 3})
        assert h.grade == "blocked"
        assert any(f.code == "untranscribed_scans" for f in h.findings)

    def test_low_validation_rate_blocks(self):
        h = evaluate({"n_documents": 1, "tables_total": 10, "tables_untrusted": 5})
        assert h.grade == "blocked"
        assert h.validation_pass_rate == 0.5

    def test_unchecked_excluded_from_pass_rate(self):
        h = evaluate({"n_documents": 1, "tables_total": 10,
                      "tables_untrusted": 0, "tables_unchecked": 8})
        # 2 checked, 0 untrusted -> 100%; unchecked are a warn
        assert h.validation_pass_rate == 1.0
        assert h.grade == "degraded"
        assert any(f.code == "unchecked_tables" for f in h.findings)

    def test_parse_fail_rate_blocks(self):
        h = evaluate({"n_documents": 1, "parse_failed": 1})
        # 1/2 = 50% > 5%
        assert h.grade == "blocked"

    def test_allow_degraded_bypasses_block(self):
        h = evaluate({"n_documents": 1, "scanned_pages_untranscribed": 1})
        assert h.grade == "blocked"
        with pytest.raises(CorpusHealthError):
            enforce_health(h, Settings())
        enforce_health(h, Settings(allow_degraded=True))

    def test_round_trip_dict(self):
        h = evaluate({"n_documents": 2, "tables_total": 1, "tables_untrusted": 1})
        clone = CorpusHealth.from_dict(h.to_dict())
        assert clone.grade == h.grade
        assert clone.tables_untrusted == 1
        assert clone.findings[0].code == h.findings[0].code


class TestUncheckedStatus:
    def test_no_totals_is_unchecked_not_trusted(self, tmp_path):
        out = tmp_path / "c.db"
        report = ingest([tmp_path / "a.pdf"], out, parse=_parse_no_totals)
        assert report.tables[0].status == "unchecked"
        assert report.validation_pass_rate == 1.0  # no checked tables
        with CorpusDB(out) as c:
            health = load_health(c)
            assert health.tables_unchecked == 1
            assert health.source == "ingest"

    def test_totals_stay_trusted(self, tmp_path):
        out = tmp_path / "c.db"
        report = ingest([tmp_path / "a.pdf"], out, parse=_parse_with_totals)
        assert report.tables[0].status == "trusted"
        with CorpusDB(out) as c:
            health = load_health(c)
            assert health.grade == "ok"
            assert health.tables_untrusted == 0

    def test_bad_total_untrusted_blocks_answer(self, tmp_path):
        out = tmp_path / "c.db"
        report = ingest([tmp_path / "a.pdf"], out, parse=_parse_bad_total)
        assert report.tables[0].status == "untrusted"
        assert report.validation_pass_rate == 0.0
        with pytest.raises(CorpusHealthError):
            corpus_env(out)
        env = corpus_env(out, settings=Settings(allow_degraded=True))
        assert env.manifest["health"]["grade"] == "blocked"

    def test_derived_health_on_legacy_artifact(self, tmp_path):
        out = tmp_path / "c.db"
        ingest([tmp_path / "a.pdf"], out, parse=_parse_with_totals)
        with CorpusDB(out, mode="rw") as c:
            c.conn.execute("DELETE FROM manifest WHERE key = 'health'")
            c.conn.commit()
        with CorpusDB(out) as c:
            health = load_health(c)
            assert health.source == "derived"
            assert health.n_documents == 1


class TestValidateEvidence:
    def test_evidence_false_without_arithmetic_or_prose(self):
        v = validate_table(RawTable(
            page=1, header=["Item", "Amount"],
            rows=[["A", "10"], ["B", "20"]], extractor="docling"))
        assert v.checks["arithmetic"].applicable == 0
        assert not v.evidence

    def test_evidence_true_with_totals(self):
        v = validate_table(RawTable(
            page=1, header=["Item", "Amount"],
            rows=[["A", "10"], ["B", "20"], ["Total", "30"]], extractor="docling"))
        assert v.evidence


class TestCostEstimate:
    def test_zero_pages(self):
        assert estimate_transcription_usd(0, "claude-haiku-4-5") == 0.0

    def test_known_model_nonzero(self):
        est = estimate_transcription_usd(10, "claude-haiku-4-5")
        assert est > 0

    def test_unknown_model_zero(self):
        assert estimate_transcription_usd(10, "not-a-real-model") == 0.0


class TestParseFailuresRecorded:
    def test_partial_parse_failure_persists(self, tmp_path):
        n = {"i": 0}

        def parse(path):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("boom")
            return _parse_with_totals(path)

        out = tmp_path / "c.db"
        report = ingest([tmp_path / "a.pdf", tmp_path / "b.pdf"], out, parse=parse)
        assert len(report.parse_failed) == 1
        assert len(report.documents) == 1
        with CorpusDB(out) as c:
            health = load_health(c)
            assert health.parse_failed == 1
            # 1 fail of 2 attempted = 50% > 5% -> blocked
            assert health.grade == "blocked"
