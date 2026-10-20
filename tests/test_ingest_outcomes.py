"""Coverage survives checkpoints and changes with the owning source generation."""
from __future__ import annotations

from pathlib import Path

import pytest

from rnsr.db.artifact import CorpusDB
from rnsr.db.schema import ARTIFACT_FORMAT_VERSION, validate_frozen, validate_integrity
from rnsr.ingest.bulk import ingest_bulk
from rnsr.ingest.health import load_health
from rnsr.ingest.lifecycle import append, replace_document
from rnsr.ingest.model import Element, ParsedDocument
from rnsr.ingest.outcomes import failures
from rnsr.ingest.pipeline import ingest


def _parsed(path, *, scanned=False, attachments=False):
    return ParsedDocument(
        Path(path).stem, str(path), ("a" if scanned else "b") * 64, 1, "test",
        elements=[] if scanned else [Element("text", "Readable source content", 1)],
        scanned_pages=[1] if scanned else [],
        pending_attachments=[("broken.pdf", b"broken")] if attachments else [])


def test_failed_attachment_is_reported_and_persisted(tmp_path):
    source = tmp_path / "mail.eml"

    def parse(path):
        if path == source:
            return _parsed(path, attachments=True)
        raise ValueError("invalid PDF attachment")

    output = tmp_path / "corpus.db"
    report = ingest(source, output, parse=parse)
    assert len(report.documents) == 1
    assert report.parse_failed == [{
        "source": str(source) + "::broken.pdf", "parent_doc_id": "mail",
        "error": "ValueError: invalid PDF attachment"}]
    with CorpusDB(output) as corpus:
        assert failures(corpus.conn) == report.parse_failed
        health = load_health(corpus)
        assert health.parse_failed == 1 and health.grade == "blocked"
        validate_frozen(corpus.conn)
        validate_integrity(corpus.conn)


def test_bulk_resume_keeps_previously_committed_scan_gap(tmp_path):
    sources = [tmp_path / "scanned.pdf", tmp_path / "good.txt"]
    for source in sources:
        source.write_text("source bytes")

    def parse(path):
        return _parsed(path, scanned=path.name == "scanned.pdf")

    def stop_after_checkpoint(message):
        if message == "ingested 1 new documents":
            raise KeyboardInterrupt("simulate interrupted ingestion")

    output = tmp_path / "corpus.db"
    with pytest.raises(KeyboardInterrupt):
        ingest_bulk(sources, output, parse=parse, workers=1, commit_every=1,
                    progress=stop_after_checkpoint)
    result = ingest_bulk(sources, output, parse=parse, workers=1, commit_every=1)
    assert result["skipped"] == 1
    assert result["scanned_pages_untranscribed"] == 1
    with CorpusDB(output) as corpus:
        health = load_health(corpus)
        assert health.n_documents == 2
        assert health.scanned_pages_total == health.scanned_pages_untranscribed == 1
        assert health.grade == "blocked"
        assert corpus.full_text("scanned") == "\n"
        validate_integrity(corpus.conn)


def test_bulk_attachment_failure_remains_visible_on_cached_run(tmp_path):
    source = tmp_path / "mail.eml"
    source.write_text("email source")

    def parse(path):
        if path == source:
            return _parsed(path, attachments=True)
        raise ValueError("invalid attachment")

    output = tmp_path / "corpus.db"
    for result in (ingest_bulk([source], output, parse=parse, workers=1),
                   ingest_bulk([source], output, parse=parse, workers=1)):
        assert result["parse_failed"] == 1
        assert result["parse_failures"] == [{
            "source": str(source) + "::broken.pdf", "parent_doc_id": "mail",
            "error": "ValueError: invalid attachment"}]


def test_successful_transcription_persists_coverage_without_gap(tmp_path):
    output = tmp_path / "corpus.db"
    ingest(tmp_path / "scanned.pdf", output,
           parse=lambda path: _parsed(path, scanned=True),
           transcriber=lambda source, pages: {1: {"blocks": [{"text": "Transcribed source"}]}})
    with CorpusDB(output) as corpus:
        health = load_health(corpus)
        assert health.scanned_pages_total == 1
        assert health.scanned_pages_untranscribed == 0
        assert health.grade == "ok"


def test_replacement_removes_obsolete_scan_and_attachment_gaps(tmp_path):
    source = tmp_path / "mail.eml"

    def initial_parse(path):
        if path == source:
            return _parsed(path, scanned=True, attachments=True)
        raise ValueError("broken attachment")

    output = tmp_path / "corpus.db"
    ingest(source, output, parse=initial_parse)
    replace_document(output, "mail", source, parse=_parsed)
    with CorpusDB(output) as corpus:
        health = load_health(corpus)
        assert health.grade == "ok"
        assert health.parse_failed == health.scanned_pages_total == health.scanned_pages_untranscribed == 0
        assert failures(corpus.conn) == []
        assert corpus.conn.execute("SELECT count(*) FROM ingest_outcomes").fetchone()[0] == 1
        validate_frozen(corpus.conn)
        validate_integrity(corpus.conn)


def test_append_preserves_existing_document_gap(tmp_path):
    output = tmp_path / "corpus.db"
    source = tmp_path / "scanned.pdf"
    ingest(source, output, parse=lambda path: _parsed(path, scanned=True))
    append(tmp_path / "good.txt", output, parse=_parsed)
    with CorpusDB(output) as corpus:
        health = load_health(corpus)
        assert health.scanned_pages_untranscribed == 1
        assert health.n_documents == 2 and health.grade == "blocked"


def test_failed_source_retry_resolves_failure_without_erasing_other_gaps(tmp_path):
    output = tmp_path / "corpus.db"
    source = tmp_path / "retry.txt"

    def parse(path):
        if path == source:
            raise ValueError("temporary parse error")
        return _parsed(path, scanned=True)

    ingest([source, tmp_path / "scanned.pdf"], output, parse=parse)
    append(source, output, parse=_parsed)
    with CorpusDB(output) as corpus:
        health = load_health(corpus)
        assert health.parse_failed == 0
        assert health.scanned_pages_untranscribed == 1
        assert health.grade == "blocked"


def test_legacy_v3_aggregate_health_remains_conservative(tmp_path):
    output = tmp_path / "legacy.db"
    ingest(tmp_path / "scanned.pdf", output,
           parse=lambda path: _parsed(path, scanned=True))
    with CorpusDB(output, mode="rw") as corpus:
        corpus.conn.execute("DROP TABLE ingest_outcomes")
        corpus.conn.execute("DELETE FROM manifest WHERE key='ingest_health_legacy'")
        corpus.conn.commit()
    # Older v3 artifacts are readable without a migration; extending one
    # retains its unallocatable aggregate gaps rather than declaring them clear.
    append(tmp_path / "good.txt", output, parse=_parsed)
    with CorpusDB(output) as corpus:
        assert corpus.conn.execute("PRAGMA user_version").fetchone()[0] == ARTIFACT_FORMAT_VERSION == 3
        assert load_health(corpus).scanned_pages_untranscribed == 1
        assert load_health(corpus).grade == "blocked"
        validate_frozen(corpus.conn)
