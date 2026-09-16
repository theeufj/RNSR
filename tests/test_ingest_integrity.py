"""Regression coverage for data preservation and artifact metadata guarantees."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from rnsr.config import Settings
from rnsr.db import fts, schema
from rnsr.db.artifact import CorpusDB
from rnsr.db.metadata import decode_table_schema
from rnsr.db.migrate import migrate_artifact
from rnsr.ingest.bulk import ingest_bulk
from rnsr.ingest.expand import expand_document
from rnsr.ingest.lifecycle import append, delete_document, file_index, replace_document
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest, ingest_text


def _table_parse(src):
    return ParsedDocument(
        Path(src).stem, str(src), "a" * 64, 3, "test",
        elements=[Element("text", "Widgets cost 10 dollars", 1)],
        tables=[RawTable(1, ["Name", "Amount"], [["Widgets", "10"]], extractor="test")])


def _assert_preserved(db, expected):
    with CorpusDB(db) as corpus:
        assert corpus.doc_dict() == expected
        schema.validate_frozen(corpus.conn)
        schema.validate_integrity(corpus.conn)
        # A trusted rw connection verifies the trigger, rather than read-only mode.
        with pytest.raises(sqlite3.IntegrityError, match="immutable"), sqlite3.connect(db) as raw:
            raw.execute("UPDATE doc_text SET text='corrupted'")


def test_interrupted_append_rolls_back_data_and_ddl(tmp_path, monkeypatch):
    import rnsr.ingest.lifecycle as lifecycle

    db = tmp_path / "c.db"
    ingest_text({"old": "Original retained statement"}, db)
    with CorpusDB(db) as corpus:
        before = corpus.doc_dict()
    src = tmp_path / "new.txt"
    src.write_text("New statement")
    original_write = lifecycle._write_docs

    def interrupted(*args, **kwargs):
        original_write(*args, **kwargs)
        raise KeyboardInterrupt("interrupted after writes")

    monkeypatch.setattr(lifecycle, "_write_docs", interrupted)
    with pytest.raises(KeyboardInterrupt):
        append(src, db)
    _assert_preserved(db, before)


def test_process_death_during_append_preserves_old_artifact(tmp_path):
    db = tmp_path / "c.db"
    ingest_text({"old": "Original retained statement"}, db)
    with CorpusDB(db) as corpus:
        before = corpus.doc_dict()
    src = tmp_path / "new.txt"
    src.write_text("Uncommitted statement")
    code = """
import os, sys
import rnsr.ingest.lifecycle as life
real = life._write_docs
def die(*args, **kwargs):
    real(*args, **kwargs)
    os._exit(73)
life._write_docs = die
life.append(sys.argv[1], sys.argv[2])
"""
    result = subprocess.run([sys.executable, "-c", code, str(src), str(db)], check=False)
    assert result.returncode == 73
    _assert_preserved(db, before)


def test_replace_parse_and_write_failures_preserve_original(tmp_path, monkeypatch):
    import rnsr.ingest.lifecycle as lifecycle

    db = tmp_path / "c.db"
    ingest_text({"old": "Original retained statement"}, db)
    with CorpusDB(db) as corpus:
        before = corpus.doc_dict()
    src = tmp_path / "bad.txt"
    src.write_text("replacement")

    def bad_parse(_):
        raise ValueError("cannot parse")

    with pytest.raises(ValueError, match="cannot parse"):
        replace_document(db, "old", src, parse=bad_parse)
    _assert_preserved(db, before)

    def bad_write(*args, **kwargs):
        raise OSError("disk is full")

    monkeypatch.setattr(lifecycle, "_write_docs", bad_write)
    with pytest.raises(OSError, match="disk is full"):
        replace_document(db, "old", src)
    _assert_preserved(db, before)


def test_ambiguous_basename_has_no_alias(tmp_path):
    paths = [tmp_path / "a" / "note.txt", tmp_path / "b" / "note.txt"]
    for i, path in enumerate(paths):
        path.parent.mkdir()
        path.write_text(f"Letter {i}")
    db = tmp_path / "c.db"
    ingest(paths, db)
    index = file_index(db)
    assert "note.txt" not in index
    assert index[str(paths[0].resolve())]["doc_id"] != index[str(paths[1].resolve())]["doc_id"]


def test_collision_allocation_rechecks_suffix():
    parsed = ParsedDocument("a", "a.txt", "x", 1, "test")
    expanded = expand_document(parsed, lambda _: None, {"a", "a_2", "a_3"})
    assert expanded[0].doc_id == "a_4"


def test_blank_trailing_pages_and_nonoverlapping_character_count(tmp_path):
    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse,
           config=Settings(chunk_chars=12, chunk_overlap=5))
    with CorpusDB(db) as corpus:
        pages = corpus.conn.execute("SELECT page,text FROM doc_text ORDER BY page").fetchall()
        assert [p[0] for p in pages] == [1, 2, 3]
        assert pages[-1][1] == "\n"
        stats = corpus.manifest_get("chunk_stats")
        assert stats["total_chars"] == len(corpus.full_text("a"))
        assert stats["total_chars"] < corpus.conn.execute(
            "SELECT SUM(length(text)) FROM chunks").fetchone()[0]


def test_bulk_duplicate_content_and_modified_time(tmp_path):
    sources = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for source in sources:
        source.write_text("Exactly identical retained source")
    db = tmp_path / "bulk.db"
    ingest_bulk(sources, db, workers=1)
    with CorpusDB(db) as corpus:
        rows = corpus.conn.execute(
            "SELECT sha256,content_sha256,source_identity,modified_at,duplicate_of FROM documents"
        ).fetchall()
        assert rows[0][0] == rows[1][0] == rows[0][1] == rows[1][1]
        assert rows[0][2] != rows[1][2]
        assert all(row[3] for row in rows)
        assert sum(row[4] is not None for row in rows) == 1


def test_foreign_keys_cells_and_duplicate_delete_policy(tmp_path):
    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    with CorpusDB(db, mode="rw") as corpus:
        conn = corpus.conn
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.execute("BEGIN IMMEDIATE")
        schema.unfreeze_corpus(conn)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            schema.insert_document(conn, doc_id="bad", source_path="bad", sha256="x",
                                   n_pages=1, parser="test", ingested_at="now", parent_doc_id="missing")
        schema.insert_document(conn, doc_id="child", source_path="child", sha256="y",
                               n_pages=1, parser="test", ingested_at="now", parent_doc_id="a")
        schema.insert_document(conn, doc_id="duplicate", source_path="dup", sha256="z",
                               n_pages=1, parser="test", ingested_at="now", duplicate_of="a")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            conn.execute("INSERT INTO cells SELECT * FROM cells LIMIT 1")
        delete_document(conn, "a")
        assert [r[0] for r in conn.execute("SELECT doc_id FROM documents")] == ["duplicate"]
        assert conn.execute("SELECT duplicate_of FROM documents").fetchone()[0] is None
        for table in ("doc_text", "chunks", "manifest_tables", "cells"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        schema.validate_integrity(conn)
        conn.rollback()


def test_annotation_column_updates_typed_schema(tmp_path):
    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    with CorpusDB(db, mode="rw") as corpus:
        assert schema.add_annotation_column(corpus.conn, "t_a_001", "label")
        corpus.conn.commit()
        entry = corpus.manifest_dict()["tables"][0]["schema"][-1]
        assert entry["name"] == "label" and entry["annotation"] is True
        schema.validate_integrity(corpus.conn)


@pytest.mark.parametrize("k", [-1, 1001, 1.5, True])
def test_fts_rejects_unbounded_or_invalid_limits(tmp_path, k):
    db = tmp_path / "c.db"
    ingest_text({"a": "Token value"}, db)
    with CorpusDB(db) as corpus, pytest.raises(ValueError, match="k must"):
        fts.match(corpus.conn, "Token", k=k)


def test_migrate_legacy_list_metadata_and_idempotency(tmp_path):
    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    with sqlite3.connect(db) as conn:
        raw = conn.execute("SELECT schema_json FROM manifest_tables").fetchone()[0]
        conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute("UPDATE manifest_tables SET schema_json=?", (json.dumps(json.loads(raw)["columns"]),))
        conn.execute("PRAGMA user_version=2")
        conn.execute("UPDATE manifest SET value='2' WHERE key='format_version'")
    migrate_artifact(db)
    migrate_artifact(db)
    with CorpusDB(db) as corpus:
        schema.validate_frozen(corpus.conn)
        schema.validate_integrity(corpus.conn)
        decoded = decode_table_schema(corpus.conn.execute(
            "SELECT schema_json FROM manifest_tables").fetchone()[0])
        assert decoded.n_data_rows == 1
        assert corpus.conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0] == 2


def test_failed_migration_is_atomic(tmp_path):
    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA user_version=2")
        conn.execute("DROP TABLE t_a_001")
    with pytest.raises(sqlite3.IntegrityError, match="Missing source table"):
        migrate_artifact(db)
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_migration_unknown_content_digest_is_not_invented(tmp_path):
    db = tmp_path / "legacy.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    identity = "b" * 64
    with sqlite3.connect(db) as conn:
        schema.unfreeze_corpus(conn)
        conn.execute("UPDATE documents SET parser='pdfium-fast',sha256=?,content_sha256=?",
                     (identity, identity))
        schema.finalize_corpus(conn)
        conn.execute("PRAGMA user_version=2")
    for _ in range(2):
        result = migrate_artifact(db)
        assert result["unknown_content_digests"] == ["a"]
        with CorpusDB(db) as corpus:
            row = corpus.conn.execute(
                "SELECT sha256,content_sha256,source_identity FROM documents").fetchone()
            assert tuple(row) == ("", None, identity)


def test_true_v1_document_columns_migrate_without_losing_pages(tmp_path):
    db = tmp_path / "legacy.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE legacy_docs (doc_id TEXT PRIMARY KEY, source_path TEXT, "
                     "sha256 TEXT,n_pages INTEGER,parser TEXT,ingested_at TEXT)")
        conn.execute("INSERT INTO legacy_docs SELECT doc_id,source_path,sha256,n_pages,"
                     "parser,ingested_at FROM documents")
        conn.execute("DROP TABLE documents")
        conn.execute("ALTER TABLE legacy_docs RENAME TO documents")
        conn.execute("DROP TABLE cells")
        conn.execute("DROP TABLE ingest_batches")
        conn.execute("PRAGMA user_version=1")
        conn.execute("UPDATE manifest SET value='1' WHERE key='format_version'")
    migrate_artifact(db)
    with CorpusDB(db) as corpus:
        assert "Widgets cost 10 dollars" in corpus.full_text("a")
        assert corpus.conn.execute("SELECT COUNT(*) FROM doc_text").fetchone()[0] == 3
        assert corpus.conn.execute("SELECT content_sha256 FROM documents").fetchone()[0] == "a" * 64
        schema.validate_frozen(corpus.conn)
        schema.validate_integrity(corpus.conn)


def test_bulk_transcriber_without_scans_still_writes_documents(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("Document has a complete text layer")
    db = tmp_path / "c.db"

    def transcriber(*args):
        raise AssertionError("no pages need transcription")

    assert ingest_bulk([src], db, workers=1, transcriber=transcriber)["new_docs"] == 1
    with CorpusDB(db) as corpus:
        assert corpus.full_text("a").startswith("Document has")


def test_bulk_attachment_family_is_one_crash_checkpoint(tmp_path, monkeypatch):
    import rnsr.ingest.bulk as bulk
    from rnsr.ingest.dispatch import parse_any_fast

    src = tmp_path / "email.txt"
    src.write_text("Container source")
    db = tmp_path / "c.db"

    def parse(path):
        parsed = parse_any_fast(path)
        if path == src:
            parsed.pending_attachments = [("child.txt", b"An attached document")]
        return parsed

    original_write = bulk.write_document
    calls = 0

    def interrupted(conn, source, parsed, config):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("before writing child")
        return original_write(conn, source, parsed, config)

    monkeypatch.setattr(bulk, "write_document", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ingest_bulk([src], db, parse=parse, commit_every=1, workers=1)
    with sqlite3.connect(db.with_suffix(".db.ingesting")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    monkeypatch.setattr(bulk, "write_document", original_write)
    assert ingest_bulk([src], db, parse=parse, commit_every=1, workers=1)["new_docs"] == 2
    with CorpusDB(db) as corpus:
        schema.validate_integrity(corpus.conn)
        assert corpus.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


def test_annotation_schema_change_rolls_back_as_one_transaction(tmp_path):
    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    with CorpusDB(db, mode="rw") as corpus:
        schema.add_annotation_column(corpus.conn, "t_a_001", "label")
        corpus.conn.rollback()
        assert "label" not in {r[1] for r in corpus.conn.execute("PRAGMA table_info(t_a_001)")}
        assert not any(c["annotation"] for c in corpus.manifest_dict()["tables"][0]["schema"])
        schema.add_annotation_column(corpus.conn, "t_a_001", "label")
        assert not schema.add_annotation_column(corpus.conn, "t_a_001", "LABEL")


def test_source_headers_cannot_shadow_rowid_or_provenance(tmp_path):
    def parse(path):
        parsed = _table_parse(path)
        parsed.tables = [RawTable(
            1, ["rowid", "source_page"], [["external id", "arbitrary value"]],
            row_pages=[1], extractor="test")]
        return parsed

    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=parse)
    with CorpusDB(db) as corpus:
        columns = [c["name"] for c in corpus.manifest_dict()["tables"][0]["schema"]]
        assert columns == ["rowid_2", "source_page_2"]
        schema.validate_integrity(corpus.conn)


def test_process_death_after_migration_ddl_rolls_back(tmp_path):
    db = tmp_path / "legacy.db"
    ingest([tmp_path / "a.pdf"], db, parse=_table_parse)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA user_version=2")
        before = list(conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name"))
    code = """
import os, sys
from rnsr.db import schema
from rnsr.db.migrate import migrate_artifact
real = schema.finalize_corpus
def die(conn):
    real(conn)
    os._exit(74)
schema.finalize_corpus = die
migrate_artifact(sys.argv[1])
"""
    result = subprocess.run([sys.executable, "-c", code, str(db)], check=False)
    assert result.returncode == 74
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert list(conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name")) == before
        schema.validate_frozen(conn)
        assert conn.execute("SELECT COUNT(*) FROM doc_text").fetchone()[0] == 3
