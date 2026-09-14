"""Artifact format_version / user_version and migrate stub."""

import sqlite3

import pytest

from rnsr.db import schema
from rnsr.db.artifact import CorpusDB
from rnsr.db.migrate import migrate_artifact
from rnsr.errors import ArtifactVersionError
from rnsr.ingest.model import Element, ParsedDocument
from rnsr.ingest.pipeline import ingest


def _parse(path):
    return ParsedDocument(
        doc_id="v", source_path=str(path), sha256="e" * 64, n_pages=1,
        parser="fake", elements=[Element("text", "hi", 1)], tables=[],
    )


def test_create_stamps_user_and_format_version(tmp_path):
    db = tmp_path / "c.db"
    ingest([tmp_path / "a.pdf"], db, parse=_parse)
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.ARTIFACT_FORMAT_VERSION
    conn.close()
    with CorpusDB(db) as c:
        assert c.manifest_get("format_version") == schema.ARTIFACT_FORMAT_VERSION


def test_open_rejects_unknown_user_version(tmp_path):
    db = tmp_path / "old.db"
    ingest([tmp_path / "a.pdf"], db, parse=_parse)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(ArtifactVersionError, match="migrate"):
        CorpusDB(db)


def test_open_rejects_missing_tables(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    with pytest.raises(ArtifactVersionError, match="missing required"):
        CorpusDB(db)


def test_migrate_stamps_current_version(tmp_path):
    db = tmp_path / "legacy.db"
    ingest([tmp_path / "a.pdf"], db, parse=_parse)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 0")
    conn.execute("DELETE FROM manifest WHERE key = 'format_version'")
    conn.commit()
    conn.close()
    with pytest.raises(ArtifactVersionError):
        CorpusDB(db)
    result = migrate_artifact(db)
    assert result["format_version"] == schema.ARTIFACT_FORMAT_VERSION
    with CorpusDB(db) as c:
        assert c.manifest_get("format_version") == schema.ARTIFACT_FORMAT_VERSION
