"""Artifact format upgrades.

v1 → v2: nullable document metadata columns, ingest_batches table, and
content-hash / parent / duplicate columns. Idempotent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rnsr.db.schema import ARTIFACT_FORMAT_VERSION, REQUIRED_TABLES

_V2_DOC_COLS = (
    ("title", "TEXT"),
    ("doc_date", "TEXT"),
    ("author", "TEXT"),
    ("modified_at", "TEXT"),
    ("content_sha256", "TEXT"),
    ("parent_doc_id", "TEXT"),
    ("duplicate_of", "TEXT"),
)

_INGEST_BATCHES = """
CREATE TABLE IF NOT EXISTS ingest_batches (
    batch_id     INTEGER PRIMARY KEY,
    created_at   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    n_docs       INTEGER NOT NULL,
    sources_json TEXT
)
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def migrate_artifact(path: str | Path) -> dict:
    """Bring ``path`` up to the current format version. Idempotent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path)
    try:
        existing = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        }
        missing = [t for t in REQUIRED_TABLES
                   if t not in existing and t != "ingest_batches"]
        if missing:
            raise RuntimeError(
                f"{path} is not a corpus.db (missing {missing}); ingest again")
        if "documents" in existing:
            have = _columns(conn, "documents")
            for name, sql_type in _V2_DOC_COLS:
                if name not in have:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {sql_type}")
        conn.executescript(_INGEST_BATCHES)
        conn.execute(f"PRAGMA user_version = {ARTIFACT_FORMAT_VERSION}")
        conn.execute(
            "INSERT INTO manifest (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("format_version", str(ARTIFACT_FORMAT_VERSION)),
        )
        conn.commit()
    finally:
        conn.close()
    return {"path": str(path), "format_version": ARTIFACT_FORMAT_VERSION}
