"""Artifact format upgrades.

Today the only upgrade is stamping ``user_version`` / ``format_version`` on
pre-versioned corpora. A future bump lands a real conversion here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rnsr.db.schema import ARTIFACT_FORMAT_VERSION, REQUIRED_TABLES


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
        missing = [t for t in REQUIRED_TABLES if t not in existing]
        if missing:
            raise RuntimeError(
                f"{path} is not a corpus.db (missing {missing}); ingest again")
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
