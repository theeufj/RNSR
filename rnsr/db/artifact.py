"""CorpusDB: typed wrapper around the corpus.db SQLite artifact."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rnsr.db import schema


class CorpusDB:
    """Open/create a corpus.db and provide manifest + text access.

    Query-time consumers open ``mode="rw"``: source data stays protected by
    the immutability triggers; only annotation columns and annotation_log
    are writable. ``mode="ro"`` maps to a SQLite read-only URI open.
    """

    def __init__(self, path: str | Path, mode: str = "ro"):
        self.path = Path(path)
        if mode not in ("ro", "rw"):
            raise ValueError(f"mode must be 'ro' or 'rw', got {mode!r}")
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        uri = f"file:{self.path}?mode={'ro' if mode == 'ro' else 'rw'}"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = sqlite3.Row

    @classmethod
    def create(cls, path: str | Path) -> CorpusDB:
        """Create a fresh artifact with the core schema (fails if file exists)."""
        p = Path(path)
        if p.exists():
            raise FileExistsError(p)
        conn = sqlite3.connect(p)
        try:
            schema.create_corpus_db(conn)
        finally:
            conn.close()
        return cls(p, mode="rw")

    # --- manifest -----------------------------------------------------------

    def manifest_get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM manifest WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def manifest_set(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO manifest (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def manifest_dict(self) -> dict[str, Any]:
        """Full manifest as a plain dict, including per-table entries (§3.5)."""
        out = {
            row["key"]: json.loads(row["value"])
            for row in self.conn.execute("SELECT key, value FROM manifest")
        }
        out["tables"] = [
            {
                **dict(row),
                "schema": json.loads(row["schema_json"]),
                "checks": json.loads(row["checks_json"]),
            }
            for row in self.conn.execute("SELECT * FROM manifest_tables ORDER BY table_name")
        ]
        for t in out["tables"]:
            t.pop("schema_json", None)
            t.pop("checks_json", None)
        return out

    # --- documents / text ---------------------------------------------------

    def doc_ids(self) -> list[str]:
        return [r["doc_id"] for r in self.conn.execute("SELECT doc_id FROM documents ORDER BY doc_id")]

    def full_text(self, doc_id: str) -> str:
        """Reassemble the full retained text of a document from doc_text pages."""
        rows = self.conn.execute(
            "SELECT text FROM doc_text WHERE doc_id = ? ORDER BY page", (doc_id,)
        ).fetchall()
        if not rows:
            raise KeyError(f"unknown doc_id: {doc_id}")
        return "".join(r["text"] for r in rows)

    def doc_dict(self) -> dict[str, str]:
        """The REPL-preloaded `doc` mapping: doc_id -> full text (§4)."""
        return {d: self.full_text(d) for d in self.doc_ids()}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> CorpusDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
