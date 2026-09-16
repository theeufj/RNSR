"""CorpusDB: typed wrapper around the corpus.db SQLite artifact."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rnsr.db import schema
from rnsr.db.metadata import decode_table_schema
from rnsr.errors import ArtifactVersionError


class CorpusDB:
    """Open/create a corpus.db and provide manifest + text access.

    Query-time consumers use read-only connections. Trusted ingestion and
    the parent annotation broker alone use ``mode="rw"``.
    """

    def __init__(self, path: str | Path, mode: str = "ro"):
        self.path = Path(path)
        if mode not in ("ro", "rw"):
            raise ValueError(f"mode must be 'ro' or 'rw', got {mode!r}")
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        uri = self.path.resolve().as_uri() + f"?mode={mode}"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = sqlite3.Row
        # mmap-backed reads: DB pages come from the OS page cache, shared
        # across every process reading the same artifact (Stage 1).
        schema.apply_read_pragmas(self.conn)
        try:
            self._validate_artifact()
        except BaseException:
            self.conn.close()
            raise

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

    def _validate_artifact(self) -> None:
        existing = {
            r[0] for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        }
        missing = [t for t in schema.REQUIRED_TABLES if t not in existing]
        if missing:
            raise ArtifactVersionError(
                f"{self.path} is missing required tables {missing}; "
                f"rebuild with `rnsr ingest` or run `rnsr migrate {self.path}`"
            )
        user_version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        fmt = self.manifest_get("format_version")
        if user_version != schema.ARTIFACT_FORMAT_VERSION or (
                fmt is not None and int(fmt) != schema.ARTIFACT_FORMAT_VERSION):
            raise ArtifactVersionError(
                f"{self.path} is format {user_version or fmt}, this rnsr "
                f"reads {schema.ARTIFACT_FORMAT_VERSION}. "
                f"Run `rnsr migrate {self.path}`"
            )

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
        # Derive mutable summaries from their authoritative rows at read time.
        out["documents"] = [dict(row) for row in self.conn.execute(
            "SELECT doc_id,source_path,n_pages,parser,title,doc_date,author,"
            "parent_doc_id,duplicate_of,content_sha256 FROM documents ORDER BY doc_id")]
        out["untrusted_tables"] = [row[0] for row in self.conn.execute(
            "SELECT table_name FROM manifest_tables WHERE status='untrusted' ORDER BY table_name")]
        out["ingest_batches"] = [dict(row) for row in self.conn.execute(
            "SELECT batch_id,created_at,kind,n_docs FROM ingest_batches ORDER BY batch_id")]
        out["tables"] = []
        for row in self.conn.execute("SELECT * FROM manifest_tables ORDER BY table_name"):
            table_schema = decode_table_schema(row["schema_json"])
            entry = {
                **dict(row),
                "schema": [c.model_dump() for c in table_schema.columns],
                "checks": json.loads(row["checks_json"]),
                "n_total_rows": table_schema.n_total_rows,
                "n_data_rows": table_schema.n_data_rows,
            }
            entry.pop("schema_json", None)
            entry.pop("checks_json", None)
            out["tables"].append(entry)
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
