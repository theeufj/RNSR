"""
Unified SQLite Store for RNSR Document Collections.

Consolidates catalog, skeletons, node content, images, and detected tables
into a single WAL-mode SQLite database per store.  Replaces the fragile
multi-file layout (catalog.json + per-doc skeleton.json + per-doc content.db)
with atomic transactions, a single file lock, and concurrent-read safety.

Usage:
    db = StoreDB("/path/to/.rnsr_store")
    db.save_document(doc_id, title, source_path, skeleton, kv_store, tables)
    skeleton, kv_store, tables = db.load_document(doc_id)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import structlog

from rnsr.exceptions import IndexingError

if TYPE_CHECKING:
    from rnsr.models import DetectedTable, SkeletonNode

logger = structlog.get_logger(__name__)

STORE_DB_FILENAME = "store.db"
STORE_DB_VERSION = "2.0"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS catalog (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    source_path TEXT,
    node_count  INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS skeletons (
    doc_id    TEXT NOT NULL,
    node_id   TEXT NOT NULL,
    parent_id TEXT,
    level     INTEGER NOT NULL DEFAULT 0,
    header    TEXT DEFAULT '',
    summary   TEXT DEFAULT '',
    child_ids TEXT DEFAULT '[]',
    page_num  INTEGER,
    metadata  TEXT DEFAULT '{}',
    PRIMARY KEY (doc_id, node_id)
);

CREATE TABLE IF NOT EXISTS node_content (
    doc_id       TEXT NOT NULL,
    node_id      TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    char_count   INTEGER NOT NULL,
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (doc_id, node_id)
);

CREATE TABLE IF NOT EXISTS node_images (
    doc_id     TEXT NOT NULL,
    node_id    TEXT NOT NULL,
    image_data BLOB NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (doc_id, node_id)
);

CREATE TABLE IF NOT EXISTS detected_tables (
    doc_id   TEXT NOT NULL,
    table_id TEXT NOT NULL,
    node_id  TEXT NOT NULL,
    page_num INTEGER,
    title    TEXT DEFAULT '',
    headers  TEXT DEFAULT '[]',
    num_rows INTEGER DEFAULT 0,
    num_cols INTEGER DEFAULT 0,
    data     TEXT DEFAULT '[]',
    PRIMARY KEY (doc_id, table_id)
);

CREATE INDEX IF NOT EXISTS idx_skeletons_doc       ON skeletons(doc_id);
CREATE INDEX IF NOT EXISTS idx_node_content_doc    ON node_content(doc_id);
CREATE INDEX IF NOT EXISTS idx_detected_tables_doc ON detected_tables(doc_id);
"""


class StoreDB:
    """Manages a single SQLite database for all document store data.

    One persistent connection is kept open with WAL journal mode so that
    concurrent readers never block writers.  All mutations are wrapped in
    transactions that roll back on error, preventing the partial-save
    states that plagued the old multi-file layout.
    """

    def __init__(self, store_path: Path | str):
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._store_path / STORE_DB_FILENAME
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()
        logger.info("store_db_initialized", db_path=str(self._db_path))

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def store_path(self) -> Path:
        return self._store_path

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Thread-safe access to the shared connection."""
        with self._lock:
            conn = self._get_conn()
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('version', ?)",
                (STORE_DB_VERSION,),
            )
            conn.commit()

    # =====================================================================
    # Document operations (atomic save / load / remove)
    # =====================================================================

    def save_document(
        self,
        doc_id: str,
        title: str,
        source_path: str | None,
        skeleton: dict[str, SkeletonNode],
        kv_store: Any,
        tables: list[DetectedTable] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atomically save a document's catalog entry, skeleton, content, and tables."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO catalog "
                "(doc_id, title, source_path, node_count, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    doc_id, title, source_path, len(skeleton),
                    datetime.now().isoformat(), json.dumps(metadata or {}),
                ),
            )

            for tbl in ("skeletons", "node_content", "node_images", "detected_tables"):
                conn.execute(f"DELETE FROM {tbl} WHERE doc_id = ?", (doc_id,))

            for node_id, node in skeleton.items():
                conn.execute(
                    "INSERT INTO skeletons "
                    "(doc_id, node_id, parent_id, level, header, summary, "
                    "child_ids, page_num, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc_id, node_id, node.parent_id, node.level,
                        node.header, node.summary,
                        json.dumps(node.child_ids), node.page_num,
                        json.dumps(node.metadata),
                    ),
                )

            for node_id in skeleton:
                content = kv_store.get(node_id)
                if content:
                    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                    conn.execute(
                        "INSERT INTO node_content "
                        "(doc_id, node_id, content, content_hash, char_count) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (doc_id, node_id, content, content_hash, len(content)),
                    )
                if hasattr(kv_store, "get_image"):
                    img = kv_store.get_image(node_id)
                    if img:
                        conn.execute(
                            "INSERT INTO node_images (doc_id, node_id, image_data) "
                            "VALUES (?, ?, ?)",
                            (doc_id, node_id, img),
                        )

            if tables:
                for t in tables:
                    conn.execute(
                        "INSERT INTO detected_tables "
                        "(doc_id, table_id, node_id, page_num, title, "
                        "headers, num_rows, num_cols, data) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            doc_id, t.id, t.node_id, t.page_num, t.title,
                            json.dumps(t.headers), t.num_rows, t.num_cols,
                            json.dumps(t.data),
                        ),
                    )

            conn.commit()
        logger.info("document_saved_to_db", doc_id=doc_id, nodes=len(skeleton))

    def load_document(
        self, doc_id: str,
    ) -> tuple[dict[str, SkeletonNode], DocKVStore, list[DetectedTable]] | None:
        """Load a document's skeleton, KV store wrapper, and tables."""
        from rnsr.models import DetectedTable, SkeletonNode

        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM catalog WHERE doc_id = ? LIMIT 1", (doc_id,)
            ).fetchone() is None:
                return None

            skeleton: dict[str, SkeletonNode] = {}
            for r in conn.execute(
                "SELECT * FROM skeletons WHERE doc_id = ?", (doc_id,)
            ):
                skeleton[r["node_id"]] = SkeletonNode(
                    node_id=r["node_id"],
                    parent_id=r["parent_id"],
                    level=r["level"],
                    header=r["header"] or "",
                    summary=r["summary"] or "",
                    child_ids=json.loads(r["child_ids"]) if r["child_ids"] else [],
                    page_num=r["page_num"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                )

            if not skeleton:
                logger.warning("load_document_empty_skeleton", doc_id=doc_id)
                return None

            kv_store = DocKVStore(self, doc_id)

            tables: list[DetectedTable] = []
            for r in conn.execute(
                "SELECT * FROM detected_tables WHERE doc_id = ?", (doc_id,)
            ):
                tables.append(DetectedTable(
                    id=r["table_id"],
                    node_id=r["node_id"],
                    page_num=r["page_num"],
                    title=r["title"] or "",
                    headers=json.loads(r["headers"]) if r["headers"] else [],
                    num_rows=r["num_rows"] or 0,
                    num_cols=r["num_cols"] or 0,
                    data=json.loads(r["data"]) if r["data"] else [],
                ))

        return skeleton, kv_store, tables

    def remove_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM catalog WHERE doc_id = ? LIMIT 1", (doc_id,)
            ).fetchone() is None:
                return False
            for tbl in ("node_content", "node_images", "skeletons",
                        "detected_tables", "catalog"):
                conn.execute(f"DELETE FROM {tbl} WHERE doc_id = ?", (doc_id,))
            conn.commit()
        return True

    def clear_documents(self) -> int:
        """Remove all document data (catalog, skeletons, content, tables)."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
            for tbl in ("node_content", "node_images", "skeletons",
                        "detected_tables", "catalog"):
                conn.execute(f"DELETE FROM {tbl}")
            conn.commit()
        return count

    # =====================================================================
    # Catalog helpers
    # =====================================================================

    def load_catalog(self) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            result: dict[str, dict[str, Any]] = {}
            for r in conn.execute("SELECT * FROM catalog"):
                result[r["doc_id"]] = {
                    "id": r["doc_id"],
                    "title": r["title"],
                    "source_path": r["source_path"],
                    "node_count": r["node_count"],
                    "created_at": r["created_at"],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                }
            return result

    def has_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM catalog WHERE doc_id = ? LIMIT 1", (doc_id,)
            ).fetchone() is not None

    # =====================================================================
    # Skeleton helpers (for collection skeleton building)
    # =====================================================================

    def get_root_summary(self, doc_id: str) -> str:
        """Root node summary with metadata extras for the collection skeleton."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary, metadata FROM skeletons "
                "WHERE doc_id = ? AND level = 0",
                (doc_id,),
            ).fetchone()
        if row is None:
            return ""

        summary = (row["summary"] or "")[:300]
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        extras: list[str] = []
        if meta.get("total_pages"):
            extras.append(f"Pages: {meta['total_pages']}")
        if meta.get("dates_found"):
            extras.append(f"Dates: {', '.join(meta['dates_found'][:3])}")
        if meta.get("references_found"):
            extras.append(f"Refs: {', '.join(meta['references_found'][:2])}")
        if extras:
            summary += " | " + " | ".join(extras)
        return summary

    # =====================================================================
    # KG path helper
    # =====================================================================

    @property
    def kg_db_path(self) -> str:
        """KnowledgeGraph should use this same DB file."""
        return str(self._db_path)

    # =====================================================================
    # Migration from legacy multi-file format
    # =====================================================================

    def migrate_from_legacy(self) -> int:
        """Import data from old catalog.json + per-doc directories.

        Returns number of documents migrated.
        """
        catalog_path = self._store_path / "catalog.json"
        if not catalog_path.exists():
            return 0

        try:
            with open(catalog_path) as f:
                old_catalog = json.load(f)
        except Exception as exc:
            logger.warning("migration_catalog_read_failed", error=str(exc))
            return 0

        docs = old_catalog.get("documents", {})
        migrated = 0

        for doc_id, info in docs.items():
            index_path = self._store_path / doc_id
            if not index_path.exists():
                continue
            try:
                from rnsr.indexing.persistence import load_index
                skeleton, kv_store, tables = load_index(index_path)
                self.save_document(
                    doc_id=doc_id,
                    title=info.get("title", doc_id),
                    source_path=info.get("source_path"),
                    skeleton=skeleton,
                    kv_store=kv_store,
                    tables=tables,
                    metadata=info.get("metadata"),
                )
                shutil.rmtree(index_path, ignore_errors=True)
                migrated += 1
            except Exception as exc:
                logger.warning(
                    "migration_doc_failed", doc_id=doc_id, error=str(exc),
                )

        old_kg = self._store_path / "workspace_kg.db"
        if old_kg.exists():
            try:
                self._migrate_kg_tables(old_kg)
                old_kg.unlink()
            except Exception as exc:
                logger.warning("migration_kg_failed", error=str(exc))

        if migrated > 0:
            catalog_path.unlink(missing_ok=True)
            (self._store_path / "collection_skeleton.json").unlink(missing_ok=True)

        if migrated:
            logger.info("migration_complete", documents=migrated)
        return migrated

    def _migrate_kg_tables(self, old_kg_path: Path) -> None:
        old_conn = sqlite3.connect(str(old_kg_path))
        old_conn.row_factory = sqlite3.Row
        try:
            with self._connect() as conn:
                existing = {
                    r[0] for r in old_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for table_name in ("entities", "mentions", "relationships", "entity_links"):
                    if table_name not in existing:
                        continue
                    rows = old_conn.execute(f"SELECT * FROM {table_name}").fetchall()
                    if not rows:
                        continue
                    cols = [
                        d[0] for d in old_conn.execute(
                            f"SELECT * FROM {table_name} LIMIT 0"
                        ).description
                    ]
                    placeholders = ",".join("?" * len(cols))
                    col_names = ",".join(cols)
                    for row in rows:
                        try:
                            conn.execute(
                                f"INSERT OR IGNORE INTO {table_name} "
                                f"({col_names}) VALUES ({placeholders})",
                                tuple(row),
                            )
                        except Exception:
                            pass
                conn.commit()
        finally:
            old_conn.close()

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DocKVStore:
    """KVStore-compatible wrapper scoped to a single document within StoreDB.

    Implements the same interface as SQLiteKVStore / InMemoryKVStore so
    navigators and other consumers can use it transparently.
    """

    def __init__(self, store_db: StoreDB, doc_id: str):
        self._db = store_db
        self._doc_id = doc_id

    @property
    def doc_id(self) -> str:
        return self._doc_id

    def put(self, node_id: str, content: str) -> str:
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        with self._db._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO node_content "
                "(doc_id, node_id, content, content_hash, char_count, updated_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (self._doc_id, node_id, content, content_hash, len(content)),
            )
            conn.commit()
        return content_hash

    def get(self, node_id: str) -> str | None:
        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT content FROM node_content "
                "WHERE doc_id = ? AND node_id = ?",
                (self._doc_id, node_id),
            ).fetchone()
        return row["content"] if row else None

    def get_batch(self, node_ids: list[str]) -> dict[str, str | None]:
        result: dict[str, str | None] = {nid: None for nid in node_ids}
        if not node_ids:
            return result
        placeholders = ",".join("?" * len(node_ids))
        with self._db._connect() as conn:
            for row in conn.execute(
                f"SELECT node_id, content FROM node_content "
                f"WHERE doc_id = ? AND node_id IN ({placeholders})",
                [self._doc_id, *node_ids],
            ):
                result[row["node_id"]] = row["content"]
        return result

    def put_image(self, node_id: str, image_bytes: bytes) -> None:
        with self._db._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO node_images "
                "(doc_id, node_id, image_data) VALUES (?, ?, ?)",
                (self._doc_id, node_id, image_bytes),
            )
            conn.commit()

    def get_image(self, node_id: str) -> bytes | None:
        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT image_data FROM node_images "
                "WHERE doc_id = ? AND node_id = ?",
                (self._doc_id, node_id),
            ).fetchone()
        return row["image_data"] if row else None

    def delete(self, node_id: str) -> bool:
        with self._db._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM node_content WHERE doc_id = ? AND node_id = ?",
                (self._doc_id, node_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def exists(self, node_id: str) -> bool:
        with self._db._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM node_content "
                "WHERE doc_id = ? AND node_id = ? LIMIT 1",
                (self._doc_id, node_id),
            ).fetchone() is not None

    def count(self) -> int:
        with self._db._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM node_content WHERE doc_id = ?",
                (self._doc_id,),
            ).fetchone()[0]

    def get_metadata(self, node_id: str) -> dict | None:
        with self._db._connect() as conn:
            row = conn.execute(
                "SELECT content_hash, char_count, created_at, updated_at "
                "FROM node_content WHERE doc_id = ? AND node_id = ?",
                (self._doc_id, node_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "content_hash": row["content_hash"],
            "char_count": row["char_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def clear(self) -> int:
        with self._db._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM node_content WHERE doc_id = ?", (self._doc_id,),
            )
            conn.execute(
                "DELETE FROM node_images WHERE doc_id = ?", (self._doc_id,),
            )
            conn.commit()
        return cursor.rowcount
