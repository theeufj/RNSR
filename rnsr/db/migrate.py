"""Atomic upgrades to artifact v3: references, content identity, typed metadata.

Migrations never read or modify original source files. Historical fast
artifacts whose digest was only a filesystem identity retain that identity
separately and mark the unknown content digest NULL instead of inventing one.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rnsr.db import fts, schema
from rnsr.db.metadata import TableSchema


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({schema.quote_ident(table)})")]


def _rebuild_core_table(conn, table):
    """Rebuild a core table using current DDL without implicit executescript commits."""
    start = schema.CORE_DDL.index(f"CREATE TABLE {table} (")
    end = schema.CORE_DDL.index("\n);", start) + 3
    ddl = schema.CORE_DDL[start:end]
    temp = table + "__migration"
    conn.execute(ddl.replace(f"CREATE TABLE {table} (", f"CREATE TABLE {temp} (", 1))
    common = [c for c in _columns(conn, table) if c in _columns(conn, temp)]
    names = ",".join(schema.quote_ident(c) for c in common)
    conn.execute(f"INSERT INTO {schema.quote_ident(temp)} ({names}) "
                 f"SELECT {names} FROM {schema.quote_ident(table)}")
    conn.execute(f"DROP TABLE {schema.quote_ident(table)}")
    conn.execute(f"ALTER TABLE {schema.quote_ident(temp)} RENAME TO {schema.quote_ident(table)}")


def _upgrade_table_metadata(conn):
    for table, raw in conn.execute(
            "SELECT table_name,schema_json FROM manifest_tables").fetchall():
        columns = _columns(conn, table)
        if not columns:
            raise sqlite3.IntegrityError(f"Missing source table: {table}")
        meta = json.loads(raw)
        entries = meta if isinstance(meta, list) else meta["columns"]
        if "_row_kind" not in columns:
            conn.execute(f"ALTER TABLE {schema.quote_ident(table)} "
                         "ADD COLUMN _row_kind TEXT NOT NULL DEFAULT 'data'")
            columns.append("_row_kind")
        total = conn.execute(f"SELECT count(*) FROM {schema.quote_ident(table)} "
                             "WHERE _row_kind IN ('total','subtotal')").fetchone()[0]
        data = conn.execute(f"SELECT count(*) FROM {schema.quote_ident(table)} "
                            "WHERE _row_kind='data'").fetchone()[0]
        annotations = {r[0] for r in conn.execute(
            'SELECT DISTINCT "column" FROM annotation_log WHERE table_name=?', (table,))}
        annotations.update(entry["name"] for entry in entries if entry.get("annotation"))
        described = {entry["name"] for entry in entries}
        for name in annotations - described:
            if name in columns:
                entries.append({"name": name, "type": "TEXT", "annotation": True})
        for entry in entries:
            if entry["name"] in annotations:
                entry["annotation"] = True
        canonical = TableSchema.model_validate({
            "columns": entries, "n_total_rows": total, "n_data_rows": data})
        conn.execute("UPDATE manifest_tables SET schema_json=? WHERE table_name=?",
                     (canonical.model_dump_json(), table))
        # Protect all retained source/provenance fields, including newly migrated ones.
        schema.unfreeze_table(conn, table)
        schema.freeze_table(conn, table, [c for c in columns if c not in annotations])


def _rebuild_cells(conn):
    conn.execute("DROP TABLE IF EXISTS cells")
    schema.ensure_cells_table(conn)
    for table, doc_id, raw in conn.execute(
            "SELECT table_name,doc_id,schema_json FROM manifest_tables").fetchall():
        metadata = TableSchema.model_validate_json(raw)
        for col in metadata.columns:
            if col.annotation:
                continue
            value_col = schema.quote_ident(col.name)
            raw_col = schema.quote_ident(col.raw_col or col.name)
            rows = conn.execute(f"SELECT rowid,{value_col},{raw_col} "
                                f"FROM {schema.quote_ident(table)}").fetchall()
            conn.executemany("INSERT INTO cells VALUES (?,?,?,?,?,?)", [
                (doc_id, table, rid, col.name,
                 None if text is None else str(text).lower(),
                 value if col.raw_col is not None else None)
                for rid, value, text in rows if value is not None or text is not None])


def migrate_artifact(path: str | Path) -> dict:
    """Upgrade a supported historical artifact or roll back the complete change."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path)
    unknown_digests = []
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > schema.ARTIFACT_FORMAT_VERSION:
            raise RuntimeError(f"Artifact format {version} is newer than this RNSR")
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = set(schema.REQUIRED_TABLES) - {"ingest_batches", "cells"}
        if missing := sorted(required - existing):
            raise RuntimeError(f"{path} is not a corpus.db (missing {missing}); ingest again")
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")
        # Canonicalize metadata before adding the SQL shape CHECK during rebuild.
        _upgrade_table_metadata(conn)
        for table in ("documents", "doc_text", "chunks", "manifest_tables"):
            _rebuild_core_table(conn, table)
        if "ingest_batches" not in existing:
            start = schema.CORE_DDL.index("CREATE TABLE ingest_batches (")
            end = schema.CORE_DDL.index("\n);", start) + 3
            conn.execute(schema.CORE_DDL[start:end])
        if version < 3:
            for doc_id, sha, content, parser in conn.execute(
                    "SELECT doc_id,sha256,content_sha256,parser FROM documents").fetchall():
                if content and content != sha:
                    conn.execute("UPDATE documents SET source_identity=?,sha256=? WHERE doc_id=?",
                                 (sha, content, doc_id))
                elif parser == "pdfium-fast":
                    unknown_digests.append(doc_id)
                    conn.execute("UPDATE documents SET source_identity=?,sha256='',"
                                 "content_sha256=NULL WHERE doc_id=?", (sha, doc_id))
                elif not content:
                    conn.execute("UPDATE documents SET content_sha256=sha256 WHERE doc_id=?", (doc_id,))
        unknown_digests = [r[0] for r in conn.execute(
            "SELECT doc_id FROM documents WHERE content_sha256 IS NULL AND sha256='' ORDER BY doc_id")]
        _rebuild_cells(conn)
        fts.populate_fts(conn)
        schema.validate_integrity(conn)
        schema.finalize_corpus(conn)
        conn.execute(f"PRAGMA user_version={schema.ARTIFACT_FORMAT_VERSION}")
        stats = {
            "n_chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "total_chars": conn.execute(
                "SELECT coalesce(sum(char_end-char_start),0) FROM doc_text").fetchone()[0],
        }
        for key, value in (("format_version", schema.ARTIFACT_FORMAT_VERSION),
                           ("unknown_content_digests", unknown_digests),
                           ("chunk_stats", stats)):
            conn.execute("INSERT INTO manifest(key,value) VALUES (?,?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (key, json.dumps(value)))
        conn.commit()
        conn.execute("PRAGMA foreign_keys=ON")
        schema.validate_frozen(conn)
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"path": str(path), "format_version": schema.ARTIFACT_FORMAT_VERSION,
            "unknown_content_digests": unknown_digests}
