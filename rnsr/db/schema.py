"""corpus.db schema: DDL, immutability triggers, data-table generation.

The artifact layout follows spec §2/§3: full retained text (doc_text),
chunks + FTS5 (§3.4), one SQL table per extracted document table with
provenance columns (§3.2), machine-derived manifest (§3.5), and an
annotation audit log (§4.1).

Immutability (§2, §4): source data is frozen by triggers — INSERT and
DELETE are blocked outright; UPDATE is blocked only for the columns that
existed at creation time (``UPDATE OF <source cols>``). Annotation columns
added later via ``ALTER TABLE ADD COLUMN`` therefore stay writable, which
is how semantic_annotate writes results back without violating the
no-eviction invariant.
"""

from __future__ import annotations

import re
import sqlite3

# Provenance columns stamped on every extracted-table row (§3.2).
PROVENANCE_COLUMNS = ("_page", "_bbox", "_extractor", "_row_kind")

# Integer stamped as PRAGMA user_version and manifest.format_version.
ARTIFACT_FORMAT_VERSION = 2
REQUIRED_TABLES = (
    "documents", "doc_text", "chunks", "manifest", "manifest_tables",
    "annotation_log", "ingest_batches",
)

DOCUMENT_COLUMNS = (
    "doc_id", "source_path", "sha256", "n_pages", "parser", "ingested_at",
    "title", "doc_date", "author", "modified_at", "content_sha256",
    "parent_doc_id", "duplicate_of",
)

CORE_DDL = """
CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    n_pages         INTEGER NOT NULL,
    parser          TEXT NOT NULL,
    ingested_at     TEXT NOT NULL,
    title           TEXT,
    doc_date        TEXT,
    author          TEXT,
    modified_at     TEXT,
    content_sha256  TEXT,
    parent_doc_id   TEXT,
    duplicate_of    TEXT
);

CREATE TABLE ingest_batches (
    batch_id     INTEGER PRIMARY KEY,
    created_at   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    n_docs       INTEGER NOT NULL,
    sources_json TEXT
);

-- Full retained text per page; the no-eviction substrate every other
-- representation (chunks, FTS, tables, embeddings) resolves back to.
CREATE TABLE doc_text (
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id),
    page       INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (doc_id, page)
);

CREATE TABLE chunks (
    chunk_id     INTEGER PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    page         INTEGER,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    heading_path TEXT,
    text         TEXT NOT NULL
);

CREATE VIRTUAL TABLE fts_chunks USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);

CREATE TABLE manifest (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL              -- JSON; machine-derived only (§9)
);

CREATE TABLE manifest_tables (
    table_name  TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    title       TEXT,
    page_start  INTEGER,
    page_end    INTEGER,
    n_rows      INTEGER NOT NULL,
    n_cols      INTEGER NOT NULL,
    schema_json TEXT NOT NULL,       -- [{name, type, coercion_rule, raw_col}]
    confidence  REAL NOT NULL,
    checks_json TEXT NOT NULL,       -- {arithmetic:…, structural:…, prose:…}
    status      TEXT NOT NULL CHECK (status IN ('trusted','reextracted','untrusted','unchecked')),
    extractor   TEXT NOT NULL
);

CREATE TABLE annotation_log (
    id            INTEGER PRIMARY KEY,
    created_at    TEXT NOT NULL,
    table_name    TEXT NOT NULL,
    column        TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    model         TEXT NOT NULL,
    where_clause  TEXT,
    batch_size    INTEGER NOT NULL,
    rows_written  INTEGER NOT NULL,
    rows_failed   INTEGER NOT NULL,
    usage_json    TEXT NOT NULL
);

CREATE UNIQUE INDEX annotation_idempotency
    ON annotation_log (table_name, column, prompt_sha256, model, ifnull(where_clause, ''));
"""

# Derived cell index (engine-poc-plan Stage 1): one indexed table over every
# typed-table cell, so rung-0 sweeps run one scan instead of un-indexed LIKE
# over every t_* table. Derived data — rebuildable, deliberately unfrozen.
# text_value is stored lowercased (it exists only to be searched).
CELLS_DDL = (
    """
    CREATE TABLE IF NOT EXISTS cells (
        doc_id     TEXT NOT NULL,
        table_name TEXT NOT NULL,
        row_idx    INTEGER NOT NULL,      -- rowid in the source t_* table
        col_name   TEXT NOT NULL,
        text_value TEXT,
        num_value  REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS cells_num ON cells(num_value) "
    "WHERE num_value IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS cells_table ON cells(table_name)",
)

# Default read mmap: the OS page cache serves DB pages, shared across every
# process that opens the same artifact (multi-worker deployments pay for the
# corpus once). 0 disables; RNSR_DB_MMAP_BYTES overrides.
DEFAULT_MMAP_BYTES = 256 << 20


def ensure_cells_table(conn: sqlite3.Connection) -> None:
    for stmt in CELLS_DDL:
        conn.execute(stmt)


def apply_read_pragmas(conn: sqlite3.Connection,
                       mmap_bytes: int | None = None) -> None:
    """Per-connection read tuning; safe on rw connections too."""
    import os

    if mmap_bytes is None:
        raw = os.environ.get("RNSR_DB_MMAP_BYTES", "")
        mmap_bytes = int(raw) if raw.strip().isdigit() else DEFAULT_MMAP_BYTES
    if mmap_bytes:
        conn.execute(f"PRAGMA mmap_size={int(mmap_bytes)}")


_IDENT_RE = re.compile(r"[^a-z0-9_]+")


def quote_ident(name: str) -> str:
    """Quote an identifier for direct inclusion in SQL."""
    return '"' + name.replace('"', '""') + '"'


def sanitize_column_name(raw: str, taken: set[str] | None = None) -> str:
    """Header text -> safe snake_case column name, deduplicated against `taken`."""
    name = _IDENT_RE.sub("_", raw.strip().lower()).strip("_") or "col"
    if name[0].isdigit():
        name = "c_" + name
    if taken is not None:
        base, i = name, 2
        while name in taken:
            name = f"{base}_{i}"
            i += 1
        taken.add(name)
    return name


def create_corpus_db(conn: sqlite3.Connection) -> None:
    """Create the core schema in an empty database (unfrozen — ingestion writes next)."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(CORE_DDL)
    ensure_cells_table(conn)
    conn.execute(f"PRAGMA user_version = {ARTIFACT_FORMAT_VERSION}")
    conn.execute(
        "INSERT INTO manifest (key, value) VALUES (?, ?)",
        ("format_version", str(ARTIFACT_FORMAT_VERSION)),
    )
    conn.commit()


def finalize_corpus(conn: sqlite3.Connection) -> None:
    """Freeze source tables once ingestion is complete.

    Data tables (t_*) are frozen individually right after their bulk insert;
    this call freezes the shared text substrate. manifest/manifest_tables/
    annotation_log stay writable (annotations and re-validation metadata).
    ``documents.duplicate_of`` stays writable so remanifest can restamp it.
    """
    for table in ("documents", "doc_text", "chunks"):
        cols = _table_columns(conn, table)
        if table == "documents":
            cols = [c for c in cols if c != "duplicate_of"]
        freeze_table(conn, table, source_columns=cols)
    conn.commit()


def unfreeze_table(conn: sqlite3.Connection, table: str) -> None:
    """Drop immutability triggers so an append/replace transaction can write."""
    for suffix in ("__no_insert", "__no_delete", "__no_update_src"):
        conn.execute(f"DROP TRIGGER IF EXISTS {quote_ident(table + suffix)}")


def unfreeze_corpus(conn: sqlite3.Connection) -> None:
    """Drop freeze triggers on the shared text substrate (not t_* tables)."""
    for table in ("documents", "doc_text", "chunks"):
        unfreeze_table(conn, table)


def record_ingest_batch(conn: sqlite3.Connection, kind: str,
                        sources: list, n_docs: int) -> None:
    """Append one ingest_batches row (create | append | replace)."""
    import json
    from datetime import UTC, datetime

    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "ingest_batches" not in names:
        return
    conn.execute(
        "INSERT INTO ingest_batches (created_at, kind, n_docs, sources_json) "
        "VALUES (?,?,?,?)",
        (datetime.now(UTC).isoformat(), kind, n_docs,
         json.dumps([str(s) for s in sources])),
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({quote_ident(table)})")]


def data_table_name(doc_id: str, seq: int) -> str:
    return f"t_{doc_id}_{seq:03d}"


def create_data_table(
    conn: sqlite3.Connection,
    table: str,
    columns: list[tuple[str, str]],
    *,
    with_source_page: bool = False,
) -> None:
    """Create an extracted-table SQL table (unfrozen — freeze after bulk insert).

    `columns` is [(name, sql_type)] for the data columns; shadow __raw
    columns must already be included by the caller. Provenance columns are
    appended automatically.
    """
    cols = [f"{quote_ident(n)} {t}" for n, t in columns]
    if with_source_page:
        cols.append('"source_page" INTEGER')
    cols += [
        '"_page" INTEGER NOT NULL',
        '"_bbox" TEXT NOT NULL',  # JSON [x0, y0, x1, y1] in page coords
        '"_extractor" TEXT NOT NULL',
        '"_row_kind" TEXT NOT NULL',  # data | total | subtotal | footnote | section
    ]
    conn.execute(f"CREATE TABLE {quote_ident(table)} ({', '.join(cols)})")


def freeze_table(conn: sqlite3.Connection, table: str, source_columns: list[str]) -> None:
    """Install immutability triggers: no INSERT/DELETE; no UPDATE of source columns.

    Call after bulk insert. Columns added later by ALTER TABLE are not in
    `source_columns`, so annotation writes remain possible.
    """
    q = quote_ident(table)
    msg = f"'{table} is immutable source data (see docdb-rlm-design-spec.md §2)'"
    conn.execute(
        f"CREATE TRIGGER {quote_ident(table + '__no_insert')} BEFORE INSERT ON {q} "
        f"BEGIN SELECT RAISE(ABORT, {msg}); END"
    )
    conn.execute(
        f"CREATE TRIGGER {quote_ident(table + '__no_delete')} BEFORE DELETE ON {q} "
        f"BEGIN SELECT RAISE(ABORT, {msg}); END"
    )
    col_list = ", ".join(quote_ident(c) for c in source_columns)
    conn.execute(
        f"CREATE TRIGGER {quote_ident(table + '__no_update_src')} "
        f"BEFORE UPDATE OF {col_list} ON {q} "
        f"BEGIN SELECT RAISE(ABORT, {msg}); END"
    )


def insert_document(conn: sqlite3.Connection, *,
                    doc_id: str, source_path: str, sha256: str,
                    n_pages: int, parser: str, ingested_at: str,
                    title: str | None = None, doc_date: str | None = None,
                    author: str | None = None, modified_at: str | None = None,
                    content_sha256: str | None = None,
                    parent_doc_id: str | None = None,
                    duplicate_of: str | None = None) -> None:
    """Insert one documents row (v2 columns). Missing optionals stay NULL."""
    conn.execute(
        f"INSERT INTO documents ({', '.join(DOCUMENT_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(DOCUMENT_COLUMNS))})",
        (doc_id, source_path, sha256, n_pages, parser, ingested_at,
         title, doc_date, author, modified_at, content_sha256,
         parent_doc_id, duplicate_of),
    )


def add_annotation_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Add a writable annotation column if absent. Returns True if added."""
    if column in _table_columns(conn, table):
        return False
    conn.execute(f"ALTER TABLE {quote_ident(table)} ADD COLUMN {quote_ident(column)}")
    return True
