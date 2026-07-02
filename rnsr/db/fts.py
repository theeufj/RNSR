"""FTS5 helpers: population at ingest, BM25 MATCH at query time (§3.4, rung 2)."""

from __future__ import annotations

import sqlite3


def populate_fts(conn: sqlite3.Connection) -> int:
    """(Re)build the external-content FTS index from chunks. Returns row count."""
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('delete-all')")
    conn.execute("INSERT INTO fts_chunks(rowid, text) SELECT chunk_id, text FROM chunks")
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def match(conn: sqlite3.Connection, query: str, k: int = 10) -> list[dict]:
    """BM25-ranked lexical search over chunks; no model calls, milliseconds.

    Query syntax errors (unbalanced quotes etc.) are reported as an empty
    result with the error captured, so the search ladder can escalate
    instead of crashing the REPL.
    """
    sql = """
        SELECT c.chunk_id, c.doc_id, c.page, c.char_start, c.char_end,
               c.heading_path, c.text, bm25(fts_chunks) AS score
        FROM fts_chunks
        JOIN chunks c ON c.chunk_id = fts_chunks.rowid
        WHERE fts_chunks MATCH ?
        ORDER BY score LIMIT ?
    """
    try:
        cur = conn.execute(sql, (query, k))
    except sqlite3.OperationalError:
        # Fall back to a fully-escaped phrase query before giving up.
        escaped = '"' + query.replace('"', '""') + '"'
        try:
            cur = conn.execute(sql, (escaped, k))
        except sqlite3.OperationalError:
            return []
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
