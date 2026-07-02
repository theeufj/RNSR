"""Manifest construction (spec §3.5).

Every manifest claim is machine-derived — no LLM-generated summaries (§9),
so the root LM's first view of the environment cannot hallucinate.
"""

from __future__ import annotations

import json
import sqlite3

from rnsr import __version__
from rnsr.db.artifact import CorpusDB
from rnsr.ingest.tables import BuiltTable
from rnsr.ingest.validate import TableValidation


def write_table_manifest(
    conn: sqlite3.Connection,
    built: BuiltTable,
    validation: TableValidation,
    status: str,
) -> None:
    conn.execute(
        "INSERT INTO manifest_tables VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            built.name,
            built.doc_id,
            built.caption,
            built.page_start,
            built.page_end,
            built.n_rows,
            built.n_cols,
            built.schema_json,
            round(validation.confidence, 4),
            json.dumps(validation.to_checks_json()),
            status,
            built.extractor,
        ),
    )


def write_corpus_manifest(corpus: CorpusDB, parser: str) -> None:
    conn = corpus.conn
    docs = [
        dict(r)
        for r in conn.execute(
            "SELECT doc_id, source_path, n_pages, parser FROM documents ORDER BY doc_id"
        )
    ]
    n_chunks, total_chars = conn.execute(
        "SELECT count(*), coalesce(sum(char_end - char_start), 0) FROM chunks"
    ).fetchone()
    untrusted = [
        r["table_name"]
        for r in conn.execute("SELECT table_name FROM manifest_tables WHERE status = 'untrusted'")
    ]
    corpus.manifest_set("documents", docs)
    corpus.manifest_set(
        "chunk_stats", {"n_chunks": n_chunks, "total_chars": total_chars}
    )
    corpus.manifest_set("untrusted_tables", untrusted)
    corpus.manifest_set("versions", {"rnsr": __version__, "parser": parser})
