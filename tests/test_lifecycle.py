"""Append/replace leave the same searchable substrate as a full ingest."""

from rnsr.eval.replay import replay
from rnsr.ingest.lifecycle import append, replace_document
from rnsr.ingest.pipeline import ingest


def _dump(path):
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        docs = sorted(r[0] for r in conn.execute("SELECT doc_id FROM documents"))
        texts = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT doc_id, group_concat(text, '\n') FROM doc_text "
                "GROUP BY doc_id")
        }
        return docs, texts
    finally:
        conn.close()


def test_append_matches_full_ingest(tmp_path):
    a = tmp_path / "alpha.txt"
    b = tmp_path / "beta.txt"
    a.write_text("alpha unique token for replay")
    b.write_text("beta unique token for replay")
    full = tmp_path / "full.db"
    part = tmp_path / "part.db"
    ingest([a, b], full)
    ingest([a], part)
    stats = append([b], part)
    assert stats["new_docs"] == 1
    assert _dump(full)[0] == _dump(part)[0]
    assert _dump(full)[1] == _dump(part)[1]
    r1 = replay(full, ["alpha unique", "beta unique"])
    r2 = replay(part, ["alpha unique", "beta unique"])
    assert r1["n_diffs"] == 0 and r2["n_diffs"] == 0


def test_replace_swaps_document_text(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("old wording")
    db = tmp_path / "c.db"
    ingest([src], db)
    src.write_text("new wording zebra")
    replace_document(db, "note", src)
    import sqlite3

    conn = sqlite3.connect(db)
    text = conn.execute("SELECT text FROM doc_text").fetchone()[0]
    conn.close()
    assert "zebra" in text
    assert "old wording" not in text


def test_ingest_batches_recorded(tmp_path):
    import sqlite3

    a = tmp_path / "a.txt"
    a.write_text("hello")
    db = tmp_path / "c.db"
    ingest([a], db)
    conn = sqlite3.connect(db)
    kinds = [r[0] for r in conn.execute("SELECT kind FROM ingest_batches")]
    conn.close()
    assert "create" in kinds
