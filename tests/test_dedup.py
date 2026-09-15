"""Exact and near-duplicate detection."""

from rnsr.ingest.dedup import NEAR_DUP_BITS, hamming, simhash64


def test_identical_text_same_fingerprint():
    assert simhash64("The expense limit is 250") == simhash64(
        "the expense limit is 250")


def test_near_duplicate_hamming():
    body = (
        "This policy supersedes all earlier drafts. Employees must obtain "
        "pre-approval for travel. The single-transaction expense limit is "
    )
    a = simhash64(body + "250 dollars. Approved and in force.")
    b = simhash64(body + "400 dollars. Approved and in force.")
    c = simhash64("Parking permits renewed. Catering for the offsite is booked.")
    assert hamming(a, b) < hamming(a, c)
    assert hamming(a, b) <= NEAR_DUP_BITS + 4


def test_detect_duplicates_stamps_older_copy(tmp_path):
    import sqlite3

    from rnsr.db import schema
    from rnsr.ingest.dedup import detect_duplicates

    conn = sqlite3.connect(":memory:")
    schema.create_corpus_db(conn)
    text = "Policy body. The limit is 250. Approved."
    for i, doc_id in enumerate(("v1", "v2")):
        schema.insert_document(
            conn, doc_id=doc_id, source_path=f"/{doc_id}.txt",
            sha256=f"{i:064x}", n_pages=1, parser="text",
            ingested_at="2026-01-01T00:00:00",
            content_sha256="same" if i == 0 else "same",
            modified_at=f"2026-01-0{i + 1}T00:00:00",
        )
        conn.execute(
            "INSERT INTO doc_text VALUES (?,?,?,?,?)",
            (doc_id, 1, 0, len(text), text),
        )
    groups = detect_duplicates(conn)
    assert groups
    kept = groups[0]["kept"]
    assert kept == "v2"
    assert conn.execute(
        "SELECT duplicate_of FROM documents WHERE doc_id='v1'"
    ).fetchone()[0] == "v2"
    conn.close()
