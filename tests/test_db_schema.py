"""db layer: schema creation, immutability triggers, FTS, artifact wrapper."""

import sqlite3

import pytest

from rnsr.db import fts, schema
from rnsr.db.artifact import CorpusDB


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    schema.create_corpus_db(c)
    yield c
    c.close()


def _seed_document(conn, doc_id="doc1", text="Alpha beta. Gamma delta epsilon."):
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.pdf", "0" * 64, 1, "test", "2026-07-02T00:00:00"),
    )
    conn.execute(
        "INSERT INTO doc_text VALUES (?, 1, 0, ?, ?)", (doc_id, len(text), text)
    )
    conn.execute(
        "INSERT INTO chunks (doc_id, page, char_start, char_end, heading_path, text) "
        "VALUES (?, 1, 0, ?, NULL, ?)",
        (doc_id, len(text), text),
    )


def _seed_data_table(conn, table="t_doc1_001"):
    schema.create_data_table(
        conn, table, [("revenue", "REAL"), ("revenue__raw", "TEXT"), ("segment", "TEXT")]
    )
    conn.execute(
        f'INSERT INTO "{table}" VALUES (?, ?, ?, ?, ?, ?)',
        (1234.0, "$1,234", "Widgets", 3, "[0,0,100,50]", "docling"),
    )
    schema.freeze_table(
        conn, table,
        source_columns=["revenue", "revenue__raw", "segment", *schema.PROVENANCE_COLUMNS],
    )
    return table


class TestImmutability:
    def test_insert_blocked_after_freeze(self, conn):
        table = _seed_data_table(conn)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(f'INSERT INTO "{table}" VALUES (1, "1", "x", 1, "[]", "t")')

    def test_delete_blocked(self, conn):
        table = _seed_data_table(conn)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(f'DELETE FROM "{table}"')

    def test_update_source_column_blocked(self, conn):
        table = _seed_data_table(conn)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(f'UPDATE "{table}" SET revenue = 999')

    def test_update_provenance_blocked(self, conn):
        table = _seed_data_table(conn)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(f'UPDATE "{table}" SET _page = 99')

    def test_annotation_column_writable(self, conn):
        table = _seed_data_table(conn)
        assert schema.add_annotation_column(conn, table, "label") is True
        conn.execute(f'UPDATE "{table}" SET label = ?', ("recurring",))
        assert conn.execute(f'SELECT label FROM "{table}"').fetchone()[0] == "recurring"

    def test_add_annotation_column_idempotent(self, conn):
        table = _seed_data_table(conn)
        schema.add_annotation_column(conn, table, "label")
        assert schema.add_annotation_column(conn, table, "label") is False

    def test_core_tables_frozen_by_finalize(self, conn):
        _seed_document(conn)
        schema.finalize_corpus(conn)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM chunks")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE doc_text SET text = 'gone'")
        # manifest stays writable after finalize
        conn.execute("INSERT INTO manifest VALUES ('k', '1')")


class TestFTS:
    def test_populate_and_match(self, conn):
        _seed_document(conn, text="The net revenue total was 1,234 dollars in fiscal 2023.")
        n = fts.populate_fts(conn)
        assert n == 1
        hits = fts.match(conn, "revenue")
        assert len(hits) == 1
        assert hits[0]["doc_id"] == "doc1"
        assert "revenue" in hits[0]["text"]

    def test_porter_stemming(self, conn):
        _seed_document(conn, text="The company was rapidly expanding its operations.")
        fts.populate_fts(conn)
        assert fts.match(conn, "operation")  # porter: operations ~ operation

    def test_bad_query_syntax_returns_empty_not_raise(self, conn):
        _seed_document(conn)
        fts.populate_fts(conn)
        assert fts.match(conn, 'unbalanced " AND ((') == []


class TestSanitize:
    def test_basic(self):
        assert schema.sanitize_column_name("Net Revenue ($M)") == "net_revenue_m"

    def test_leading_digit(self):
        assert schema.sanitize_column_name("2023 Total") == "c_2023_total"

    def test_collision(self):
        taken: set[str] = set()
        a = schema.sanitize_column_name("Total", taken)
        b = schema.sanitize_column_name("Total", taken)
        assert (a, b) == ("total", "total_2")

    def test_empty(self):
        assert schema.sanitize_column_name("  ") == "col"


class TestArtifact:
    def test_create_open_roundtrip(self, tmp_path):
        db_path = tmp_path / "corpus.db"
        with CorpusDB.create(db_path) as corpus:
            _seed_document(corpus.conn)
            corpus.manifest_set("chunk_stats", {"n_chunks": 1})
            corpus.conn.commit()
        with CorpusDB(db_path, mode="ro") as corpus:
            assert corpus.doc_ids() == ["doc1"]
            assert corpus.full_text("doc1").startswith("Alpha beta")
            assert corpus.manifest_get("chunk_stats") == {"n_chunks": 1}
            assert corpus.doc_dict()["doc1"] == corpus.full_text("doc1")

    def test_ro_mode_blocks_writes(self, tmp_path):
        db_path = tmp_path / "corpus.db"
        CorpusDB.create(db_path).close()
        with CorpusDB(db_path, mode="ro") as corpus, pytest.raises(sqlite3.OperationalError):
            corpus.manifest_set("k", 1)

    def test_create_refuses_overwrite(self, tmp_path):
        db_path = tmp_path / "corpus.db"
        CorpusDB.create(db_path).close()
        with pytest.raises(FileExistsError):
            CorpusDB.create(db_path)

    def test_manifest_dict_includes_tables(self, tmp_path):
        db_path = tmp_path / "corpus.db"
        with CorpusDB.create(db_path) as corpus:
            _seed_document(corpus.conn)
            corpus.conn.execute(
                "INSERT INTO manifest_tables VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("t_doc1_001", "doc1", "Revenue by segment", 3, 3, 10, 2,
                 '[{"name": "revenue", "type": "REAL"}]', 0.93,
                 '{"arithmetic": {"passed": true}}', "trusted", "docling"),
            )
            m = corpus.manifest_dict()
            assert m["tables"][0]["table_name"] == "t_doc1_001"
            assert m["tables"][0]["schema"][0]["name"] == "revenue"
            assert m["tables"][0]["checks"]["arithmetic"]["passed"] is True
