"""tables.py: DDL generation, shadow columns, provenance, multi-page merge."""

import json
import sqlite3

import pytest

from rnsr.db import schema
from rnsr.ingest.model import RawTable
from rnsr.ingest.tables import build_data_table, headers_match, merge_multipage


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    schema.create_corpus_db(c)
    yield c
    c.close()


def _financial_table(**kw):
    defaults = dict(
        page=3,
        bbox=(50.0, 100.0, 550.0, 300.0),
        header=["Segment", "Revenue ($M)", "Margin %"],
        rows=[
            ["Widgets", "$1,234", "45%"],
            ["Gadgets", "$2,000", "30%"],
            ["Total", "$3,234", "75%"],
        ],
        extractor="docling",
        caption="Revenue by segment",
    )
    defaults.update(kw)
    return RawTable(**defaults)


class TestBuildDataTable:
    def test_columns_and_types(self, conn):
        built = build_data_table(conn, "doc1", 1, _financial_table())
        assert built.name == "t_doc1_001"
        cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(t_doc1_001)")}
        assert cols["segment"] == "TEXT"
        assert cols["revenue_m"] == "INTEGER"
        assert cols["revenue_m__raw"] == "TEXT"
        assert cols["margin"] in ("INTEGER", "REAL")
        assert cols["_page"] == "INTEGER"
        assert cols["_bbox"] == "TEXT"
        assert cols["_extractor"] == "TEXT"
        # text column has no shadow
        assert "segment__raw" not in cols

    def test_values_and_shadows(self, conn):
        build_data_table(conn, "doc1", 1, _financial_table())
        rows = conn.execute(
            "SELECT segment, revenue_m, revenue_m__raw FROM t_doc1_001 ORDER BY rowid"
        ).fetchall()
        assert rows[0] == ("Widgets", 1234, "$1,234")
        assert rows[2] == ("Total", 3234, "$3,234")

    def test_provenance_stamped(self, conn):
        build_data_table(conn, "doc1", 1, _financial_table())
        page, bbox, extractor = conn.execute(
            "SELECT _page, _bbox, _extractor FROM t_doc1_001 LIMIT 1"
        ).fetchone()
        assert page == 3
        assert json.loads(bbox) == [50.0, 100.0, 550.0, 300.0]
        assert extractor == "docling"

    def test_frozen_after_build(self, conn):
        build_data_table(conn, "doc1", 1, _financial_table())
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE t_doc1_001 SET revenue_m = 0")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM t_doc1_001")

    def test_schema_entries_record_rules(self, conn):
        built = build_data_table(conn, "doc1", 1, _financial_table())
        by_name = {e["name"]: e for e in built.schema_entries}
        assert by_name["revenue_m"]["coercion_rule"]["features"] == ["currency"]
        assert by_name["revenue_m"]["raw_col"] == "revenue_m__raw"
        assert by_name["segment"]["coercion_rule"] is None

    def test_style_override_text_forces_text(self, conn):
        built = build_data_table(
            conn, "doc1", 1, _financial_table(), style_overrides={"revenue_m": "text"}
        )
        by_name = {e["name"]: e for e in built.schema_entries}
        assert by_name["revenue_m"]["type"] == "TEXT"
        # raw strings preserved as the column value itself
        val = conn.execute("SELECT revenue_m FROM t_doc1_001 LIMIT 1").fetchone()[0]
        assert val == "$1,234"

    def test_style_override_eu(self, conn):
        t = _financial_table(
            header=["Item", "Amount"],
            rows=[["A", "1.234,50"], ["B", "2.000,00"]],
        )
        built = build_data_table(conn, "doc1", 1, t, style_overrides={"amount": "eu"})
        assert conn.execute("SELECT amount FROM t_doc1_001").fetchall() == [(1234.5,), (2000.0,)]
        by_name = {e["name"]: e for e in built.schema_entries}
        assert by_name["amount"]["coercion_rule"]["style"] == "eu"

    def test_ragged_rows_padded(self, conn):
        t = _financial_table(rows=[["OnlyName"], ["Both", "$5", "1%"]])
        build_data_table(conn, "doc1", 1, t)
        rows = conn.execute("SELECT segment, revenue_m FROM t_doc1_001 ORDER BY rowid").fetchall()
        assert rows[0] == ("OnlyName", None)

    def test_built_metadata(self, conn):
        built = build_data_table(conn, "doc1", 2, _financial_table())
        assert (built.page_start, built.page_end) == (3, 3)
        assert built.n_rows == 3 and built.n_cols == 3
        assert built.caption == "Revenue by segment"
        assert not built.multipage


class TestMultipageMerge:
    def _fragment(self, page, rows):
        return RawTable(page=page, header=["Item", "Amount"], rows=rows, extractor="docling")

    def test_merges_consecutive_repeated_headers(self):
        parts = [
            self._fragment(1, [["A", "1"], ["B", "2"]]),
            self._fragment(2, [["C", "3"]]),
            self._fragment(3, [["D", "4"]]),
        ]
        merged = merge_multipage(parts)
        assert len(merged) == 1
        assert len(merged[0].rows) == 4
        assert merged[0].row_pages == [1, 1, 2, 3]

    def test_different_headers_not_merged(self):
        parts = [
            self._fragment(1, [["A", "1"]]),
            RawTable(page=2, header=["Name", "Total"], rows=[["X", "9"]]),
        ]
        assert len(merge_multipage(parts)) == 2

    def test_nonadjacent_pages_not_merged(self):
        parts = [self._fragment(1, [["A", "1"]]), self._fragment(5, [["B", "2"]])]
        assert len(merge_multipage(parts)) == 2

    def test_header_match_normalizes_whitespace_case(self):
        assert headers_match(["Net  Revenue"], ["net revenue"])
        assert not headers_match(["Revenue"], ["Revenue", "Extra"])

    def test_merged_table_gets_source_page_column(self, conn):
        parts = [
            self._fragment(1, [["A", "1"], ["B", "2"]]),
            self._fragment(2, [["C", "3"]]),
        ]
        merged = merge_multipage(parts)
        built = build_data_table(conn, "doc1", 1, merged[0])
        assert built.multipage
        rows = conn.execute(
            "SELECT item, source_page, _page FROM t_doc1_001 ORDER BY rowid"
        ).fetchall()
        assert rows == [("A", 1, 1), ("B", 1, 1), ("C", 2, 2)]
        assert (built.page_start, built.page_end) == (1, 2)
