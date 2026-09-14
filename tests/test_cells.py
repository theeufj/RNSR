"""Stage 1 (engine-poc-plan): derived cells index, zone-map stats, mmap.

The invariants that matter: the cells path and the legacy per-table sweep
return the same rows for the same probes; artifacts without cells (old, or
ingested with cells_index off) keep working on the legacy path; zone stats
ride in the full manifest but never in the prompt-side compact view.
"""

import sqlite3

import pytest

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.env.search import Ladder
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


def _parse(path):
    return ParsedDocument(
        doc_id="acme", source_path=str(path), sha256="b" * 64, n_pages=1,
        parser="fake",
        elements=[Element("text", "ACME results. Revenue was $3,234 million.", 1)],
        tables=[RawTable(
            page=1,
            header=["Segment", "Revenue ($M)"],
            rows=[["Widgets", "$1,234"], ["Gadgets", "$2,000"],
                  ["Total", "$3,234"]],
            extractor="docling",
        )],
    )


@pytest.fixture
def corpus(tmp_path):
    out = tmp_path / "corpus.db"
    ingest([tmp_path / "acme.pdf"], out, parse=_parse)
    return out


@pytest.fixture
def corpus_nocells(tmp_path):
    out = tmp_path / "nocells.db"
    ingest([tmp_path / "acme.pdf"], out, parse=_parse,
           config=Settings(cells_index=False))
    return out


def _ladder(db) -> tuple[sqlite3.Connection, Ladder]:
    conn = sqlite3.connect(db)
    with CorpusDB(db) as c:
        manifest = c.manifest_dict()
        doc = c.doc_dict()
    return conn, Ladder(conn=conn, doc=doc, manifest=manifest,
                        rpc=lambda payload: {})


class TestCellsIndex:
    def test_cells_populated_with_text_and_numeric(self, corpus):
        conn = sqlite3.connect(corpus)
        rows = conn.execute(
            "SELECT col_name, text_value, num_value FROM cells "
            "ORDER BY row_idx, col_name").fetchall()
        # 3 rows x 2 columns
        assert len(rows) == 6
        by_col = {}
        for col, text, num in rows:
            by_col.setdefault(col, []).append((text, num))
        assert [t for t, _ in by_col["segment"]] == ["widgets", "gadgets", "total"]
        assert all(n is None for _, n in by_col["segment"])
        # numeric cells carry the coerced number AND the lowered raw string
        assert [n for _, n in by_col["revenue_m"]] == [1234.0, 2000.0, 3234.0]
        assert [t for t, _ in by_col["revenue_m"]] == ["$1,234", "$2,000", "$3,234"]

    def test_row_idx_matches_source_rowid(self, corpus):
        conn = sqlite3.connect(corpus)
        (table,) = conn.execute(
            "SELECT DISTINCT table_name FROM cells").fetchone()
        for row_idx, text in conn.execute(
                "SELECT row_idx, text_value FROM cells "
                "WHERE col_name='segment'"):
            src = conn.execute(
                f'SELECT segment FROM "{table}" WHERE rowid=?',
                (row_idx,)).fetchone()[0]
            assert src.lower() == text

    def test_cells_off_leaves_index_empty(self, corpus_nocells):
        conn = sqlite3.connect(corpus_nocells)
        assert conn.execute("SELECT count(*) FROM cells").fetchone()[0] == 0


class TestRung0Parity:
    def test_cells_and_legacy_paths_find_the_same_rows(self, corpus,
                                                       corpus_nocells):
        def probe(db, query):
            conn, ladder = _ladder(db)
            hits = ladder.search(query, rung=0)
            conn.close()
            return {(h["table"].split("__")[0], h["provenance"]["rowid"],
                     h["rows"]["segment"]) for h in hits}

        for query in ("revenue 3,234", "widgets segment", "gadgets 2,000"):
            cells_hits = probe(corpus, query)
            legacy_hits = probe(corpus_nocells, query)
            assert {h[2] for h in cells_hits} == {h[2] for h in legacy_hits}, query

    def test_cells_path_is_taken_when_populated(self, corpus, corpus_nocells):
        conn, ladder = _ladder(corpus)
        assert ladder._cells_ready() is True
        conn.close()
        conn, ladder = _ladder(corpus_nocells)
        assert ladder._cells_ready() is False   # empty index -> legacy path
        conn.close()

    def test_untrusted_tables_stay_excluded(self, corpus):
        conn, ladder = _ladder(corpus)
        for t in ladder.manifest["tables"]:
            t["status"] = "untrusted"
        assert ladder.search("widgets segment", rung=0) == []
        conn.close()

    def test_routing_gate_matches_legacy(self, corpus):
        # a term that overlaps no column name/caption (and no numbers)
        # must yield NO rung-0 hits — the ladder escalates to prose
        # instead of stopping on a weak table hit (both paths agree)
        conn, ladder = _ladder(corpus)
        assert ladder.search("gadgets", rung=0) == []
        ladder._cells_ok = False
        assert ladder.search("gadgets", rung=0) == []
        conn.close()

    def test_multi_column_match_no_duplicates_no_starvation(self, tmp_path):
        # the live Stage-1 bug: a row matching the query in SEVERAL columns
        # produced duplicate hits that exhausted the limit before
        # alphabetically-later tables were reached
        def parse(path):
            return ParsedDocument(
                doc_id="aaa", source_path=str(path), sha256="c" * 64,
                n_pages=1, parser="fake",
                elements=[Element("text", "filler", 1)],
                tables=[
                    RawTable(page=1, header=["Detail", "Detail value"],
                             rows=[["marriage date", "marriage 2014"]] * 8,
                             extractor="anydoc"),
                    # different header so multipage merge keeps it separate;
                    # 'detail_extra' still passes the routing gate
                    RawTable(page=1, header=["Detail extra", "Value"],
                             rows=[["certificate", "marriage 14/02/2014"]],
                             extractor="anydoc"),
                ],
            )

        out = tmp_path / "dup.db"
        ingest([tmp_path / "aaa.pdf"], out, parse=parse)
        conn, ladder = _ladder(out)
        hits = ladder.search("detail marriage certificate", rung=0, k=10)
        keys = [(h["table"], h["provenance"]["rowid"]) for h in hits]
        assert len(keys) == len(set(keys)), "duplicate rows returned"
        assert any(t.endswith("_002") for t, _ in keys), \
            "second table starved out of the results"
        conn.close()

    def test_hit_shape_is_unchanged(self, corpus):
        conn, ladder = _ladder(corpus)
        (hit,) = ladder.search("segment gadgets", rung=0, k=1)
        assert hit["rung"] == 0 and hit["kind"] == "sql"
        assert hit["rows"]["segment"] == "Gadgets"
        assert "revenue_m__raw" not in hit["text"]   # shadows stay out of text
        assert hit["provenance"]["table"] == hit["table"]
        conn.close()


class TestZoneMaps:
    def test_numeric_and_text_stats_in_manifest(self, corpus):
        with CorpusDB(corpus) as c:
            (table,) = c.manifest_dict()["tables"]
        stats = {e["name"]: e["stats"] for e in table["schema"]}
        assert stats["revenue_m"] == {"min": 1234.0, "max": 3234.0, "n_null": 0}
        assert stats["segment"]["n_distinct"] == 3
        assert "Widgets" in stats["segment"]["sample"]

    def test_stats_stay_out_of_the_prompt_manifest(self, corpus):
        from rnsr.harness.prompts.base import compact_manifest

        with CorpusDB(corpus) as c:
            compact = compact_manifest(c.manifest_dict())
        blob = str(compact)
        assert "n_distinct" not in blob and "Widgets" not in blob


class TestMmap:
    def test_read_pragma_applied_on_corpusdb(self, corpus):
        with CorpusDB(corpus) as c:
            assert c.conn.execute("PRAGMA mmap_size").fetchone()[0] > 0

    def test_env_override_disables(self, corpus, monkeypatch):
        monkeypatch.setenv("RNSR_DB_MMAP_BYTES", "0")
        with CorpusDB(corpus) as c:
            assert c.conn.execute("PRAGMA mmap_size").fetchone()[0] == 0
