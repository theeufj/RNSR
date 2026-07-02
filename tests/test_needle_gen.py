"""needle_gen: PDFs are real, golds exact, and the docdb path answers by SQL."""

import pytest

from rnsr.eval.datasets.needle_gen import generate_needle_set


def test_deterministic(tmp_path):
    a = generate_needle_set(tmp_path / "a", n_docs=1, questions_per_doc=2)
    b = generate_needle_set(tmp_path / "b", n_docs=1, questions_per_doc=2)
    assert [(i.question, i.gold) for i in a] == [(i.question, i.gold) for i in b]


def test_items_shape(tmp_path):
    items = generate_needle_set(tmp_path, n_docs=2, questions_per_doc=3)
    assert len(items) == 6
    assert all(i.task_class == "numeric" for i in items)
    assert all(i.sources[0].exists() for i in items)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_needle_reachable_via_sql_after_ingest(tmp_path):
    pytest.importorskip("docling")
    from rnsr.db.artifact import CorpusDB
    from rnsr.ingest.pipeline import ingest

    items = generate_needle_set(tmp_path / "pdfs", n_docs=1, tables_per_doc=2,
                                questions_per_doc=1)
    item = items[0]
    report = ingest(item.sources, tmp_path / "corpus.db")
    assert report.validation_pass_rate >= 0.99, report.to_json()

    gold = int(item.gold)
    with CorpusDB(tmp_path / "corpus.db") as corpus:
        found = False
        for t in corpus.manifest_dict()["tables"]:
            numeric_cols = [c["name"] for c in t["schema"] if c["type"] != "TEXT"]
            for col in numeric_cols:
                hit = corpus.conn.execute(
                    f'SELECT 1 FROM "{t["table_name"]}" WHERE "{col}" = ?', (gold,)
                ).fetchone()
                if hit:
                    found = True
        assert found, f"needle {gold} not reachable via SQL"
