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


class TestMatterGen:
    def test_deterministic_and_golds_consistent(self, tmp_path):
        from rnsr.eval.datasets.matter_gen import MatterFacts, generate_matter

        a = generate_matter(tmp_path / "a", n_filler=2, filler_chars=2000, seed=5)
        b = generate_matter(tmp_path / "b", n_filler=2, filler_chars=2000, seed=5)
        assert [(i.question, i.gold) for i in a] == [(i.question, i.gold) for i in b]

        f = MatterFacts(5)
        by_class = {i.task_class: i for i in a}
        # invoice arithmetic golds match the fact system
        totals = next(i for i in a if "ALL tax invoices" in i.question)
        assert totals.gold == f"${f.total_invoiced:,}"
        outstanding = next(i for i in a if "still owed" in i.question)
        assert outstanding.gold == f"${sum(x['amount'] for x in f.outstanding):,}"
        # cure deadline is breach + cure_days
        from datetime import timedelta
        deadline = next(i for i in a if "drop-dead date" in i.question)
        assert deadline.gold == (f.breach_date + timedelta(days=f.cure_days)).strftime("%-d %B %Y")
        assert "absent" in by_class and "cross-doc" in by_class

    def test_pdfs_written_and_sources_attached(self, tmp_path):
        from rnsr.eval.datasets.matter_gen import generate_matter

        items = generate_matter(tmp_path / "m", n_filler=3, filler_chars=2000, seed=7)
        assert all(len(i.sources) == items[0].meta["n_docs"] for i in items)
        assert all(p.exists() for p in items[0].sources)
        # operative docs present
        names = {p.name for p in items[0].sources}
        assert "01_master_services_agreement.pdf" in names
        assert "21_breach_notice.pdf" in names

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_small_matter_ingests_and_needle_reachable(self, tmp_path):
        pytest.importorskip("docling")
        from rnsr.db.artifact import CorpusDB
        from rnsr.eval.datasets.matter_gen import MatterFacts, generate_matter
        from rnsr.ingest.pipeline import ingest

        items = generate_matter(tmp_path / "m", n_filler=2, filler_chars=1500, seed=5)
        report = ingest(items[0].sources, tmp_path / "matter.db")
        assert report.validation_pass_rate >= 0.7, report.to_json()
        f = MatterFacts(5)
        with CorpusDB(tmp_path / "matter.db") as corpus:
            docs = corpus.doc_dict()
            joined = " ".join(docs.values())
            assert f"{f.indemnity_cap_0:,}" in joined      # original cap retained
            assert f"{f.indemnity_cap_2:,}" in joined      # amendment retained
