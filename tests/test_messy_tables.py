"""Messy-table generator, row-kind classification, labelled scorer."""

from rnsr.eval.datasets.messy_tables import generate_messy_tables, load_labelled_tables
from rnsr.eval.tables_score import score_labelled_tables
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.validate import classify_row_kind


class TestClassifyRowKind:
    def test_total_and_subtotal(self):
        assert classify_row_kind(["Total", "100"]) == "total"
        assert classify_row_kind(["Subtotal", "40"]) == "subtotal"
        assert classify_row_kind(["Widgets", "10"]) == "data"

    def test_footnote_and_section(self):
        assert classify_row_kind(["* see note", "", ""]) == "footnote"
        assert classify_row_kind(["(1)", None, None]) == "footnote"
        assert classify_row_kind(["Operating activities", "", ""]) == "section"

    def test_net_income_is_not_a_total(self):
        assert classify_row_kind(["Net income", "50"]) == "data"


class TestGenerator:
    def test_writes_labels_and_pdfs(self, tmp_path):
        path = generate_messy_tables(tmp_path)
        spec = load_labelled_tables(tmp_path)
        assert path.name == "labels.json"
        assert len(spec["tables"]) >= 3
        docs = {t["doc"] for t in spec["tables"]}
        for name in docs:
            assert (tmp_path / name).exists()


class TestScorer:
    def test_scores_synthetic_via_fake_parse(self, tmp_path, monkeypatch):
        generate_messy_tables(tmp_path)

        def parse(path):
            name = path.stem
            if "us_invoice" in name:
                rows = [["Widgets", "1,200.00"], ["Gadgets", "800.00"],
                        ["Total", "2,000.00"]]
                header = ["Item", "Amount ($)"]
            elif "eu_ledger" in name:
                rows = [["Miete", "1.234,50"], ["Strom", "800,00"],
                        ["Subtotal", "2.034,50"], ["Korrektur", "(34,50)"],
                        ["Total", "2.000,00"]]
                header = ["Konto", "Betrag"]
            elif "multipage" in name:
                rows = ([[f"Item {i}", str(i), str(i * 10)] for i in range(1, 21)]
                        + [["* see terms", "", ""], ["Total", "210", "2100"]])
                header = ["Line", "Qty", "Price"]
            else:
                rows = [["2026-01-03", "Salary", "3,400.00"],
                        ["2026-01-08", "Rent", "(1,800.00)"],
                        ["Total", "", "1,600.00"]]
                header = ["Date", "Description", "Amount"]
            return ParsedDocument(
                doc_id=name.replace("-", "_"),
                source_path=str(path),
                sha256="b" * 64,
                n_pages=1,
                parser="fake",
                elements=[Element("text", name, 1)],
                tables=[RawTable(page=1, header=header, rows=rows, extractor="fake")],
            )

        import rnsr.eval.tables_score as ts

        def ingest_with_parse(sources, out_db, *, config=None, **kw):
            from rnsr.ingest.pipeline import ingest
            return ingest(sources, out_db, config=config, parse=parse)

        monkeypatch.setattr(ts, "ingest", ingest_with_parse)
        report = score_labelled_tables(tmp_path)
        required = [r for r in report["results"] if r["must_pass"]]
        assert required
        assert all(r["passed"] for r in required), report["results"]
