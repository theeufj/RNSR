"""Miss-cause classifier and office-gen exposures."""

import pytest

from rnsr.eval.autopsy import (
    CAUSES,
    autopsy_run,
    classify_miss,
    format_only_miss,
    write_ledger,
)
from rnsr.eval.datasets.office_gen import generate_office
from rnsr.eval.metrics import EvalResult, summarize
from rnsr.harness.trajectory import TrajectoryWriter


def _result(**kw) -> EvalResult:
    base = dict(qid="q", task_class="lookup", predicted="wrong", gold="right",
                correct=False, status="final", latency_s=1.0, cost_usd=0.0,
                sub_calls=0, iterations=1)
    base.update(kw)
    return EvalResult(**base)


class TestFormatOnly:
    def test_prefix_and_tie_set(self):
        assert format_only_miss("Label: location", "location")
        assert format_only_miss(
            "location",
            "description and abstract concept, location, numeric value",
        )
        assert format_only_miss("Answer: 42", "42")
        assert not format_only_miss("Paris", "London")
        assert not format_only_miss(None, "42")


class TestClassify:
    def test_correct_is_ok(self):
        item = classify_miss(_result(predicted="right", gold="right", correct=True))
        assert item.cause == "ok"

    def test_gold_error_from_reviewer(self):
        item = classify_miss(_result(), reviewer_mark="gold-error")
        assert item.cause == "gold"

    def test_budget_when_not_final(self):
        item = classify_miss(_result(status="budget_exhausted"))
        assert item.cause == "budget"
        item = classify_miss(_result(status="recovered"))
        assert item.cause == "budget"

    def test_format_tie_set(self):
        item = classify_miss(_result(
            predicted="location",
            gold="description and abstract concept, location, numeric value",
        ))
        assert item.cause == "format"

    def test_ingest_untranscribed_scan(self):
        item = classify_miss(
            _result(gold="$312", predicted="unknown"),
            meta={"gold_doc": "receipt_scan", "exposure": "scanned_page",
                  "gold_page": 1},
            health={"scanned_pages_untranscribed": [
                {"doc_id": "receipt_scan", "pages": [1]}]},
        )
        assert item.cause == "ingest"

    def test_ingest_missing_attachment(self):
        item = classify_miss(
            _result(gold="$47", predicted="NOT_FOUND"),
            meta={"gold_doc": "cfo_q3_invoice", "exposure": "attachment",
                  "parent_doc": "cfo_q3_invoice", "child_doc": "invoice_widget"},
            manifest={"documents": [{"doc_id": "cfo_q3_invoice"}]},
        )
        assert item.cause == "ingest"

    def test_ingest_lost_sheet_identity(self):
        item = classify_miss(
            _result(gold="55000", predicted="sum of everything"),
            meta={"gold_doc": "budget", "exposure": "sheet_identity",
                  "sheet_name": "Q2"},
            manifest={"tables": [{"table_name": "t_budget_001", "title": None}]},
        )
        assert item.cause == "ingest"

    def test_retrieval_when_gold_doc_never_seen(self):
        records = [
            {"kind": "search_rung", "query": "revenue", "hits": 2},
            {"kind": "cell", "stdout": "doc=notes page=1 filler"},
            {"kind": "final", "value": "x", "verification": {"quotes": []}},
        ]
        item = classify_miss(
            _result(gold="$1,240,000", predicted="n/a"),
            records,
            meta={"gold_doc": "memo_q3_review"},
        )
        assert item.cause == "retrieval"

    def test_reasoning_when_gold_text_seen(self):
        records = [
            {"kind": "cell",
             "stdout": "doc=memo_q3_review Q3 revenue closed at $1,240,000"},
            {"kind": "final", "value": "980000",
             "verification": {"quotes": [
                 {"quote": "$1,240,000", "doc_id": "memo_q3_review", "page": 1}]}},
        ]
        item = classify_miss(
            _result(gold="$1,240,000", predicted="980000"),
            records,
            meta={"gold_doc": "memo_q3_review"},
        )
        assert item.cause == "reasoning"


class TestSummarizeCauseTable:
    def test_cause_x_class(self):
        results = [
            _result(qid="a", correct=True, predicted="right", gold="right"),
            _result(qid="b", task_class="lookup", cause="retrieval"),
            _result(qid="c", task_class="lookup", cause="reasoning"),
            _result(qid="d", task_class="aggregation", cause="ingest"),
        ]
        s = summarize(results)
        assert s["cause_counts"] == {
            "retrieval": 1, "reasoning": 1, "ingest": 1,
        }
        assert s["cause_x_class"]["lookup"]["retrieval"] == 1
        assert s["cause_x_class"]["aggregation"]["ingest"] == 1

    def test_retrieval_recall(self):
        results = [
            _result(qid="a", correct=True, predicted="right", gold="right",
                    retrieval_hit=True),
            _result(qid="b", retrieval_hit=False),
            _result(qid="c", retrieval_hit=True),
        ]
        s = summarize(results)
        assert s["retrieval_recall"] == pytest.approx(2 / 3)


class TestAutopsyRun:
    def test_writes_ledger_from_results_jsonl(self, tmp_path):
        import json

        results = [
            _result(qid="ok1", predicted="right", gold="right", correct=True),
            _result(qid="miss1", predicted="location",
                    gold="location, numeric value", task_class="tie"),
        ]
        (tmp_path / "results.jsonl").write_text(
            "".join(json.dumps(r.to_dict()) + "\n" for r in results))
        with TrajectoryWriter(tmp_path / "trajectories", "miss1") as w:
            w.event("start", question="least common?")
            w.event("final", value="location")
            w.event("end", status="final")
        ledger = autopsy_run(tmp_path)
        assert ledger["n_miss"] == 1
        assert ledger["cause_counts"]["format"] == 1
        written = write_ledger(ledger, tmp_path / "out")
        assert "format" in (tmp_path / "out" / "loss-ledger.md").read_text()
        assert written["json"].endswith("autopsy.json")


class TestOfficeGen:
    def test_deterministic_and_exposures(self, tmp_path):
        a = generate_office(tmp_path / "a", seed=7)
        b = generate_office(tmp_path / "b", seed=7)
        assert [i.gold for i in a] == [i.gold for i in b]
        assert [i.qid for i in a] == [i.qid for i in b]
        exposures = {i.meta.get("exposure") for i in a}
        assert {"docx", "sheet_identity", "attachment",
                "supersession", "scanned_page"} <= exposures
        assert any(i.expect == "absent" for i in a)
        files = {p.name for p in (tmp_path / "a").iterdir()}
        assert "memo_q3_review.docx" in files
        assert "budget.xlsx" in files
        assert "cfo_q3_invoice.eml" in files
        assert "receipt_scan.pdf" in files
        assert "policy_v1_superseded.pdf" in files
        assert "policy_v2_current.pdf" in files

    def test_idempotent(self, tmp_path):
        generate_office(tmp_path, seed=3)
        first = (tmp_path / "budget.xlsx").read_bytes()
        generate_office(tmp_path, seed=3)
        assert (tmp_path / "budget.xlsx").read_bytes() == first

    def test_all_causes_named(self):
        assert CAUSES == ("gold", "budget", "format", "ingest",
                          "retrieval", "reasoning")
