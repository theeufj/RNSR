"""review-import and answers.xlsx / report card."""

import csv

from rnsr.eval.report_card import write_report_card
from rnsr.eval.review_import import import_review
from rnsr.eval.xlsx_out import write_answers_xlsx


def test_import_review_writes_golden_and_diff(tmp_path):
    review = tmp_path / "review.csv"
    with open(review, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "answer", "tier",
                                           "reviewer_mark", "note"])
        w.writeheader()
        w.writerow({"qid": "q001", "answer": "wrong", "tier": "low",
                    "reviewer_mark": "wrong",
                    "note": "Prefer the latest policy version when they conflict."})
        w.writerow({"qid": "q002", "answer": "ok", "tier": "high",
                    "reviewer_mark": "ok", "note": ""})
    result = import_review(review, out_dir=tmp_path / "out")
    assert result["n_miss"] == 1
    assert result["n_ok"] == 1
    assert (tmp_path / "out" / "golden" / "from_review.json").exists()
    assert (tmp_path / "out" / "playbook.diff.json").exists()


def test_answers_xlsx_and_report_card(tmp_path):
    path = write_answers_xlsx(
        tmp_path / "answers.xlsx",
        [{"question": "Q?", "value": "42", "tier": "high",
          "doc": "memo", "page": "1", "quote": "1/1"}],
        question_col="question",
    )
    assert path.exists() and path.stat().st_size > 0
    report = write_report_card(
        tmp_path,
        report={"questions": 1, "tier_counts": {"high": 1, "medium": 0, "low": 0},
                "provider": {"spend_usd": 0.1}, "wall_s": 1},
        health={"grade": "ok", "findings": []},
        ledger={"cause_counts": {"reasoning": 0}, "accuracy": 1.0},
    )
    text = report.read_text()
    assert "high: 1" in text
    assert "grade: ok" in text or "health: ok" in text
