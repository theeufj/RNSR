"""arms_bench report builder and trajectory counters, offline."""
from __future__ import annotations

import csv
import json

from rnsr.eval.arms_bench import build_report, trajectory_stats


def _csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_report_and_trajectory_counters(tmp_path):
    matter = tmp_path / "matter"
    matter.mkdir()
    (matter / "answer_key.json").write_text(json.dumps([
        {"id": "a", "question_type": "aggregation", "answer": "$10.00", "sources": ["2024-01-01_Inv_A.pdf"]},
        {"id": "b", "question_type": "value_lookup", "answer": "x", "sources": ["Letter_B.docx"]}]))
    _csv(matter / "questions.csv", [{"ground_truth_question": "qa", "field_id": "a"},
                                    {"ground_truth_question": "qb", "field_id": "b"}],
         ["ground_truth_question", "field_id"])
    out = tmp_path / "out"
    for arm, verdicts in (("classic", {"a": "OK", "b": "DIFF"}), ("docdb", {"a": "OK", "b": "OK"})):
        score = out / arm / "score"
        score.mkdir(parents=True)
        _csv(score / "comparison.csv",
             [{"field_id": k, "verdict": v, "golden": "g", "answer": "ans", "judge_status": "rejected"} for k, v in verdicts.items()],
             ["field_id", "verdict", "golden", "answer", "judge_status"])
        _csv(score / "answers_status.csv",
             [{"row": 0, "query_id": "a", "tier": "high", "quotes_verified": "1/1"},
              {"row": 1, "query_id": "b", "tier": "low", "quotes_verified": "0/0"}],
             ["row", "query_id", "tier", "quotes_verified"])
        base = out / arm / ("out" if arm == "docdb" else "")
        base.mkdir(parents=True, exist_ok=True)
        (base / "run_report.json").write_text(json.dumps({"provider": {"spend_usd": 1.5, "requests": 7}, "wall_s": 60}))
    traj = out / "work" / "trajectories"
    traj.mkdir(parents=True)
    with open(traj / "b1.jsonl", "w") as fh:
        for rec in [{"kind": "root_call_completed", "cost_usd": 0.2},
                    {"kind": "final", "value": {"q000": "10", "q001": "x"}, "verification": {
                        "q000": {"passed": True, "check": "source_bound_calculation", "quotes": [
                            {"quote": "Amount 10", "matched": True, "doc_id": "2024_01_01_inv_a"}],
                            "records": [{"kind": "aggregate", "n_rows": 2, "documents": ["2024_01_01_inv_a"]}]},
                        "q001": {"passed": True, "check": "lexical_source_match", "quotes": [
                            {"quote": "x", "matched": True, "doc_id": "other_doc"}]}}},
                    {"kind": "cell", "code": "rows = db.execute('SELECT 1')\nimport re\nre.search('x', 'y')"},
                    {"kind": "cell", "code": "hits = search('invoice total', k=5)\nr = calculate('sum', ids)"},
                    {"kind": "embed_batch"}, {"kind": "claim_review"}, {"kind": "claim_review_halted"},
                    {"kind": "search_rung", "rung": 2}]:
            fh.write(json.dumps(rec) + "\n")
    st = trajectory_stats("docdb", out)
    assert st["root_calls"] == 1 and st["cells"] == 2
    assert st["cells_sql"] == 1 and st["cells_search"] == 1 and st["cells_calc"] == 1   # re.search not counted
    assert st["embed_batches"] == 1 and st["review_halts"] == 1 and st["rungs"] == {"2": 1}

    summary = build_report(matter, out, ["classic", "docdb"], judge_passes=1, caps={"RNSR_MAX_ROOT_ITERS": "16"})
    assert summary["totals"]["classic"]["correct"] == 1 and summary["totals"]["docdb"]["correct"] == 2
    assert summary["by_type"]["value_lookup"] == {"classic": [0, 1], "docdb": [1, 1]}
    assert [m["id"] for m in summary["misses"]] == ["b"]
    cit = summary["totals"]["docdb"]["citations"]
    assert cit["verified_citation"] == 2 and cit["key_source_citation"] == 1
    assert cit["replayable_derivation"] == 1 and cit["derivation_coverage"] == 1.0
    assert summary["totals"]["classic"]["citations"]["verified_citation"] == 0
    md = (out / "report.md").read_text()
    assert "| Classic (flat string in REPL) | 1 / 2 | 50.0% |" in md
    assert "| `aggregation` (1) | 1 | 1 |" in md
    assert "1/0/1" in md   # tiers H/M/L for the fixture
