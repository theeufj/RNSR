"""Public-benchmark arms: stratified subset and the cross-system report."""
from __future__ import annotations

import json
from pathlib import Path

from rnsr.eval.datasets.base import EvalItem
from rnsr.eval.datasets.financebench import stratified_subset
from rnsr.eval.public_arms_report import build


def _item(i, doc, cls):
    return EvalItem(qid=f"q{i:02d}", question="?", gold="1", task_class=cls, sources=[Path(f"/x/{doc}.pdf")],
                    meta={"doc_name": doc})


def test_stratified_subset_is_balanced_fixed_and_recorded(tmp_path):
    items = [_item(i, f"doc{i % 7}", "numeric" if i % 3 else "textual") for i in range(40)]
    rec = tmp_path / "subset.json"
    a = stratified_subset(items, 10, seed=3, record=rec)
    assert len(a) == 10 and sum(1 for i in a if i.task_class == "numeric") == 5
    assert json.loads(rec.read_text())["qids"] == [i.qid for i in a]
    # A second call honours the record even with a different seed.
    b = stratified_subset(items, 10, seed=99, record=rec)
    assert [i.qid for i in b] == [i.qid for i in a]


def test_report_compares_only_common_qids(tmp_path):
    def rows(system, correct_map, extra_qid=None):
        d = tmp_path / f"fb-{system}"
        d.mkdir()
        lines = [{"qid": q, "task_class": "numeric" if q == "q1" else "textual", "predicted": "x", "gold": "g",
                  "correct": ok, "status": "final", "latency_s": 1.0, "cost_usd": 0.5, "sub_calls": 0,
                  "iterations": 1, "scored_by": "string", "tier": "medium" if system == "docdb" else None}
                 for q, ok in correct_map.items()]
        if extra_qid:
            lines.append({**lines[0], "qid": extra_qid})
        (d / "results.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    rows("stuffing", {"q1": False, "q2": True})
    rows("docdb", {"q1": True, "q2": True}, extra_qid="q9")
    s = build(tmp_path, "fb", ["stuffing", "docdb"])
    assert s["n"] == 2 and s["totals"]["stuffing"]["correct"] == 1 and s["totals"]["docdb"]["correct"] == 2
    assert s["by_class"]["numeric"] == {"stuffing": [0, 1], "docdb": [1, 1]}
    assert [m["qid"] for m in s["misses"]] == ["q1"]
    md = (tmp_path / "fb-report.md").read_text()
    assert "| DocDB | 2 / 2 | 100.0% |" in md and "| `q1` | numeric |" in md
