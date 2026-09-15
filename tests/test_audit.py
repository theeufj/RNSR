"""audit-export evidence packet and review scoring."""

import json

from rnsr.eval.audit import export_audit, extract_evidence, score_review
from rnsr.harness.trajectory import TrajectoryWriter


def test_extract_evidence_omits_cell_transcript():
    records = [
        {"kind": "start", "question": "What is revenue?"},
        {"kind": "cell", "code": "print(db.execute('SELECT amt FROM t').fetchall())",
         "stdout": "secret row dump"},
        {"kind": "final", "value": "42",
         "verification": {"quotes": [
             {"quote": "revenue was 42", "matched": True, "doc_id": "a", "page": 1}
         ]}},
        {"kind": "end", "status": "final"},
    ]
    ev = extract_evidence(records, qid="q000", health={"grade": "ok"})
    assert ev["answer"] == "42"
    assert ev["verified_quotes"][0]["matched"] is True
    assert ev["pages"] == [1]
    assert ev["health"]["grade"] == "ok"
    assert "secret row dump" not in json.dumps(ev)
    dumped = json.dumps(ev)
    assert "SELECT" in dumped or ev["sql"]


def test_export_and_score_review(tmp_path):
    work = tmp_path / "work"
    traj = work / "trajectories"
    with TrajectoryWriter(traj, "q000") as w:
        w.event("start", question="Q?")
        w.event("cell", code="SELECT 1", stdout="1")
        w.event("final", value="yes",
                verification={"quotes": [
                    {"quote": "yes", "matched": True, "doc_id": "a"}]})
        w.event("end", status="final")
    out = tmp_path / "packet"
    result = export_audit(work, out)
    ev = json.loads((out / "evidence" / "q000.json").read_text())
    assert ev["qid"] == "q000"
    assert ev["answer"] == "yes"
    review = (out / "review.csv").read_text()
    assert "qid,answer,tier,reviewer_mark,note" in review
    assert result["n"] == 1

    sheet = out / "review.csv"
    sheet.write_text(
        "qid,answer,tier,reviewer_mark,note\n"
        "q000,yes,low,wrong,should be no\n"
        "q001,no,high,correct,\n"
    )
    scored = score_review(sheet)
    assert scored["misses"] == 1
    assert scored["correct"] == 1
    assert scored["miss_list"][0]["qid"] == "q000"
