"""OOLONG label diagnostics: gold lines parse, labels join by instance text,
and the cross-tab separates labelling from counting."""
import json
import sqlite3

from rnsr.eval.oolong_report import (
    build,
    docdb_labels,
    instance_of,
    label_diagnostics,
    parse_gold_lines,
    text_corpus_path,
)

LABELLED = (
    "The following lines contain 3 general-knowledge questions, one per line.\n\n"
    "Date: Feb 02, 2023 || User: 76153 || Instance: What is one of the languages of the Sioux ? || Label: entity\n"
    "Date: May 09, 2023 || User: 77482 || Instance: Who founded the Red Cross ? || Label: human being\n"
    "Date: Nov 11, 2024 || User: 17297 || Instance: How many legs does a spider have ? || Label: numeric value\n")
CONTEXT = LABELLED.replace(" || Label: entity", "").replace(" || Label: human being", "").replace(" || Label: numeric value", "")


def test_gold_lines_parse_and_instance_extraction():
    gold = parse_gold_lines(LABELLED)
    assert [g["label"] for g in gold] == ["entity", "human being", "numeric value"]
    assert gold[1]["user"] == "77482" and gold[1]["instance"] == "Who founded the Red Cross ?"
    assert instance_of("Date: x || User: 1 || Instance: Who founded the Red Cross ?") == "Who founded the Red Cross ?"
    assert instance_of("The following lines contain") is None


def _corpus(tmp_path, labels):
    db = text_corpus_path(CONTEXT, tmp_path)
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE t_context_001 (line_no INTEGER, text TEXT, cls TEXT)')
    for i, (line, lab) in enumerate(zip(CONTEXT.splitlines()[2:], labels, strict=True)):
        conn.execute("INSERT INTO t_context_001 VALUES (?,?,?)", (i + 3, line, lab))
    conn.commit()
    conn.close()
    return db


def _run(tmp_path, labels, correct):
    rd = tmp_path / "oolong-docdb"
    (rd / "trajectories").mkdir(parents=True)
    _corpus(tmp_path, labels)
    cls = {"certified": True, "table": "t_context_001", "column": "cls", "operation": "count",
           "allowed_labels": ["entity", "human being", "numeric value"], "where": None}
    (rd / "trajectories" / "q1.jsonl").write_text(json.dumps(
        {"kind": "final", "value": "2", "verification": {"check": "classification_aggregate", "classification": cls}}) + "\n")
    row = {"qid": "q1", "correct": correct, "task_class": "counting", "predicted": "2", "gold": "1",
           "trajectory_path": str(rd / "trajectories" / "q1.jsonl")}
    (rd / "results.jsonl").write_text(json.dumps(row) + "\n")
    return rd, row


def test_label_diagnostics_separate_labelling_from_counting(tmp_path):
    rd, row = _run(tmp_path, ["entity", "entity", "numeric value"], correct=False)   # one label wrong
    items = {"q1": {"task": "NUMERIC_ONE_CLASS", "context_len": 1024, "context": CONTEXT, "labelled": LABELLED}}
    d = label_diagnostics(items, {"q1": row}, rd, tmp_path)
    item = d["per_item"]["q1"]
    assert item["labels"] == "read" and item["n_labelled"] == 3 and abs(item["label_accuracy"] - 2 / 3) < 1e-9
    assert not item["counts_exact"] and not item["ranking_kept"]
    assert d["cross_tab"] == {"ranking_changed|answer_wrong": 1}
    assert d["confusion"]["human being"] == {"entity": 1}
    assert docdb_labels(text_corpus_path(CONTEXT, tmp_path), "t_context_001", "missing_col", None) is None


def test_build_writes_strata_and_skips_non_label_classifications(tmp_path):
    rd, row = _run(tmp_path, ["entity", "human being", "numeric value"], correct=True)
    # a second arm with the same qid and no labels
    other = tmp_path / "oolong-stuffing"
    other.mkdir()
    (other / "results.jsonl").write_text(json.dumps({**row, "correct": False, "trajectory_path": None}) + "\n")
    items = {"q1": {"task": "NUMERIC_ONE_CLASS", "context_len": 1024, "context": CONTEXT, "labelled": LABELLED}}
    s = build(tmp_path, ["stuffing", "docdb"], items, cache_dir=tmp_path)
    assert s["strata"]["task"]["NUMERIC_ONE_CLASS"] == {"stuffing": [0, 1], "docdb": [1, 1]}
    assert s["labels"]["docdb"]["cross_tab"] == {"counts_exact|answer_right": 1}
    md = (tmp_path / "oolong-strata.md").read_text()
    assert "| `1024` | 1 | 0 / 1 | 1 / 1 |" in md and "keep no per-line labels" in md
