"""Rung-0 cells vs legacy replay — search semantics are agent contract."""

import json
from pathlib import Path

from rnsr.eval.replay import load_queries, replay

FIXTURE = Path(__file__).parent / "fixtures" / "replay"


def test_committed_fixture_matches_baseline():
    db = FIXTURE / "corpus.db"
    assert db.exists(), "committed replay corpus.db is missing"
    queries = load_queries(FIXTURE / "queries.json")
    assert queries
    baseline = json.loads((FIXTURE / "baseline.json").read_text())
    report = replay(db, queries, baseline=baseline)
    assert report["n_diffs"] == 0, report["diffs"]


def test_load_queries_keeps_rung0_only(tmp_path):
    path = tmp_path / "traj.jsonl"
    path.write_text(
        json.dumps({"kind": "search_rung", "rung": 0, "query": "Widgets"}) + "\n"
        + json.dumps({"kind": "search_rung", "rung": 2, "query": "ignored"}) + "\n"
        + json.dumps({"kind": "cell", "query": "nope"}) + "\n"
    )
    assert load_queries(path) == ["Widgets"]
