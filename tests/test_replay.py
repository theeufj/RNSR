"""Rung-0 cells vs legacy replay — search semantics are agent contract."""

import json
import shutil
from pathlib import Path

from rnsr.db.migrate import migrate_artifact
from rnsr.eval.replay import load_queries, replay

FIXTURE = Path(__file__).parent / "fixtures" / "replay"


def test_committed_fixture_matches_baseline(tmp_path):
    original = FIXTURE / "corpus.db"
    assert original.exists(), "committed replay corpus.db is missing"
    # The historical replay artifact also exercises the supported format migration.
    db = tmp_path / "replay.db"
    shutil.copyfile(original, db)
    migrate_artifact(db)
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
