"""Replay rung-0 search queries against cells and legacy paths.

Rung-0 hit semantics are part of the agent contract. A faster cells path
that returns a slightly different row set has already dropped golden-matter
accuracy. This module compares both paths to each other and to a frozen
baseline, and exits non-zero on any diff.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rnsr.db.artifact import CorpusDB
from rnsr.env.lazydoc import LazyDoc
from rnsr.env.search import Ladder


def load_queries(*sources: str | Path) -> list[str]:
    """Collect unique queries from trajectory JSONL files or a queries.json."""
    queries: list[str] = []
    for src in sources:
        path = Path(src)
        raw = path.read_text()
        if path.suffix == ".json":
            data = json.loads(raw)
            if isinstance(data, list):
                queries.extend(str(x) for x in data)
            elif isinstance(data, dict):
                queries.extend(str(q) for q in data.get("queries", []))
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "search_rung" and rec.get("rung", 0) == 0:
                queries.append(rec["query"])
    return list(dict.fromkeys(queries))


def _rowset(hits: list[dict]) -> list[tuple]:
    out = []
    for h in hits:
        prov = h.get("provenance") or {}
        out.append((h.get("table"), prov.get("rowid")))
    return out


def replay(db: str | Path, queries: list[str], *, k: int = 10,
           baseline: dict | None = None) -> dict:
    """Compare cells vs legacy (and optional baseline) for each query."""
    db = Path(db)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        with CorpusDB(db) as c:
            manifest = c.manifest_dict()
        ladder = Ladder(conn=conn, doc=LazyDoc(conn), manifest=manifest,
                        rpc=lambda _p: {})
        diffs: list[dict] = []
        snapshot: dict[str, list] = {}
        for q in queries:
            ladder._cells_ok = True
            cells = _rowset(ladder.search(q, rung=0, k=k))
            ladder._cells_ok = False
            legacy = _rowset(ladder.search(q, rung=0, k=k))
            snapshot[q] = cells
            kind = None
            if set(cells) != set(legacy):
                kind = "set"
            elif cells != legacy:
                kind = "order"
            expected = None if baseline is None else baseline.get(q)
            if expected is not None:
                exp = [tuple(x) for x in expected]
                if set(cells) != set(exp) or cells != exp:
                    kind = kind or "baseline"
            if kind:
                diffs.append({
                    "query": q, "kind": kind,
                    "cells": cells, "legacy": legacy, "baseline": expected,
                })
        return {
            "n": len(queries),
            "diffs": diffs,
            "n_diffs": len(diffs),
            "snapshot": snapshot,
        }
    finally:
        conn.close()


def write_baseline(path: str | Path, snapshot: dict) -> None:
    Path(path).write_text(json.dumps(
        {q: rows for q, rows in snapshot.items()}, indent=2))
