"""Per-query rung-0 parity drill: cells path vs legacy path, same artifact.

Transient dev tool for Stage 1 verification:
    python benchmarks/parity_drill.py
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    from bench_stage1 import TS6, amplify_tables, bench_ingest, derive_queries

    from rnsr.db.artifact import CorpusDB
    from rnsr.env.lazydoc import LazyDoc
    from rnsr.env.search import Ladder

    amplify = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "c.db"
        bench_ingest(Path(TS6), db, cells=True)
        if amplify > 1:
            amplify_tables(db, amplify)
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        with CorpusDB(db) as c:
            manifest = c.manifest_dict()
        ladder = Ladder(conn=conn, doc=LazyDoc(conn), manifest=manifest,
                        rpc=lambda p: {})
        queries = derive_queries(conn, 60)
        presence_diffs = n_cells = n_legacy = 0
        for q in queries:
            ladder._cells_ok = True
            cells = {(h["table"], h["provenance"]["rowid"])
                     for h in ladder.search(q, rung=0, k=10)}
            ladder._cells_ok = False
            legacy = {(h["table"], h["provenance"]["rowid"])
                      for h in ladder.search(q, rung=0, k=10)}
            n_cells += bool(cells)
            n_legacy += bool(legacy)
            if cells != legacy:     # exact row-set parity, not just presence
                presence_diffs += 1
                print("DIFF", repr(q))
                print("  cells :", sorted(cells)[:4], len(cells))
                print("  legacy:", sorted(legacy)[:4], len(legacy))
        print(f"queries with presence-diff: {presence_diffs}/{len(queries)}")
        print(f"queries with hits — cells: {n_cells}, legacy: {n_legacy}")


if __name__ == "__main__":
    main()
