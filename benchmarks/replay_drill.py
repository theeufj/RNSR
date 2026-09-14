"""Replay real trajectory search queries against both rung-0 paths on the
actual run artifact, diffing exact row sets. Transient Stage-1 dev tool.

    python benchmarks/replay_drill.py <corpus.db> <trajectory.jsonl>...
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnsr.db.artifact import CorpusDB  # noqa: E402
from rnsr.env.lazydoc import LazyDoc  # noqa: E402
from rnsr.env.search import Ladder  # noqa: E402


def main() -> None:
    db, *trajs = sys.argv[1:]
    queries: list[str] = []
    for t in trajs:
        with open(t) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "search_rung":
                    queries.append(r["query"])
    queries = list(dict.fromkeys(queries))

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    with CorpusDB(db) as c:
        manifest = c.manifest_dict()
    ladder = Ladder(conn=conn, doc=LazyDoc(conn), manifest=manifest,
                    rpc=lambda p: {})
    diffs = 0
    for q in queries:
        ladder._cells_ok = True
        cells = [(h["table"], h["provenance"]["rowid"])
                 for h in ladder.search(q, rung=0, k=10)]
        ladder._cells_ok = False
        legacy = [(h["table"], h["provenance"]["rowid"])
                  for h in ladder.search(q, rung=0, k=10)]
        if set(cells) != set(legacy):
            diffs += 1
            print("DIFF", repr(q[:100]))
            print("  cells-only :", sorted(set(cells) - set(legacy))[:5])
            print("  legacy-only:", sorted(set(legacy) - set(cells))[:5])
        elif cells != legacy:
            diffs += 1
            print("ORDER-DIFF", repr(q[:100]))
    print(f"{diffs} diffs across {len(queries)} replayed queries")


if __name__ == "__main__":
    main()
