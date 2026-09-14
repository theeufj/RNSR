"""Stage-1 baseline benchmark (docs/engine-poc-plan.md, Stage 1 entry/exit).

Measures, on a real corpus directory (default: the 999-document Test Set 6):

  - ingest throughput (fast text tier, serial parse for determinism)
  - rung-0 sweep latency (manifest-guided SQL over the typed tables)
  - rung-1 grep latency (regex over doc text via LazyDoc)
  - rung-2 FTS5 latency
  - full-text scan throughput (text serving via LazyDoc)
  - peak RSS per phase

Queries are derived deterministically from the corpus itself (sampled
chunk tokens + numeric cell values), so old/new runs compare like with
like. Results land in benchmarks/results/<label>.json; compare two files
with --compare.

    python benchmarks/bench_stage1.py --label baseline
    python benchmarks/bench_stage1.py --label cells --compare baseline
"""

from __future__ import annotations

import argparse
import json
import random
import re
import resource
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnsr.config import Settings  # noqa: E402
from rnsr.db.artifact import CorpusDB  # noqa: E402
from rnsr.env.lazydoc import LazyDoc  # noqa: E402
from rnsr.env.search import Ladder  # noqa: E402

TS6 = ("testMatter/Test Set 6 - Initiating Application (999 Documents)/"
       "Matter Documents")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]{4,}")


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _summary(samples_ms: list[float]) -> dict:
    return {
        "n": len(samples_ms),
        "p50_ms": round(statistics.median(samples_ms), 3),
        "p95_ms": round(sorted(samples_ms)[int(0.95 * (len(samples_ms) - 1))], 3),
        "total_ms": round(sum(samples_ms), 1),
    }


def derive_queries(conn: sqlite3.Connection, n: int, seed: int = 7) -> list[str]:
    """Deterministic query set from the corpus: term pairs + numeric needles."""
    rng = random.Random(seed)
    # ORDER BY doc_id/char_start, not chunk_id: parallel ingest assigns
    # chunk_ids in completion order, which varies run to run — queries must
    # be identical across separately ingested artifacts to compare fairly
    chunks = [r[0] for r in conn.execute(
        "SELECT text FROM chunks WHERE length(text) > 200 "
        "ORDER BY doc_id, char_start LIMIT 4000")]
    rng.shuffle(chunks)
    queries: list[str] = []
    for text in chunks:
        tokens = [t for t in _TOKEN.findall(text)
                  if not t.isupper() and t.lower() not in ("which", "there")]
        if len(tokens) < 4:
            continue
        pair = rng.sample(tokens[:40], 2)
        numbers = re.findall(r"\d[\d,]{2,}(?:\.\d+)?", text)
        q = " ".join(pair) + (f" {rng.choice(numbers)}" if numbers else "")
        queries.append(q)
        if len(queries) >= n:
            break
    return queries


def amplify_tables(db: Path, factor: int) -> None:
    """Duplicate every typed table (and its manifest/cells rows) factor-1
    times: a synthetic scale-up that stresses exactly what Stage 1 changes —
    rung-0 cost as a function of table count."""
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    base = [dict(r) for r in conn.execute("SELECT * FROM manifest_tables")]
    has_cells = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='cells'").fetchone())
    for t in base:
        for i in range(factor - 1):
            new = f"{t['table_name']}__amp{i}"
            conn.execute(f'CREATE TABLE "{new}" AS '
                         f'SELECT * FROM "{t["table_name"]}"')
            conn.execute(
                "INSERT INTO manifest_tables VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (new, t["doc_id"], t["title"], t["page_start"], t["page_end"],
                 t["n_rows"], t["n_cols"], t["schema_json"], t["confidence"],
                 t["checks_json"], t["status"], t["extractor"]))
            if has_cells:
                conn.execute(
                    "INSERT INTO cells SELECT doc_id, ?, row_idx, col_name, "
                    "text_value, num_value FROM cells WHERE table_name = ?",
                    (new, t["table_name"]))
    conn.commit()
    conn.close()


def bench_ingest(corpus_dir: Path, out_db: Path, *, cells: bool) -> dict:
    from rnsr.ingest.bulk import ingest_bulk
    from rnsr.ingest.dispatch import is_ingestable

    files = sorted(p for p in corpus_dir.rglob("*")
                   if p.is_file() and not p.name.startswith(".")
                   and is_ingestable(p))
    t0 = time.monotonic()
    stats = ingest_bulk(files, out_db, config=Settings(cells_index=cells))
    wall = time.monotonic() - t0
    return {
        "files": len(files),
        "wall_s": round(wall, 2),
        "files_per_s": round(len(files) / wall, 1),
        "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))},
        "artifact_mb": round(out_db.stat().st_size / 1e6, 1),
        "peak_rss_mb": rss_mb(),
    }


def bench_queries(corpus_db: Path, n_queries: int) -> dict:
    from rnsr.db.schema import apply_read_pragmas

    conn = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True)
    apply_read_pragmas(conn)
    with CorpusDB(corpus_db) as c:
        manifest = c.manifest_dict()
    doc = LazyDoc(conn)
    ladder = Ladder(conn=conn, doc=doc, manifest=manifest,
                    rpc=lambda payload: {})
    queries = derive_queries(conn, n_queries)

    out: dict = {"n_queries": len(queries)}
    for rung, key in ((0, "rung0_sql"), (1, "rung1_grep"), (2, "rung2_fts")):
        samples, hits = [], 0
        for q in queries:
            t0 = time.perf_counter()
            try:
                got = ladder.search(q, rung=rung)
            except Exception:
                got = []
            samples.append((time.perf_counter() - t0) * 1000)
            hits += bool(got)
        out[key] = {**_summary(samples), "queries_with_hits": hits}

    # text serving: full sequential scan through LazyDoc
    t0 = time.monotonic()
    total_chars = sum(len(doc[d]) for d in doc)
    out["text_scan"] = {
        "docs": len(doc),
        "chars": total_chars,
        "wall_s": round(time.monotonic() - t0, 3),
        "mb_per_s": round(total_chars / 1e6 / max(time.monotonic() - t0, 1e-9), 1),
    }
    out["counts"] = {
        "tables": conn.execute(
            "SELECT count(*) FROM manifest_tables").fetchone()[0],
        "chunks": conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "cells_rows": (conn.execute("SELECT count(*) FROM cells").fetchone()[0]
                       if conn.execute("SELECT 1 FROM sqlite_master WHERE "
                                       "name='cells'").fetchone() else 0),
    }
    out["peak_rss_mb"] = rss_mb()
    conn.close()
    return out


def compare(a: dict, b: dict, a_name: str, b_name: str) -> None:
    print(f"\n{'metric':<28}{a_name:>14}{b_name:>14}")
    rows = [
        ("ingest wall_s", a["ingest"]["wall_s"], b["ingest"]["wall_s"]),
        ("rung0 p50_ms", a["query"]["rung0_sql"]["p50_ms"],
         b["query"]["rung0_sql"]["p50_ms"]),
        ("rung0 p95_ms", a["query"]["rung0_sql"]["p95_ms"],
         b["query"]["rung0_sql"]["p95_ms"]),
        ("rung0 hits", a["query"]["rung0_sql"]["queries_with_hits"],
         b["query"]["rung0_sql"]["queries_with_hits"]),
        ("rung1 p50_ms", a["query"]["rung1_grep"]["p50_ms"],
         b["query"]["rung1_grep"]["p50_ms"]),
        ("fts p50_ms", a["query"]["rung2_fts"]["p50_ms"],
         b["query"]["rung2_fts"]["p50_ms"]),
        ("text scan MB/s", a["query"]["text_scan"]["mb_per_s"],
         b["query"]["text_scan"]["mb_per_s"]),
        ("peak RSS MB", a["query"]["peak_rss_mb"], b["query"]["peak_rss_mb"]),
    ]
    for name, va, vb in rows:
        print(f"{name:<28}{va:>14}{vb:>14}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=TS6)
    ap.add_argument("--label", required=True)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--compare", help="label of a previous run to diff against")
    ap.add_argument("--amplify-tables", type=int, default=1,
                    help="duplicate typed tables N-fold after ingest "
                         "(synthetic scale-up of table count)")
    ap.add_argument("--no-cells", action="store_true",
                    help="ingest without the derived cells index "
                         "(legacy rung-0 path)")
    args = ap.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        out_db = Path(td) / "bench_corpus.db"
        result = {
            "label": args.label,
            "corpus": args.corpus,
            "amplify_tables": args.amplify_tables,
            "cells_index": not args.no_cells,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ingest": bench_ingest(Path(args.corpus), out_db,
                                   cells=not args.no_cells),
        }
        if args.amplify_tables > 1:
            amplify_tables(out_db, args.amplify_tables)
        result["query"] = bench_queries(out_db, args.queries)

    path = results_dir / f"{args.label}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {path}")

    if args.compare:
        prev = json.loads((results_dir / f"{args.compare}.json").read_text())
        compare(prev, result, args.compare, args.label)


if __name__ == "__main__":
    main()
