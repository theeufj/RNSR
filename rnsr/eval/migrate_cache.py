"""Carry historical eval corpora forward into a shared cache.

The eval harness caches one corpus.db per source set under a name derived
from the source files and the ingestion identity. Artifacts built by earlier
revisions are valid parses but predate the `cells` and `ingest_batches`
tables and carry an older cache name. This tool copies each old artifact to
the name the current harness would compute for the same sources, migrates
it in place (`rnsr.db.migrate`), and verifies it, so a benchmark rerun can
reuse hours of Docling work instead of repeating it.

    python -m rnsr.eval.migrate_cache --from runs/financebench/financebench-docdb/corpora \
        --to runs/public-arms/corpora

Re-ingestion is still the right choice when the parser itself changed in a
way that matters to the questions; the write-up must say which was done.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.db.migrate import migrate_artifact
from rnsr.db.schema import validate_frozen
from rnsr.eval.harness import _cache_name


def carry_forward(src_dir: Path, dst_dir: Path, settings: Settings | None = None) -> dict:
    settings = settings or Settings.from_env()
    dst_dir.mkdir(parents=True, exist_ok=True)
    report = {"copied": 0, "skipped_existing": 0, "missing_sources": [], "failed": []}
    for old in sorted(src_dir.glob("*.db")):
        conn = sqlite3.connect(f"file:{old}?mode=ro", uri=True)
        try:
            sources = [Path(r[0]) for r in conn.execute("SELECT source_path FROM documents ORDER BY source_path")]
        finally:
            conn.close()
        if not sources or not all(s.exists() for s in sources):
            report["missing_sources"].append(old.name)
            continue
        new = dst_dir / _cache_name(sources, settings)
        if new.exists():
            report["skipped_existing"] += 1
            continue
        shutil.copy(old, new)
        try:
            migrate_artifact(new)
            with CorpusDB(new) as corpus:
                validate_frozen(corpus.conn)
                n_cells = corpus.conn.execute("SELECT count(*) FROM cells").fetchone()[0]
                n_docs = corpus.conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            if n_docs != len(sources):
                raise RuntimeError(f"{n_docs} documents for {len(sources)} sources")
            report["copied"] += 1
            report.setdefault("cells", []).append(n_cells)
        except Exception as exc:
            new.unlink(missing_ok=True)
            report["failed"].append(f"{old.name}: {type(exc).__name__}: {exc}"[:200])
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--to", dest="dst", type=Path, required=True)
    args = ap.parse_args(argv)
    report = carry_forward(args.src, args.dst)
    cells = report.pop("cells", [])
    print({**report, "cells_min": min(cells) if cells else None, "cells_max": max(cells) if cells else None})


if __name__ == "__main__":
    main()
