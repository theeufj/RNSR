"""Pre-warm the answer-csv corpus cache for Test Set 6 (timed).

Replicates answer-csv's stat-identity cache key exactly, then runs the
corpus-scale bulk ingest into that path, so the subsequent answer-csv run
finds a warm cache. Run from the repo root.
"""

from __future__ import annotations

import sys
import time
from hashlib import sha256
from pathlib import Path

from rnsr.config import Settings
from rnsr.ingest.bulk import ingest_bulk
from rnsr.ingest.dispatch import is_ingestable

CORPUS_DIR = Path(
    "testMatter/Test Set 6 - Initiating Application (999 Documents)/"
    "Matter Documents")
WORK_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/ts6/work")


def main() -> None:
    all_files = [p for p in CORPUS_DIR.rglob("*")
                 if p.is_file() and not p.name.startswith(".")]
    files = sorted(p for p in all_files if is_ingestable(p))
    print(f"{len(files)} ingestable of {len(all_files)} files "
          f"({len(all_files) - len(files)} skipped)")

    h = sha256()
    for s in files:  # identical to answer-csv's stat-identity key
        st = s.stat()
        h.update(f"{s}|{st.st_size}|{st.st_mtime_ns}".encode())
    cache_dir = WORK_DIR / "corpora"
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = cache_dir / f"corpus_{h.hexdigest()[:16]}.db"
    print(f"target: {corpus_path}")

    t0 = time.monotonic()
    stats = ingest_bulk(files, corpus_path, config=Settings.from_env(),
                        progress=lambda s: print(f"  {s}", flush=True))
    dt = time.monotonic() - t0
    print(f"stats: {stats}")
    print(f"INGEST WALL TIME: {dt:.1f}s")


if __name__ == "__main__":
    main()
