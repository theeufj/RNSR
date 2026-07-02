"""Per-query trajectory log: every cell, observation, sub-call batch, rung
event, and FINAL candidate, as JSONL under runs/{run_id}/ (§6, §8).

Trajectories live in run directories, not corpus.db — the artifact stays a
portable corpus index whose only query-time writes are annotations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class TrajectoryWriter:
    def __init__(self, run_dir: str | Path, query_id: str):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{query_id}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 — lifetime spans the query; closed in close()
        self._t0 = time.monotonic()

    def event(self, kind: str, **data) -> None:
        record = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **data}
        self._f.write(json.dumps(record, default=repr) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
