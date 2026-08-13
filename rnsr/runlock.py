"""Single-writer lock for a work directory.

The checkpoint file (answers_partial.jsonl) and the corpus artifact both
assume one writer. Two runs pointed at the same --work-dir interleave their
checkpoint appends and contend on the same SQLite file, so a resume can
read back a mixture of two jobs. Nothing prevented that.

The lock is advisory and process-scoped: an exclusive flock held for the
life of the run, with the holder's pid and start time recorded so the error
message names who to look for. A stale lock from a killed process is
released by the OS automatically, which is why this is flock rather than a
hand-rolled pid file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from rnsr.errors import RNSRError


class WorkDirBusy(RNSRError):
    """Another process holds the lock on this work directory."""


class WorkDirLock:
    """Exclusive advisory lock on <work_dir>/.rnsr.lock."""

    def __init__(self, work_dir: str | Path, *, label: str = ""):
        self.dir = Path(work_dir)
        self.path = self.dir / ".rnsr.lock"
        self.label = label
        self._f = None

    def acquire(self) -> WorkDirLock:
        self.dir.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 — held for the run
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:             # pragma: no cover - non-POSIX
            pass                        # no advisory locking available
        except OSError as e:
            f.seek(0)
            holder = f.read()[:200].strip()
            f.close()
            raise WorkDirBusy(
                f"work directory {self.dir} is in use by another rnsr run "
                f"({holder or 'unknown holder'}). Concurrent runs would "
                "interleave the same checkpoint and corpus writes — use a "
                "separate --work-dir, or wait for that run to finish."
            ) from e
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"pid": os.getpid(), "label": self.label,
                            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        f.flush()
        self._f = f
        return self

    def release(self) -> None:
        if self._f is None:
            return
        try:
            import fcntl

            fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):   # pragma: no cover
            pass
        self._f.close()
        self._f = None

    def __enter__(self) -> WorkDirLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
