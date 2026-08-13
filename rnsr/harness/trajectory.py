"""Per-query trajectory log: every cell, observation, sub-call batch, rung
event, and FINAL candidate, as JSONL under runs/{run_id}/ (§6, §8).

Trajectories live in run directories, not corpus.db — the artifact stays a
portable corpus index whose only query-time writes are annotations.

Trajectories are the system's best forensic asset and its largest data
liability: they quote client documents verbatim. Three controls make that
a decision rather than an accident, all off by default so research runs are
unchanged:

  content mode   full (default) | redacted | metadata. Redacted swaps
                 document-bearing values for a length + digest descriptor,
                 which still supports "did these two runs see the same
                 text?" without keeping the text. Metadata drops them.
  encryption     with a Fernet key configured, every line is encrypted at
                 rest; read_trajectory() and `rnsr trajectory` decrypt.
  retention      prune_trajectories() deletes logs past an age, so a
                 matter's working files do not accumulate indefinitely.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

# Keys whose values quote the corpus, the question, or model reasoning over
# them. Everything else (statuses, counts, model names, caps) is operational
# metadata and survives redaction so the forensic skeleton stays readable.
_CONTENT_KEYS = frozenset({
    "question", "code", "stdout", "error", "reply", "gap", "value", "final",
    "query", "candidates", "choice", "verification", "flagged", "prompt",
})
_REDACT_OVER_CHARS = 200


def _describe(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
    return f"<redacted {len(text)} chars sha256:{digest}>"


def redact(record: dict, mode: str) -> dict:
    """Apply a content mode to one event record."""
    if mode == "full":
        return record
    out: dict = {}
    for key, value in record.items():
        sensitive = key in _CONTENT_KEYS or (
            isinstance(value, str) and len(value) > _REDACT_OVER_CHARS)
        if not sensitive or value is None:
            out[key] = value
        elif mode == "metadata":
            continue
        else:
            out[key] = _describe(value)
    return out


class _Cipher:
    """Fernet wrapper. cryptography is an optional extra, so the failure to
    encrypt is loud: silently writing plaintext when an operator asked for
    encryption would be the worst outcome."""

    def __init__(self, key: str):
        try:
            from cryptography.fernet import Fernet
        except ImportError as e:      # pragma: no cover - depends on extras
            raise RuntimeError(
                "trajectory encryption needs the 'secure' extra: "
                "pip install 'rnsr[secure]'") from e
        self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, line: str) -> str:
        return self._fernet.encrypt(line.encode()).decode()

    def decrypt(self, line: str) -> str:
        return self._fernet.decrypt(line.encode()).decode()


class TrajectoryWriter:
    def __init__(self, run_dir: str | Path, query_id: str, *,
                 content: str = "full", key: str = ""):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.content = content or "full"
        self._cipher = _Cipher(key) if key else None
        suffix = ".jsonl.enc" if self._cipher else ".jsonl"
        self.path = self.dir / f"{query_id}{suffix}"
        self._f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 — lifetime spans the query; closed in close()
        self._t0 = time.monotonic()

    def event(self, kind: str, **data) -> None:
        record = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **data}
        line = json.dumps(redact(record, self.content), default=repr)
        if self._cipher:
            line = self._cipher.encrypt(line)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_trajectory(path: str | Path, key: str = "") -> list[dict]:
    """Read a trajectory, decrypting when it was written with a key."""
    path = Path(path)
    cipher = _Cipher(key) if key else None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if cipher:
            line = cipher.decrypt(line)
        records.append(json.loads(line))
    return records


def prune_trajectories(run_dir: str | Path, max_age_days: float) -> int:
    """Delete trajectories older than max_age_days. Returns the count removed.

    Retention is enforced where the files are written rather than left to an
    external cron, so a matter's working directory does not quietly become
    an indefinite archive of privileged text.
    """
    if not max_age_days or max_age_days <= 0:
        return 0
    root = Path(run_dir)
    if not root.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for path in list(root.rglob("*.jsonl")) + list(root.rglob("*.jsonl.enc")):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed
