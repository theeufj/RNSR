"""In-memory background-job registry.

The API runs heavy operations (indexing, knowledge-graph extraction) in a
worker thread via :func:`asyncio.to_thread`. This registry tracks their
status so clients can poll ``GET /jobs/{job_id}``.

Jobs are kept in memory only — restarting the server clears them. That's a
deliberate trade-off: persisted job state would need careful crash-recovery,
and re-indexing is idempotent, so the simpler design is preferable.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class Job:
    job_id: str
    kind: str  # e.g. "index", "build_knowledge_graph"
    doc_id: str | None
    status: JobStatus = "queued"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobRegistry:
    """Thread-safe in-memory job tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, kind: str, doc_id: str | None = None) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:16], kind=kind, doc_id=doc_id)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "running"
                job.started_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "completed"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.result = result

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.error = error

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())
