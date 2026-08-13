"""HTTP service surface for rnsr (optional 'service' extra).

The CLI was the only entry point, so every consumer had to shell out and
parse a CSV, and nothing could report whether the process was healthy. This
module adds the smallest surface that makes rnsr operable as a service:

    GET  /healthz      process is up (liveness)
    GET  /readyz       a provider key resolves and the corpus is readable
    GET  /metrics      counters, percentiles, provider spend
    POST /jobs         submit questions; returns a job id immediately
    GET  /jobs/{id}    status, answers, per-field agreement
    GET  /jobs         recent jobs

Deliberately single-node: jobs live in this process's memory and run as
asyncio tasks. That is honest about what it is — a service wrapper around
the same runner the CLI uses, suitable behind one worker per matter. A
multi-node deployment needs a real queue, and pretending otherwise with a
database-backed job table here would imply guarantees this does not have.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rnsr.config import Settings
from rnsr.obs import configure_logging, get_logger, log, metrics

_LOG = get_logger("service")


class JobRequest(BaseModel):
    """Defined at module scope on purpose: this module uses postponed
    annotations, and FastAPI resolves endpoint hints against module globals
    only — a model nested inside create_app() is read as a query parameter."""

    questions: list[str] = Field(min_length=1)
    corpus_db: str
    batch_size: int = Field(default=8, ge=1, le=64)
    consensus: int = Field(default=1, ge=1, le=5)


@dataclass
class Job:
    id: str
    questions: list[str]
    corpus_db: str
    batch_size: int = 8
    consensus: int = 1
    state: str = "queued"          # queued | running | done | failed
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    answers: list[str | None] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    agreement: list[float | None] = field(default_factory=list)
    error: str | None = None

    def public(self, *, include_answers: bool = True) -> dict:
        out: dict[str, Any] = {
            "id": self.id, "state": self.state, "questions": len(self.questions),
            "batch_size": self.batch_size, "consensus": self.consensus,
            "submitted_at": self.submitted_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "error": self.error,
        }
        if include_answers and self.state == "done":
            out["results"] = [
                {"question": q, "answer": a, "status": s, "agreement": g}
                for q, a, s, g in zip(self.questions, self.answers,
                                      self.statuses, self.agreement,
                                      strict=False)
            ]
        return out


class JobStore:
    """Bounded in-memory job registry."""

    def __init__(self, max_jobs: int = 200):
        self.max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        # tasks are held so the event loop keeps a strong reference; a bare
        # create_task() can be garbage-collected mid-job
        self._tasks: set[asyncio.Task] = set()

    def track(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def add(self, job: Job) -> Job:
        self._jobs[job.id] = job
        while len(self._jobs) > self.max_jobs:
            oldest = min(self._jobs.values(), key=lambda j: j.submitted_at)
            if oldest.state in ("queued", "running"):
                break              # never evict work in progress
            del self._jobs[oldest.id]
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.submitted_at,
                      reverse=True)[:limit]


async def run_job(job: Job, settings: Settings, run_dir: Path) -> None:
    """Answer a job's questions with the same runner the CLI uses."""
    from rnsr.db.artifact import CorpusDB
    from rnsr.harness.loop import EnvSpec, RootRunner
    from rnsr.llm.router import Router

    job.state, job.started_at = "running", time.time()
    log(_LOG, logging.INFO, "job.start", job_id=job.id,
        questions=len(job.questions), consensus=job.consensus)
    try:
        with CorpusDB(job.corpus_db) as corpus:
            manifest = corpus.manifest_dict()
        env = EnvSpec(mode="docdb", corpus_db=job.corpus_db, manifest=manifest)
        router = Router(settings)
        root, sub = router.resolve("root"), router.resolve("sub")
        embed_client, embed_model = None, ""
        try:
            embed = router.resolve("embed")
            embed_client, embed_model = embed.client, embed.model
        except RuntimeError:
            pass
        runner = RootRunner(root_client=root.client, root_model=root.model,
                            sub_client=sub.client, sub_model=sub.model,
                            embed_client=embed_client, embed_model=embed_model,
                            settings=settings)

        n = len(job.questions)
        job.answers = [None] * n
        job.statuses = ["pending"] * n
        job.agreement = [None] * n
        size = max(1, job.batch_size)
        for start in range(0, n, size):
            group = list(range(start, min(start + size, n)))
            pairs = [(f"q{i:03d}", job.questions[i]) for i in group]
            if job.consensus > 1:
                cr = await runner.run_batch_consensus(
                    pairs, env, run_dir=run_dir / job.id,
                    query_id=f"b{group[0]:03d}", passes=job.consensus)
                for i, (qid, _) in zip(group, pairs, strict=True):
                    answer = cr.answers[qid]
                    job.answers[i] = answer.value
                    job.statuses[i] = answer.resolved_by
                    job.agreement[i] = answer.agreement
            else:
                br = await runner.run_batch(pairs, env, run_dir=run_dir / job.id,
                                            query_id=f"b{group[0]:03d}")
                for i, (qid, _) in zip(group, pairs, strict=True):
                    job.answers[i] = br.answers.get(qid)
                    job.statuses[i] = (br.result.status if br.answers.get(qid)
                                       else "unanswered")
        job.state = "done"
        metrics().incr("jobs_finished", state="done")
    except Exception as e:
        job.state, job.error = "failed", f"{type(e).__name__}: {e}"[:300]
        metrics().incr("jobs_finished", state="failed")
        log(_LOG, logging.ERROR, "job.failed", job_id=job.id, error=job.error)
    finally:
        job.finished_at = time.time()
        log(_LOG, logging.INFO, "job.end", job_id=job.id, state=job.state,
            duration_s=round(job.finished_at - (job.started_at or 0), 2))


def create_app(settings: Settings | None = None, *,
               run_dir: Path | None = None) -> Any:
    """Build the FastAPI app. Imported lazily so the core install stays lean."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:      # pragma: no cover - depends on extras
        raise RuntimeError("rnsr serve needs the 'service' extra: "
                           "pip install 'rnsr[service]'") from e

    settings = settings or Settings.from_env()
    configure_logging(settings)
    run_dir = Path(run_dir or settings.run_dir) / "service"
    store = JobStore()
    app = FastAPI(title="rnsr", version="1", summary="DocDB-RLM answering service")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": app.version}

    @app.get("/readyz")
    async def readyz() -> dict:
        from rnsr.llm.router import available_providers

        providers = available_providers()
        checks = {"providers": providers, "run_dir_writable": _writable(run_dir)}
        ready = bool(providers) and checks["run_dir_writable"]
        if not ready:
            raise HTTPException(status_code=503,
                                detail={"status": "not ready", **checks})
        return {"status": "ready", **checks}

    @app.get("/metrics")
    async def metrics_endpoint() -> dict:
        from rnsr.llm import governor

        return {"metrics": metrics().snapshot(),
                "provider": governor.current().snapshot()}

    @app.post("/jobs", status_code=202)
    async def submit(req: JobRequest) -> dict:
        if not Path(req.corpus_db).exists():
            raise HTTPException(status_code=400,
                                detail=f"corpus_db not found: {req.corpus_db}")
        job = store.add(Job(id=uuid.uuid4().hex[:12], questions=req.questions,
                            corpus_db=req.corpus_db, batch_size=req.batch_size,
                            consensus=req.consensus))
        metrics().incr("jobs_submitted")
        # a task, not a BackgroundTask: answering a form takes minutes and
        # must not hold the submitting connection open
        store.track(asyncio.create_task(run_job(job, settings, run_dir)))
        return job.public()

    @app.get("/jobs")
    async def list_jobs(limit: int = 50) -> dict:
        return {"jobs": [j.public(include_answers=False)
                         for j in store.recent(limit)]}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job.public()

    app.state.store = store
    app.state.settings = settings
    return app


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def serve(host: str = "127.0.0.1", port: int = 8000,
          settings: Settings | None = None) -> None:      # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, log_config=None)
