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

One process is one authorization scope: all holders of its bearer token
can access its jobs. Use separate instances/tokens/corpus roots per matter
or tenant. All routes except /healthz require RNSR_SERVICE_TOKEN, and corpus
paths are confined to RNSR_SERVICE_CORPUS_ROOT. TLS belongs at the proxy.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
    abstain_below: Literal["off", "low", "medium", "high"] = "off"


class JobResult(BaseModel):
    question: str
    answer: str | None
    status: str
    agreement: float | None
    tier: str | None
    contested: bool


class JobSummary(BaseModel):
    id: str
    state: Literal["queued", "running", "done", "failed"]
    questions: int
    batch_size: int
    consensus: int
    submitted_at: float
    started_at: float | None
    finished_at: float | None


class JobResponse(JobSummary):
    error: str | None = None
    results: list[JobResult] | None = None


class JobListing(BaseModel):
    jobs: list[JobSummary]


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    providers: list[str]
    run_dir_writable: bool


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
    provider: dict[str, Any]


@dataclass
class Job:
    id: str
    questions: list[str]
    corpus_db: str
    batch_size: int = 8
    consensus: int = 1
    abstain_below: Literal["off", "low", "medium", "high"] = "off"
    state: Literal["queued", "running", "done", "failed"] = "queued"
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    answers: list[str | None] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    agreement: list[float | None] = field(default_factory=list)
    tiers: list[str | None] = field(default_factory=list)
    contested: list[bool] = field(default_factory=list)
    error: str | None = None


def job_summary(job: Job) -> JobSummary:
    return JobSummary(id=job.id, state=job.state, questions=len(job.questions),
                      batch_size=job.batch_size, consensus=job.consensus,
                      submitted_at=job.submitted_at, started_at=job.started_at,
                      finished_at=job.finished_at)


def job_response(job: Job) -> JobResponse:
    results = None
    if job.state == "done":
        results = [JobResult(question=q, answer=a, status=s, agreement=g,
                             tier=t, contested=c)
                   for q, a, s, g, t, c in zip(
                       job.questions, job.answers, job.statuses, job.agreement,
                       job.tiers or [None] * len(job.questions),
                       job.contested or [False] * len(job.questions), strict=True)]
    return JobResponse(**job_summary(job).model_dump(), error=job.error, results=results)


class JobStore:
    """Bounded in-memory job registry."""

    def __init__(self, max_jobs: int = 200):
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        self.max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        # tasks are held so the event loop keeps a strong reference; a bare
        # create_task() can be garbage-collected mid-job
        self._tasks: set[asyncio.Task] = set()

    def track(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def add(self, job: Job) -> Job:
        while len(self._jobs) >= self.max_jobs:
            finished = [j for j in self._jobs.values() if j.state in ("done", "failed")]
            if not finished:
                raise OverflowError("job capacity reached; retry when running jobs finish")
            oldest = min(finished, key=lambda j: j.submitted_at)
            del self._jobs[oldest.id]
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.submitted_at,
                      reverse=True)[:limit]


async def run_job(job: Job, settings: Settings, run_dir: Path) -> None:
    """Answer a job's questions with the same runner the CLI uses."""
    from rnsr.sdk import answer_batch

    job.state, job.started_at = "running", time.time()
    log(_LOG, logging.INFO, "job.start", job_id=job.id,
        questions=len(job.questions), consensus=job.consensus)
    try:
        results = await answer_batch(
            job.questions, job.corpus_db,
            batch_size=job.batch_size, consensus=job.consensus,
            retry_solo=True, settings=settings,
            abstain_below=job.abstain_below,
            run_dir=run_dir / job.id,
        )
        job.answers = [r.answer for r in results]
        job.statuses = [r.status for r in results]
        job.agreement = [r.agreement for r in results]
        job.tiers = [r.tier for r in results]
        job.contested = [r.contested for r in results]
        job.state = "done"
        metrics().incr("jobs_finished", state="done")
    except Exception as e:
        job.state, job.error = "failed", f"{type(e).__name__}: job failed"
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
        from fastapi import Depends, FastAPI, Header, HTTPException, Query
    except ImportError as e:      # pragma: no cover - depends on extras
        raise RuntimeError("rnsr serve needs the 'service' extra: "
                           "pip install 'rnsr[service]'") from e

    settings = settings or Settings.from_env()
    if not settings.service_token.strip():
        raise ValueError("RNSR_SERVICE_TOKEN is required to start the HTTP service")
    if settings.service_corpus_root is None:
        raise ValueError("RNSR_SERVICE_CORPUS_ROOT is required to start the HTTP service")
    corpus_root = settings.service_corpus_root.resolve(strict=True)
    if not corpus_root.is_dir():
        raise ValueError("RNSR_SERVICE_CORPUS_ROOT must be a directory")
    configure_logging(settings)
    run_dir = Path(run_dir or settings.run_dir) / "service"
    store = JobStore(settings.service_max_jobs)
    app = FastAPI(title="rnsr", version="1", summary="DocDB-RLM answering service",
                  docs_url=None, redoc_url=None, openapi_url=None)

    async def authorize(authorization: str = Header(default="")) -> None:
        scheme, _, credential = authorization.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
                credential.encode(), settings.service_token.encode()):
            raise HTTPException(status_code=401, detail="valid bearer token required",
                                headers={"WWW-Authenticate": "Bearer"})

    protected = [Depends(authorize)]

    @app.get("/healthz")
    async def healthz() -> HealthResponse:
        return {"status": "ok", "version": app.version}

    @app.get("/readyz", dependencies=protected)
    async def readyz() -> ReadyResponse:
        from rnsr.llm.router import available_providers

        providers = available_providers()
        checks = {"providers": providers, "run_dir_writable": _writable(run_dir)}
        ready = bool(providers) and checks["run_dir_writable"]
        if not ready:
            raise HTTPException(status_code=503,
                                detail={"status": "not ready", **checks})
        return {"status": "ready", **checks}

    @app.get("/metrics", dependencies=protected)
    async def metrics_endpoint() -> MetricsResponse:
        from rnsr.llm import governor

        return {"metrics": metrics().snapshot(),
                "provider": governor.current().snapshot()}

    @app.post("/jobs", status_code=202, dependencies=protected)
    async def submit(req: JobRequest) -> JobResponse:
        try:
            requested = Path(req.corpus_db)
            corpus = (requested if requested.is_absolute() else corpus_root / requested).resolve(
                strict=True)
            if not corpus.is_relative_to(corpus_root) or not corpus.is_file():
                raise ValueError("corpus must be a file under the configured root")
        except (OSError, ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail="corpus_db unavailable or outside allowed root") from e
        try:
            job = store.add(Job(id=uuid.uuid4().hex, questions=req.questions,
                                corpus_db=str(corpus), batch_size=req.batch_size,
                                consensus=req.consensus, abstain_below=req.abstain_below))
        except OverflowError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        metrics().incr("jobs_submitted")
        # a task, not a BackgroundTask: answering a form takes minutes and
        # must not hold the submitting connection open
        store.track(asyncio.create_task(run_job(job, settings, run_dir)))
        return job_response(job)

    @app.get("/jobs", dependencies=protected)
    async def list_jobs(limit: int = Query(default=50, ge=1, le=200)) -> JobListing:
        return JobListing(jobs=[job_summary(j) for j in store.recent(limit)])

    @app.get("/jobs/{job_id}", dependencies=protected)
    async def get_job(job_id: str) -> JobResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job_response(job)

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
