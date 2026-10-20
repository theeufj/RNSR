"""HTTP service surface for rnsr (optional 'service' extra).

The CLI was the only entry point, so every consumer had to shell out and
parse a CSV, and nothing could report whether the process was healthy. This
module adds the smallest surface that makes rnsr operable as a service:

    GET  /healthz      process is up (liveness)
    GET  /readyz       configured root/sub roles resolve locally; storage is ready
    GET  /metrics      counters, percentiles, provider spend
    POST /jobs         submit questions; returns a job id immediately
    GET  /jobs/{id}    status, answers, per-field agreement
    GET  /jobs         recent jobs
    GET  /jobs/{id}/audit/{ref}    operational trajectory events (authorized)

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
import math
import os
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
    evidence: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    health: dict[str, Any] | None = None
    claim_review: dict[str, Any] | None = None
    audit_refs: list[str] = Field(default_factory=list)


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
    roles: dict[str, str]
    corpus_root_readable: bool


class AuditResponse(BaseModel):
    reference: str
    content: Literal["metadata"] = "metadata"
    events: list[dict[str, Any]]


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
    evidence: list[dict | None] = field(default_factory=list)
    citations: list[list[dict]] = field(default_factory=list)
    health: list[dict | None] = field(default_factory=list)
    claim_review: list[dict | None] = field(default_factory=list)
    audit_refs: list[list[str]] = field(default_factory=list)
    audit_paths: dict[str, Path] = field(default_factory=dict, repr=False)
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
                             tier=t, contested=c, evidence=_public_data(ev),
                             citations=_public_data(citations), health=_public_data(health),
                             claim_review=_public_data(review), audit_refs=refs)
                   for q, a, s, g, t, c, ev, citations, health, review, refs in zip(
                       job.questions, job.answers, job.statuses, job.agreement,
                       job.tiers or [None] * len(job.questions),
                       job.contested or [False] * len(job.questions),
                       job.evidence or [None] * len(job.questions),
                       job.citations or [[] for _ in job.questions],
                       job.health or [None] * len(job.questions),
                       job.claim_review or [None] * len(job.questions),
                       job.audit_refs or [[] for _ in job.questions], strict=True)]
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
    from rnsr.harness.trajectory import prepare_trajectory_dir
    from rnsr.sdk import answer_batch

    job.state, job.started_at = "running", time.time()
    log(_LOG, logging.INFO, "job.start", job_id=job.id,
        questions=len(job.questions), consensus=job.consensus)
    try:
        run_dir = prepare_trajectory_dir(run_dir, settings.trajectory_retention_days)
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
        job.evidence = [r.evidence for r in results]
        job.citations = [r.citations for r in results]
        job.health = [r.health for r in results]
        job.claim_review = [r.claim_review for r in results]
        job_root = (run_dir / job.id).resolve()
        references: dict[Path, str] = {}
        for result in results:
            refs = []
            for raw_path in result.trajectory_paths:
                path = Path(raw_path).resolve()
                if not path.is_relative_to(job_root) or not path.is_file():
                    continue
                if path not in references:
                    audit_id = uuid.uuid4().hex
                    references[path] = f"/jobs/{job.id}/audit/{audit_id}"
                    job.audit_paths[audit_id] = path
                refs.append(references[path])
            job.audit_refs.append(list(dict.fromkeys(refs)))
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
        from rnsr.llm.router import Router, available_providers

        providers = available_providers()
        roles = {}
        try:
            router = Router(settings)
            if router.provider not in providers:
                raise RuntimeError("configured provider key unavailable")
            for role in ("root", "sub"):
                resolved = router.resolve(role)  # constructs clients; makes no request
                if getattr(resolved.client, "provider", router.provider) not in providers:
                    raise RuntimeError("configured role provider key unavailable")
                roles[role] = resolved.model
        except Exception:
            # Provider errors can include credentials or host details. Readiness
            # reports only which local role configurations were validated.
            pass
        checks = {"providers": providers, "roles": roles,
                  "run_dir_writable": _writable(run_dir),
                  "corpus_root_readable": _readable(corpus_root)}
        ready = len(roles) == 2 and checks["run_dir_writable"] and checks["corpus_root_readable"]
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

    @app.get("/jobs/{job_id}/audit/{audit_id}", dependencies=protected)
    async def get_audit(job_id: str, audit_id: str) -> AuditResponse:
        from rnsr.harness.trajectory import read_trajectory

        job = store.get(job_id)
        path = job.audit_paths.get(audit_id) if job else None
        job_root = (run_dir / job_id).resolve()
        if (path is None or not path.resolve().is_relative_to(job_root)
                or not path.is_file()):
            raise HTTPException(status_code=404, detail="no such audit record")
        try:
            records = read_trajectory(path, settings.trajectory_key if path.name.endswith(".enc") else "")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="audit record unavailable") from exc
        # Citations/support signals accompany results. The HTTP trajectory view
        # is an operational timeline, even when local research logs retain full
        # text, so code/errors/source paths cannot leak through this endpoint.
        return AuditResponse(reference=f"/jobs/{job_id}/audit/{audit_id}",
                             events=[_audit_metadata(record) for record in records])

    app.state.store = store
    app.state.settings = settings
    return app


def _public_data(value):
    """Drop filesystem locations from nested source contexts and review metadata."""
    if isinstance(value, dict):
        return {key: _public_data(item) for key, item in value.items()
                if key not in ("path", "paths", "corpus_db")
                and (key in ("heading_path", "heading_paths")
                     or not key.endswith(("_path", "_paths")))}
    if isinstance(value, list):
        return [_public_data(item) for item in value]
    return value


def _audit_metadata(record: dict) -> dict:
    # Child log RPCs are untrusted, including values under operational names.
    # Enumerations and exact scalar types keep a forged phase/status/count from
    # smuggling client content or host paths into the public timeline.
    enums = {
        "kind": {"start", "end", "cell", "sub_batch", "embed_batch", "search_rung",
                 "negative_audit", "claim_review", "claim_review_decision", "root_call_started",
                 "root_call_completed", "root_call_failed", "unverified_recovery", "budget_breached",
                 "no_code", "sandbox_restarted", "final_observation_review", "completeness_pushback",
                 "budget_warning", "damping", "final", "error", "recovery", "env_log"},
        "status": {"final", "recovered", "budget_exhausted", "error", "reviewed",
                   "completed", "skipped", "supported", "unverified", "disabled", "not_applicable"},
        "mode": {"docdb", "classic"},
        "phase": {"queue", "provider", "review", "adjudication"},
        "timeout_scope": {"overall", "request"},
    }
    numbers = {"t", "timeout_s", "elapsed_s", "queue_wait_s", "spend_usd", "wall_s"}
    counts = {"attempt", "iterations", "root_iters", "sub_calls", "n", "failed", "rpc_count",
              "rung", "batch_size", "input_tokens", "output_tokens"}
    flags = {"retryable", "ok", "supported"}
    out = {}
    for key, item in record.items():
        valid = ((key in enums and isinstance(item, str) and item in enums[key])
                 or (key in numbers and type(item) in (int, float) and math.isfinite(item) and item >= 0)
                 or (key in counts and type(item) is int and item >= 0)
                 or (key in flags and type(item) is bool))
        if valid:
            out[key] = item
    out.setdefault("kind", "env_log")
    return out


def _readable(path: Path) -> bool:
    try:
        with os.scandir(path) as entries:
            next(entries, None)
        return True
    except OSError:
        return False


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
