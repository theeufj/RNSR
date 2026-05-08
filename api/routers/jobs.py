"""Background-job status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_jobs
from ..jobs import JobRegistry
from ..schemas import JobListResponse, JobResponse


router = APIRouter(prefix="/jobs", tags=["jobs"])


def _to_response(job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        kind=job.kind,
        doc_id=job.doc_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=job.result,
        error=job.error,
    )


@router.get("", response_model=JobListResponse, summary="List all background jobs")
def list_jobs(jobs: JobRegistry = Depends(get_jobs)) -> JobListResponse:
    return JobListResponse(jobs=[_to_response(j) for j in jobs.list()])


@router.get("/{job_id}", response_model=JobResponse, summary="Get job status")
def get_job(job_id: str, jobs: JobRegistry = Depends(get_jobs)) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return _to_response(job)
