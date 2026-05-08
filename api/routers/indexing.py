"""Explicit indexing endpoint backed by background jobs.

Indexing (especially knowledge-graph extraction) can take minutes for large
PDFs. The API exposes it as an asynchronous job: ``POST /documents/{id}/index``
returns a ``job_id``, then ``GET /jobs/{job_id}`` reports progress.

The Q&A endpoints (`/ask`, `/ask_advanced`, ...) will *also* trigger indexing
on demand if needed — this endpoint is for callers that want to pre-warm the
cache or rebuild the knowledge graph without firing a query.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends

from ..dependencies import AppState, get_document, get_state
from ..registry import DocumentRecord
from ..schemas import IndexRequest, JobResponse


router = APIRouter(prefix="/documents", tags=["indexing"])


def _job_to_response(job) -> JobResponse:
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


def _run_indexing(
    state: AppState,
    rec: DocumentRecord,
    job_id: str,
    build_kg: bool,
    force_reindex: bool,
) -> None:
    """Run the actual ingestion + indexing in a worker thread."""
    state.jobs.mark_running(job_id)
    try:
        skeleton, kv_store = state.client._get_or_create_index(
            __import__("pathlib").Path(rec.path),
            force_reindex,
        )
        state.registry.update(rec.doc_id, indexed=True)

        kg_entities = 0
        kg_relationships = 0
        if build_kg:
            cache_key = state.client._get_cache_key(
                __import__("pathlib").Path(rec.path)
            )
            kg = state.client._get_or_create_knowledge_graph(
                cache_key=cache_key,
                skeleton=skeleton,
                kv_store=kv_store,
                doc_id=rec.name,
            )
            stats = kg.get_stats()
            kg_entities = int(stats.get("entity_count", 0))
            kg_relationships = int(stats.get("relationship_count", 0))
            state.registry.update(rec.doc_id, knowledge_graph_built=True)

        state.jobs.mark_completed(
            job_id,
            result={
                "doc_id": rec.doc_id,
                "nodes": len(skeleton),
                "knowledge_graph_built": build_kg,
                "kg_entities": kg_entities,
                "kg_relationships": kg_relationships,
            },
        )
    except Exception as exc:  # noqa: BLE001 -- we want to capture all errors
        state.jobs.mark_failed(job_id, f"{type(exc).__name__}: {exc}")


@router.post(
    "/{doc_id}/index",
    response_model=JobResponse,
    status_code=202,
    summary="Build skeleton index (and optionally knowledge graph) in the background",
)
async def trigger_index(
    doc_id: str,
    req: IndexRequest,
    background_tasks: BackgroundTasks,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> JobResponse:
    """Kick off indexing for a document. Returns immediately with a job handle."""
    job = state.jobs.create("index", doc_id=doc_id)

    async def _runner() -> None:
        lock = state.indexing_locks[doc_id]
        async with lock:
            await asyncio.to_thread(
                _run_indexing,
                state,
                rec,
                job.job_id,
                req.build_knowledge_graph,
                req.force_reindex,
            )

    background_tasks.add_task(_runner)
    return _job_to_response(job)
