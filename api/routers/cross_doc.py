"""Cross-document Q&A endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import AppState, get_state
from ..schemas import CrossDocRequest, CrossDocResponse


router = APIRouter(tags=["qa"])


@router.post(
    "/ask/cross-document",
    response_model=CrossDocResponse,
    summary="Ask a question that spans multiple registered documents",
)
async def ask_cross_document(
    req: CrossDocRequest,
    state: AppState = Depends(get_state),
) -> CrossDocResponse:
    paths: list[str] = []
    missing: list[str] = []
    for doc_id in req.doc_ids:
        rec = state.registry.get(doc_id)
        if rec is None:
            missing.append(doc_id)
        else:
            paths.append(rec.path)

    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown document IDs: {missing}",
        )
    if not paths:
        raise HTTPException(status_code=400, detail="No documents provided.")

    result = await asyncio.to_thread(
        state.client.ask_cross_document,
        paths,
        req.question,
        store_path=None,
        metadata=req.metadata,
    )

    raw = result if isinstance(result, dict) else {}
    return CrossDocResponse(
        answer=raw.get("answer"),
        documents_used=raw.get("documents_used") or [],
        entities_involved=raw.get("entities_involved") or [],
        raw=raw,
    )
