"""Document structure analysis and outline endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query

from ..dependencies import AppState, get_document, get_state
from ..registry import DocumentRecord
from ..schemas import (
    OutlineEntry,
    OutlineResponse,
    StructureResponse,
)


router = APIRouter(prefix="/documents", tags=["structure"])


@router.get(
    "/{doc_id}/structure",
    response_model=StructureResponse,
    summary="Document hierarchy stats (section count, depth, length distribution)",
)
async def analyze_structure(
    doc_id: str,
    force_reindex: bool = False,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> StructureResponse:
    info = await asyncio.to_thread(
        state.client.analyze_document_structure,
        rec.path,
        force_reindex,
    )
    state.registry.update(rec.doc_id, indexed=True)
    return StructureResponse(**info)


@router.get(
    "/{doc_id}/outline",
    response_model=OutlineResponse,
    summary="Table of contents — headers and summaries up to a depth",
)
async def get_outline(
    doc_id: str,
    max_depth: int = Query(default=2, ge=1, le=10),
    force_reindex: bool = False,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> OutlineResponse:
    entries = await asyncio.to_thread(
        state.client.get_document_outline,
        rec.path,
        max_depth,
        force_reindex,
    )
    state.registry.update(rec.doc_id, indexed=True)
    return OutlineResponse(entries=[OutlineEntry(**e) for e in entries])
