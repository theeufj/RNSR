"""Question-answering endpoints (synchronous)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import AppState, get_document, get_state
from ..registry import DocumentRecord
from ..schemas import (
    AskAdvancedRequest,
    AskAdvancedResponse,
    AskRequest,
    AskResponse,
    AskTextRequest,
    AskTextResponse,
    AskVisionRequest,
    AskVisionResponse,
)


router = APIRouter(tags=["qa"])


def _normalise_answer(value) -> str | None:
    """``RNSRClient.ask`` returns a string; advanced/vision return dicts.

    Defensive helper that always extracts a string answer if present.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("answer")
    return str(value)


# ---------------------------------------------------------------------------
# Document-scoped Q&A
# ---------------------------------------------------------------------------
@router.post(
    "/documents/{doc_id}/ask",
    response_model=AskResponse,
    summary="Simple Q&A — returns just the answer",
)
async def ask(
    doc_id: str,
    req: AskRequest,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> AskResponse:
    try:
        result = await asyncio.to_thread(
            state.client.ask,
            rec.path,
            req.question,
            use_knowledge_graph=req.use_knowledge_graph,
            force_reindex=req.force_reindex,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc

    state.registry.update(rec.doc_id, indexed=True)
    if req.use_knowledge_graph:
        state.registry.update(rec.doc_id, knowledge_graph_built=True)

    return AskResponse(
        answer=_normalise_answer(result) or "",
        doc_id=doc_id,
    )


@router.post(
    "/documents/{doc_id}/ask_advanced",
    response_model=AskAdvancedResponse,
    summary="Full RLM Navigator Q&A — returns answer, confidence, and trace",
)
async def ask_advanced(
    doc_id: str,
    req: AskAdvancedRequest,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> AskAdvancedResponse:
    try:
        result = await asyncio.to_thread(
            state.client.ask_advanced,
            rec.path,
            req.question,
            use_rlm=req.use_rlm,
            use_knowledge_graph=req.use_knowledge_graph,
            enable_pre_filtering=req.enable_pre_filtering,
            enable_verification=req.enable_verification,
            max_recursion_depth=req.max_recursion_depth,
            force_reindex=req.force_reindex,
            metadata=req.metadata,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc

    state.registry.update(rec.doc_id, indexed=True)
    if req.use_knowledge_graph:
        state.registry.update(rec.doc_id, knowledge_graph_built=True)

    return AskAdvancedResponse(
        answer=_normalise_answer(result),
        confidence=(result.get("confidence") if isinstance(result, dict) else None),
        doc_id=doc_id,
        raw=(result if isinstance(result, dict) else {"answer": result}),
    )


@router.post(
    "/documents/{doc_id}/ask_vision",
    response_model=AskVisionResponse,
    summary="Vision-based Q&A on PDF page images",
)
async def ask_vision(
    doc_id: str,
    req: AskVisionRequest,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> AskVisionResponse:
    try:
        result = await asyncio.to_thread(
            state.client.ask_vision,
            rec.path,
            req.question,
            req.use_hybrid,
            req.metadata,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Vision dependencies are not installed: {exc}. "
                "Install with: pip install 'rnsr[vision]'"
            ),
        ) from exc

    raw = result if isinstance(result, dict) else {}
    return AskVisionResponse(
        answer=raw.get("answer"),
        confidence=raw.get("confidence"),
        selected_pages=raw.get("selected_pages"),
        method_used=raw.get("method_used"),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Standalone (no document required)
# ---------------------------------------------------------------------------
@router.post(
    "/ask/text",
    response_model=AskTextResponse,
    tags=["qa"],
    summary="Q&A over raw text — no PDF needed",
)
async def ask_text(
    req: AskTextRequest,
    state: AppState = Depends(get_state),
) -> AskTextResponse:
    answer = await asyncio.to_thread(
        state.client.ask_text,
        req.text,
        req.question,
        req.cache_key,
    )
    return AskTextResponse(answer=_normalise_answer(answer) or "")
