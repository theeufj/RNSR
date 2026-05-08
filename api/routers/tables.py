"""Table listing and SQL-like querying."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import AppState, get_document, get_state
from ..registry import DocumentRecord
from ..schemas import (
    TableAggregateRequest,
    TableAggregateResponse,
    TableInfo,
    TableListResponse,
    TableQueryRequest,
    TableQueryResponse,
)


router = APIRouter(prefix="/documents", tags=["tables"])


@router.get(
    "/{doc_id}/tables",
    response_model=TableListResponse,
    summary="List tables auto-detected during ingestion",
)
async def list_tables(
    doc_id: str,
    force_reindex: bool = False,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> TableListResponse:
    tables = await asyncio.to_thread(
        state.client.list_tables,
        rec.path,
        force_reindex,
    )
    state.registry.update(rec.doc_id, indexed=True)
    return TableListResponse(tables=[TableInfo(**t) for t in tables])


@router.post(
    "/{doc_id}/tables/{table_id}/query",
    response_model=TableQueryResponse,
    summary="SQL-like SELECT/WHERE/ORDER BY/LIMIT over a detected table",
)
async def query_table(
    doc_id: str,
    table_id: str,
    req: TableQueryRequest,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> TableQueryResponse:
    try:
        rows = await asyncio.to_thread(
            state.client.query_table,
            rec.path,
            table_id,
            req.columns,
            req.where,
            req.order_by,
            req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TableQueryResponse(rows=rows)


@router.post(
    "/{doc_id}/tables/{table_id}/aggregate",
    response_model=TableAggregateResponse,
    summary="Aggregate (sum/avg/count/min/max) a numeric column",
)
async def aggregate_table(
    doc_id: str,
    table_id: str,
    req: TableAggregateRequest,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> TableAggregateResponse:
    try:
        value = await asyncio.to_thread(
            state.client.aggregate_table,
            rec.path,
            table_id,
            req.column,
            req.operation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TableAggregateResponse(
        operation=req.operation,
        column=req.column,
        value=float(value),
    )
