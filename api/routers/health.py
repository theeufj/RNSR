"""Health and metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

import rnsr

from ..dependencies import AppState, get_state
from ..schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Liveness probe; also reports basic application metadata."""
    return HealthResponse(
        status="ok",
        rnsr_version=getattr(rnsr, "__version__", "unknown"),
        documents_registered=len(state.registry.list()),
        storage_dir=str(state.storage_dir),
    )
