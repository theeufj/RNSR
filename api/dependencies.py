"""Shared FastAPI dependencies and application state.

The state is built once during application startup (see ``main.lifespan``) and
exposed to handlers via the dependency-injection helpers below.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from rnsr import RNSRClient

from .jobs import JobRegistry
from .registry import DocumentRecord, DocumentRegistry


@dataclass
class AppState:
    """Aggregates everything handlers need from the running app."""

    client: RNSRClient
    registry: DocumentRegistry
    jobs: JobRegistry
    storage_dir: Path
    # Per-document indexing locks to prevent concurrent re-indexing
    indexing_locks: dict[str, asyncio.Lock]


def build_app_state() -> AppState:
    """Construct the singleton ``AppState`` from environment variables.

    Environment variables:
        RNSR_API_STORAGE_DIR: Directory for uploads, registry, and the RNSR
                              cache. Defaults to ``./.rnsr_api_storage``.
        RNSR_LLM_PROVIDER:    Optional LLM provider override.
        RNSR_LLM_MODEL:       Optional model override.
        RNSR_API_KEY:         Optional API key for the chosen provider
                              (falls back to provider-specific env vars).
    """
    storage_dir = Path(
        os.getenv("RNSR_API_STORAGE_DIR", "./.rnsr_api_storage")
    ).resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = storage_dir / "rnsr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = RNSRClient(
        cache_dir=cache_dir,
        llm_provider=os.getenv("RNSR_LLM_PROVIDER"),
        llm_model=os.getenv("RNSR_LLM_MODEL"),
        api_key=os.getenv("RNSR_API_KEY"),
    )

    return AppState(
        client=client,
        registry=DocumentRegistry(storage_dir),
        jobs=JobRegistry(),
        storage_dir=storage_dir,
        indexing_locks=defaultdict(asyncio.Lock),
    )


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------
def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "rnsr_state", None)
    if state is None:
        raise HTTPException(
            status_code=503,
            detail="Application state is not initialised. Did the lifespan run?",
        )
    return state


def get_client(request: Request) -> RNSRClient:
    return get_state(request).client


def get_registry(request: Request) -> DocumentRegistry:
    return get_state(request).registry


def get_jobs(request: Request) -> JobRegistry:
    return get_state(request).jobs


def get_document(request: Request, doc_id: str) -> DocumentRecord:
    """Resolve a ``doc_id`` path parameter into a :class:`DocumentRecord`.

    Raises ``404`` if the document is unknown, ``410`` if it was registered
    but the underlying file no longer exists.
    """
    rec = get_registry(request).get(doc_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    if not Path(rec.path).exists():
        raise HTTPException(
            status_code=410,
            detail=f"Document file no longer exists at {rec.path}",
        )
    return rec
