"""FastAPI application entrypoint.

Run locally::

    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import build_app_state
from .routers import (
    cross_doc,
    documents,
    health,
    indexing,
    jobs,
    qa,
    structure,
    tables,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the singleton ``AppState`` on startup."""
    app.state.rnsr_state = build_app_state()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="RNSR API",
        description=(
            "REST wrapper around the RNSR (Recursive Neural-Symbolic Retriever) "
            "library. Upload PDFs and ask questions about them — with optional "
            "knowledge-graph extraction, vision-based analysis, table queries, "
            "and cross-document reasoning."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    cors_origins = os.getenv("RNSR_API_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(indexing.router)
    app.include_router(jobs.router)
    app.include_router(qa.router)
    app.include_router(cross_doc.router)
    app.include_router(structure.router)
    app.include_router(tables.router)

    return app


app = create_app()
