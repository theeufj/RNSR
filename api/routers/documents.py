"""Document registration, listing, and deletion."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import (
    AppState,
    get_client,
    get_document,
    get_registry,
    get_state,
)
from ..registry import DocumentRecord, DocumentRegistry
from ..schemas import (
    DeleteDocumentResponse,
    DocumentInfoResponse,
    DocumentListResponse,
    DocumentRegisterRequest,
    DocumentResponse,
)


router = APIRouter(prefix="/documents", tags=["documents"])


def _to_response(rec: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        doc_id=rec.doc_id,
        name=rec.name,
        path=rec.path,
        size_bytes=rec.size_bytes,
        source=rec.source,  # type: ignore[arg-type]
        created_at=rec.created_at,
        indexed=rec.indexed,
        knowledge_graph_built=rec.knowledge_graph_built,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=201,
    summary="Register a document by upload OR by server-side path",
)
async def create_document(
    file: UploadFile | None = File(
        default=None,
        description="PDF file upload (multipart). Mutually exclusive with `path`.",
    ),
    path: str | None = Form(
        default=None,
        description="Server-side path to an existing PDF. Mutually exclusive with `file`.",
    ),
    registry: DocumentRegistry = Depends(get_registry),
) -> DocumentResponse:
    """Either upload a PDF (multipart `file`) or reference an existing one by `path`.

    Exactly one of the two must be provided. The response contains a `doc_id`
    that all subsequent endpoints use to refer to this document.
    """
    if (file is None) == (path is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of `file` (upload) or `path` (server-side).",
        )

    if file is not None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no name.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        rec = registry.register_upload(file.filename, content)
    else:
        try:
            rec = registry.register_path(Path(path))  # type: ignore[arg-type]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _to_response(rec)


@router.get("", response_model=DocumentListResponse, summary="List registered documents")
def list_documents(
    registry: DocumentRegistry = Depends(get_registry),
) -> DocumentListResponse:
    return DocumentListResponse(
        documents=[_to_response(r) for r in registry.list()]
    )


@router.get(
    "/{doc_id}",
    response_model=DocumentInfoResponse,
    summary="Get registration + cache info for a document",
)
def get_document_info(
    doc_id: str,
    rec: DocumentRecord = Depends(get_document),
    state: AppState = Depends(get_state),
) -> DocumentInfoResponse:
    """Fetch metadata for a single document and report whether RNSR has it cached."""
    info = state.client.get_document_info(rec.path)
    return DocumentInfoResponse(
        doc_id=rec.doc_id,
        path=rec.path,
        cached=bool(info.get("cached")),
        nodes=info.get("nodes"),
        indexed=rec.indexed,
        knowledge_graph_built=rec.knowledge_graph_built,
    )


@router.delete(
    "/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Remove a document from the registry",
)
def delete_document(
    doc_id: str,
    delete_files: bool = False,
    registry: DocumentRegistry = Depends(get_registry),
) -> DeleteDocumentResponse:
    """Remove a document from the registry.

    By default, the underlying PDF (and any RNSR cache) is left in place. Pass
    ``?delete_files=true`` to also remove uploaded PDF bytes from the storage
    directory. Documents registered by ``path`` (not uploaded) are never
    deleted from disk regardless of this flag.
    """
    if registry.get(doc_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")

    removed = registry.remove(doc_id, delete_files=delete_files)
    return DeleteDocumentResponse(
        doc_id=doc_id,
        deleted=removed,
        files_removed=delete_files,
    )
