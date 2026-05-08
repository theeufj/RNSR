"""Pydantic request/response schemas for the API.

These mirror the shape of values returned by ``RNSRClient`` methods. Most
client methods return raw ``dict[str, Any]`` (LLM trace + metadata varies),
so several response models intentionally use ``dict[str, Any]`` for the
free-form parts and only structure the fields we always provide.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class DocumentRegisterRequest(BaseModel):
    """Register a PDF that already exists on the API server's filesystem."""

    path: str = Field(..., description="Absolute or relative path to a PDF on the API host.")


class DocumentResponse(BaseModel):
    doc_id: str
    name: str
    path: str
    size_bytes: int
    source: Literal["upload", "path"]
    created_at: str
    indexed: bool
    knowledge_graph_built: bool


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentInfoResponse(BaseModel):
    doc_id: str
    path: str
    cached: bool
    nodes: int | None = None
    indexed: bool
    knowledge_graph_built: bool


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted: bool
    files_removed: bool


# ---------------------------------------------------------------------------
# Indexing / jobs
# ---------------------------------------------------------------------------
class IndexRequest(BaseModel):
    build_knowledge_graph: bool = Field(
        default=True,
        description=(
            "Build the entity-extraction knowledge graph in addition to the "
            "skeleton index. Slower (LLM calls per node) but improves accuracy."
        ),
    )
    force_reindex: bool = Field(
        default=False,
        description="Re-process even if a cached index exists.",
    )


class JobResponse(BaseModel):
    job_id: str
    kind: str
    doc_id: str | None = None
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """Simple Q&A request, mapped to ``RNSRClient.ask``."""

    question: str = Field(..., min_length=1)
    use_knowledge_graph: bool = True
    force_reindex: bool = False


class AskResponse(BaseModel):
    answer: str
    doc_id: str | None = None


class AskAdvancedRequest(BaseModel):
    """Full ``RNSRClient.ask_advanced`` parameters."""

    question: str = Field(..., min_length=1)
    use_rlm: bool = True
    use_knowledge_graph: bool = True
    enable_pre_filtering: bool = True
    enable_verification: bool = False
    max_recursion_depth: int = Field(default=3, ge=1, le=10)
    force_reindex: bool = False
    metadata: dict[str, Any] | None = None


class AskAdvancedResponse(BaseModel):
    answer: str | None = None
    confidence: float | None = None
    doc_id: str | None = None
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="The full result dict returned by RNSRClient.ask_advanced.",
    )


class AskVisionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    use_hybrid: bool = True
    metadata: dict[str, Any] | None = None


class AskVisionResponse(BaseModel):
    answer: str | None = None
    confidence: float | None = None
    selected_pages: list[int] | None = None
    method_used: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class AskTextRequest(BaseModel):
    text: str | list[str] = Field(..., description="Raw text or a list of text chunks.")
    question: str = Field(..., min_length=1)
    cache_key: str | None = Field(
        default=None,
        description="Optional stable cache key. Auto-generated from content hash if omitted.",
    )


class AskTextResponse(BaseModel):
    answer: str


class CrossDocRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class CrossDocResponse(BaseModel):
    answer: str | None = None
    documents_used: list[str] = Field(default_factory=list)
    entities_involved: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structure / outline
# ---------------------------------------------------------------------------
class StructureResponse(BaseModel):
    path: str
    section_count: int
    max_depth: int
    level_distribution: dict[int, int]
    total_characters: int
    average_section_length: int


class OutlineEntry(BaseModel):
    id: str
    header: str
    level: int
    summary: str
    child_count: int


class OutlineResponse(BaseModel):
    entries: list[OutlineEntry]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
class TableInfo(BaseModel):
    id: str
    node_id: str
    page_num: int | None = None
    title: str
    headers: list[str]
    num_rows: int
    num_cols: int


class TableListResponse(BaseModel):
    tables: list[TableInfo]


class TableQueryRequest(BaseModel):
    columns: list[str] | None = None
    where: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Filter conditions. Either a literal value (case-insensitive substring match) "
            'or {"op": "==|!=|>|>=|<|<=|contains", "value": <val>}.'
        ),
    )
    order_by: str | None = Field(
        default=None,
        description="Column name to sort by. Prefix with '-' for descending.",
    )
    limit: int | None = Field(default=None, ge=1)


class TableQueryResponse(BaseModel):
    rows: list[dict[str, Any]]


class TableAggregateRequest(BaseModel):
    column: str
    operation: Literal["sum", "avg", "count", "min", "max"]


class TableAggregateResponse(BaseModel):
    operation: str
    column: str
    value: float


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"]
    rnsr_version: str
    documents_registered: int
    storage_dir: str
