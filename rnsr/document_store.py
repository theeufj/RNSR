"""
Document Store - Multi-Document Management

Provides a high-level interface for managing multiple indexed documents.
Handles persistence, loading, and querying across a document collection.

Usage:
    from rnsr import DocumentStore
    
    # Create or open a document store
    store = DocumentStore("./my_documents/")
    
    # Add documents
    store.add_document("contract.pdf")
    store.add_document("report.pdf", metadata={"year": 2024})
    
    # Query a specific document
    answer = store.query("contract", "What are the payment terms?")
    
    # List all documents
    for doc in store.list_documents():
        print(f"{doc['id']}: {doc['title']}")
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import structlog

from rnsr.exceptions import IndexingError
from rnsr.indexing.collection_skeleton import (
    CollectionSkeletonBuilder,
    is_doc_stub,
)
from rnsr.indexing.expandable_skeleton import ExpandableSkeleton
from rnsr.indexing.kv_store import KVStore, LazyKVStore, SQLiteKVStore
from rnsr.indexing.persistence import (
    save_index,
    load_index,
    get_index_info,
    delete_index,
)
from rnsr.indexing.skeleton_index import build_skeleton_index
from rnsr.indexing.store_db import StoreDB
from rnsr.ingestion import ingest_document
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


# =============================================================================
# Document Metadata Extraction
# =============================================================================

# Common date patterns in legal / business documents
_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[/-]\w{3,9}[/-]\d{2,4})\b"),          # 11-Nov-24, 25/Jul/22
    re.compile(r"\b(\d{1,2}[/ ]\w{3,9}[, ]+\d{4})\b"),           # 25 July 2022, 11 November 2024
    re.compile(r"\b(\w{3,9} \d{1,2},? \d{4})\b"),                 # July 25, 2022
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"),                 # 11/12/2024
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),                       # 2024-11-29
]

_REF_PATTERNS = [
    re.compile(r"(?:Our Ref|Your Ref|Ref|Reference)[:\s]+([^\n]{3,40})", re.IGNORECASE),
    re.compile(r"(?:File No|File Number|Matter No)[.:\s]+([^\n]{3,30})", re.IGNORECASE),
]


def _extract_document_metadata(
    skeleton: dict[str, "SkeletonNode"],
    kv_store: "KVStore",
) -> dict[str, Any]:
    """Pull dates, reference numbers, and page info from a document's content.

    The extracted metadata is intentionally kept lightweight — it is
    stored on the skeleton root node so the navigator can surface it
    without reading every section.
    """
    meta: dict[str, Any] = {}

    # Find root and leaf nodes
    root = None
    max_page = 0
    for node in skeleton.values():
        if node.level == 0:
            root = node
        if node.page_num and node.page_num > max_page:
            max_page = node.page_num

    if max_page > 0:
        meta["total_pages"] = max_page

    # Gather text from the first few nodes (headers / opening) for date & ref extraction
    sample_texts: list[str] = []
    if root:
        content = kv_store.get(root.node_id)
        if content:
            sample_texts.append(content[:2000])
        for cid in root.child_ids[:3]:
            child_content = kv_store.get(cid)
            if child_content:
                sample_texts.append(child_content[:1000])

    combined = "\n".join(sample_texts)

    # Extract dates
    dates_found: list[str] = []
    for pat in _DATE_PATTERNS:
        dates_found.extend(pat.findall(combined))
    if dates_found:
        meta["dates_found"] = dates_found[:5]

    # Extract references
    refs_found: list[str] = []
    for pat in _REF_PATTERNS:
        refs_found.extend(m.strip() for m in pat.findall(combined))
    if refs_found:
        meta["references_found"] = refs_found[:5]

    return meta


def _get_source_page_count(source_path: Path) -> int | None:
    """Return the true page count from the source file.

    Only reliable for PDF files (via PyMuPDF). DOCX ``docProps/app.xml``
    is not used because Word only updates it on save, making it
    frequently stale for programmatically generated documents.
    """
    if source_path.suffix.lower() != ".pdf":
        return None
    try:
        import fitz
        doc = fitz.open(source_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return None


@dataclass
class DocumentInfo:
    """Information about an indexed document."""
    
    id: str
    title: str
    source_path: str | None
    node_count: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchProgress:
    """Progress update emitted after each file during batch ingestion."""

    completed: int
    total: int
    current_file: str
    doc_id: str | None
    status: str  # "success" | "skipped" | "error"
    error: str | None = None


@dataclass
class BatchResult:
    """Aggregate result of a batch ingestion run."""

    total: int
    succeeded: int
    skipped: int
    failed: int
    doc_ids: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class DocumentStore:
    """
    Manages a collection of indexed documents.
    
    Provides:
    - Add/remove documents
    - Persistent storage
    - Query individual documents
    - List and search documents
    
    Example:
        store = DocumentStore("./documents/")
        store.add_document("contract.pdf")
        answer = store.query("contract", "What are the terms?")
    """
    
    def __init__(self, store_path: str | Path, root_path: str | Path | None = None):
        """
        Initialize or open a document store.
        
        Args:
            store_path: Directory for storing document indexes.
            root_path: Original root directory of the document collection.
                When set, the folder hierarchy relative to this path is
                used to build a collection skeleton for hierarchical
                navigation.  When ``None``, documents are placed flat
                under the root (scoped / matter mode).
        """
        self.store_path = Path(store_path).resolve()
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.root_path = Path(root_path).resolve() if root_path else None
        
        # Unified SQLite store (WAL mode, single file)
        self._db = StoreDB(self.store_path)

        # Auto-migrate legacy multi-file stores
        self._db.migrate_from_legacy()

        self._catalog: dict[str, DocumentInfo] = {}
        
        # Lock for thread-safe catalog and skeleton mutations
        self._lock = threading.Lock()
        
        # Collection skeleton (lazy loaded)
        self._collection_skeleton: dict[str, SkeletonNode] | None = None

        # Cache loaded document indexes to avoid re-opening SQLite per query
        self._doc_cache: dict[str, tuple] = {}
        
        # Load catalog from unified DB
        self._load_catalog()
        
        logger.info(
            "document_store_initialized",
            path=str(self.store_path),
            documents=len(self._catalog),
        )
    
    def _load_catalog(self) -> None:
        """Load the document catalog from the unified DB."""
        catalog_data = self._db.load_catalog()
        self._catalog = {
            doc_id: DocumentInfo(**info)
            for doc_id, info in catalog_data.items()
        }

    def _save_catalog(self) -> None:
        """No-op: catalog is persisted by StoreDB.save_document / remove_document."""

    # ------------------------------------------------------------------
    # Collection skeleton helpers
    # ------------------------------------------------------------------

    def _get_doc_root_summary(self, doc_id: str) -> str:
        """Extract the root-node summary for a document from the unified DB.

        When the root node carries extracted metadata (dates, references,
        page count) the summary is augmented so the navigator can see
        this information at the collection level without drilling in.
        """
        return self._db.get_root_summary(doc_id)

    def _build_collection_skeleton(self) -> dict[str, SkeletonNode]:
        """Build and save the collection skeleton from the current catalog."""
        catalog_dict: dict[str, dict[str, Any]] = {}
        for doc_id, info in self._catalog.items():
            catalog_dict[doc_id] = {
                "title": info.title,
                "source_path": info.source_path or "",
                "summary": self._get_doc_root_summary(doc_id),
                "node_count": info.node_count,
            }

        builder = CollectionSkeletonBuilder(catalog_dict, root_path=self.root_path)
        nodes = builder.build()
        builder.save(self.store_path)
        self._collection_skeleton = nodes
        return nodes

    def _load_collection_skeleton(self) -> dict[str, SkeletonNode] | None:
        """Load the collection skeleton from disk if it exists."""
        nodes = CollectionSkeletonBuilder.load(self.store_path)
        if nodes:
            self._collection_skeleton = nodes
        return nodes or None

    def _update_collection_skeleton_add(self, doc_id: str, info: DocumentInfo) -> None:
        """Incrementally add a doc stub to the collection skeleton."""
        if self._collection_skeleton is None:
            self._load_collection_skeleton()
        if self._collection_skeleton is None:
            self._build_collection_skeleton()
            return

        catalog_dict: dict[str, dict[str, Any]] = {}
        for did, inf in self._catalog.items():
            catalog_dict[did] = {
                "title": inf.title,
                "source_path": inf.source_path or "",
                "summary": self._get_doc_root_summary(did),
                "node_count": inf.node_count,
            }

        builder = CollectionSkeletonBuilder(catalog_dict, root_path=self.root_path)
        builder._nodes = dict(self._collection_skeleton)
        builder.add_doc_stub(
            doc_id=doc_id,
            title=info.title,
            summary=self._get_doc_root_summary(doc_id),
            source_path=info.source_path,
            node_count=info.node_count,
        )
        builder.save(self.store_path)
        self._collection_skeleton = builder._nodes

    def _update_collection_skeleton_remove(self, doc_id: str) -> None:
        """Incrementally remove a doc stub from the collection skeleton."""
        if self._collection_skeleton is None:
            self._load_collection_skeleton()
        if self._collection_skeleton is None:
            return

        catalog_dict: dict[str, dict[str, Any]] = {}
        for did, inf in self._catalog.items():
            catalog_dict[did] = {
                "title": inf.title,
                "source_path": inf.source_path or "",
                "node_count": inf.node_count,
            }
        builder = CollectionSkeletonBuilder(catalog_dict, root_path=self.root_path)
        builder._nodes = dict(self._collection_skeleton)
        builder.remove_doc_stub(doc_id)
        builder.save(self.store_path)
        self._collection_skeleton = builder._nodes

    def add_document(
        self,
        source: str | Path,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add and index a document.
        
        Args:
            source: Path to PDF file
            doc_id: Optional custom ID (defaults to filename hash)
            title: Optional title (defaults to filename)
            metadata: Optional metadata dictionary
            
        Returns:
            Document ID
            
        Example:
            doc_id = store.add_document("report.pdf", metadata={"year": 2024})
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise IndexingError(f"Source file not found: {source_path}")
        
        # Generate ID if not provided
        if doc_id is None:
            # Hash of filename + file size for uniqueness
            hash_input = f"{source_path.name}_{source_path.stat().st_size}"
            doc_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        # Check if already exists
        with self._lock:
            if doc_id in self._catalog:
                logger.warning("document_already_exists", doc_id=doc_id)
                return doc_id
        
        # Ingest document (outside lock -- this is the expensive step)
        logger.info("ingesting_document", source=str(source_path))
        result = ingest_document(str(source_path))
        
        # Build skeleton index
        skeleton, kv_store = build_skeleton_index(result.tree)

        # Extract document-level metadata (dates, references, page counts)
        # and attach it to the root skeleton node so the navigator can
        # surface it without reading every section.
        doc_meta = _extract_document_metadata(skeleton, kv_store)

        # Get authoritative page count from the source file itself
        # (overrides the heuristic max-page_num from _extract_document_metadata)
        page_count = _get_source_page_count(source_path)
        if page_count:
            doc_meta["total_pages"] = page_count
            result.tree.page_count = page_count

        if doc_meta:
            for node in skeleton.values():
                if node.level == 0:
                    node.metadata.update(doc_meta)
                    break
        
        # Atomically save skeleton + content + catalog to unified DB
        doc_title = title or source_path.stem
        self._db.save_document(
            doc_id=doc_id,
            title=doc_title,
            source_path=str(source_path),
            skeleton=skeleton,
            kv_store=kv_store,
            tables=result.tables,
            metadata=metadata,
        )
        
        # Update in-memory catalog (under lock for thread safety)
        info = DocumentInfo(
            id=doc_id,
            title=doc_title,
            source_path=str(source_path),
            node_count=len(skeleton),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        with self._lock:
            self._catalog[doc_id] = info
            self._update_collection_skeleton_add(doc_id, info)
        
        logger.info(
            "document_added",
            doc_id=doc_id,
            title=info.title,
            nodes=info.node_count,
        )
        
        return doc_id
    
    def add_from_text(
        self,
        text: str | list[str],
        doc_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add and index a document from raw text.
        
        Args:
            text: Text content or list of text chunks
            doc_id: Document ID
            title: Optional title
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        from rnsr.ingestion import build_tree_from_text
        
        # Check if already exists
        if doc_id in self._catalog:
            logger.warning("document_already_exists", doc_id=doc_id)
            return doc_id
        
        # Build tree from text
        tree = build_tree_from_text(text)
        
        # Build skeleton index
        skeleton, kv_store = build_skeleton_index(tree)
        
        # Atomically save to unified DB
        doc_title = title or doc_id
        self._db.save_document(
            doc_id=doc_id,
            title=doc_title,
            source_path=None,
            skeleton=skeleton,
            kv_store=kv_store,
            metadata=metadata,
        )
        
        # Update in-memory catalog
        info = DocumentInfo(
            id=doc_id,
            title=doc_title,
            source_path=None,
            node_count=len(skeleton),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._catalog[doc_id] = info
        self._update_collection_skeleton_add(doc_id, info)
        
        logger.info(
            "document_added_from_text",
            doc_id=doc_id,
            title=info.title,
            nodes=info.node_count,
        )
        
        return doc_id
    
    # All file types the ingestion pipeline can handle natively.
    SUPPORTED_EXTENSIONS = {"*.pdf", "*.md", "*.txt", "*.text", "*.markdown", "*.docx"}

    def batch_ingest(
        self,
        sources: str | Path | list[str | Path],
        recursive: bool = False,
        glob_pattern: str | None = None,
        metadata: dict[str, Any] | None = None,
        skip_existing: bool = True,
        max_workers: int = 4,
        build_kg: bool = True,
        on_progress: Callable[[BatchProgress], None] | None = None,
    ) -> BatchResult:
        """
        Ingest multiple documents in batch.

        Accepts a folder path, a list of file paths, or a single file path.
        Each file is ingested independently so one failure does not abort the
        rest of the batch.

        Args:
            sources: Directory path, single file path, or list of file paths.
            recursive: When *sources* is a directory, recurse into
                subdirectories.
            glob_pattern: Glob used to discover files inside a directory.
                Defaults to all supported types (pdf, docx, md, txt).
                Accepts comma-separated patterns like ``"*.pdf,*.docx"``.
            metadata: Metadata dict applied to every ingested document.
            skip_existing: Skip files whose generated doc_id is already in the
                catalog.
            max_workers: Number of parallel ingestion workers (default 4).
                Set to 1 for sequential processing.
            build_kg: Call :meth:`build_workspace_kg` and
                :meth:`link_entities_across_documents` after ingestion
                (default ``True``).
            on_progress: Optional callback invoked after each file is
                processed.

        Returns:
            A :class:`BatchResult` summarising the run.

        Example::

            result = store.batch_ingest("./contracts/", recursive=True)
            print(f"{result.succeeded}/{result.total} documents ingested")
        """
        patterns = self._parse_glob_patterns(glob_pattern)
        files = self._resolve_sources(sources, recursive, patterns)

        if not files:
            logger.warning("batch_ingest_no_files", sources=str(sources))
            return BatchResult(total=0, succeeded=0, skipped=0, failed=0)

        logger.info(
            "batch_ingest_start",
            total_files=len(files),
            max_workers=max_workers,
        )

        start = time.monotonic()
        succeeded = 0
        skipped = 0
        failed = 0
        doc_ids: list[str] = []
        errors: list[dict[str, str]] = []
        completed = 0

        def _ingest_one(file_path: Path) -> tuple[str, str | None, str | None]:
            """Returns (status, doc_id_or_none, error_or_none)."""
            fpath = Path(file_path)
            candidate_id = hashlib.md5(
                f"{fpath.name}_{fpath.stat().st_size}".encode()
            ).hexdigest()[:12]

            if skip_existing and candidate_id in self._catalog:
                return ("skipped", candidate_id, None)

            try:
                doc_id = self.add_document(
                    fpath, doc_id=candidate_id, metadata=metadata
                )
                return ("success", doc_id, None)
            except Exception as exc:
                return ("error", None, str(exc))

        if max_workers <= 1:
            for fpath in files:
                status, doc_id, error = _ingest_one(fpath)
                completed += 1
                if status == "success":
                    succeeded += 1
                    doc_ids.append(doc_id)  # type: ignore[arg-type]
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    errors.append({"file": str(fpath), "error": error or ""})

                if on_progress:
                    on_progress(BatchProgress(
                        completed=completed,
                        total=len(files),
                        current_file=str(fpath),
                        doc_id=doc_id,
                        status=status,
                        error=error,
                    ))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_path = {
                    pool.submit(_ingest_one, fp): fp for fp in files
                }
                _INGEST_TIMEOUT = 600  # 10 min per document
                for future in as_completed(future_to_path):
                    fpath = future_to_path[future]
                    try:
                        status, doc_id, error = future.result(timeout=_INGEST_TIMEOUT)
                    except TimeoutError:
                        logger.warning("ingest_timeout", file=str(fpath), timeout_s=_INGEST_TIMEOUT)
                        status, doc_id, error = "failed", None, f"Timed out after {_INGEST_TIMEOUT}s"
                    completed += 1
                    if status == "success":
                        succeeded += 1
                        doc_ids.append(doc_id)  # type: ignore[arg-type]
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                        errors.append({"file": str(fpath), "error": error or ""})

                    if on_progress:
                        on_progress(BatchProgress(
                            completed=completed,
                            total=len(files),
                            current_file=str(fpath),
                            doc_id=doc_id,
                            status=status,
                            error=error,
                        ))

        elapsed = time.monotonic() - start

        result = BatchResult(
            total=len(files),
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
            doc_ids=doc_ids,
            errors=errors,
            elapsed_seconds=round(elapsed, 2),
        )

        logger.info(
            "batch_ingest_complete",
            total=result.total,
            succeeded=result.succeeded,
            skipped=result.skipped,
            failed=result.failed,
            elapsed=result.elapsed_seconds,
        )

        # Build or rebuild the collection skeleton
        if doc_ids:
            self._build_collection_skeleton()

        if build_kg and doc_ids:
            logger.info("batch_ingest_building_kg", doc_count=len(doc_ids))
            self.build_workspace_kg(doc_ids=doc_ids)
            self.link_entities_across_documents(doc_ids=doc_ids)

        return result

    @classmethod
    def _parse_glob_patterns(cls, glob_pattern: str | None) -> list[str]:
        """Turn *glob_pattern* into a list of individual globs.

        ``None`` → all supported extensions.
        Comma-separated strings like ``"*.pdf,*.docx"`` are split.
        """
        if glob_pattern is None:
            return sorted(cls.SUPPORTED_EXTENSIONS)
        return [p.strip() for p in glob_pattern.split(",") if p.strip()]

    @staticmethod
    def _resolve_sources(
        sources: str | Path | list[str | Path],
        recursive: bool,
        glob_patterns: list[str],
    ) -> list[Path]:
        """Normalise *sources* into a de-duplicated list of file paths."""
        seen: set[Path] = set()
        resolved: list[Path] = []

        def _add(p: Path) -> None:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                resolved.append(p)

        def _glob_dir(directory: Path) -> None:
            for pattern in glob_patterns:
                full = f"**/{pattern}" if recursive else pattern
                for fp in sorted(directory.glob(full)):
                    _add(fp)

        if isinstance(sources, (str, Path)):
            source_path = Path(sources)
            if source_path.is_dir():
                _glob_dir(source_path)
            elif source_path.is_file():
                _add(source_path)
            return resolved

        for s in sources:
            p = Path(s)
            if p.is_dir():
                _glob_dir(p)
            elif p.is_file():
                _add(p)
        return resolved

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if doc_id not in self._catalog:
                return False

            self._db.remove_document(doc_id)

            # Invalidate cache
            self._doc_cache.pop(doc_id, None)

            # Remove from in-memory catalog
            del self._catalog[doc_id]
            self._update_collection_skeleton_remove(doc_id)

        logger.info("document_removed", doc_id=doc_id)
        return True
    
    def clear_all(self) -> int:
        """
        Remove all documents and reset the catalog.
        
        Returns:
            Number of documents removed
        """
        count = self._db.clear_documents()

        # Clear KG tables (they live in the same DB)
        kg = self.get_workspace_kg()
        try:
            kg.clear()
        except Exception:
            pass

        # Remove collection skeleton file
        from rnsr.indexing.collection_skeleton import COLLECTION_SKELETON_FILE
        cs_path = self.store_path / COLLECTION_SKELETON_FILE
        if cs_path.exists():
            cs_path.unlink()
        self._collection_skeleton = None
        
        self._catalog.clear()
        self._doc_cache.clear()
        
        logger.info("store_cleared", documents_removed=count)
        return count
    
    def get_document(
        self,
        doc_id: str,
    ) -> tuple[dict[str, SkeletonNode], KVStore] | None:
        """
        Load a document's index (cached after first load).

        Args:
            doc_id: Document ID

        Returns:
            Tuple of (skeleton, kv_store, tables) or None if not found.
            Returns None (with a warning log) when the index exists in
            the catalog but cannot be loaded from the database.
        """
        if doc_id not in self._catalog:
            return None

        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]

        try:
            result = self._db.load_document(doc_id)
            if result is None:
                logger.warning("get_document_not_in_db", doc_id=doc_id)
                return None
            self._doc_cache[doc_id] = result
            return result
        except Exception as exc:
            logger.warning(
                "get_document_load_failed",
                doc_id=doc_id,
                error=str(exc),
            )
            return None
    
    def query(
        self,
        doc_id: str,
        question: str,
    ) -> str:
        """
        Query a document.
        
        Args:
            doc_id: Document ID
            question: Question to ask
            
        Returns:
            Answer string
            
        Example:
            answer = store.query("contract_123", "What are the payment terms?")
        """
        from rnsr.agent import run_navigator
        
        index_result = self.get_document(doc_id)
        if index_result is None:
            raise IndexingError(f"Document not found: {doc_id}")

        skeleton, kv_store = index_result[:2]
        tables = index_result[2] if len(index_result) > 2 else None
        nav_result = run_navigator(question, skeleton, kv_store, tables=tables)
        return nav_result.get("answer", "No answer found.")
    
    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the store.
        
        Returns:
            List of document info dictionaries
        """
        return [info.to_dict() for info in self._catalog.values()]
    
    def get_document_info(self, doc_id: str) -> DocumentInfo | None:
        """
        Get information about a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentInfo or None if not found
        """
        return self._catalog.get(doc_id)
    
    def search_documents(
        self,
        query: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentInfo]:
        """
        Search documents by title or metadata.
        
        Args:
            query: Optional text to search in titles
            metadata_filter: Optional metadata key-value pairs to match
            
        Returns:
            List of matching DocumentInfo objects
        """
        results = []
        
        for info in self._catalog.values():
            # Title search
            if query and query.lower() not in info.title.lower():
                continue
            
            # Metadata filter
            if metadata_filter:
                match = all(
                    info.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                )
                if not match:
                    continue
            
            results.append(info)
        
        return results
    
    # =========================================================================
    # Workspace Knowledge Graph and Cross-Document Entity Linking
    # =========================================================================

    def get_workspace_kg(self) -> "KnowledgeGraph":
        """
        Get or create the workspace-wide knowledge graph.
        
        The KG tables live inside the unified ``store.db`` and accumulate
        entities from all documents added to the store. Use it together with
        :meth:`link_entities_across_documents` and :meth:`query_cross_document`
        to enable cross-document reasoning.
        
        Returns:
            A file-backed ``KnowledgeGraph`` sharing the unified DB file.
            
        Example:
            kg = store.get_workspace_kg()
            print(kg.get_stats())
        """
        from rnsr.indexing.knowledge_graph import KnowledgeGraph
        
        return KnowledgeGraph(self._db.kg_db_path)

    def build_workspace_kg(
        self,
        doc_ids: list[str] | None = None,
        max_workers: int = 4,
        batch_size: int = 8,
    ) -> "KnowledgeGraph":
        """
        Build (or extend) the workspace KG from indexed documents.

        Extracts entities and relationships from each document's skeleton
        nodes and merges them into the workspace KG.

        Sections shorter than ``batch_size`` threshold characters are
        batched together into a single LLM call to reduce API round-trips.
        Documents are processed in parallel using *max_workers* threads.

        Args:
            doc_ids: Specific document IDs to process (default: all).
            max_workers: Number of parallel document extraction threads.
            batch_size: Number of small sections to combine into one LLM call.

        Returns:
            The populated workspace ``KnowledgeGraph``.
        """
        from rnsr.indexing.knowledge_graph import KnowledgeGraph
        from rnsr.extraction import extract_entities_and_relationships
        from rnsr.extraction.rlm_unified_extractor import (
            extract_entities_and_relationships_batch,
        )

        kg = self.get_workspace_kg()
        target_ids = doc_ids or list(self._catalog.keys())

        _SMALL_SECTION_CHARS = 1500

        def _process_document(doc_id: str) -> int:
            """Extract entities from one document. Returns entity count."""
            index_result = self.get_document(doc_id)
            if index_result is None:
                logger.warning("doc_not_found_for_kg", doc_id=doc_id)
                return 0

            skeleton, kv_store = index_result[:2]

            small_items: list[tuple[str, str, str, str, str | None]] = []
            entity_count = 0

            for node_id, node in skeleton.items():
                content = kv_store.get(node_id) or ""
                if len(content.strip()) < 50:
                    continue

                if len(content) <= _SMALL_SECTION_CHARS:
                    small_items.append(
                        (node_id, doc_id, node.header, content, None)
                    )
                    if len(small_items) >= batch_size:
                        entity_count += _flush_batch(small_items, kg)
                        small_items.clear()
                else:
                    try:
                        result = extract_entities_and_relationships(
                            node_id=node_id,
                            doc_id=doc_id,
                            header=node.header,
                            content=content,
                        )
                        for entity in result.entities:
                            kg.add_entity(entity)
                        entity_count += len(result.entities)
                        for rel in result.relationships:
                            kg.add_relationship(rel)
                    except Exception as exc:
                        logger.debug(
                            "workspace_kg_node_error",
                            doc_id=doc_id,
                            node_id=node_id,
                            error=str(exc),
                        )

            if small_items:
                entity_count += _flush_batch(small_items, kg)

            return entity_count

        def _flush_batch(
            items: list[tuple[str, str, str, str, str | None]],
            kg_: "KnowledgeGraph",
        ) -> int:
            count = 0
            try:
                results = extract_entities_and_relationships_batch(items)
                for r in results:
                    for entity in r.entities:
                        kg_.add_entity(entity)
                    count += len(r.entities)
                    for rel in r.relationships:
                        kg_.add_relationship(rel)
            except Exception as exc:
                logger.debug("batch_extraction_fallback", error=str(exc))
                for nid, did, h, c, ac in items:
                    try:
                        r = extract_entities_and_relationships(
                            nid, did, h, c, ancestor_context=ac,
                        )
                        for entity in r.entities:
                            kg_.add_entity(entity)
                        count += len(r.entities)
                        for rel in r.relationships:
                            kg_.add_relationship(rel)
                    except Exception:
                        pass
            return count

        if max_workers <= 1 or len(target_ids) <= 1:
            for doc_id in target_ids:
                _process_document(doc_id)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_process_document, did): did
                    for did in target_ids
                }
                _KG_DOC_TIMEOUT = 300  # 5 min per doc KG
                for future in as_completed(futures):
                    try:
                        future.result(timeout=_KG_DOC_TIMEOUT)
                    except TimeoutError:
                        logger.warning(
                            "workspace_kg_doc_timeout",
                            doc_id=futures[future],
                            timeout_s=_KG_DOC_TIMEOUT,
                        )
                    except Exception as exc:
                        logger.debug(
                            "workspace_kg_doc_error",
                            doc_id=futures[future],
                            error=str(exc),
                        )

        logger.info(
            "workspace_kg_built",
            documents=len(target_ids),
            stats=kg.get_stats(),
        )

        # Build document profiles from KG entities (no extra LLM calls)
        self._build_document_profiles(kg, target_ids)

        return kg

    def _build_document_profiles(
        self,
        kg: "KnowledgeGraph",
        doc_ids: list[str],
    ) -> None:
        """Extract structured profiles from KG entities for each document."""
        from rnsr.indexing.document_profile import extract_profile

        for doc_id in doc_ids:
            info = self._catalog.get(doc_id)
            if info is None:
                continue

            # Get root and tail content for regex fallback
            index_result = self.get_document(doc_id)
            root_content: str | None = None
            tail_content: str | None = None
            page_count: int | None = None
            if index_result:
                skeleton, kv_store = index_result[:2]
                for node in skeleton.values():
                    if node.level == 0:
                        root_content = kv_store.get(node.node_id)
                        if root_content:
                            root_content = root_content[:2000]
                        for cid in node.child_ids[:2]:
                            c = kv_store.get(cid)
                            if c:
                                root_content = (root_content or "") + "\n" + c[:1000]
                        page_count_meta = node.metadata.get("total_pages")
                        if page_count_meta:
                            page_count = int(page_count_meta)
                        break
                # Get tail content from the last node
                all_nodes = list(skeleton.values())
                if len(all_nodes) > 1:
                    last_node = all_nodes[-1]
                    tail_content = kv_store.get(last_node.node_id)
                    if tail_content:
                        tail_content = tail_content[-1000:]

            try:
                profile = extract_profile(
                    kg=kg,
                    doc_id=doc_id,
                    title=info.title,
                    root_content=root_content,
                    tail_content=tail_content,
                    page_count=page_count,
                )
                # Store profile in catalog metadata
                info.metadata["profile"] = profile.model_dump(exclude_none=True)
                self._db.update_catalog_metadata(doc_id, info.metadata)
            except Exception as exc:
                logger.debug(
                    "profile_extraction_failed",
                    doc_id=doc_id,
                    error=str(exc),
                )

    def get_document_profiles(self) -> dict[str, dict]:
        """Return all document profiles keyed by doc_id."""
        profiles: dict[str, dict] = {}
        for doc_id, info in self._catalog.items():
            profile = info.metadata.get("profile")
            if profile:
                profiles[doc_id] = profile
        return profiles

    def link_entities_across_documents(
        self,
        doc_ids: list[str] | None = None,
    ) -> list:
        """
        Link entities in *doc_ids* against the full KG index.

        Uses an index-lookup strategy — O(n * k) where k is the average
        number of KG matches per entity — instead of comparing every
        document pair which is O(n^2) and cannot scale.

        Each document's entities are looked up in the KG; the
        ``EntityLinker`` handles exact, fuzzy, and alias matching via
        the KG's own search index.

        Args:
            doc_ids: Document IDs whose entities should be linked
                     (default: all documents in the store).

        Returns:
            List of ``EntityLink`` objects created.
        """
        from rnsr.extraction.entity_linker import EntityLinker

        kg = self.get_workspace_kg()
        linker = EntityLinker(kg)
        target_ids = doc_ids or list(self._catalog.keys())
        all_links: list = []

        for doc_id in target_ids:
            try:
                entities = kg.find_entities_in_document(doc_id)
                if not entities:
                    continue
                links = linker.link_entities(entities)
                all_links.extend(links)
            except Exception as exc:
                logger.debug(
                    "entity_link_error",
                    doc_id=doc_id,
                    error=str(exc),
                )

        logger.info(
            "entities_linked",
            documents=len(target_ids),
            links_created=len(all_links),
        )
        return all_links

    # Threshold: collections above this size use hierarchical navigation
    _HIERARCHICAL_THRESHOLD = 15

    def query_cross_document(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        use_short_answer: bool = False,
    ) -> dict[str, Any]:
        """
        Ask a question that spans multiple documents.

        For small / scoped collections (≤ ``_HIERARCHICAL_THRESHOLD``),
        uses the existing ``CrossDocNavigator`` which eagerly loads every
        document.

        For large / hierarchical collections, navigates the collection
        skeleton (folder → doc stub → section) with a single
        ``RLMNavigator``.  Documents are loaded lazily — only the ones
        the navigator decides to "enter" are expanded from disk.

        Args:
            question: The cross-document question.
            doc_ids: Limit to specific documents (default: all).
            use_short_answer: When True, instruct navigators to produce
                minimal answers (key phrases only).

        Returns:
            Result dictionary with ``answer``, ``documents_used``, etc.
        """
        target_ids = doc_ids or list(self._catalog.keys())

        # Use hierarchical navigation for large collections
        use_hierarchical = (
            len(target_ids) > self._HIERARCHICAL_THRESHOLD
            and doc_ids is None  # hierarchical only when querying all docs
            and self._has_collection_skeleton()
        )

        if use_hierarchical:
            return self._query_hierarchical(question, use_short_answer=use_short_answer)
        return self._query_cross_doc_eager(question, target_ids, use_short_answer=use_short_answer)

    def _has_collection_skeleton(self) -> bool:
        """Check whether a collection skeleton exists (in memory or on disk)."""
        if self._collection_skeleton:
            return True
        loaded = self._load_collection_skeleton()
        return loaded is not None

    def _query_hierarchical(
        self, question: str, *, use_short_answer: bool = False,
    ) -> dict[str, Any]:
        """Navigate the collection skeleton with a single RLMNavigator."""
        from rnsr.agent.rlm_navigator import RLMNavigator, RLMConfig
        from rnsr.client import _get_cached_llm_fn

        if self._collection_skeleton is None:
            self._load_collection_skeleton()
        assert self._collection_skeleton is not None

        lazy_kv = LazyKVStore(self.store_path)

        # Populate overlay with collection-level summaries
        for nid, node in self._collection_skeleton.items():
            lazy_kv.put(nid, node.summary)

        skel = ExpandableSkeleton(
            self._collection_skeleton,
            lazy_kv,
            self.store_path,
            max_loaded=5,
            store_db=self._db,
        )

        kg = self.get_workspace_kg()

        config = RLMConfig(
            max_iterations=50,
            max_recursion_depth=5,
            enable_pre_filtering=True,
            enable_verification=False,
        )
        navigator = RLMNavigator(
            skeleton=skel,  # type: ignore[arg-type]
            kv_store=lazy_kv,
            config=config,
            knowledge_graph=kg,
        )
        navigator.set_llm_function(_get_cached_llm_fn())

        nav_metadata = {"use_short_answer": True} if use_short_answer else None
        nav_result = navigator.navigate(question, metadata=nav_metadata)

        docs_used = list(skel._expanded_docs.keys())

        return {
            "answer": nav_result.get("answer", "No answer found."),
            "documents_used": docs_used,
            "entities_involved": [],
            "total_nodes_visited": nav_result.get("visited_nodes", 0),
            "total_iterations": nav_result.get("iteration", 0),
            "confidence": nav_result.get("confidence", 0.0),
        }

    def _query_cross_doc_eager(
        self,
        question: str,
        target_ids: list[str],
        *,
        use_short_answer: bool = False,
    ) -> dict[str, Any]:
        """Original eager-loading cross-document query (for small collections)."""
        from rnsr.agent.cross_doc_navigator import (
            create_cross_doc_navigator,
        )
        from rnsr.agent.rlm_navigator import RLMNavigator, RLMConfig
        from rnsr.agent.kg_resolver import KGResolver
        from rnsr.client import _get_cached_llm_fn

        kg = self.get_workspace_kg()

        # KG-first resolution: try answering from profiles and entities
        # before spinning up expensive per-document navigation.
        profiles = self.get_document_profiles()
        resolver = KGResolver(kg, profiles)
        resolution = resolver.try_resolve(question, doc_ids=target_ids)
        if resolution.resolved and resolution.answer:
            return {
                "answer": resolution.answer,
                "documents_used": target_ids,
                "entities_involved": [],
                "total_nodes_visited": 0,
                "total_iterations": 0,
                "confidence": resolution.confidence,
                "kg_resolved": True,
            }

        cross_nav = create_cross_doc_navigator(kg)
        cross_nav._kg_resolver = resolver

        loaded = 0
        for doc_id in target_ids:
            index_result = self.get_document(doc_id)
            if index_result is None:
                logger.warning("cross_doc_skip_document", doc_id=doc_id)
                continue
            skeleton, kv_store = index_result[:2]
            tables = index_result[2] if len(index_result) > 2 else None

            config = RLMConfig()
            if use_short_answer:
                config.use_short_answer = True

            doc_title = doc_id
            info = self._catalog.get(doc_id)
            if info and info.title:
                doc_title = info.title

            # Attach document profile for identity anchoring
            doc_profile = profiles.get(doc_id) if profiles else None

            navigator = RLMNavigator(
                skeleton=skeleton,
                kv_store=kv_store,
                knowledge_graph=kg,
                config=config,
                tables=tables,
                doc_profile=doc_profile,
                doc_title=doc_title,
            )
            navigator.set_llm_function(_get_cached_llm_fn())

            cross_nav.register_document(
                doc_id, skeleton, kv_store,
                navigator=navigator, title=doc_title,
            )
            loaded += 1

        if loaded == 0:
            return {
                "answer": "No documents could be loaded for querying.",
                "documents_used": [],
                "entities_involved": [],
                "total_nodes_visited": 0,
                "total_iterations": 0,
                "confidence": 0.0,
            }

        result = cross_nav.query(question)
        return {
            "answer": result.answer if hasattr(result, "answer") else str(result),
            "documents_used": [
                r.doc_id for r in (result.document_results if hasattr(result, "document_results") else [])
            ],
            "entities_involved": [
                e.canonical_name for e in (result.entities_involved if hasattr(result, "entities_involved") else [])
            ],
            "total_nodes_visited": getattr(result, "total_nodes_visited", 0),
            "total_iterations": getattr(result, "total_iterations", 0),
            "confidence": getattr(result, "confidence", 0.0),
        }

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __len__(self) -> int:
        """Number of documents in the store."""
        return len(self._catalog)
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document exists."""
        return doc_id in self._catalog
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over document IDs."""
        return iter(self._catalog.keys())
