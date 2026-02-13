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
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import structlog

from rnsr.exceptions import IndexingError
from rnsr.indexing.kv_store import KVStore, SQLiteKVStore
from rnsr.indexing.persistence import (
    save_index,
    load_index,
    get_index_info,
    delete_index,
)
from rnsr.indexing.skeleton_index import build_skeleton_index
from rnsr.ingestion import ingest_document
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


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
    
    def __init__(self, store_path: str | Path):
        """
        Initialize or open a document store.
        
        Args:
            store_path: Directory for storing document indexes
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self._catalog_path = self.store_path / "catalog.json"
        self._catalog: dict[str, DocumentInfo] = {}
        
        # Load existing catalog if present
        if self._catalog_path.exists():
            self._load_catalog()
        
        logger.info(
            "document_store_initialized",
            path=str(self.store_path),
            documents=len(self._catalog),
        )
    
    def _load_catalog(self) -> None:
        """Load the document catalog from disk."""
        with open(self._catalog_path) as f:
            data = json.load(f)
        
        self._catalog = {
            doc_id: DocumentInfo(**info)
            for doc_id, info in data.get("documents", {}).items()
        }
    
    def _save_catalog(self) -> None:
        """Save the document catalog to disk."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "documents": {
                doc_id: info.to_dict()
                for doc_id, info in self._catalog.items()
            }
        }
        
        with open(self._catalog_path, "w") as f:
            json.dump(data, f, indent=2)
    
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
        if doc_id in self._catalog:
            logger.warning("document_already_exists", doc_id=doc_id)
            return doc_id
        
        # Ingest document
        logger.info("ingesting_document", source=str(source_path))
        result = ingest_document(str(source_path))
        
        # Build skeleton index
        skeleton, kv_store = build_skeleton_index(result.tree)
        
        # Save to store
        index_path = self.store_path / doc_id
        save_index(skeleton, kv_store, index_path)
        
        # Update catalog
        info = DocumentInfo(
            id=doc_id,
            title=title or source_path.stem,
            source_path=str(source_path),
            node_count=len(skeleton),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._catalog[doc_id] = info
        self._save_catalog()
        
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
        
        # Save to store
        index_path = self.store_path / doc_id
        save_index(skeleton, kv_store, index_path)
        
        # Update catalog
        info = DocumentInfo(
            id=doc_id,
            title=title or doc_id,
            source_path=None,
            node_count=len(skeleton),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._catalog[doc_id] = info
        self._save_catalog()
        
        logger.info(
            "document_added_from_text",
            doc_id=doc_id,
            title=info.title,
            nodes=info.node_count,
        )
        
        return doc_id
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if doc_id not in self._catalog:
            return False
        
        # Delete index files
        index_path = self.store_path / doc_id
        delete_index(index_path)
        
        # Remove from catalog
        del self._catalog[doc_id]
        self._save_catalog()
        
        logger.info("document_removed", doc_id=doc_id)
        return True
    
    def clear_all(self) -> int:
        """
        Remove all documents and reset the catalog.
        
        Returns:
            Number of documents removed
        """
        doc_ids = list(self._catalog.keys())
        for doc_id in doc_ids:
            index_path = self.store_path / doc_id
            delete_index(index_path)
        
        # Also remove the workspace KG if it exists
        kg_path = self.store_path / "workspace_kg.db"
        if kg_path.exists():
            kg_path.unlink()
        
        count = len(self._catalog)
        self._catalog.clear()
        self._save_catalog()
        
        logger.info("store_cleared", documents_removed=count)
        return count
    
    def get_document(
        self,
        doc_id: str,
    ) -> tuple[dict[str, SkeletonNode], KVStore] | None:
        """
        Load a document's index.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (skeleton, kv_store) or None if not found
        """
        if doc_id not in self._catalog:
            return None
        
        index_path = self.store_path / doc_id
        return load_index(index_path)
    
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
        
        skeleton, kv_store = index_result
        nav_result = run_navigator(question, skeleton, kv_store)
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
        
        This KG persists in ``<store_path>/workspace_kg.db`` and accumulates
        entities from all documents added to the store. Use it together with
        :meth:`link_entities_across_documents` and :meth:`query_cross_document`
        to enable cross-document reasoning.
        
        Returns:
            A file-backed ``KnowledgeGraph`` shared across all documents.
            
        Example:
            kg = store.get_workspace_kg()
            print(kg.get_stats())
        """
        from rnsr.indexing.knowledge_graph import KnowledgeGraph
        
        kg_path = self.store_path / "workspace_kg.db"
        return KnowledgeGraph(str(kg_path))

    def build_workspace_kg(
        self,
        doc_ids: list[str] | None = None,
        max_workers: int = 8,
    ) -> "KnowledgeGraph":
        """
        Build (or rebuild) the workspace KG from indexed documents.
        
        Extracts entities and relationships from each document and merges
        them into the workspace KG. Then runs entity linking across all
        document pairs to discover shared entities.
        
        Args:
            doc_ids: Specific document IDs to process (default: all).
            max_workers: Parallel extraction threads per document.
            
        Returns:
            The populated workspace ``KnowledgeGraph``.
        """
        from rnsr.indexing.knowledge_graph import KnowledgeGraph
        from rnsr.extraction import extract_entities_and_relationships

        kg = self.get_workspace_kg()
        target_ids = doc_ids or list(self._catalog.keys())

        for doc_id in target_ids:
            index_result = self.get_document(doc_id)
            if index_result is None:
                logger.warning("doc_not_found_for_kg", doc_id=doc_id)
                continue

            skeleton, kv_store = index_result[:2]

            # Extract entities from each node
            for node_id, node in skeleton.items():
                content = kv_store.get(node_id) or ""
                if len(content.strip()) < 50:
                    continue

                try:
                    result = extract_entities_and_relationships(
                        node_id=node_id,
                        doc_id=doc_id,
                        header=node.header,
                        content=content,
                    )
                    for entity in result.entities:
                        kg.add_entity(entity)
                    for rel in result.relationships:
                        kg.add_relationship(rel)
                except Exception as exc:
                    logger.debug(
                        "workspace_kg_node_error",
                        doc_id=doc_id,
                        node_id=node_id,
                        error=str(exc),
                    )

        logger.info(
            "workspace_kg_built",
            documents=len(target_ids),
            stats=kg.get_stats(),
        )
        return kg

    def link_entities_across_documents(
        self,
        doc_ids: list[str] | None = None,
    ) -> list:
        """
        Run entity linking across all document pairs in the workspace KG.
        
        Discovers that e.g. "GeoV William Sorenssen" in Doc A is the same
        entity as "G. Sorenssen" in Doc B, and stores the link in the KG.
        
        Args:
            doc_ids: Specific document IDs to link (default: all).
            
        Returns:
            List of ``EntityLink`` objects created.
        """
        from rnsr.extraction.entity_linker import EntityLinker

        kg = self.get_workspace_kg()
        linker = EntityLinker(kg)
        target_ids = doc_ids or list(self._catalog.keys())
        all_links = []

        for i, d1 in enumerate(target_ids):
            for d2 in target_ids[i + 1:]:
                try:
                    links = linker.link_across_documents(d1, d2)
                    all_links.extend(links)
                except Exception as exc:
                    logger.debug(
                        "entity_link_error",
                        doc_1=d1,
                        doc_2=d2,
                        error=str(exc),
                    )

        logger.info(
            "entities_linked",
            document_pairs=len(target_ids) * (len(target_ids) - 1) // 2,
            links_created=len(all_links),
        )
        return all_links

    def query_cross_document(
        self,
        question: str,
        doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Ask a question that spans multiple documents.
        
        Uses the ``CrossDocNavigator`` to decompose the question, resolve
        entities to documents, navigate each document, and synthesize a
        combined answer.
        
        Args:
            question: The cross-document question.
            doc_ids: Limit to specific documents (default: all).
            
        Returns:
            Result dictionary with ``answer``, ``documents_used``, etc.
        """
        from rnsr.agent.cross_doc_navigator import (
            create_cross_doc_navigator,
        )
        from rnsr.agent.rlm_navigator import RLMNavigator, RLMConfig
        from rnsr.client import _get_cached_llm_fn

        kg = self.get_workspace_kg()
        cross_nav = create_cross_doc_navigator(kg)
        target_ids = doc_ids or list(self._catalog.keys())

        for doc_id in target_ids:
            index_result = self.get_document(doc_id)
            if index_result is None:
                continue
            skeleton, kv_store = index_result[:2]

            navigator = RLMNavigator(
                skeleton=skeleton,
                kv_store=kv_store,
                knowledge_graph=kg,
                config=RLMConfig(),
            )
            navigator.set_llm_function(_get_cached_llm_fn())
            cross_nav.register_document(doc_id, skeleton, kv_store, navigator=navigator)

        result = cross_nav.query(question)
        return {
            "answer": result.answer if hasattr(result, "answer") else str(result),
            "documents_used": [
                r.doc_id for r in (result.document_results if hasattr(result, "document_results") else [])
            ],
            "entities_involved": [
                e.canonical_name for e in (result.entities_involved if hasattr(result, "entities_involved") else [])
            ],
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
