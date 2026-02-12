"""
RNSR Client - Simple High-Level API

Provides the simplest possible interface for using RNSR.
Handles all the complexity of ingestion, indexing, and navigation.

This is the state-of-the-art hybrid retrieval system combining:
- PageIndex: Vectorless, reasoning-based tree search
- RLMs: REPL environment with recursive sub-LLM calls
- Vision: OCR-free image-based document analysis

Usage:
    from rnsr import RNSRClient
    
    # One-line document Q&A
    client = RNSRClient()
    answer = client.ask("contract.pdf", "What are the payment terms?")
    
    # With caching (faster for repeated queries)
    client = RNSRClient(cache_dir="./cache")
    answer = client.ask("contract.pdf", "What are the terms?")
    answer2 = client.ask("contract.pdf", "Who are the parties?")  # Uses cache
    
    # Advanced: RLM Navigator with full features
    result = client.ask_advanced(
        "complex_report.pdf",
        "Compare revenue in Q1 vs Q2",
        use_rlm=True,
        enable_verification=True,
    )
    
    # Vision-based analysis (for scanned docs, charts)
    result = client.ask_vision(
        "scanned_document.pdf",
        "What is shown in the chart?",
    )
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable

import structlog

from rnsr.agent import run_navigator
from rnsr.document_store import DocumentStore
from rnsr.indexing import build_skeleton_index, save_index, load_index
from rnsr.indexing.kv_store import InMemoryKVStore, KVStore
from rnsr.indexing.knowledge_graph import KnowledgeGraph
from rnsr.ingestion import ingest_document
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)

# Cached LLM instances: key (provider, model) or (None, None) for auto-detect
_cached_llm: Any = None
_cached_llm_key: tuple[str | None, str | None] = (None, None)
_cached_llm_fns: dict[tuple[str | None, str | None], Callable[[str], str]] = {}


def _get_cached_llm(provider: str | None = None, model: str | None = None) -> Any:
    """Get a cached LLM instance. When provider/model are set, use them; else auto-detect."""
    global _cached_llm, _cached_llm_key
    key = (provider, model)
    if _cached_llm is not None and _cached_llm_key == key:
        return _cached_llm
    from rnsr.llm import LLMProvider, get_llm
    kwargs: dict[str, Any] = {}
    if provider is not None:
        kwargs["provider"] = LLMProvider(provider) if isinstance(provider, str) else provider
    if model is not None:
        kwargs["model"] = model
    _cached_llm = get_llm(**kwargs)
    _cached_llm_key = key
    return _cached_llm


def _get_cached_llm_fn(provider: str | None = None, model: str | None = None) -> Callable[[str], str]:
    """Get a cached LLM complete function for navigator use. Respects provider/model when set."""
    key = (provider, model)
    if key not in _cached_llm_fns:
        llm = _get_cached_llm(provider, model)
        _cached_llm_fns[key] = lambda p, _llm=llm: str(_llm.complete(p))
    return _cached_llm_fns[key]


class RNSRClient:
    """
    High-level client for RNSR document Q&A.
    
    This is the simplest way to use RNSR. It handles:
    - Document ingestion
    - Skeleton index building
    - Optional caching/persistence
    - Navigation and answer generation
    
    Example:
        # Basic usage (no caching)
        client = RNSRClient()
        answer = client.ask("document.pdf", "What is the main topic?")
        
        # With caching (recommended for production)
        client = RNSRClient(cache_dir="./rnsr_cache")
        answer = client.ask("document.pdf", "What is the main topic?")
        
        # From raw text
        answer = client.ask_text(
            "This is my document content...",
            "What is this about?"
        )
    """
    
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        """
        Initialize the RNSR client.
        
        Args:
            cache_dir: Optional directory for caching indexes.
                       If provided, indexes are persisted and reused.
            llm_provider: LLM provider ("openai", "anthropic", "gemini", "ollama")
            llm_model: LLM model name
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
        # In-memory cache for session
        self._session_cache: dict[str, tuple[dict[str, SkeletonNode], KVStore]] = {}
        
        # Knowledge graph cache (keyed by document cache_key)
        self._kg_cache: dict[str, KnowledgeGraph] = {}
        
        # Cached navigator instances for reuse across queries
        self._navigator_cache: dict[str, Any] = {}
        
        # Tables cache (keyed by document cache_key)
        self._tables_cache: dict[str, list] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "rnsr_client_initialized",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
    
    def ask(
        self,
        document: str | Path,
        question: str,
        use_knowledge_graph: bool = True,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """
        Ask a question about a PDF document.
        
        By default, a knowledge graph is built (or reused from cache) to give
        the navigator entity awareness, which significantly improves accuracy.
        Set ``use_knowledge_graph=False`` to skip this step for faster but
        less accurate results.
        
        Args:
            document: Path to PDF file
            question: Question to ask
            use_knowledge_graph: Build and use knowledge graph with entity
                                 extraction (default True).
            force_reindex: If True, re-process even if cached
            
        Returns:
            Answer string from the navigator
            
        Example:
            answer = client.ask("contract.pdf", "What are the payment terms?")
        """
        # Delegate to ask_advanced which supports knowledge graph via RLMNavigator
        result = self.ask_advanced(
            document=document,
            question=question,
            use_knowledge_graph=use_knowledge_graph,
            force_reindex=force_reindex,
        )
        return result.get("answer", "No answer found.")
    
    def ask_text(
        self,
        text: str | list[str],
        question: str,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Ask a question about raw text.
        
        Args:
            text: Text content or list of text chunks
            question: Question to ask
            cache_key: Optional key for caching (if cache_dir is set)
            
        Returns:
            Answer string
            
        Example:
            answer = client.ask_text(
                "The company was founded in 2020...",
                "When was the company founded?"
            )
        """
        from rnsr.ingestion import build_tree_from_text
        
        # Generate cache key from content if not provided
        if cache_key is None:
            content = text if isinstance(text, str) else "\n".join(text)
            cache_key = f"text_{hashlib.md5(content[:1000].encode()).hexdigest()[:12]}"
        
        # Check caches
        if cache_key in self._session_cache:
            skeleton, kv_store = self._session_cache[cache_key]
        elif self.cache_dir and (self.cache_dir / cache_key).exists():
            skeleton, kv_store, tables = load_index(self.cache_dir / cache_key)
            self._session_cache[cache_key] = (skeleton, kv_store)
            if tables:
                self._tables_cache[cache_key] = tables
        else:
            # Build index
            tree = build_tree_from_text(text)
            skeleton, kv_store = build_skeleton_index(tree)
            self._session_cache[cache_key] = (skeleton, kv_store)
            
            # Persist if cache_dir is set
            if self.cache_dir:
                save_index(skeleton, kv_store, self.cache_dir / cache_key)
        
        result = run_navigator(question, skeleton, kv_store)
        return result.get("answer", "No answer found.")
    
    def ask_multiple(
        self,
        document: str | Path,
        questions: list[str],
        force_reindex: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ask multiple questions about a document.
        
        More efficient than calling ask() multiple times because
        the document is only indexed once.
        
        Args:
            document: Path to PDF file
            questions: List of questions
            force_reindex: If True, re-process even if cached
            
        Returns:
            List of answers
            
        Example:
            answers = client.ask_multiple(
                "contract.pdf",
                ["What are the terms?", "Who are the parties?"]
            )
        """
        doc_path = Path(document)
        skeleton, kv_store = self._get_or_create_index(doc_path, force_reindex)
        
        return [
            run_navigator(q, skeleton, kv_store).get("answer", "No answer found.")
            for q in questions
        ]
    
    def get_document_info(self, document: str | Path) -> dict[str, Any]:
        """
        Get information about a document without querying it.
        
        Args:
            document: Path to PDF file
            
        Returns:
            Dictionary with document metadata
        """
        doc_path = Path(document)
        cache_key = self._get_cache_key(doc_path)
        
        # Check if cached
        if cache_key in self._session_cache:
            skeleton, _ = self._session_cache[cache_key]
            return {
                "path": str(doc_path),
                "cached": True,
                "nodes": len(skeleton),
            }
        
        if self.cache_dir and (self.cache_dir / cache_key).exists():
            from rnsr.indexing import get_index_info
            info = get_index_info(self.cache_dir / cache_key)
            info["path"] = str(doc_path)
            info["cached"] = True
            return info
        
        return {
            "path": str(doc_path),
            "cached": False,
            "exists": doc_path.exists(),
        }
    
    def clear_cache(self) -> int:
        """
        Clear all cached indexes.
        
        Returns:
            Number of caches cleared
        """
        count = len(self._session_cache)
        self._session_cache.clear()
        
        if self.cache_dir:
            import shutil
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    count += 1
        
        logger.info("cache_cleared", count=count)
        return count
    
    def _get_cache_key(self, doc_path: Path) -> str:
        """Generate a cache key for a document."""
        stat = doc_path.stat()
        hash_input = f"{doc_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _get_or_create_index(
        self,
        doc_path: Path,
        force_reindex: bool,
    ) -> tuple[dict[str, SkeletonNode], KVStore]:
        """Get cached index or create new one."""
        cache_key = self._get_cache_key(doc_path)
        
        # Check session cache first
        if not force_reindex and cache_key in self._session_cache:
            logger.debug("using_session_cache", key=cache_key)
            return self._session_cache[cache_key]
        
        # Check persistent cache
        if not force_reindex and self.cache_dir:
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                try:
                    logger.debug("using_persistent_cache", key=cache_key)
                    skeleton, kv_store, tables = load_index(cache_path)
                    self._session_cache[cache_key] = (skeleton, kv_store)
                    if tables:
                        self._tables_cache[cache_key] = tables
                    return skeleton, kv_store
                except Exception as exc:
                    logger.warning(
                        "cache_load_failed",
                        key=cache_key,
                        error=str(exc),
                    )
                    # Fall through to re-index from scratch
        
        # Create new index
        logger.info("creating_new_index", path=str(doc_path))
        result = ingest_document(str(doc_path))
        skeleton, kv_store = build_skeleton_index(result.tree)
        
        # Store in session cache
        self._session_cache[cache_key] = (skeleton, kv_store)
        
        # Store detected tables in cache
        if result.tables:
            self._tables_cache[cache_key] = result.tables
            logger.info(
                "tables_cached",
                cache_key=cache_key,
                table_count=len(result.tables),
            )
        
        # Persist if cache_dir is set (including tables)
        if self.cache_dir:
            save_index(
                skeleton, 
                kv_store, 
                self.cache_dir / cache_key,
                tables=result.tables,
            )
        
        return skeleton, kv_store
    
    # =========================================================================
    # Knowledge Graph Support
    # =========================================================================

    def _get_or_create_knowledge_graph(
        self,
        cache_key: str,
        skeleton: dict[str, SkeletonNode],
        kv_store: KVStore,
        doc_id: str = "document",
        max_workers: int | None = None,
    ) -> KnowledgeGraph:
        """
        Get cached knowledge graph or create a new one with extracted entities.
        
        Uses the RLMUnifiedExtractor to extract entities and relationships
        from each skeleton node **in parallel**.  The extractor is LLM-driven
        and adaptive: it discovers entity/relationship types from the document
        content and persists learned types to ``~/.rnsr/`` for future runs.
        
        Args:
            cache_key: Cache key for the document.
            skeleton: The skeleton index.
            kv_store: The key-value store with content.
            doc_id: Document identifier.
            max_workers: Max parallel extraction threads.  Defaults to
                         ``min(8, node_count)`` which is a good balance
                         between speed and LLM API rate-limits.
            
        Returns:
            KnowledgeGraph with extracted entities and relationships.
        """
        # Check cache first
        if cache_key in self._kg_cache:
            logger.debug("using_cached_knowledge_graph", key=cache_key)
            return self._kg_cache[cache_key]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from rnsr.extraction import extract_entities_and_relationships
        from rnsr.extraction import RLMUnifiedResult
        from rnsr.extraction.rlm_unified_extractor import (
            extract_entities_and_relationships_batch,
        )

        # Create new knowledge graph (in-memory for now)
        kg = KnowledgeGraph(":memory:")

        # -----------------------------------------------------------------
        # Build ancestor breadcrumb + subject context for each node.
        #
        # For a node like:
        #   root > Form 80 Data > PRIMARY APPLICANT DETAILS > Identity Documents
        # we build:
        #   breadcrumb = "Form 80 Data > PRIMARY APPLICANT DETAILS > Identity Documents"
        #   subject_hint = content from the first leaf sibling that contains a person's name
        #
        # This lets the LLM know that "Identity Documents" belongs to the
        # primary applicant (GeoV William Sorenssen), not to a nameless entity.
        # -----------------------------------------------------------------
        def _find_personal_info_content(container_id: str) -> str:
            """Drill into a container section to find Personal Information.
            
            Given a section like "PRIMARY APPLICANT DETAILS" that has children
            (Personal Information, Contact Details, Identity Documents), find
            the child that contains the person's name / identity details.
            """
            container = skeleton.get(container_id)
            if not container or not container.child_ids:
                # Leaf node — return its own content
                return (kv_store.get(container_id) or "")[:400]

            # Look for a child with a header containing "personal", "info",
            # or just take the first child with substantial content.
            for child_id in container.child_ids:
                child = skeleton.get(child_id)
                if not child:
                    continue
                h_lower = child.header.lower()
                if "personal" in h_lower or "info" in h_lower or "name" in h_lower:
                    content = kv_store.get(child_id) or ""
                    if len(content) > 50:
                        return content[:400]

            # Fallback: first child with enough content
            for child_id in container.child_ids:
                content = kv_store.get(child_id) or ""
                if len(content) > 50:
                    return content[:400]

            return ""

        def _build_ancestor_context(node_id: str) -> str:
            """Walk up the tree and build a breadcrumb + subject hint.
            
            The subject hint resolves generic labels like "PRIMARY APPLICANT"
            to actual names by finding the Personal Information section
            under the relevant applicant container.
            
            For a document structured as:
              Form 80 Data
                PRIMARY APPLICANT DETAILS
                  Personal Information  ← "GeoV William Sorenssen"
                  Contact Details
                  Identity Documents
                EDUCATION HISTORY      ← sibling, needs subject hint
                TRAVEL HISTORY         ← sibling, needs subject hint
            
            The subject hint for EDUCATION HISTORY finds the nearest
            preceding applicant section (PRIMARY APPLICANT DETAILS), drills
            into its "Personal Information" child, and extracts the name.
            """
            parts: list[str] = []
            current = node_id
            while current:
                node = skeleton.get(current)
                if not node:
                    break
                parts.append(node.header)
                current = node.parent_id
            parts.reverse()
            breadcrumb = " > ".join(parts)

            subject_hint = ""
            node = skeleton.get(node_id)
            if node and node.parent_id:
                parent = skeleton.get(node.parent_id)
                if parent and parent.child_ids:
                    # Strategy 1: Direct sibling lookup (works when this node
                    # is a child of e.g. "PRIMARY APPLICANT DETAILS")
                    first_child_id = parent.child_ids[0]
                    if first_child_id != node_id:
                        first_child = skeleton.get(first_child_id)
                        if first_child and first_child.child_ids:
                            # First sibling is a container — drill into it
                            subject_hint = _find_personal_info_content(first_child_id)
                        else:
                            content = kv_store.get(first_child_id) or ""
                            if len(content) > 50:
                                subject_hint = content[:400]

                    # Strategy 2: If this is a top-level section (e.g.
                    # EDUCATION HISTORY under Form 80 Data), find the
                    # PRIMARY APPLICANT section and drill into it.
                    # Also works for deeper nodes (e.g. "Current Employment"
                    # under "EMPLOYMENT HISTORY") — we walk up through
                    # grandparent siblings too.
                    #
                    # We prefer "primary applicant" over any other
                    # applicant section, because forms like Form 80
                    # list the secondary applicant for reference but all
                    # subsequent sections are about the primary.
                    if not subject_hint:
                        search_node = node
                        while not subject_hint and search_node and search_node.parent_id:
                            search_parent = skeleton.get(search_node.parent_id)
                            if not search_parent or not search_parent.child_ids:
                                break

                            # First pass: look for "primary applicant"
                            # Second pass: any applicant/personal/client
                            for keyword_set in [
                                ("primary",),
                                ("applicant", "personal", "client"),
                            ]:
                                if subject_hint:
                                    break
                                for sib_id in search_parent.child_ids:
                                    if sib_id == node_id:
                                        continue
                                    sib = skeleton.get(sib_id)
                                    if not sib:
                                        continue
                                    h_lower = sib.header.lower()
                                    if any(kw in h_lower for kw in keyword_set):
                                        subject_hint = _find_personal_info_content(sib_id)
                                        if subject_hint:
                                            break

                            # Move up one level
                            search_node = search_parent

            lines = [f"Document path: {breadcrumb}"]
            if subject_hint:
                lines.append(
                    f"Subject of this section (from Personal Information):\n{subject_hint}"
                )
            return "\n".join(lines)

        # Prepare work items: (node_id, header, content, ancestor_context)
        # Skip nodes with very short content — these are container/header
        # nodes (e.g. "PRIMARY APPLICANT DETAILS" with 25 chars) that have
        # no extractable entities.  The extractor already skips <50 chars,
        # but filtering here avoids scheduling overhead and LLM init.
        MIN_CONTENT_LENGTH = 50
        all_items = [
            (
                node_id,
                node.header,
                kv_store.get(node_id) or "",
                _build_ancestor_context(node_id),
            )
            for node_id, node in skeleton.items()
        ]
        work_items = [
            item for item in all_items
            if len(item[2].strip()) >= MIN_CONTENT_LENGTH
        ]
        skipped = len(all_items) - len(work_items)
        if skipped:
            logger.info(
                "skipped_short_nodes",
                skipped=skipped,
                threshold=MIN_CONTENT_LENGTH,
                total=len(all_items),
                remaining=len(work_items),
            )

        import os
        import json as _json
        if max_workers is None:
            default_workers = int(os.getenv("RNSR_EXTRACTION_WORKERS", "16"))
            max_workers = min(default_workers, len(work_items)) or 1

        # Per-node extraction cache — allows resumption of interrupted runs
        # and avoids re-extracting unchanged nodes.
        from rnsr.extraction.models import Entity, Relationship

        node_cache_dir: Path | None = None
        if self.cache_dir:
            node_cache_dir = self.cache_dir / cache_key / "nodes"
            node_cache_dir.mkdir(parents=True, exist_ok=True)

        def _load_cached_result(node_id: str) -> RLMUnifiedResult | None:
            """Try to load a cached extraction result for a node."""
            if node_cache_dir is None:
                return None
            cache_file = node_cache_dir / f"{node_id}.json"
            if not cache_file.exists():
                return None
            try:
                data = _json.loads(cache_file.read_text())
                entities = [Entity.model_validate(e) for e in data.get("entities", [])]
                relationships = [Relationship.model_validate(r) for r in data.get("relationships", [])]
                result = RLMUnifiedResult(
                    node_id=node_id,
                    doc_id=doc_id,
                    entities=entities,
                    relationships=relationships,
                    code_executed=True,
                )
                return result
            except Exception as exc:
                logger.debug("node_cache_load_failed", node_id=node_id, error=str(exc))
                return None

        def _save_cached_result(node_id: str, result: RLMUnifiedResult) -> None:
            """Save an extraction result to the per-node cache."""
            if node_cache_dir is None:
                return
            try:
                data = {
                    "entities": [e.model_dump(mode="json") for e in result.entities],
                    "relationships": [r.model_dump(mode="json") for r in result.relationships],
                }
                cache_file = node_cache_dir / f"{node_id}.json"
                cache_file.write_text(_json.dumps(data, default=str))
            except Exception as exc:
                logger.debug("node_cache_save_failed", node_id=node_id, error=str(exc))

        def _extract_node(item):
            """Run extraction for a single node (executed in worker thread)."""
            node_id, header, content, ancestor_context = item
            # Check per-node cache first
            cached = _load_cached_result(node_id)
            if cached is not None:
                logger.debug("node_extraction_cached", node_id=node_id)
                return cached
            result = extract_entities_and_relationships(
                node_id=node_id,
                doc_id=doc_id,
                header=header,
                content=content,
                ancestor_context=ancestor_context,
            )
            _save_cached_result(node_id, result)
            return result

        def _extract_batch(batch_items):
            """Extract a batch of small nodes in a single LLM call."""
            # Check cache for each item first
            cached_results = []
            uncached = []
            for item in batch_items:
                node_id = item[0]
                cached = _load_cached_result(node_id)
                if cached is not None:
                    logger.debug("node_extraction_cached", node_id=node_id)
                    cached_results.append(cached)
                else:
                    uncached.append(item)

            if not uncached:
                return cached_results

            # Build batch input: (node_id, doc_id, header, content, ancestor_context)
            batch_input = [
                (nid, doc_id, hdr, cnt, ac) for nid, hdr, cnt, ac in uncached
            ]
            results = extract_entities_and_relationships_batch(batch_input)

            # Cache each result
            for item, result in zip(uncached, results):
                _save_cached_result(item[0], result)

            return cached_results + results

        # ── Batching: group small nodes (< 500 chars) into batches of up to 5 ──
        BATCH_CONTENT_THRESHOLD = 500
        BATCH_SIZE = 5

        small_items = [
            item for item in work_items
            if len(item[2].strip()) < BATCH_CONTENT_THRESHOLD
        ]
        large_items = [
            item for item in work_items
            if len(item[2].strip()) >= BATCH_CONTENT_THRESHOLD
        ]

        # Create batches from small items
        batches = [
            small_items[i:i + BATCH_SIZE]
            for i in range(0, len(small_items), BATCH_SIZE)
        ]

        if batches:
            logger.info(
                "batching_small_nodes",
                small_nodes=len(small_items),
                large_nodes=len(large_items),
                batches=len(batches),
            )

        entity_count = 0
        relationship_count = 0
        nodes_done = 0
        total_nodes = len(work_items)

        logger.info(
            "knowledge_graph_extraction_started",
            cache_key=cache_key,
            node_count=total_nodes,
            max_workers=max_workers,
        )

        def _add_result_to_kg(result: RLMUnifiedResult):
            """Add extraction result to the knowledge graph."""
            nonlocal entity_count, relationship_count, nodes_done
            nodes_done += 1
            node_id = result.node_id
            node_entities_added = 0
            node_rels_added = 0

            for entity in result.entities:
                try:
                    kg.add_entity(entity)
                    entity_count += 1
                    node_entities_added += 1
                except Exception as ent_err:
                    logger.warning(
                        "add_entity_failed",
                        entity_id=getattr(entity, "id", "?"),
                        entity_name=getattr(entity, "canonical_name", "?"),
                        error=str(ent_err),
                    )

            for relationship in result.relationships:
                try:
                    kg.add_relationship(relationship)
                    relationship_count += 1
                    node_rels_added += 1
                except Exception as rel_err:
                    logger.warning(
                        "add_relationship_failed",
                        rel_id=getattr(relationship, "id", "?"),
                        error=str(rel_err),
                    )

            entity_names = [e.canonical_name for e in result.entities]
            skel_node = skeleton.get(node_id)
            header_text = skel_node.header if skel_node else node_id
            logger.info(
                "node_extraction_complete",
                progress=f"{nodes_done}/{total_nodes}",
                node_id=node_id,
                header=header_text[:60],
                entities_added=node_entities_added,
                relationships_added=node_rels_added,
                entity_names=entity_names,
                running_total_entities=entity_count,
                running_total_relationships=relationship_count,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Submit large nodes individually
            future_to_meta: dict = {}
            for item in large_items:
                fut = pool.submit(_extract_node, item)
                future_to_meta[fut] = ("single", item[0])

            # Submit batches of small nodes
            for batch in batches:
                fut = pool.submit(_extract_batch, batch)
                batch_ids = [item[0] for item in batch]
                future_to_meta[fut] = ("batch", batch_ids)

            # Collect all results as they complete
            for future in as_completed(future_to_meta):
                kind, meta = future_to_meta[future]
                try:
                    if kind == "single":
                        result = future.result()
                        _add_result_to_kg(result)
                    else:
                        results = future.result()
                        for result in results:
                            _add_result_to_kg(result)
                except Exception as exc:
                    if kind == "single":
                        nodes_done += 1
                    logger.warning(
                        "extraction_failed",
                        kind=kind,
                        meta=meta,
                        progress=f"{nodes_done}/{total_nodes}",
                        error=str(exc),
                    )

        logger.info(
            "knowledge_graph_created",
            cache_key=cache_key,
            entity_count=entity_count,
            relationship_count=relationship_count,
            node_count=total_nodes,
        )

        # Cache the knowledge graph
        self._kg_cache[cache_key] = kg

        return kg
    
    # =========================================================================
    # Advanced Navigation Methods
    # =========================================================================
    
    def ask_advanced(
        self,
        document: str | Path,
        question: str,
        use_rlm: bool = True,
        use_knowledge_graph: bool = True,
        enable_pre_filtering: bool = True,
        enable_verification: bool = False,
        max_recursion_depth: int = 3,
        force_reindex: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Advanced Q&A using the full RLM Navigator with Knowledge Graph.
        
        This replicates the benchmark's zero-hallucination performance by using:
        - Knowledge Graph with entity extraction (companies, people, dates, etc.)
        - Direct RLMNavigator class with cached LLM for consistency
        - Pre-filtering with keyword extraction before LLM calls
        - Deep recursive sub-LLM calls (configurable depth)
        
        Args:
            document: Path to PDF file.
            question: Question to ask.
            use_rlm: Use RLM Navigator (True) or standard navigator (False).
            use_knowledge_graph: Build and use knowledge graph with entity 
                                 extraction. This significantly improves accuracy
                                 by giving the navigator entity awareness.
            enable_pre_filtering: Use keyword filtering before ToT evaluation.
            enable_verification: Verify answers with strict critic loop.
                                Note: Set to False by default as this can be
                                overly strict. Enable for maximum accuracy.
            max_recursion_depth: Max depth for recursive sub-LLM calls.
            force_reindex: Re-process even if cached.
            metadata: Optional metadata (e.g., multiple choice options).
            
        Returns:
            Full result dictionary with answer, confidence, trace.
            
        Example:
            # Basic usage (matches benchmark performance)
            result = client.ask_advanced(
                "contract.pdf",
                "What are the payment terms?",
            )
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            
            # With strict verification (for maximum accuracy)
            result = client.ask_advanced(
                "legal_document.pdf",
                "What is the liability cap?",
                enable_verification=True,
            )
        """
        doc_path = Path(document)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        # Get cache key for this document
        cache_key = self._get_cache_key(doc_path)
        
        # Get or create index
        skeleton, kv_store = self._get_or_create_index(doc_path, force_reindex)
        
        if use_rlm:
            from rnsr.agent.rlm_navigator import RLMNavigator, RLMConfig
            
            # Build knowledge graph if enabled (key to benchmark performance)
            knowledge_graph = None
            if use_knowledge_graph:
                knowledge_graph = self._get_or_create_knowledge_graph(
                    cache_key=cache_key,
                    skeleton=skeleton,
                    kv_store=kv_store,
                    doc_id=doc_path.name,
                )
            
            # Create navigator key for caching
            nav_key = f"{cache_key}_rlm_{use_knowledge_graph}_{enable_verification}"
            
            # Get or create navigator (reuse for multiple queries on same doc)
            if nav_key not in self._navigator_cache or force_reindex:
                config = RLMConfig(
                    max_recursion_depth=max_recursion_depth,
                    enable_pre_filtering=enable_pre_filtering,
                    enable_verification=enable_verification,
                )
                
                # Get detected tables for SQL-like queries during navigation
                tables = self._tables_cache.get(cache_key, [])
                
                navigator = RLMNavigator(
                    skeleton=skeleton,
                    kv_store=kv_store,
                    knowledge_graph=knowledge_graph,
                    config=config,
                    tables=tables,
                )
                
                # Use cached LLM function; respect client llm_provider/llm_model when set
                navigator.set_llm_function(_get_cached_llm_fn(self.llm_provider, self.llm_model))
                
                self._navigator_cache[nav_key] = navigator
                
                logger.info(
                    "rlm_navigator_created",
                    cache_key=cache_key,
                    use_knowledge_graph=use_knowledge_graph,
                    enable_verification=enable_verification,
                )
            else:
                navigator = self._navigator_cache[nav_key]
                logger.debug("using_cached_navigator", key=nav_key)
            
            # Run the navigation
            result = navigator.navigate(question, metadata=metadata)
            
            return result
        else:
            # Use standard navigator (simpler, no knowledge graph)
            result = run_navigator(question, skeleton, kv_store, metadata=metadata)
            return result
    
    def ask_vision(
        self,
        document: str | Path,
        question: str,
        use_hybrid: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Vision-based Q&A working directly on page images.
        
        This is ideal for:
        - Scanned documents where OCR quality is poor
        - Documents with charts, graphs, or diagrams
        - Image-heavy presentations or reports
        
        Args:
            document: Path to PDF file.
            question: Question to ask.
            use_hybrid: Combine text+vision analysis (True) or vision-only (False).
            metadata: Optional metadata (e.g., multiple choice options).
            
        Returns:
            Result dictionary with answer, confidence, selected_pages.
            
        Example:
            result = client.ask_vision(
                "financial_report.pdf",
                "What does the revenue chart show?",
            )
            print(f"Answer: {result['answer']}")
            print(f"Pages analyzed: {result['selected_pages']}")
        """
        doc_path = Path(document)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        from rnsr.ingestion.vision_retrieval import (
            VisionConfig,
            create_vision_navigator,
            create_hybrid_navigator,
        )
        
        config = VisionConfig()
        
        if use_hybrid:
            # Get text-based index for hybrid mode
            try:
                skeleton, kv_store = self._get_or_create_index(doc_path, False)
            except Exception:
                skeleton, kv_store = None, None
            
            navigator = create_hybrid_navigator(
                doc_path,
                skeleton=skeleton,
                kv_store=kv_store,
                vision_config=config,
            )
            
            result = navigator.navigate(question, metadata)
            return {
                "answer": result.get("combined_answer"),
                "confidence": result.get("confidence", 0),
                "method_used": result.get("method_used"),
                "text_result": result.get("text_result"),
                "vision_result": result.get("vision_result"),
            }
        else:
            # Vision-only mode
            navigator = create_vision_navigator(doc_path, config)
            return navigator.navigate(question, metadata)
    
    def analyze_document_structure(
        self,
        document: str | Path,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """
        Analyze a document's structure without querying it.
        
        Returns detailed information about the document hierarchy,
        sections, and content distribution.
        
        Args:
            document: Path to PDF file.
            force_reindex: Re-process even if cached.
            
        Returns:
            Dictionary with structure analysis.
            
        Example:
            info = client.analyze_document_structure("contract.pdf")
            print(f"Sections: {info['section_count']}")
            print(f"Max depth: {info['max_depth']}")
        """
        doc_path = Path(document)
        skeleton, kv_store = self._get_or_create_index(doc_path, force_reindex)
        
        # Analyze structure
        section_count = 0
        max_depth = 0
        total_chars = 0
        level_counts: dict[int, int] = {}
        
        for node_id, node in skeleton.items():
            section_count += 1
            max_depth = max(max_depth, node.level)
            level_counts[node.level] = level_counts.get(node.level, 0) + 1
            
            content = kv_store.get(node_id)
            if content:
                total_chars += len(content)
        
        return {
            "path": str(doc_path),
            "section_count": section_count,
            "max_depth": max_depth,
            "level_distribution": level_counts,
            "total_characters": total_chars,
            "average_section_length": total_chars // max(section_count, 1),
        }
    
    def get_document_outline(
        self,
        document: str | Path,
        max_depth: int = 2,
        force_reindex: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get a document outline (table of contents).
        
        Args:
            document: Path to PDF file.
            max_depth: Maximum heading depth to include.
            force_reindex: Re-process even if cached.
            
        Returns:
            List of section dictionaries with header and level.
            
        Example:
            outline = client.get_document_outline("report.pdf")
            for section in outline:
                indent = "  " * section['level']
                print(f"{indent}{section['header']}")
        """
        doc_path = Path(document)
        skeleton, _ = self._get_or_create_index(doc_path, force_reindex)
        
        outline = []
        for node in skeleton.values():
            if node.level <= max_depth:
                outline.append({
                    "id": node.node_id,
                    "header": node.header,
                    "level": node.level,
                    "summary": node.summary[:100] if node.summary else "",
                    "child_count": len(node.child_ids),
                })
        
        # Sort by node_id to maintain document order
        outline.sort(key=lambda x: x["id"])
        return outline
    
    # =========================================================================
    # Table Query Methods
    # =========================================================================
    
    def list_tables(
        self,
        document: str | Path,
        force_reindex: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List all tables detected in a document.
        
        Tables are automatically detected during ingestion and can be
        queried using SQL-like syntax.
        
        Args:
            document: Path to PDF file.
            force_reindex: Re-process even if cached.
            
        Returns:
            List of table metadata dictionaries with id, title, headers, num_rows, etc.
            
        Example:
            tables = client.list_tables("financial_report.pdf")
            for t in tables:
                print(f"{t['id']}: {t['title']} ({t['num_rows']} rows)")
        """
        doc_path = Path(document)
        cache_key = self._get_cache_key(doc_path)
        
        # Ensure document is indexed (populates tables cache)
        self._get_or_create_index(doc_path, force_reindex)
        
        tables = self._tables_cache.get(cache_key, [])
        
        return [
            {
                "id": t.id,
                "node_id": t.node_id,
                "page_num": t.page_num,
                "title": t.title,
                "headers": t.headers,
                "num_rows": t.num_rows,
                "num_cols": t.num_cols,
            }
            for t in tables
        ]
    
    def query_table(
        self,
        document: str | Path,
        table_id: str,
        columns: list[str] | None = None,
        where: dict | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        force_reindex: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Run a SQL-like query on a specific table.
        
        Args:
            document: Path to PDF file.
            table_id: The table ID to query (from list_tables()).
            columns: List of column names to select (None = all).
            where: Filter conditions as {column: value} or {column: {"op": ">=", "value": 100}}.
                   Supported operators: "==", "!=", ">", ">=", "<", "<=", "contains"
            order_by: Column name to sort by (prefix with "-" for descending).
            limit: Maximum number of rows to return.
            force_reindex: Re-process even if cached.
            
        Returns:
            List of row dictionaries matching the query.
            
        Example:
            # Get all rows from a table
            rows = client.query_table("report.pdf", "table_001")
            
            # Filter and sort
            rows = client.query_table(
                "report.pdf",
                "table_001",
                columns=["Name", "Revenue"],
                where={"Revenue": {"op": ">=", "value": 1000}},
                order_by="-Revenue",
                limit=10,
            )
        """
        doc_path = Path(document)
        cache_key = self._get_cache_key(doc_path)
        
        # Ensure document is indexed
        self._get_or_create_index(doc_path, force_reindex)
        
        tables = self._tables_cache.get(cache_key, [])
        
        # Find the target table
        target_table = None
        for t in tables:
            if t.id == table_id:
                target_table = t
                break
        
        if not target_table:
            available = [t.id for t in tables]
            raise ValueError(f"Table '{table_id}' not found. Available tables: {available}")
        
        headers = target_table.headers
        data = target_table.data
        
        # Convert data rows to dicts
        rows = []
        for row_data in data:
            row_dict = {}
            for i, value in enumerate(row_data):
                col_name = headers[i] if i < len(headers) else f"col_{i}"
                row_dict[col_name] = value
            rows.append(row_dict)
        
        # Apply WHERE filter
        if where:
            filtered_rows = []
            for row in rows:
                match = True
                for col, condition in where.items():
                    cell_value = row.get(col, "")
                    
                    if isinstance(condition, dict):
                        op = condition.get("op", "==")
                        cond_value = condition.get("value")
                        
                        try:
                            cell_num = float(str(cell_value).replace(",", "").replace("$", "").strip())
                            cond_num = float(cond_value)
                            
                            if op == "==" and cell_num != cond_num:
                                match = False
                            elif op == "!=" and cell_num == cond_num:
                                match = False
                            elif op == ">" and cell_num <= cond_num:
                                match = False
                            elif op == ">=" and cell_num < cond_num:
                                match = False
                            elif op == "<" and cell_num >= cond_num:
                                match = False
                            elif op == "<=" and cell_num > cond_num:
                                match = False
                        except (ValueError, TypeError):
                            if op == "==" and str(cell_value) != str(cond_value):
                                match = False
                            elif op == "!=" and str(cell_value) == str(cond_value):
                                match = False
                            elif op == "contains" and str(cond_value).lower() not in str(cell_value).lower():
                                match = False
                    else:
                        if str(condition).lower() not in str(cell_value).lower():
                            match = False
                
                if match:
                    filtered_rows.append(row)
            rows = filtered_rows
        
        # Apply ORDER BY
        if order_by:
            descending = order_by.startswith("-")
            col = order_by.lstrip("-")
            
            def sort_key(row):
                val = row.get(col, "")
                try:
                    return float(str(val).replace(",", "").replace("$", "").strip())
                except (ValueError, TypeError, AttributeError):
                    return str(val).lower()
            
            rows = sorted(rows, key=sort_key, reverse=descending)
        
        # Apply LIMIT
        if limit and limit > 0:
            rows = rows[:limit]
        
        # Apply column selection
        if columns:
            rows = [{col: row.get(col, "") for col in columns} for row in rows]
        
        return rows
    
    def aggregate_table(
        self,
        document: str | Path,
        table_id: str,
        column: str,
        operation: str,
        force_reindex: bool = False,
    ) -> float | int:
        """
        Run an aggregation operation on a table column.
        
        Args:
            document: Path to PDF file.
            table_id: The table ID to query.
            column: The column name to aggregate.
            operation: One of "sum", "avg", "count", "min", "max".
            force_reindex: Re-process even if cached.
            
        Returns:
            Numeric aggregation result.
            
        Example:
            total = client.aggregate_table("report.pdf", "table_001", "Revenue", "sum")
            avg = client.aggregate_table("report.pdf", "table_001", "Price", "avg")
        """
        doc_path = Path(document)
        cache_key = self._get_cache_key(doc_path)
        
        # Ensure document is indexed
        self._get_or_create_index(doc_path, force_reindex)
        
        tables = self._tables_cache.get(cache_key, [])
        
        # Find the target table
        target_table = None
        for t in tables:
            if t.id == table_id:
                target_table = t
                break
        
        if not target_table:
            available = [t.id for t in tables]
            raise ValueError(f"Table '{table_id}' not found. Available tables: {available}")
        
        headers = target_table.headers
        data = target_table.data
        
        # Find column index
        col_idx = None
        for i, h in enumerate(headers):
            if h.lower() == column.lower():
                col_idx = i
                break
        
        if col_idx is None:
            raise ValueError(f"Column '{column}' not found. Available columns: {headers}")
        
        # Extract numeric values
        values = []
        for row_data in data:
            if col_idx < len(row_data):
                val = row_data[col_idx]
                try:
                    clean_val = str(val).replace(",", "").replace("$", "").replace("%", "").strip()
                    if clean_val:
                        values.append(float(clean_val))
                except (ValueError, TypeError, AttributeError):
                    pass
        
        if not values:
            raise ValueError(f"No numeric values found in column '{column}'.")
        
        operation = operation.lower()
        
        if operation == "sum":
            return sum(values)
        elif operation == "avg":
            return sum(values) / len(values)
        elif operation == "count":
            return len(values)
        elif operation == "min":
            return min(values)
        elif operation == "max":
            return max(values)
        else:
            raise ValueError(f"Unknown operation '{operation}'. Use: sum, avg, count, min, max.")