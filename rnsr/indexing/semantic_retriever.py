"""
Semantic Retriever - Query-driven node selection using embeddings.

This module provides retrieval-based navigation that reduces complexity
from O(N) (evaluating all nodes) to O(log N) (retrieving top-k relevant nodes).
"""

from __future__ import annotations

from typing import Any

import structlog

from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


class SemanticRetriever:
    """
    Retrieves the most relevant nodes given a query using semantic search.
    
    Uses embeddings and cosine similarity to rank nodes by relevance,
    reducing search complexity from O(N) to O(log N).
    """
    
    def __init__(
        self,
        skeleton_nodes: dict[str, SkeletonNode],
        llm_provider: str = "gemini",
    ):
        """
        Initialize the retriever.
        
        Args:
            skeleton_nodes: Dictionary of node_id -> SkeletonNode.
            llm_provider: LLM provider for embeddings.
        """
        self.skeleton_nodes = skeleton_nodes
        self.llm_provider = llm_provider
        self._index = None
        self._embed_model = None
        
        logger.info(
            "semantic_retriever_initialized",
            total_nodes=len(skeleton_nodes),
            provider=llm_provider,
        )
    
    def _initialize_index(self) -> None:
        """Initialize the vector index lazily."""
        if self._index is not None:
            return
        
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.embeddings.gemini import GeminiEmbedding
            from rnsr.indexing.skeleton_index import create_llama_index_nodes
            
            # Create embedding model
            self._embed_model = GeminiEmbedding(
                model_name="models/text-embedding-005"
            )
            
            # Create index nodes (summaries only!)
            llama_nodes = create_llama_index_nodes(self.skeleton_nodes)
            
            # Build vector index
            self._index = VectorStoreIndex(
                nodes=llama_nodes,
                embed_model=self._embed_model,
            )
            
            logger.info(
                "vector_index_built",
                nodes_indexed=len(llama_nodes),
            )
            
        except ImportError as e:
            logger.warning(
                "vector_index_unavailable",
                error=str(e),
                fallback="will_use_bm25",
            )
            self._index = None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        parent_id: str | None = None,
    ) -> list[SkeletonNode]:
        """
        Retrieve the most relevant nodes for a query.
        
        Args:
            query: The question or search query.
            top_k: Number of results to return.
            parent_id: Optional parent node to restrict search to children.
            
        Returns:
            List of SkeletonNode objects ranked by relevance.
        """
        # Initialize index on first use
        self._initialize_index()
        
        # Filter candidates if parent_id specified
        candidates = self._get_candidates(parent_id)
        
        if self._index is not None:
            # Use vector search
            return self._retrieve_vector(query, top_k, candidates)
        else:
            # Fallback to BM25/keyword search
            return self._retrieve_bm25(query, top_k, candidates)
    
    def _get_candidates(
        self,
        parent_id: str | None,
    ) -> dict[str, SkeletonNode]:
        """Get candidate nodes to search over."""
        if parent_id is None:
            return self.skeleton_nodes
        
        # Filter to children of parent
        parent = self.skeleton_nodes.get(parent_id)
        if parent is None:
            return self.skeleton_nodes
        
        return {
            cid: self.skeleton_nodes[cid]
            for cid in parent.child_ids
            if cid in self.skeleton_nodes
        }
    
    def _retrieve_vector(
        self,
        query: str,
        top_k: int,
        candidates: dict[str, SkeletonNode],
    ) -> list[SkeletonNode]:
        """Retrieve using vector similarity."""
        try:
            if self._index is None:
                return self._retrieve_bm25(query, top_k, candidates)
            
            retriever = self._index.as_retriever(similarity_top_k=top_k * 2)
            results = retriever.retrieve(query)
            
            # Filter to candidates and return SkeletonNodes
            relevant = []
            for result in results:
                # Access metadata safely
                node_id = result.node.metadata.get("node_id") if hasattr(result.node, "metadata") else None
                if node_id and node_id in candidates:
                    relevant.append(candidates[node_id])
                    if len(relevant) >= top_k:
                        break
            
            logger.info(
                "vector_retrieval_complete",
                query_words=len(query.split()),
                results=len(relevant),
            )
            
            return relevant
            
        except Exception as e:
            logger.warning(
                "vector_retrieval_failed",
                error=str(e),
                fallback="bm25",
            )
            return self._retrieve_bm25(query, top_k, candidates)
    
    def _retrieve_bm25(
        self,
        query: str,
        top_k: int,
        candidates: dict[str, SkeletonNode],
    ) -> list[SkeletonNode]:
        """Fallback retrieval using BM25/keyword matching."""
        from collections import Counter
        
        query_terms = set(query.lower().split())
        
        # Score each candidate by term overlap
        scores = []
        for node_id, node in candidates.items():
            # Combine header and summary for matching
            text = f"{node.header or ''} {node.summary}".lower()
            text_terms = text.split()
            
            # Count matching terms
            matches = sum(1 for term in query_terms if term in text_terms)
            
            # Boost for header matches
            header_matches = sum(
                1 for term in query_terms
                if term in (node.header or "").lower()
            )
            
            score = matches + (header_matches * 2)
            
            if score > 0:
                scores.append((score, node))
        
        # Sort by score descending
        scores.sort(reverse=True, key=lambda x: x[0])
        
        results = [node for score, node in scores[:top_k]]
        
        logger.info(
            "bm25_retrieval_complete",
            query_terms=len(query_terms),
            candidates=len(candidates),
            results=len(results),
        )
        
        return results


def create_retriever(
    skeleton_nodes: dict[str, SkeletonNode],
    llm_provider: str = "gemini",
) -> SemanticRetriever:
    """
    Convenience function to create a semantic retriever.
    
    Args:
        skeleton_nodes: Dictionary of node_id -> SkeletonNode.
        llm_provider: LLM provider for embeddings.
        
    Returns:
        SemanticRetriever instance.
    """
    return SemanticRetriever(skeleton_nodes, llm_provider)
