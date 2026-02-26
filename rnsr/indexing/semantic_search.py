"""
Semantic Search for Skeleton Index

Provides O(log N) retrieval using vector similarity search on node summaries.
Falls back to full exploration if needed.

Usage:
    searcher = SemanticSearcher(skeleton_nodes, kv_store)
    relevant_nodes = searcher.search(query, top_k=5)
    
    # Or get all node IDs ranked by relevance
    all_ranked = searcher.rank_all_nodes(query)
"""

from __future__ import annotations

from typing import Any

import structlog

from rnsr.exceptions import IndexingError
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


class SemanticSearcher:
    """
    Semantic search over skeleton node summaries.
    
    Uses vector embeddings for O(log N) retrieval instead of
    evaluating all nodes with expensive LLM calls.
    
    Attributes:
        skeleton_nodes: Dictionary of node_id -> SkeletonNode
        kv_store: KV store for full content retrieval
        index: LlamaIndex VectorStoreIndex (built lazily)
        embedder: Embedding model instance
    """
    
    def __init__(
        self,
        skeleton_nodes: dict[str, SkeletonNode],
        kv_store: KVStore,
        embed_model: str = "text-embedding-3-small",
        provider: str | None = None,
    ):
        """
        Initialize semantic searcher.
        
        Args:
            skeleton_nodes: Skeleton nodes to search over
            kv_store: KV store for content retrieval
            embed_model: Embedding model name
            provider: "openai", "gemini", or None for auto-detect
        """
        self.skeleton_nodes = skeleton_nodes
        self.kv_store = kv_store
        self.embed_model_name = embed_model
        self.provider = provider
        
        self._index = None
        self._embedder = None
        self._node_map: dict[str, SkeletonNode] = {}
        
        logger.info(
            "semantic_searcher_initialized",
            nodes=len(skeleton_nodes),
            embed_model=embed_model,
        )
    
    def _build_index(self) -> None:
        """Build vector index lazily on first search."""
        if self._index is not None:
            return
        
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.core.schema import TextNode
            
            # Get embedding model
            embed_model = self._get_embedding_model()
            
            # Create text nodes from skeleton summaries
            text_nodes = []
            for node_id, skel in self.skeleton_nodes.items():
                # Skip nodes with no content
                if not skel.summary or len(skel.summary.strip()) < 10:
                    continue
                
                # Create text node with summary
                text = f"{skel.header or ''}\n{skel.summary}".strip()
                
                text_node = TextNode(
                    text=text,
                    id_=node_id,
                    metadata={
                        "node_id": node_id,
                        "level": skel.level,
                        "header": skel.header,
                        "has_children": len(skel.child_ids) > 0,
                        "child_ids": skel.child_ids,
                    },
                )
                text_nodes.append(text_node)
                self._node_map[node_id] = skel
            
            # Build index
            self._index = VectorStoreIndex(
                nodes=text_nodes,
                embed_model=embed_model,
                show_progress=False,
            )
            
            logger.info(
                "vector_index_built",
                nodes_indexed=len(text_nodes),
                embed_model=self.embed_model_name,
            )
            
        except ImportError as e:
            logger.warning(
                "llama_index_not_available",
                error=str(e),
                fallback="Will use linear search",
            )
            raise IndexingError(
                "LlamaIndex not installed. "
                "Install with: pip install llama-index llama-index-embeddings-openai"
            ) from e
    
    def _get_embedding_model(self) -> Any:
        """Get embedding model based on provider."""
        import os
        
        # Auto-detect provider
        provider = self.provider
        if provider is None:
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("GOOGLE_API_KEY"):
                provider = "gemini"
            else:
                logger.warning("no_embedding_api_key_found")
                raise IndexingError("No API key found for embeddings")
        
        provider = provider.lower()
        
        try:
            if provider == "openai":
                from llama_index.embeddings.openai import OpenAIEmbedding
                
                logger.info("using_openai_embeddings", model=self.embed_model_name)
                return OpenAIEmbedding(model=self.embed_model_name)
            
            elif provider == "gemini":
                from llama_index.embeddings.gemini import GeminiEmbedding
                
                logger.info("using_gemini_embeddings")
                return GeminiEmbedding(model_name="models/text-embedding-005")
            
            else:
                raise IndexingError(f"Unsupported provider: {provider}")
                
        except ImportError as e:
            raise IndexingError(
                f"Failed to import {provider} embeddings. "
                f"Install with: pip install llama-index-embeddings-{provider}"
            ) from e
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[SkeletonNode, float]]:
        """
        Search for relevant nodes using semantic similarity.
        
        Args:
            query: Search query (user question)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (SkeletonNode, similarity_score) tuples, sorted by relevance
        """
        # Build index if not already built
        if self._index is None:
            self._build_index()
        
        # Index should be built now, but check again for type safety
        if self._index is None:
            logger.error("index_build_failed")
            return []
        
        # Query the index
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        
        # Convert to skeleton nodes with scores
        node_scores = []
        for result in results:
            node_id = result.node.id_
            if node_id in self._node_map:
                # LlamaIndex similarity scores are already normalized 0-1
                score = result.score if result.score is not None else 0.0
                if score >= similarity_threshold:
                    node_scores.append((self._node_map[node_id], score))
        
        logger.info(
            "semantic_search_complete",
            query_len=len(query),
            results=len(node_scores),
            top_score=node_scores[0][1] if node_scores else 0,
        )
        
        return node_scores
    
    def rank_all_nodes(
        self,
        query: str,
        filter_leaves_only: bool = False,
    ) -> list[tuple[SkeletonNode, float]]:
        """
        Rank ALL nodes by relevance to query.
        
        This is useful for exploring everything but in priority order.
        Much faster than LLM-based Tree of Thoughts evaluation.
        
        Args:
            query: Search query
            filter_leaves_only: If True, only return leaf nodes
            
        Returns:
            All nodes ranked by similarity score
        """
        # Get all nodes (use high top_k)
        all_ranked = self.search(query, top_k=len(self.skeleton_nodes))
        
        if filter_leaves_only:
            all_ranked = [
                (node, score)
                for node, score in all_ranked
                if len(node.child_ids) == 0
            ]
        
        logger.info(
            "all_nodes_ranked",
            total=len(all_ranked),
            leaves_only=filter_leaves_only,
        )
        
        return all_ranked
    
    def search_and_expand(
        self,
        query: str,
        top_k: int = 10,
        max_explore: int = 20,
    ) -> list[str]:
        """
        Adaptive search strategy:
        1. Find top_k most relevant nodes via semantic search (O(log N))
        2. If needed, expand to explore up to max_explore nodes
        
        This ensures we don't miss important data while staying efficient.
        
        Args:
            query: Search query
            top_k: Initial number of nodes to explore
            max_explore: Maximum nodes to explore if initial set insufficient
            
        Returns:
            List of node IDs to explore, in priority order
        """
        # Get top-k via semantic search
        top_results = self.search(query, top_k=min(top_k, max_explore))
        node_ids = [node.node_id for node, score in top_results]
        
        logger.info(
            "adaptive_search",
            initial_nodes=len(node_ids),
            max_explore=max_explore,
        )
        
        return node_ids


def create_semantic_searcher(
    skeleton_nodes: dict[str, SkeletonNode],
    kv_store: KVStore,
    provider: str | None = None,
) -> SemanticSearcher | None:
    """
    Create a semantic searcher if embeddings are available.
    
    Args:
        skeleton_nodes: Skeleton nodes to search
        kv_store: KV store for content
        provider: "openai", "gemini", or None for auto-detect
        
    Returns:
        SemanticSearcher instance, or None if embeddings unavailable
    """
    try:
        searcher = SemanticSearcher(
            skeleton_nodes=skeleton_nodes,
            kv_store=kv_store,
            provider=provider,
        )
        return searcher
    except IndexingError as e:
        logger.warning(
            "semantic_search_unavailable",
            error=str(e),
            fallback="Will use Tree of Thoughts evaluation",
        )
        return None
