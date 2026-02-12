"""
Skeleton Index - Summary-Only Vector Index with External Content

The Skeleton Index pattern implements a two-layer retrieval approach:

1. **Skeleton Layer** (Vector Index): Contains ONLY summaries and metadata
   - Each IndexNode's .text field contains a 50-100 word summary
   - Child node IDs stored in metadata for navigation
   - Used for initial retrieval and expand/traverse decisions

2. **Content Layer** (KV Store): Contains full text content
   - Stored separately to prevent context pollution
   - Only fetched during synthesis when explicitly needed
   - Accessed via node_id pointers

Agent Decision Protocol:
    if summary_answers_question(node.text):
        # EXPAND: Fetch full content from KV Store
        content = kv_store.get(node.node_id)
        store_as_variable(content)
    else:
        # TRAVERSE: Navigate to child nodes
        children = [get_node(cid) for cid in node.child_ids]
        continue_navigation(children)
"""

from __future__ import annotations

from typing import Any

import structlog

from rnsr.exceptions import IndexingError
from rnsr.indexing.kv_store import InMemoryKVStore, KVStore, SQLiteKVStore
from rnsr.models import DocumentNode, DocumentTree, SkeletonNode

logger = structlog.get_logger(__name__)


def generate_summary(
    content: str,
    max_words: int = 100,
    child_headers: list[str] | None = None,
) -> str:
    """
    Generate a summary for a node's content.
    
    For **parent / group nodes** with children, produces a table-of-contents
    style summary listing child section headers.  This gives ToT far better
    signal than the first 100 words of (possibly empty) group content.
    
    For **leaf nodes**, uses an extractive approach (first *max_words* words)
    to preserve key facts, entities, and concrete details.
    
    Args:
        content: Full text content.
        max_words: Maximum words in summary.
        child_headers: Optional list of child-node headers.  When provided
            (and non-empty), the summary becomes a table-of-contents.
        
    Returns:
        Summary text.
    """
    # Table-of-contents summary for parent nodes
    if child_headers:
        toc = ", ".join(h for h in child_headers if h)
        return f"Contains: {toc}"

    if not content:
        return ""
    
    words = content.split()
    
    if len(words) <= max_words:
        return content
    
    # EXTRACTIVE SUMMARY: Take first max_words to preserve:
    # - Opening sentences (often contain key context)
    # - Named entities (people, places, concepts)
    # - Concrete facts (numbers, dates, specific actions)
    # This gives ToT better signal than arbitrary truncation
    return " ".join(words[:max_words]) + "..."


async def generate_summary_llm(
    content: str,
    llm: Any = None,
    max_words: int = 75,
    provider: str | None = None,
) -> str:
    """
    Generate a summary using an LLM.

    Supports OpenAI, Anthropic, Gemini, Ollama, auto-detect.

    Args:
        content: Full text content.
        llm: LlamaIndex LLM instance (optional). If None, creates one.
        max_words: Target word count.
        provider: LLM provider ("openai", "anthropic", "gemini", "ollama", or None for auto).
        
    Returns:
        LLM-generated summary.
    """
    if not content or len(content.strip()) < 50:
        return content
    
    # If no LLM provided, try to create one
    if llm is None:
        llm = _get_llm_for_summary(provider)
        if llm is None:
            return generate_summary(content, max_words)
    
    prompt = f"""Summarize the following text in {max_words} words or less.

IMPORTANT: Use an EXTRACTIVE approach - preserve:
- Key facts, entities, names, and concrete details (who, what, when, where)
- Specific actions, events, and outcomes
- Numbers, dates, and measurements
- The main subject and what happens to/with it

Avoid:
- Vague generalizations ("discusses various topics")
- Meta-commentary ("this section explains...")
- Abstractions without specifics

TEXT:
{content}

EXTRACTIVE SUMMARY:"""
    
    try:
        response = await llm.acomplete(prompt)
        return str(response).strip()
    except Exception as e:
        logger.warning("llm_summary_failed", error=str(e))
        return generate_summary(content, max_words)


def _get_llm_for_summary(provider: str | None = None) -> Any:
    """
    Get an LLM instance for summary generation via central rnsr.llm.

    Supports: OpenAI, Anthropic, Gemini, Ollama, auto-detect.

    Args:
        provider: "openai", "anthropic", "gemini", "ollama", or None for auto-detect.

    Returns:
        LlamaIndex-compatible LLM, or None if unavailable.
    """
    try:
        from rnsr.llm import LLMProvider, get_llm
        p = LLMProvider.AUTO if provider is None else LLMProvider(provider.lower())
        return get_llm(provider=p, enable_fallback=False)
    except (ValueError, KeyError) as e:
        logger.warning("llm_unavailable", provider=provider, error=str(e))
        return None
    except ImportError as e:
        logger.warning("llm_import_failed", provider=provider, error=str(e))
        return None


class SkeletonIndexBuilder:
    """
    Builds a Skeleton Index from a DocumentTree.
    
    The index consists of:
    1. SkeletonNode objects (summaries + metadata)
    2. KV Store entries (full content)
    
    Attributes:
        kv_store: Key-value store for full content.
        nodes: Dictionary of node_id -> SkeletonNode.
    """
    
    def __init__(self, kv_store: KVStore | None = None):
        """
        Initialize the builder.
        
        Args:
            kv_store: KV store instance. Defaults to InMemoryKVStore.
        """
        self.kv_store = kv_store or InMemoryKVStore()
        self.nodes: dict[str, SkeletonNode] = {}
        
        logger.info("skeleton_builder_initialized")
    
    def build_from_tree(self, tree: DocumentTree) -> dict[str, SkeletonNode]:
        """
        Build a skeleton index from a DocumentTree.
        
        Args:
            tree: The document tree to index.
            
        Returns:
            Dictionary mapping node_id to SkeletonNode.
        """
        self.nodes.clear()
        
        logger.info(
            "building_skeleton_index",
            doc_id=tree.id,
            total_nodes=tree.total_nodes,
        )
        
        # Recursively process the tree
        self._process_node(tree.root, parent_id=None)
        
        logger.info(
            "skeleton_index_complete",
            indexed_nodes=len(self.nodes),
            kv_entries=self.kv_store.count(),
        )
        
        return self.nodes
    
    def _process_node(
        self,
        node: DocumentNode,
        parent_id: str | None,
    ) -> SkeletonNode:
        """
        Recursively process a document node.
        
        1. Store full content in KV Store
        2. Generate summary
        3. Create SkeletonNode
        4. Process children
        """
        # Store full content in KV Store
        full_content = self._collect_content(node)
        if full_content:
            self.kv_store.put(node.id, full_content)
        
        # Generate summary.
        # For parent nodes with children, produce a table-of-contents
        # listing the child headers so ToT can make informed decisions.
        child_headers = [c.header for c in node.children] if node.children else None
        summary = generate_summary(full_content, child_headers=child_headers)
        
        # Propagate node-level metadata (e.g. is_table) into the skeleton
        skel_metadata: dict[str, Any] = {
            "has_children": len(node.children) > 0,
            "content_chars": len(full_content),
        }
        if hasattr(node, "metadata") and node.metadata:
            skel_metadata.update(node.metadata)

        # Create skeleton node
        skeleton = SkeletonNode(
            node_id=node.id,
            parent_id=parent_id,
            level=node.level,
            header=node.header,
            summary=summary,  # ONLY summary in index
            child_ids=[c.id for c in node.children],
            page_num=node.page_num,
            metadata=skel_metadata,
        )
        
        self.nodes[node.id] = skeleton
        
        # Process children
        for child in node.children:
            self._process_node(child, parent_id=node.id)
        
        return skeleton
    
    def _collect_content(self, node: DocumentNode) -> str:
        """
        Collect content from a node (header + body content).
        """
        parts = []
        
        if node.header:
            parts.append(node.header)
        
        if node.content:
            parts.append(node.content)
        
        return "\n\n".join(parts)
    
    def get_node(self, node_id: str) -> SkeletonNode | None:
        """Get a skeleton node by ID."""
        return self.nodes.get(node_id)
    
    def get_content(self, node_id: str) -> str | None:
        """Get full content for a node from KV Store."""
        return self.kv_store.get(node_id)
    
    def get_children(self, node_id: str) -> list[SkeletonNode]:
        """Get child skeleton nodes."""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        
        return [
            self.nodes[cid]
            for cid in node.child_ids
            if cid in self.nodes
        ]
    
    def get_root(self) -> SkeletonNode | None:
        """Get the root node (level 0)."""
        for node in self.nodes.values():
            if node.level == 0:
                return node
        return None


def build_skeleton_index(
    tree: DocumentTree,
    kv_store: KVStore | None = None,
) -> tuple[dict[str, SkeletonNode], KVStore]:
    """
    Convenience function to build a skeleton index.
    
    Args:
        tree: Document tree to index.
        kv_store: Optional KV store (defaults to InMemoryKVStore).
        
    Returns:
        Tuple of (skeleton_nodes dict, kv_store).
        
    Example:
        tree = ingest_document("contract.pdf").tree
        skeleton, kv = build_skeleton_index(tree)
        
        # Navigate skeleton
        root = skeleton[tree.root.id]
        for child_id in root.child_ids:
            child = skeleton[child_id]
            print(f"{child.header}: {child.summary}")
            
            # Only fetch full content when needed
            if need_full_content:
                content = kv.get(child_id)
    """
    kv_store = kv_store or InMemoryKVStore()
    builder = SkeletonIndexBuilder(kv_store)
    nodes = builder.build_from_tree(tree)
    return nodes, kv_store


# For LlamaIndex integration
def create_llama_index_nodes(
    skeleton_nodes: dict[str, SkeletonNode],
) -> list:
    """
    Create LlamaIndex IndexNode objects from skeleton nodes.
    
    Each IndexNode's .text field contains ONLY the summary,
    with child_ids in metadata for navigation.
    
    Returns:
        List of LlamaIndex IndexNode objects.
    """
    try:
        from llama_index.core.schema import IndexNode
    except ImportError:
        raise IndexingError(
            "LlamaIndex not installed. "
            "Install with: pip install llama-index"
        )
    
    llama_nodes = []
    
    for skel in skeleton_nodes.values():
        # IndexNode.text = summary ONLY (not full content!)
        node = IndexNode(
            text=skel.summary,
            index_id=skel.node_id,
            obj={
                "node_id": skel.node_id,
                "parent_id": skel.parent_id,
                "level": skel.level,
                "header": skel.header,
                "child_ids": skel.child_ids,
                "has_children": len(skel.child_ids) > 0,
            },
        )
        llama_nodes.append(node)
    
    logger.info("llama_nodes_created", count=len(llama_nodes))
    return llama_nodes
