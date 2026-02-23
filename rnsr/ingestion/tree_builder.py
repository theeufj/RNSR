"""
Tree Builder - Document Tree Assembly

This module builds a hierarchical document tree from classified spans.
Uses a stack-based parser to:
1. Process blocks in reading order (top-to-bottom, left-to-right)
2. Assign body text to the nearest preceding header
3. Output nested DocumentNode structure

The tree structure enables:
- Hierarchical navigation by the agent
- Section-based summarization
- Context-aware retrieval

Multi-Document Support:
- Detects document boundaries in combined PDFs
- Creates separate subtrees for each logical document
- Preserves internal hierarchy within each document
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from rnsr.models import ClassifiedSpan, DocumentNode, DocumentTree

if TYPE_CHECKING:
    from rnsr.ingestion.document_boundary import DocumentSegment

logger = structlog.get_logger(__name__)


class TreeBuilder:
    """
    Builds a hierarchical document tree from classified spans.
    
    Uses a stack-based approach where:
    - Headers push new nodes onto the stack (at appropriate level)
    - Body text appends to the current node's content
    - The stack maintains the current path in the hierarchy
    """

    def __init__(self):
        """Initialize the Tree Builder."""
        self._current_page = -1

    def build_tree(
        self,
        spans: list[ClassifiedSpan],
        document_title: str = "",
    ) -> DocumentTree:
        """
        Build a document tree from classified spans.
        
        Args:
            spans: List of ClassifiedSpan (headers and body text).
            document_title: Optional title for the document.
            
        Returns:
            DocumentTree with hierarchical structure.
        """
        if not spans:
            logger.warning("empty_spans_list")
            root = DocumentNode(id="root", level=0, header="Document")
            return DocumentTree(
                title=document_title or "Untitled",
                root=root,
                total_nodes=1,
            )
        
        # Sort spans by reading order (page, then top-to-bottom, left-to-right)
        sorted_spans = self._sort_by_reading_order(spans)
        
        # Initialize root node
        root = DocumentNode(
            id="root",
            level=0,
            header=document_title or self._extract_title(sorted_spans),
        )
        
        # Stack tracks current path: [(level, node), ...]
        # Start with root at level 0
        stack: list[tuple[int, DocumentNode]] = [(0, root)]
        
        # Process each span
        for span in sorted_spans:
            if span.role == "header":
                self._process_header(span, stack)
            else:
                self._process_body(span, stack)
        
        # Post-process: merge small nodes into parent
        self._merge_small_nodes(root)
        
        # Post-process: recursively decompose oversized leaf nodes
        self._split_large_nodes(root)
        
        # Count total nodes
        total_nodes = self._count_nodes(root)
        
        logger.info(
            "tree_built",
            total_nodes=total_nodes,
            max_depth=self._get_max_depth(root),
        )
        
        return DocumentTree(
            title=root.header,
            root=root,
            total_nodes=total_nodes,
        )

    def _sort_by_reading_order(
        self, 
        spans: list[ClassifiedSpan],
    ) -> list[ClassifiedSpan]:
        """
        Sort spans by reading order: page number, then y position, then x position.
        """
        return sorted(
            spans,
            key=lambda s: (
                s.page_num,
                s.bbox.y0,  # Top to bottom
                s.bbox.x0,  # Left to right
            ),
        )

    def _extract_title(self, spans: list[ClassifiedSpan]) -> str:
        """
        Extract document title from the first H1 header.
        """
        for span in spans:
            if span.role == "header" and span.header_level == 1:
                return span.text.strip()
        
        # Fallback: use first header of any level
        for span in spans:
            if span.role == "header":
                return span.text.strip()
        
        return "Untitled Document"

    def _process_header(
        self,
        span: ClassifiedSpan,
        stack: list[tuple[int, DocumentNode]],
    ) -> None:
        """
        Process a header span by adding a new node to the tree.
        
        The stack is adjusted so that:
        - Headers pop nodes until finding a parent with lower level
        - Then push the new header node
        """
        level = span.header_level
        
        # Pop stack until we find a parent with level < this header
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        
        # Create new node for this header
        new_node = DocumentNode(
            id=f"sec_{str(uuid4())[:6]}",
            level=level,
            header=span.text.strip(),
            page_num=span.page_num,
            bbox=span.bbox,
        )
        
        # Add as child of current top of stack
        parent_node = stack[-1][1]
        parent_node.children.append(new_node)
        
        # Push onto stack
        stack.append((level, new_node))
        
        logger.debug(
            "header_added",
            level=level,
            header=span.text[:50],
            parent=parent_node.header[:30] if parent_node.header else "root",
        )

    def _process_body(
        self,
        span: ClassifiedSpan,
        stack: list[tuple[int, DocumentNode]],
    ) -> None:
        """
        Process body text by appending to the current node's content.
        """
        if not stack:
            return
        
        current_node = stack[-1][1]
        
        # Append text with appropriate spacing
        if current_node.content:
            # Check if we need a new paragraph (different page or significant y gap)
            current_node.content += " " + span.text.strip()
        else:
            current_node.content = span.text.strip()
        
        # Update page number if not set
        if current_node.page_num is None:
            current_node.page_num = span.page_num

    # Thresholds for merging small nodes
    MIN_CONTENT_LENGTH = 50  # Nodes with less content get merged
    MIN_HEADER_LENGTH = 5   # Headers shorter than this are likely captions
    CAPTION_PATTERNS = ["figure", "table", "chart", "diagram", "fig.", "tab.", "exhibit"]
    
    def _merge_small_nodes(self, node: DocumentNode) -> None:
        """
        Post-process: merge small child nodes into parent.
        
        Nodes are merged if:
        - Content is shorter than MIN_CONTENT_LENGTH chars
        - Header looks like a caption (Figure X, Table X, etc.)
        - Node has no children of its own
        """
        if not node.children:
            return
        
        # First, recursively process all children
        for child in node.children:
            self._merge_small_nodes(child)
        
        # Now merge small children into this node
        merged_count = 0
        children_to_keep = []
        
        for child in node.children:
            should_merge = self._should_merge_into_parent(child)
            
            if should_merge:
                # Merge child content into parent
                child_text = f"\n\n[{child.header}]\n{child.content}" if child.content else f"\n\n[{child.header}]"
                if node.content:
                    node.content += child_text
                else:
                    node.content = child_text.strip()
                
                # Also merge any grandchildren up
                children_to_keep.extend(child.children)
                merged_count += 1
                
                logger.debug(
                    "node_merged",
                    child_header=child.header[:30],
                    parent_header=node.header[:30],
                    reason="small_content" if len(child.content or "") < self.MIN_CONTENT_LENGTH else "caption",
                )
            else:
                children_to_keep.append(child)
        
        node.children = children_to_keep
        
        if merged_count > 0:
            logger.info(
                "nodes_merged_into_parent",
                merged_count=merged_count,
                parent=node.header[:30],
                remaining_children=len(children_to_keep),
            )
    
    def _should_merge_into_parent(self, node: DocumentNode) -> bool:
        """
        Determine if a node should be merged into its parent.
        """
        # Don't merge nodes that have children (they're section containers)
        if node.children:
            return False
        
        content_len = len(node.content or "")
        header_lower = node.header.lower().strip()
        
        # Merge if content is very short
        if content_len < self.MIN_CONTENT_LENGTH:
            return True
        
        # Merge if header looks like a caption
        if any(pattern in header_lower for pattern in self.CAPTION_PATTERNS):
            return True
        
        # Merge if header is very short (likely a label, not a section)
        if len(node.header.strip()) < self.MIN_HEADER_LENGTH:
            return True
        
        return False
    
    # Threshold for recursive decomposition of oversized leaf nodes.
    # Leaf nodes with more content than this are split into navigable
    # sub-sections so Tree of Thoughts can traverse into them.
    MAX_LEAF_CHARS = 8000

    # Maximum recursion depth for splitting.  Prevents infinite recursion
    # when _semantic_chunk_text cannot produce chunks smaller than the
    # threshold (e.g. a single dense table row).
    MAX_SPLIT_DEPTH = 5

    def _split_large_nodes(self, node: DocumentNode, _depth: int = 0) -> None:
        """
        Post-process: recursively decompose leaf nodes that exceed MAX_LEAF_CHARS.

        Uses semantic chunking (header detection -> paragraph splitting ->
        sentence splitting) to create meaningful sub-sections.  This ensures
        the Tree of Thoughts navigator can traverse into large document
        sections rather than receiving them as monolithic blobs.

        The decomposition is recursive: if a chunk is still too large after
        the first split, it will be split again on the next pass (up to
        MAX_SPLIT_DEPTH levels to prevent infinite recursion).
        """
        # First, recurse into existing children
        for child in node.children:
            self._split_large_nodes(child, _depth)

        # Only split leaf nodes (no children) that have oversized content
        if node.children or not node.content:
            return

        content_len = len(node.content)
        if content_len <= self.MAX_LEAF_CHARS:
            return

        # Guard against infinite recursion when chunker can't reduce size
        if _depth >= self.MAX_SPLIT_DEPTH:
            logger.warning(
                "split_depth_limit_reached",
                node_id=node.id[:120],
                header=node.header[:50],
                chars=content_len,
                depth=_depth,
            )
            return

        from rnsr.ingestion.text_builder import (
            _semantic_chunk_text,
            _infer_header_from_content,
        )

        segments = _semantic_chunk_text(node.content, chunk_size=self.MAX_LEAF_CHARS)

        # Only split if we got multiple meaningful segments
        if len(segments) < 2:
            return

        logger.info(
            "splitting_large_node",
            node_id=node.id[:120],
            header=node.header[:50],
            original_chars=content_len,
            new_children=len(segments),
            depth=_depth,
        )

        # Create child nodes from segments
        for i, seg in enumerate(segments):
            header = seg.header or _infer_header_from_content(seg.text)
            child = DocumentNode(
                id=f"{node.id}_chunk_{i:03d}",
                level=node.level + 1,
                header=header,
                content=seg.text,
                page_num=node.page_num,
            )
            node.children.append(child)

        # Parent becomes a container: keep a brief summary for context;
        # full content now lives in the children.
        node.content = node.content[:200] + "..."

        # Recurse into newly created children in case any are still too large
        for child in node.children:
            self._split_large_nodes(child, _depth + 1)

    def _count_nodes(self, node: DocumentNode) -> int:
        """Recursively count all nodes in the tree."""
        count = 1  # Count this node
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node: DocumentNode, current_depth: int = 0) -> int:
        """Get the maximum depth of the tree."""
        if not node.children:
            return current_depth
        
        return max(
            self._get_max_depth(child, current_depth + 1)
            for child in node.children
        )


def build_document_tree(
    spans: list[ClassifiedSpan],
    title: str = "",
) -> DocumentTree:
    """
    Convenience function to build a document tree from classified spans.
    
    Args:
        spans: List of ClassifiedSpan from header classification.
        title: Optional document title.
        
    Returns:
        DocumentTree with hierarchical structure.
        
    Example:
        analysis, raw_spans = analyze_font_histogram("doc.pdf")
        classified = classify_headers(raw_spans, analysis)
        tree = build_document_tree(classified)
    """
    builder = TreeBuilder()
    return builder.build_tree(spans, title)


def build_multi_document_tree(
    segments: list[DocumentSegment],
    container_title: str = "",
) -> DocumentTree:
    """
    Build a tree from multiple document segments.
    
    Creates a structure where:
    - Root node represents the container (e.g., the PDF file)
    - Each document segment becomes a level-1 child
    - Internal structure of each document is preserved below that
    
    Args:
        segments: List of DocumentSegment from boundary detection
        container_title: Title for the root container node
        
    Returns:
        DocumentTree with multi-document structure
    """
    from rnsr.ingestion.header_classifier import classify_headers
    from rnsr.ingestion.font_histogram import FontHistogramAnalyzer
    
    if not segments:
        root = DocumentNode(id="root", level=0, header=container_title or "Documents")
        return DocumentTree(
            title=container_title or "Documents",
            root=root,
            total_nodes=1,
        )
    
    # If only one segment, build a regular tree
    if len(segments) == 1:
        builder = TreeBuilder()
        # Need to classify spans first
        analyzer = FontHistogramAnalyzer()
        analysis = analyzer.analyze_spans(segments[0].spans)
        classified = classify_headers(segments[0].spans, analysis)
        return builder.build_tree(classified, segments[0].title or container_title)
    
    # Create container root node
    root = DocumentNode(
        id="root",
        level=0,
        header=container_title or f"{len(segments)} Documents",
    )
    
    total_nodes = 1
    builder = TreeBuilder()
    analyzer = FontHistogramAnalyzer()
    
    for i, segment in enumerate(segments):
        # Create a document node for this segment
        doc_title = segment.title or f"Document {i + 1}"
        
        logger.debug(
            "building_document_subtree",
            doc_index=i,
            title=doc_title[:50],
            span_count=len(segment.spans),
            pages=f"{segment.start_page}-{segment.end_page}",
        )
        
        # Analyze and classify spans for this segment
        if segment.spans:
            analysis = analyzer.analyze_spans(segment.spans)
            classified = classify_headers(segment.spans, analysis)
            
            # Build subtree for this document
            subtree = builder.build_tree(classified, doc_title)
            
            # The subtree's root becomes a child of our container root
            # But we need to shift levels down by 1
            # Pass doc_index to generate unique IDs and avoid collision with container root
            doc_node = _shift_node_levels(subtree.root, level_shift=1, doc_index=i)
            doc_node.header = doc_title  # Ensure title is preserved
            
            root.children.append(doc_node)
            total_nodes += subtree.total_nodes
        else:
            # Empty segment - just create a placeholder
            doc_node = DocumentNode(
                id=f"doc_{i:03d}",
                level=1,
                header=doc_title,
                page_num=segment.start_page,
            )
            root.children.append(doc_node)
            total_nodes += 1
    
    logger.info(
        "multi_document_tree_built",
        documents=len(segments),
        total_nodes=total_nodes,
    )
    
    return DocumentTree(
        title=container_title or f"{len(segments)} Documents",
        root=root,
        total_nodes=total_nodes,
    )


def _shift_node_levels(node: DocumentNode, level_shift: int, doc_index: int = 0) -> DocumentNode:
    """
    Recursively shift all node levels by a fixed amount.
    
    Used when embedding a subtree under a higher-level node.
    Generates new unique IDs to avoid collisions with the container root.
    """
    # Generate new unique ID to avoid collision with container root
    if node.id == "root":
        new_id = f"doc_{doc_index:03d}"
    else:
        new_id = f"d{doc_index}_{node.id}"
    
    new_node = DocumentNode(
        id=new_id,
        level=node.level + level_shift,
        header=node.header,
        content=node.content,
        page_num=node.page_num,
        bbox=node.bbox,
        children=[],
    )
    
    for child in node.children:
        new_node.children.append(_shift_node_levels(child, level_shift, doc_index))
    
    return new_node


def tree_to_dict(tree: DocumentTree) -> dict:
    """
    Convert a DocumentTree to a nested dictionary for serialization.
    
    This produces the JSON structure specified in the plan.
    """
    def node_to_dict(node: DocumentNode) -> dict:
        return {
            "id": node.id,
            "type": "section" if node.level > 0 else "document",
            "level": node.level,
            "header": node.header,
            "content": node.content,
            "page_num": node.page_num,
            "children": [node_to_dict(child) for child in node.children],
        }
    
    return {
        "id": tree.id,
        "title": tree.title,
        "total_nodes": tree.total_nodes,
        "ingestion_tier": tree.ingestion_tier,
        "ingestion_method": tree.ingestion_method,
        "root": node_to_dict(tree.root),
    }
