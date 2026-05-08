"""
Recursive XY-Cut Algorithm - Visual-Geometric Segmentation

Implements the Recursive XY-Cut (RXYC) algorithm from Section 4.1.1:
"A top-down page segmentation technique that is particularly effective 
for discovering document structure without relying on text content."

The algorithm:
1. Treats document page as a binary image
2. Calculates projection profiles (sum of black pixels) along X and Y axes
3. Identifies "valleys" (whitespace gaps) as natural separators
4. Recursively cuts at widest valleys to produce a tree of bounding boxes
5. Larger boxes (detected early) = major structural elements
6. Smaller, deeply nested boxes = paragraphs/cells

Use this for:
- Multi-column layouts
- Complex L-shaped text wraps  
- Documents with visual structure but no font variance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BoundingRegion:
    """A rectangular region on a page."""
    
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int = 0
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def contains(self, other: "BoundingRegion") -> bool:
        """Check if this region contains another."""
        return (
            self.x0 <= other.x0 and
            self.y0 <= other.y0 and
            self.x1 >= other.x1 and
            self.y1 >= other.y1
        )


@dataclass
class SegmentNode:
    """A node in the XY-Cut segmentation tree."""
    
    region: BoundingRegion
    children: list["SegmentNode"] = field(default_factory=list)
    text: str = ""
    node_type: str = "region"  # "region", "text_block", "header", "body"
    depth: int = 0
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RecursiveXYCutter:
    """
    Implements the Recursive XY-Cut algorithm for document segmentation.
    
    Per Section 6.2 of the research paper:
    "A major failure mode of simple parsing is complex layouts 
    (e.g., a figure spanning two columns, or an L-shaped text wrap). 
    The Recursive XY-Cut handles this."
    """
    
    def __init__(
        self,
        min_gap_ratio: float = 0.02,  # Minimum gap as ratio of page dimension
        min_region_ratio: float = 0.01,  # Minimum region size ratio
        max_depth: int = 10,  # Maximum recursion depth
        valley_threshold: float = 0.1,  # Threshold for valley detection
    ):
        """
        Initialize the XY-Cutter.
        
        Args:
            min_gap_ratio: Minimum whitespace gap size as ratio of page size.
            min_region_ratio: Minimum region size to consider.
            max_depth: Maximum recursion depth.
            valley_threshold: Threshold for detecting valleys in projection.
        """
        self.min_gap_ratio = min_gap_ratio
        self.min_region_ratio = min_region_ratio
        self.max_depth = max_depth
        self.valley_threshold = valley_threshold
    
    def segment_pdf(self, pdf_path: Path | str) -> list[SegmentNode]:
        """
        Segment all pages of a PDF using XY-Cut.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of SegmentNode trees (one per page).
        """
        import fitz
        
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        
        page_trees = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.debug("segmenting_page", page=page_num)
            tree = self.segment_page(page, page_num)
            page_trees.append(tree)
        
        doc.close()
        
        logger.info(
            "xy_cut_complete",
            pages=len(page_trees),
            total_regions=sum(self._count_nodes(t) for t in page_trees),
        )
        
        return page_trees
    
    def segment_page(self, page: Any, page_num: int = 0) -> SegmentNode:
        """
        Segment a single page using XY-Cut.
        
        Args:
            page: A fitz.Page object.
            page_num: Page number for metadata.
            
        Returns:
            Root SegmentNode with hierarchy of regions.
        """
        import fitz
        
        # Get page dimensions
        rect = page.rect
        page_width = rect.width
        page_height = rect.height
        
        # Create initial region (full page)
        root_region = BoundingRegion(
            x0=0, y0=0,
            x1=page_width, y1=page_height,
            page_num=page_num,
        )
        
        # Render page to pixmap for projection analysis
        # Use lower resolution for speed (72 dpi = 1x)
        mat = fitz.Matrix(1, 1)  # 72 dpi
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array (grayscale)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, 3)
        gray = np.mean(img, axis=2)
        
        # Binarize (invert so text = 1, background = 0)
        binary = (gray < 240).astype(np.float32)
        
        # Calculate minimum dimensions
        min_gap_x = int(page_width * self.min_gap_ratio)
        min_gap_y = int(page_height * self.min_gap_ratio)
        min_region_w = int(page_width * self.min_region_ratio)
        min_region_h = int(page_height * self.min_region_ratio)
        
        # Recursive cut
        root = SegmentNode(region=root_region, depth=0)
        self._recursive_cut(
            binary, root,
            0, 0, pix.width, pix.height,
            min_gap_x, min_gap_y,
            min_region_w, min_region_h,
            page_width / pix.width,  # Scale factor
            page_height / pix.height,
        )
        
        return root
    
    def _recursive_cut(
        self,
        binary: np.ndarray,
        parent: SegmentNode,
        x0: int, y0: int, x1: int, y1: int,
        min_gap_x: int, min_gap_y: int,
        min_region_w: int, min_region_h: int,
        scale_x: float, scale_y: float,
    ) -> None:
        """
        Recursively cut a region.
        
        Per the research paper algorithm:
        1. Calculate projection profiles
        2. Find valleys (gaps of whitespace)
        3. Split horizontally first (Y-cut), then vertically (X-cut)
        4. Recurse on sub-regions
        """
        if parent.depth >= self.max_depth:
            return
        
        width = x1 - x0
        height = y1 - y0
        
        # Check minimum size
        if width < min_region_w or height < min_region_h:
            return
        
        # Extract region
        region_pixels = binary[y0:y1, x0:x1]
        
        if region_pixels.size == 0:
            return
        
        # Calculate projection profiles
        y_proj = np.sum(region_pixels, axis=1)  # Horizontal projection
        x_proj = np.sum(region_pixels, axis=0)  # Vertical projection
        
        # Try horizontal cut first (Y-cut - splits top/bottom)
        y_valleys = self._find_valleys(y_proj, min_gap_y)
        
        if y_valleys:
            # Split at the widest valley
            best_valley = max(y_valleys, key=lambda v: v[1] - v[0])
            cut_y = (best_valley[0] + best_valley[1]) // 2
            
            # Create two child regions
            if cut_y - y0 > min_region_h:
                top_region = BoundingRegion(
                    x0=x0 * scale_x, y0=y0 * scale_y,
                    x1=x1 * scale_x, y1=cut_y * scale_y,
                    page_num=parent.region.page_num,
                )
                top_node = SegmentNode(region=top_region, depth=parent.depth + 1)
                parent.children.append(top_node)
                self._recursive_cut(
                    binary, top_node,
                    x0, y0, x1, cut_y,
                    min_gap_x, min_gap_y,
                    min_region_w, min_region_h,
                    scale_x, scale_y,
                )
            
            if y1 - cut_y > min_region_h:
                bottom_region = BoundingRegion(
                    x0=x0 * scale_x, y0=cut_y * scale_y,
                    x1=x1 * scale_x, y1=y1 * scale_y,
                    page_num=parent.region.page_num,
                )
                bottom_node = SegmentNode(region=bottom_region, depth=parent.depth + 1)
                parent.children.append(bottom_node)
                self._recursive_cut(
                    binary, bottom_node,
                    x0, cut_y, x1, y1,
                    min_gap_x, min_gap_y,
                    min_region_w, min_region_h,
                    scale_x, scale_y,
                )
            return
        
        # No horizontal cut found - try vertical (X-cut - splits columns)
        x_valleys = self._find_valleys(x_proj, min_gap_x)
        
        if x_valleys:
            # Split at the widest valley
            best_valley = max(x_valleys, key=lambda v: v[1] - v[0])
            cut_x = (best_valley[0] + best_valley[1]) // 2
            
            # Create two child regions
            if cut_x - x0 > min_region_w:
                left_region = BoundingRegion(
                    x0=x0 * scale_x, y0=y0 * scale_y,
                    x1=cut_x * scale_x, y1=y1 * scale_y,
                    page_num=parent.region.page_num,
                )
                left_node = SegmentNode(region=left_region, depth=parent.depth + 1)
                parent.children.append(left_node)
                self._recursive_cut(
                    binary, left_node,
                    x0, y0, cut_x, y1,
                    min_gap_x, min_gap_y,
                    min_region_w, min_region_h,
                    scale_x, scale_y,
                )
            
            if x1 - cut_x > min_region_w:
                right_region = BoundingRegion(
                    x0=cut_x * scale_x, y0=y0 * scale_y,
                    x1=x1 * scale_x, y1=y1 * scale_y,
                    page_num=parent.region.page_num,
                )
                right_node = SegmentNode(region=right_region, depth=parent.depth + 1)
                parent.children.append(right_node)
                self._recursive_cut(
                    binary, right_node,
                    cut_x, y0, x1, y1,
                    min_gap_x, min_gap_y,
                    min_region_w, min_region_h,
                    scale_x, scale_y,
                )
            return
        
        # No cuts possible - this is a leaf (text block)
        parent.node_type = "text_block"
    
    def _find_valleys(
        self,
        projection: np.ndarray,
        min_gap: int,
    ) -> list[tuple[int, int]]:
        """
        Find valleys (whitespace gaps) in a projection profile.
        
        A valley is a contiguous region where the projection is below threshold.
        
        Args:
            projection: 1D array of projection values.
            min_gap: Minimum gap size to consider.
            
        Returns:
            List of (start, end) tuples for each valley.
        """
        if len(projection) == 0:
            return []
        
        # Normalize projection
        max_val = np.max(projection)
        if max_val == 0:
            return []
        
        normalized = projection / max_val
        
        # Find regions below threshold (valleys)
        is_valley = normalized < self.valley_threshold
        
        valleys = []
        in_valley = False
        valley_start = 0
        
        for i, is_v in enumerate(is_valley):
            if is_v and not in_valley:
                # Start of valley
                valley_start = i
                in_valley = True
            elif not is_v and in_valley:
                # End of valley
                if i - valley_start >= min_gap:
                    valleys.append((valley_start, i))
                in_valley = False
        
        # Handle valley at end
        if in_valley and len(projection) - valley_start >= min_gap:
            valleys.append((valley_start, len(projection)))
        
        return valleys
    
    def _count_nodes(self, node: SegmentNode) -> int:
        """Count total nodes in a tree."""
        return 1 + sum(self._count_nodes(c) for c in node.children)
    
    def extract_text_in_regions(
        self,
        page: Any,
        root: SegmentNode,
    ) -> None:
        """
        Extract text content for each leaf region.
        
        Args:
            page: A fitz.Page object.
            root: Root SegmentNode from segment_page().
        """
        self._extract_text_recursive(page, root)
    
    def _extract_text_recursive(
        self,
        page: Any,
        node: SegmentNode,
    ) -> None:
        """Recursively extract text for leaf nodes."""
        import fitz
        
        if node.is_leaf:
            # Extract text from this region
            rect = fitz.Rect(
                node.region.x0,
                node.region.y0,
                node.region.x1,
                node.region.y1,
            )
            node.text = page.get_text("text", clip=rect).strip()
        else:
            for child in node.children:
                self._extract_text_recursive(page, child)


def segment_pdf_with_xy_cut(pdf_path: Path | str) -> list[SegmentNode]:
    """
    Convenience function to segment a PDF using XY-Cut.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of SegmentNode trees (one per page).
        
    Example:
        trees = segment_pdf_with_xy_cut("document.pdf")
        for page_tree in trees:
            for leaf in get_leaves(page_tree):
                print(leaf.text)
    """
    cutter = RecursiveXYCutter()
    return cutter.segment_pdf(pdf_path)


def get_leaves(node: SegmentNode) -> list[SegmentNode]:
    """Get all leaf nodes from a segment tree."""
    if node.is_leaf:
        return [node]
    
    leaves = []
    for child in node.children:
        leaves.extend(get_leaves(child))
    return leaves


def analyze_document_with_xycut(
    pdf_path: Path | str,
    use_layoutlm: bool = True,
) -> Any:
    """
    Analyze document using XY-Cut + LayoutLM visual classification.
    
    Combines geometric segmentation (XY-Cut) with visual analysis (LayoutLM)
    to create a hierarchical document tree.
    
    Args:
        pdf_path: Path to PDF file.
        use_layoutlm: Use LayoutLM to classify block types (Header/Body/Title).
        
    Returns:
        DocumentTree with visually-detected structure.
    """
    from rnsr.models import DocumentNode, DocumentTree
    import fitz
    
    pdf_path = Path(pdf_path)
    
    # Segment with XY-Cut
    cutter = RecursiveXYCutter()
    page_trees = cutter.segment_pdf(pdf_path)
    
    # Extract text for each region
    doc = fitz.open(pdf_path)
    for page_num, tree in enumerate(page_trees):
        cutter.extract_text_in_regions(doc[page_num], tree)
    
    # Optionally classify blocks with LayoutLM
    if use_layoutlm:
        try:
            from rnsr.ingestion.layout_model import classify_layout_blocks
            from PIL import Image
            
            for page_num, tree in enumerate(page_trees):
                # Render page as image
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Get all leaf regions
                leaves = get_leaves(tree)
                bboxes = [(leaf.region.x0, leaf.region.y0, leaf.region.x1, leaf.region.y1) 
                          for leaf in leaves]
                texts = [leaf.text for leaf in leaves]
                
                # Classify with LayoutLM
                if bboxes:
                    results = classify_layout_blocks(img, bboxes, texts)
                    
                    # Update node types based on classification
                    for leaf, result in zip(leaves, results):
                        leaf.node_type = result["label"].lower()
                        
        except Exception as e:
            logger.warning("layoutlm_classification_failed", error=str(e))
    
    doc.close()
    
    root = DocumentNode(id="root", level=0, header=pdf_path.stem)

    all_leaves = []
    for page_tree in page_trees:
        for leaf in get_leaves(page_tree):
            if leaf.text.strip():
                all_leaves.append(leaf)

    body_leaves = [
        (i, leaf) for i, leaf in enumerate(all_leaves)
        if leaf.node_type not in ("header", "title")
    ]

    body_headers: list[str] = []
    if body_leaves:
        from rnsr.ingestion.semantic_fallback import generate_synthetic_headers_batch

        items = [(leaf.text, idx + 1) for idx, (_orig_idx, leaf) in enumerate(body_leaves)]
        body_headers = generate_synthetic_headers_batch(items)

    body_header_by_orig_idx = {
        orig_idx: body_headers[batch_idx]
        for batch_idx, (orig_idx, _leaf) in enumerate(body_leaves)
    }

    for section_num, leaf in enumerate(all_leaves, start=1):
        is_header = leaf.node_type in ("header", "title")
        if is_header:
            section = DocumentNode(
                id=f"sec_{section_num:03d}",
                level=1,
                header=leaf.text.strip(),
                page_num=leaf.region.page_num,
            )
        else:
            section = DocumentNode(
                id=f"sec_{section_num:03d}",
                level=1,
                header=body_header_by_orig_idx[section_num - 1] or f"Section {section_num}",
                content=leaf.text,
                page_num=leaf.region.page_num,
            )
        root.children.append(section)

    section_num = len(all_leaves)
    
    return DocumentTree(
        title=pdf_path.stem,
        root=root,
        total_nodes=section_num + 1,
        ingestion_tier=1,
        ingestion_method="xy_cut_layoutlm" if use_layoutlm else "xy_cut",
    )


