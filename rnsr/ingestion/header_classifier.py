"""
Header Classifier - Detect and Classify Headers by Level

This module classifies text spans into:
- Headers (H1, H2, H3) based on font size magnitude
- Body text (most frequent font size)
- Captions/footnotes (smaller than body)

Header Level Mapping (adaptive):
- Defaults: >= 24pt: H1, 18-23pt: H2, 14-17pt: H3
- Learns from document types (legal, academic, marketing, etc.)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
import structlog
from sklearn.cluster import KMeans

from rnsr.models import ClassifiedSpan, FontAnalysis, SpanInfo

logger = structlog.get_logger(__name__)


# =============================================================================
# Learned Header Thresholds Registry
# =============================================================================

DEFAULT_HEADER_THRESHOLDS_PATH = Path.home() / ".rnsr" / "learned_header_thresholds.json"


class LearnedHeaderThresholds:
    """
    Registry for learning document-type-specific header thresholds.
    
    Different document types use different conventions:
    - Legal briefs: 12pt bold = header
    - Academic papers: 11pt = everything
    - Marketing: 36pt+ = titles
    
    This class learns optimal thresholds from document analysis.
    """
    
    # Default thresholds
    DEFAULT_H1_MIN = 24.0
    DEFAULT_H2_MIN = 18.0
    DEFAULT_H3_MIN = 14.0
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
    ):
        """
        Initialize the header thresholds registry.
        
        Args:
            storage_path: Path to JSON file for persistence.
            auto_save: Whether to save after changes.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_HEADER_THRESHOLDS_PATH
        self.auto_save = auto_save
        
        self._lock = Lock()
        self._document_types: dict[str, dict[str, Any]] = {}
        self._dirty = False
        
        self._load()
    
    def _load(self) -> None:
        """Load learned thresholds from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._document_types = data.get("document_types", {})
            
            logger.info(
                "header_thresholds_loaded",
                document_types=len(self._document_types),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_header_thresholds", error=str(e))
    
    def _save(self) -> None:
        """Save to storage."""
        if not self._dirty:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "document_types": self._document_types,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
        except Exception as e:
            logger.warning("failed_to_save_header_thresholds", error=str(e))
    
    def record_thresholds(
        self,
        document_type: str,
        h1_min: float,
        h2_min: float,
        h3_min: float,
        body_size: float = 12.0,
        document_example: str = "",
    ) -> None:
        """
        Record observed header thresholds for a document type.
        
        The system averages thresholds across multiple documents
        of the same type to learn optimal values.
        
        Args:
            document_type: Type of document (legal, academic, marketing, etc.)
            h1_min: Observed H1 minimum size.
            h2_min: Observed H2 minimum size.
            h3_min: Observed H3 minimum size.
            body_size: Observed body text size.
            document_example: Example document filename.
        """
        document_type = document_type.lower().strip()
        
        if not document_type:
            return
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if document_type not in self._document_types:
                self._document_types[document_type] = {
                    "count": 0,
                    "h1_min_sum": 0.0,
                    "h2_min_sum": 0.0,
                    "h3_min_sum": 0.0,
                    "body_size_sum": 0.0,
                    "first_seen": now,
                    "last_seen": now,
                    "examples": [],
                }
                logger.info("new_document_type_learned", document_type=document_type)
            
            dt = self._document_types[document_type]
            dt["count"] += 1
            dt["h1_min_sum"] += h1_min
            dt["h2_min_sum"] += h2_min
            dt["h3_min_sum"] += h3_min
            dt["body_size_sum"] += body_size
            dt["last_seen"] = now
            
            if document_example and len(dt["examples"]) < 3:
                dt["examples"].append(document_example)
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def get_thresholds(
        self,
        document_type: str | None = None,
    ) -> dict[str, float]:
        """
        Get header thresholds for a document type.
        
        Args:
            document_type: Type of document. If None, returns defaults.
            
        Returns:
            Dict with h1_min, h2_min, h3_min, body_size.
        """
        if not document_type:
            return {
                "h1_min": self.DEFAULT_H1_MIN,
                "h2_min": self.DEFAULT_H2_MIN,
                "h3_min": self.DEFAULT_H3_MIN,
                "body_size": 12.0,
            }
        
        document_type = document_type.lower().strip()
        
        with self._lock:
            if document_type in self._document_types:
                dt = self._document_types[document_type]
                count = dt["count"]
                
                if count > 0:
                    return {
                        "h1_min": dt["h1_min_sum"] / count,
                        "h2_min": dt["h2_min_sum"] / count,
                        "h3_min": dt["h3_min_sum"] / count,
                        "body_size": dt["body_size_sum"] / count,
                    }
        
        # Return defaults if not found
        return self.get_thresholds(None)
    
    def detect_document_type(
        self,
        body_size: float,
        max_font_size: float,
        has_legal_terms: bool = False,
        has_academic_structure: bool = False,
    ) -> str:
        """
        Attempt to detect document type from characteristics.
        
        Args:
            body_size: Most common font size.
            max_font_size: Largest font in document.
            has_legal_terms: Whether document contains legal terminology.
            has_academic_structure: Whether document has academic structure.
            
        Returns:
            Detected document type string.
        """
        # Simple heuristics
        if has_legal_terms:
            if body_size <= 12 and max_font_size <= 16:
                return "legal_brief"
            return "legal_general"
        
        if has_academic_structure:
            return "academic"
        
        if max_font_size >= 36:
            return "marketing"
        
        if body_size >= 14:
            return "presentation"
        
        return "general"
    
    def get_known_document_types(self) -> list[str]:
        """Get list of document types we have learned."""
        with self._lock:
            return list(self._document_types.keys())
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about learned thresholds."""
        with self._lock:
            return {
                "document_types_count": len(self._document_types),
                "document_types": list(self._document_types.keys()),
                "total_documents_analyzed": sum(
                    dt["count"] for dt in self._document_types.values()
                ),
            }


# Global header thresholds registry
_global_header_thresholds: LearnedHeaderThresholds | None = None


def get_learned_header_thresholds() -> LearnedHeaderThresholds:
    """Get the global learned header thresholds registry."""
    global _global_header_thresholds
    
    if _global_header_thresholds is None:
        custom_path = os.getenv("RNSR_HEADER_THRESHOLDS_PATH")
        _global_header_thresholds = LearnedHeaderThresholds(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_header_thresholds


class HeaderClassifier:
    """
    Classifies text spans into headers and body text.
    
    Uses font size analysis and optional k-means clustering
    to determine header levels. Supports adaptive thresholds
    that learn from document types.
    """

    # Default thresholds for header levels (in points)
    H1_MIN_SIZE = 24.0
    H2_MIN_SIZE = 18.0
    H3_MIN_SIZE = 14.0
    
    # Font tolerance: minimum size difference to be considered a header
    # This prevents slight font variations (e.g., figure captions) from creating new sections
    FONT_TOLERANCE = 2.0  # points
    
    # Caption/figure patterns that should NOT be treated as section headers
    CAPTION_PATTERNS = [
        "figure", "fig.", "fig ", 
        "table", "tab.", "tab ",
        "chart", "diagram", "exhibit",
        "graph", "image", "photo",
        "note:", "notes:", "source:",
    ]

    def __init__(
        self,
        use_clustering: bool = True,
        n_header_levels: int = 3,
        document_type: str | None = None,
        enable_threshold_learning: bool = True,
        font_tolerance: float = 2.0,
    ):
        """
        Initialize the Header Classifier.
        
        Args:
            use_clustering: Whether to use k-means clustering for header levels.
            n_header_levels: Number of header levels to detect (default: 3).
            document_type: Optional document type for adaptive thresholds.
            enable_threshold_learning: Whether to learn thresholds from documents.
            font_tolerance: Minimum size difference (pts) to consider as header.
        """
        self.use_clustering = use_clustering
        self.n_header_levels = n_header_levels
        self.document_type = document_type
        self.enable_threshold_learning = enable_threshold_learning
        self.font_tolerance = font_tolerance
        
        # Get learned thresholds registry
        self._threshold_registry = get_learned_header_thresholds() if enable_threshold_learning else None
        
        # Set thresholds based on document type
        self._update_thresholds(document_type)
    
    def _update_thresholds(self, document_type: str | None) -> None:
        """Update thresholds based on document type."""
        if self._threshold_registry and document_type:
            thresholds = self._threshold_registry.get_thresholds(document_type)
            self.H1_MIN_SIZE = thresholds["h1_min"]
            self.H2_MIN_SIZE = thresholds["h2_min"]
            self.H3_MIN_SIZE = thresholds["h3_min"]
        else:
            # Use class defaults
            self.H1_MIN_SIZE = HeaderClassifier.H1_MIN_SIZE
            self.H2_MIN_SIZE = HeaderClassifier.H2_MIN_SIZE
            self.H3_MIN_SIZE = HeaderClassifier.H3_MIN_SIZE
    
    def set_document_type(self, document_type: str) -> None:
        """Set document type and update thresholds."""
        self.document_type = document_type
        self._update_thresholds(document_type)
    
    def learn_from_analysis(
        self,
        analysis: FontAnalysis,
        detected_h1_size: float | None = None,
        detected_h2_size: float | None = None,
        detected_h3_size: float | None = None,
        document_name: str = "",
    ) -> None:
        """
        Learn thresholds from document analysis.
        
        Call this after processing a document to record observed values.
        """
        if not self._threshold_registry or not self.document_type:
            return
        
        # Use detected sizes or infer from analysis
        h1_size = detected_h1_size or analysis.header_threshold + 6
        h2_size = detected_h2_size or analysis.header_threshold + 3
        h3_size = detected_h3_size or analysis.header_threshold
        
        self._threshold_registry.record_thresholds(
            document_type=self.document_type,
            h1_min=h1_size,
            h2_min=h2_size,
            h3_min=h3_size,
            body_size=analysis.body_size,
            document_example=document_name,
        )

    def classify_spans(
        self,
        spans: list[SpanInfo],
        analysis: FontAnalysis,
    ) -> list[ClassifiedSpan]:
        """
        Classify all spans into headers or body text.
        
        Args:
            spans: List of SpanInfo from font analysis.
            analysis: FontAnalysis with body size and threshold.
            
        Returns:
            List of ClassifiedSpan with role and header_level assigned.
        """
        if not spans:
            return []
        
        # First pass: identify potential headers
        potential_headers: list[SpanInfo] = []
        for span in spans:
            if self._is_header_candidate(span, analysis):
                potential_headers.append(span)
        
        # Determine header levels
        if potential_headers and self.use_clustering:
            level_mapping = self._cluster_header_levels(potential_headers)
        else:
            level_mapping = {}
        
        # Classify all spans
        classified: list[ClassifiedSpan] = []
        for span in spans:
            role, level = self._classify_single_span(span, analysis, level_mapping)
            
            classified_span = ClassifiedSpan(
                text=span.text,
                font_size=span.font_size,
                font_name=span.font_name,
                is_bold=span.is_bold,
                is_italic=span.is_italic,
                bbox=span.bbox,
                page_num=span.page_num,
                role=role,
                header_level=level,
            )
            classified.append(classified_span)
        
        # Log classification stats
        header_count = sum(1 for s in classified if s.role == "header")
        logger.info(
            "spans_classified",
            total=len(classified),
            headers=header_count,
            body=len(classified) - header_count,
        )
        
        return classified

    def _is_header_candidate(
        self, 
        span: SpanInfo, 
        analysis: FontAnalysis,
    ) -> bool:
        """
        Determine if a span is a header candidate.
        
        A span is a header candidate if:
        1. Font size > header_threshold by at least font_tolerance, OR
        2. Bold text at or above body size (bold at body size is a common
           heading style in court judgments, contracts, and legal documents)
        
        A span is NOT a header candidate if:
        1. Text matches caption patterns (Figure X, Table X, etc.)
        2. Font size difference is within tolerance (noise)
        """
        text_lower = span.text.lower().strip()
        
        # Skip caption-like text (Figure X, Table X, etc.)
        if self._is_caption_text(text_lower):
            return False
        
        # Require significant size difference (font tolerance)
        size_above_threshold = span.font_size - analysis.header_threshold
        
        # Size-based detection with tolerance
        if size_above_threshold >= self.font_tolerance:
            return True
        
        # Bold text at or above body size is treated as a header.
        # Many documents (court judgments, legal agreements) use bold at
        # body size for section headings.
        if span.is_bold and span.font_size >= analysis.body_size:
            return True
        
        return False
    
    # Header candidates that are pure numbers, currency, or table column
    # fragments (e.g. "2,018", "$1,234.5", "($ million)", "currency ∆%")
    # are common false positives in financial filings, where bold table
    # cells trip the font-size threshold. Detected here so they're
    # rejected up-front rather than ending up as section nodes that the
    # downstream navigator can never match against query keywords.
    _NUMERIC_HEADER_RE = re.compile(
        r"""
        ^                       # full-string match
        \(?                     # optional opening paren (negative numbers)
        [\$€£¥]?\s*             # optional currency symbol
        -?                      # optional sign
        \d{1,3}(?:[,\s]\d{3})*  # integer part with optional thousands sep
        (?:\.\d+)?              # optional fractional part
        \s*\)?                  # optional closing paren
        \s*[%]?                 # optional trailing percent
        \s*$
        """,
        re.VERBOSE,
    )
    _COLUMN_LABEL_HINTS = (
        "$ million",
        "$ millions",
        "in millions",
        "in thousands",
        "constant currency",
        "currency ∆",
        "currency change",
        "% growth",
        "% change",
        "year ended",
        "twelve months",
    )

    def _is_caption_text(self, text_lower: str) -> bool:
        """
        Check if text looks like a caption/label rather than a section header.
        
        Captions include: Figure X, Table X, Chart X, etc.
        These should not create new tree nodes even if they have different fonts.
        """
        # Check for caption patterns at start of text
        for pattern in self.CAPTION_PATTERNS:
            if text_lower.startswith(pattern):
                return True
        
        # Check for numeric-only or very short labels
        clean_text = text_lower.strip()
        if len(clean_text) < 3:
            return True  # Too short to be a meaningful header
        
        # Check for patterns like "1." or "A)" which are list items, not headers
        if len(clean_text) < 5 and any(c in clean_text for c in ".):"):
            return True

        # Reject pure numeric / currency / percent strings (table cells)
        if self._NUMERIC_HEADER_RE.match(clean_text):
            return True

        # Reject column-label fragments common in financial tables
        if any(hint in clean_text for hint in self._COLUMN_LABEL_HINTS):
            return True

        return False

    def _classify_single_span(
        self,
        span: SpanInfo,
        analysis: FontAnalysis,
        level_mapping: dict[float, int],
    ) -> tuple[Literal["header", "body", "caption", "footnote"], int]:
        """
        Classify a single span.
        
        Returns:
            Tuple of (role, header_level).
        """
        # Check if it's a header
        if self._is_header_candidate(span, analysis):
            # Use clustering-based level if available
            if span.font_size in level_mapping:
                level = level_mapping[span.font_size]
            else:
                # Fall back to absolute thresholds
                level = self._get_level_by_size(span.font_size)
            
            return ("header", level)
        
        # Check for captions/footnotes (smaller than body)
        caption_threshold = analysis.body_size - (analysis.body_size * 0.2)
        if span.font_size < caption_threshold:
            return ("caption", 0)
        
        # Default to body text
        return ("body", 0)

    def _get_level_by_size(self, font_size: float) -> int:
        """
        Get header level based on absolute font size thresholds.
        
        Args:
            font_size: The font size in points.
            
        Returns:
            Header level (1, 2, or 3).
        """
        if font_size >= self.H1_MIN_SIZE:
            return 1
        elif font_size >= self.H2_MIN_SIZE:
            return 2
        else:
            return 3

    def _cluster_header_levels(
        self, 
        headers: list[SpanInfo],
    ) -> dict[float, int]:
        """
        Use k-means clustering to determine header levels from actual data.
        
        This adapts to documents with non-standard font sizes.
        
        Args:
            headers: List of spans identified as headers.
            
        Returns:
            Dict mapping font_size to header_level (1, 2, or 3).
        """
        if len(headers) < self.n_header_levels:
            # Not enough headers to cluster
            return {}
        
        # Get unique font sizes
        unique_sizes = list(set(h.font_size for h in headers))
        
        if len(unique_sizes) < self.n_header_levels:
            # Fewer unique sizes than levels - assign directly
            sorted_sizes = sorted(unique_sizes, reverse=True)
            return {size: i + 1 for i, size in enumerate(sorted_sizes)}
        
        # Perform k-means clustering
        try:
            X = np.array(unique_sizes).reshape(-1, 1)
            n_clusters = min(self.n_header_levels, len(unique_sizes))
            
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42,
                n_init=10,
            ).fit(X)
            
            # Map clusters to levels by size (largest = H1)
            cluster_centers = [
                (i, kmeans.cluster_centers_[i][0]) 
                for i in range(n_clusters)
            ]
            cluster_centers.sort(key=lambda x: -x[1])  # Descending by size
            
            cluster_to_level = {
                cluster: level + 1 
                for level, (cluster, _) in enumerate(cluster_centers)
            }
            
            # Map each unique size to its level
            size_to_level = {}
            for size in unique_sizes:
                cluster = kmeans.predict([[size]])[0]
                size_to_level[size] = cluster_to_level[cluster]
            
            logger.debug(
                "header_levels_clustered",
                mapping=size_to_level,
            )
            
            return size_to_level
            
        except Exception as e:
            logger.warning("clustering_failed", error=str(e))
            return {}


def classify_headers(
    spans: list[SpanInfo],
    analysis: FontAnalysis,
    use_clustering: bool = True,
    font_tolerance: float = 2.0,
) -> list[ClassifiedSpan]:
    """
    Convenience function to classify spans into headers and body text.
    
    Args:
        spans: List of SpanInfo from font analysis.
        analysis: FontAnalysis with body size and threshold.
        use_clustering: Whether to use k-means for header levels.
        font_tolerance: Minimum size difference (pts) to be considered a header.
            Higher values = fewer headers, less splitting.
            Lower values = more headers, more granular tree.
        
    Returns:
        List of ClassifiedSpan with roles assigned.
        
    Example:
        analysis, spans = analyze_font_histogram("doc.pdf")
        classified = classify_headers(spans, analysis)
        headers = [s for s in classified if s.role == "header"]
    """
    classifier = HeaderClassifier(
        use_clustering=use_clustering, 
        font_tolerance=font_tolerance,
    )
    return classifier.classify_spans(spans, analysis)
