"""
Analysis Module -- higher-level analysis tools built on top of RNSR's
extraction and knowledge graph infrastructure.

Includes:
- Contradiction detection within and across documents
- Timeline extraction (re-exported from extraction module)
"""

from rnsr.analysis.contradiction_detector import (
    FactContradiction,
    detect_document_contradictions,
    detect_cross_document_contradictions,
)
from rnsr.extraction.timeline_extractor import (
    TimelineEvent,
    extract_timeline,
    format_timeline,
)

__all__ = [
    "FactContradiction",
    "detect_document_contradictions",
    "detect_cross_document_contradictions",
    "TimelineEvent",
    "extract_timeline",
    "format_timeline",
]
