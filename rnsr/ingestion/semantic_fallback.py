"""
Semantic Fallback - TIER 2: For Flat Text Documents

When the Font Histogram Analyzer detects no font variance (flat text),
this module uses LlamaIndex's SemanticSplitterNodeParser to generate
"synthetic" sections based on embedding shifts.

Use this fallback when:
- Document has uniform font size throughout
- No headers can be detected via font analysis
- Document is machine-generated with no formatting

Enhancement: Before semantic splitting, we try PATTERN-BASED header detection
to find textual markers like "1. HEADING", "ARTICLE I:", "Section 1:", etc.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import structlog

from rnsr.models import DocumentNode, DocumentTree

logger = structlog.get_logger(__name__)


# =============================================================================
# Pattern-Based Header Detection (before semantic fallback)
# =============================================================================

# Patterns that indicate a section header in legal/contract documents
# Each tuple: (pattern, level_override or None, header_group_indices)
HEADER_PATTERNS = [
    # Major sections: "1. PARTIES AND RECITALS" - must have capital word after dot
    (r'^(\d{1,2})[\.\)]\s+([A-Z][A-Z\s]{2,}(?:[A-Za-z\s]*)?)$', 1),
    # Sub-sections: "1.1 Service Provider" or "2.1.1 Details"
    (r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][A-Za-z\s]+)', 2),
    # Roman numerals: "I. HEADING" or "II. HEADING"
    (r'^([IVXLC]{1,4})[\.\)]\s+([A-Z][A-Z\s]{2,}(?:[A-Za-z\s]*)?)$', 1),
    # ARTICLE format: "ARTICLE I: Title" or "ARTICLE 1"
    (r'^ARTICLE\s+([IVXLC\d]+)[\.:]\s*(.*)$', 1),
    # Section format: "Section 1:" or "SECTION 1."
    (r'^[Ss][Ee][Cc][Tt][Ii][Oo][Nn]\s+(\d+)[\.:]\s*(.*)$', 1),
    # Exhibit/Appendix: "EXHIBIT A: TECHNICAL SPECIFICATIONS" 
    (r'^(EXHIBIT|APPENDIX)\s+([A-Z\d]+)[\.:]\s*(.*)$', 1),
    # All caps line (minimum 10 chars, standalone, typically a header)
    (r'^([A-Z][A-Z\s]{10,})$', 1),
    # Markdown-style headers - level based on # count
    (r'^(#{1,3})\s+(.+)$', None),  # Level determined by # count
]


def _try_pattern_based_headers(text: str, title: str) -> DocumentTree | None:
    """
    Try to detect section headers using regex patterns.
    
    This is more accurate than semantic splitting for documents that have
    textual markers like "1. INTRODUCTION" even without font variance.
    
    Returns:
        DocumentTree if headers were found, None otherwise.
    """
    lines = text.split('\n')
    sections = []
    current_section = {"header": None, "content": [], "level": 1}
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            current_section["content"].append("")
            continue
        
        # Check if this line matches a header pattern
        is_header = False
        matched_header = None
        matched_level = 1
        
        for pattern, level_override in HEADER_PATTERNS:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                # Extract the header text using the original line
                matched_header = line_stripped
                
                # Clean up markdown prefixes (## Header -> Header)
                if matched_header.startswith('#'):
                    matched_header = matched_header.lstrip('#').strip()
                
                # Clean up the header
                matched_header = matched_header.strip()
                if len(matched_header) < 4:  # Too short to be a header
                    continue
                    
                # Avoid false positives: line must be short enough to be a header
                if len(matched_header) > 100:
                    continue
                    
                is_header = True
                
                # Determine level
                if level_override is not None:
                    matched_level = level_override
                elif line_stripped.startswith('#'):
                    # Markdown: count # for level
                    matched_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                break
        
        if is_header and matched_header:
            # Save current section if it has content
            if current_section["header"]:
                sections.append(current_section)
            
            # Start new section
            current_section = {
                "header": matched_header,
                "content": [],
                "level": matched_level,
            }
        else:
            current_section["content"].append(line)
    
    # Don't forget the last section
    if current_section["header"]:
        sections.append(current_section)
    
    # Need at least 3 sections for this to be useful
    if len(sections) < 3:
        logger.debug("pattern_based_headers_insufficient", found=len(sections))
        return None
    
    logger.info("pattern_based_headers_detected", count=len(sections))
    
    # Build hierarchical tree from detected sections
    root = DocumentNode(id="root", level=0, header=title)
    total_nodes = 1
    
    # Track the last node at each level for proper nesting
    level_stack: list[DocumentNode] = [root]
    
    for i, sec in enumerate(sections):
        content = '\n'.join(sec["content"]).strip()
        node = DocumentNode(
            id=f"section_{i}",
            level=sec["level"],
            header=sec["header"],
            content=content,
        )
        total_nodes += 1
        
        # Find the right parent: the most recent node with a lower level
        while len(level_stack) > 1 and level_stack[-1].level >= sec["level"]:
            level_stack.pop()
        
        # Add this node as a child of the appropriate parent
        parent = level_stack[-1]
        parent.children.append(node)
        
        # Push this node onto the stack in case it has children
        level_stack.append(node)
    
    return DocumentTree(
        title=title,
        root=root,
        total_nodes=total_nodes,
        ingestion_tier=2,
        ingestion_method="pattern_based_headers",
    )


def extract_raw_text(pdf_path: Path | str) -> str:
    """
    Extract all text from a PDF as a single string.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Full text content of the document.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    # get_text() returns str when called with no args or "text"
    full_text = "\n\n".join(str(page.get_text()) for page in doc)
    doc.close()
    
    return full_text


def try_semantic_splitter_ingestion(
    pdf_path: Path | str,
    embed_provider: str | None = None,
) -> DocumentTree:
    """
    TIER 2 Fallback: Use semantic splitting for flat text documents.
    
    When Font Histogram detects no font variance, this method:
    1. First tries PATTERN-BASED header detection (regex for "1. HEADING", etc.)
    2. If that fails, falls back to embedding-based semantic splitting
    3. Generates synthetic section headers as last resort
    
    Args:
        pdf_path: Path to the PDF file.
        embed_provider: Embedding provider ("openai", "gemini", or None for auto).
        
    Returns:
        DocumentTree with detected or synthetic sections.
    """
    pdf_path = Path(pdf_path)
    
    logger.info("using_semantic_splitter", path=str(pdf_path))
    
    # Extract raw text
    full_text = extract_raw_text(pdf_path)
    
    if not full_text.strip():
        logger.warning("no_text_extracted", path=str(pdf_path))
        # Return minimal tree
        root = DocumentNode(id="root", level=0, header="Document")
        return DocumentTree(
            title="Empty Document",
            root=root,
            total_nodes=1,
            ingestion_tier=2,
            ingestion_method="semantic_splitter",
        )
    
    # =========================================================================
    # STEP 1: Try pattern-based header detection FIRST
    # This catches "1. PARTIES AND RECITALS", "ARTICLE I:", etc.
    # =========================================================================
    pattern_tree = _try_pattern_based_headers(full_text, pdf_path.stem)
    if pattern_tree:
        logger.info(
            "pattern_based_headers_success",
            sections=pattern_tree.total_nodes - 1,
            path=str(pdf_path),
        )
        return pattern_tree
    
    logger.debug("pattern_based_headers_failed_trying_semantic")
    
    # =========================================================================
    # STEP 2: Fall back to semantic splitting
    # =========================================================================
    # Try to import LlamaIndex components
    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        
        # Get embedding model (supports OpenAI, Gemini, auto-detect)
        embed_model = _get_embedding_model(embed_provider)
        
        # Create semantic splitter
        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=95,
            buffer_size=1,
        )
        
        # Split document
        llama_doc = Document(text=full_text)
        nodes = splitter.get_nodes_from_documents([llama_doc])
        
        logger.info(
            "semantic_split_complete",
            chunks=len(nodes),
        )
        
        # Build tree from semantic chunks
        return _build_tree_from_semantic_nodes(nodes, pdf_path.stem)
        
    except ImportError as e:
        logger.warning(
            "llama_index_not_available",
            error=str(e),
            fallback="simple_chunking",
        )
        # Fall back to simple chunking
        return _simple_chunk_fallback(full_text, pdf_path.stem)


def _get_embedding_model(provider: str | None = None):
    """
    Get embedding model via central rnsr.llm.

    Supports: OpenAI, Gemini, Ollama, auto-detect.

    Args:
        provider: "openai", "gemini", "ollama", or None for auto-detect.

    Returns:
        LlamaIndex-compatible embedding model.
    """
    from rnsr.llm import LLMProvider, get_embed_model
    p = LLMProvider.AUTO if provider is None else LLMProvider(provider.lower())
    return get_embed_model(provider=p)


def _build_tree_from_semantic_nodes(nodes: list, title: str) -> DocumentTree:
    """
    Build a two-level tree structure from semantic splitter nodes.
    
    This creates a hierarchy for plain text to give the navigator agent
    a more meaningful structure to traverse.
    """
    root = DocumentNode(
        id="root",
        level=0,
        header=title,
    )
    
    logger.info("generating_synthetic_headers", count=len(nodes))
    
    # Helper to generate header (wrapped for potential parallelization later)
    # For now, we add logging so the user knows it's not frozen
    def get_header(text: str, index: int) -> str:
        h = _generate_synthetic_header(text, index)
        if h: 
            return h
        return f"Section {index}"

    # For plain text, create a two-level hierarchy.
    group_size = 5  # Segments per group
    if len(nodes) < group_size * 1.5:  # Don't group if it results in tiny groups
        # Create a flat tree if not enough segments for meaningful grouping
        for i, node in enumerate(nodes, 1):
            if i % 5 == 0:
                logger.info("processing_node", current=i, total=len(nodes))
                
            text = node.text.strip()
            synthetic_header = get_header(text, i)
            section = DocumentNode(
                id=f"sec_{i:03d}",
                level=1,
                header=synthetic_header,
                content=text,
            )
            root.children.append(section)
    else:
        # Create parent nodes to add hierarchy
        num_groups = (len(nodes) + group_size - 1) // group_size
        logger.info("processing_groups", total_groups=num_groups)
        
        for i in range(num_groups):
            logger.info("processing_group", current=i+1, total=num_groups)
            
            start_index = i * group_size
            end_index = start_index + group_size
            group_nodes = nodes[start_index:end_index]

            # Use the header of the first node in the group for the parent
            parent_header_text = group_nodes[0].text.strip()
            parent_header = get_header(parent_header_text, i + 1)

            parent_node = DocumentNode(
                id=f"group_{i}",
                level=1,
                header=parent_header,
            )

            for j, node in enumerate(group_nodes):
                text = node.text.strip()
                child_header = f"Paragraph {j + 1}"
                
                child_node = DocumentNode(
                    id=f"sec_{(start_index + j):03d}",
                    level=2,
                    header=child_header,
                    content=text,
                )
                parent_node.children.append(child_node)
            
            root.children.append(parent_node)

    return DocumentTree(
        title=title,
        root=root,
        total_nodes=len(nodes) + 1, # This is an approximation
        ingestion_tier=2,
        ingestion_method="semantic_splitter",
    )


def _simple_chunk_fallback(text: str, title: str, chunk_size: int = 1000) -> DocumentTree:
    """
    Simple chunking fallback when LlamaIndex is not available.
    
    Splits text into fixed-size chunks.
    """
    logger.info("using_simple_chunking", chunk_size=chunk_size)
    
    root = DocumentNode(
        id="root",
        level=0,
        header=title,
    )
    
    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    
    # Group paragraphs into chunks
    current_chunk = ""
    chunk_num = 0

    # Helper to generate header safely
    def get_header(text: str, index: int) -> str:
        h = _generate_synthetic_header(text, index)
        if h: 
            return h
        return f"Section {index}"
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunk_num += 1
                if chunk_num % 5 == 0:
                    logger.info("processing_chunk", current=chunk_num)
                    
                section = DocumentNode(
                    id=f"sec_{chunk_num:03d}",
                    level=1,
                    header=get_header(current_chunk, chunk_num),
                    content=current_chunk,
                )
                root.children.append(section)
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add final chunk
    if current_chunk:
        chunk_num += 1
        section = DocumentNode(
            id=f"sec_{chunk_num:03d}",
            level=1,
            header=get_header(current_chunk, chunk_num),
            content=current_chunk,
        )
        root.children.append(section)
    
    return DocumentTree(
        title=title,
        root=root,
        total_nodes=chunk_num + 1,
        ingestion_tier=2,
        ingestion_method="semantic_splitter",
    )


def _generate_synthetic_header(text: str, section_num: int) -> str:
    """
    Generate a synthetic header from text content using LLM.
    
    Per the research paper (Section 6.3): "For each identified section, 
    we execute an LLM call with prompt: 'Generate a descriptive, 
    hierarchical title for it. Return ONLY the title.'"
    
    Falls back to heuristic extraction if LLM fails.
    """
    # Try LLM-based header generation first
    try:
        header = _generate_header_via_llm(text, section_num)
        if header:
            return header
    except Exception as e:
        logger.debug("llm_header_generation_failed", error=str(e))
    
    # Fallback: heuristic extraction
    return _generate_header_heuristic(text, section_num)


def _generate_header_via_llm(text: str, section_num: int) -> str | None:
    """
    Use LLM to generate a concise, descriptive header for a text section.
    
    This implements the Synthetic Header Generation from Section 4.2.2:
    "The 'Title' of each node in this semantic tree is generated generatively:
    we feed the text of the cluster to a summarization LLM with the prompt 
    'Generate a concise 5-word header for this text section.'"
    """
    from rnsr.llm import get_llm
    
    # Truncate text to avoid token limits (first 1500 chars should be enough for context)
    text_sample = text[:1500] if len(text) > 1500 else text
    
    prompt = f"""Read the following text segment and generate a descriptive, hierarchical title for it.
The title should be concise (3-7 words) and capture the main topic of this section.

Text:
{text_sample}

Return ONLY the title, nothing else. Example format: "Section 3: Liability Limitations" or "Payment Terms and Conditions" """

    try:
        # Use centralized provider with retry logic
        llm = get_llm()
        # Note: LlamaIndex LLM.complete() usually returns a CompletionResponse,
        # but our custom Gemini wrapper returns a string. str() handles both.
        response = llm.complete(prompt)
        header = str(response).strip().strip('"').strip("'")
        
        # Validate: should be reasonable length
        if 3 <= len(header) <= 100:
            logger.debug("llm_header_generated", header=header[:50])
            return header
            
    except Exception as e:
        logger.debug("synthetic_header_generation_failed", error=str(e))
    
    return None


def _generate_header_heuristic(text: str, section_num: int) -> str:
    """
    Fallback: Generate header from first sentence/words when LLM unavailable.
    """
    # Get first sentence or first N words
    words = text.split()[:10]
    
    if not words:
        return f"Section {section_num}"
    
    header = " ".join(words)
    
    # Truncate at sentence end if present
    for punct in ".!?":
        if punct in header:
            header = header.split(punct)[0] + punct
            break
    
    # Ensure reasonable length
    if len(header) > 60:
        header = header[:57] + "..."
    
    return header
