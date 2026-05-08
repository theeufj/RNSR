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

import os
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
# Each tuple: (pattern, level_override, requires_uppercase)
# requires_uppercase: when True, the matched text must be verified as
# actually uppercase after matching (to counteract re.IGNORECASE).
HEADER_PATTERNS = [
    # Major sections: "1. PARTIES AND RECITALS" - must have capital word after dot
    (r'^(\d{1,2})[\.\)]\s+([A-Z][A-Z\s]{2,}(?:[A-Za-z\s]*)?)$', 1, False),
    # Sub-sections: "1.1 Service Provider" or "2.1.1 Details"
    (r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][A-Za-z\s]+)', 2, False),
    # Roman numerals: "I. HEADING" or "II. HEADING"
    (r'^([IVXLC]{1,4})[\.\)]\s+([A-Z][A-Z\s]{2,}(?:[A-Za-z\s]*)?)$', 1, False),
    # ARTICLE format: "ARTICLE I: Title" or "ARTICLE 1"
    (r'^ARTICLE\s+([IVXLC\d]+)[\.:]\s*(.*)$', 1, False),
    # Section format: "Section 1:" or "SECTION 1."
    (r'^[Ss][Ee][Cc][Tt][Ii][Oo][Nn]\s+(\d+)[\.:]\s*(.*)$', 1, False),
    # Exhibit/Appendix: "EXHIBIT A: TECHNICAL SPECIFICATIONS"
    (r'^(EXHIBIT|APPENDIX)\s+([A-Z\d]+)[\.:]\s*(.*)$', 1, False),
    # All caps line (minimum 10 chars, standalone, typically a header).
    # Needs requires_uppercase=True because we apply re.IGNORECASE globally.
    (r'^([A-Z][A-Z\s]{10,})$', 1, True),
    # Markdown-style headers - level based on # count
    (r'^(#{1,3})\s+(.+)$', None, False),
]

# Lines that should never be treated as section headers even if they
# match a pattern (common salutations, closings, generic phrases).
_HEADER_BLACKLIST = re.compile(
    r"^(?:yours? (?:faithfully|sincerely|truly)|"
    r"dear\b|kind regards|best regards|"
    r"thank(?:s|ing) you|regards|"
    r"mr |mrs |ms |dr |prof )",
    re.IGNORECASE,
)


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
    current_section: dict = {"header": None, "content": [], "level": 1}
    preamble_lines: list[str] = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            current_section["content"].append("")
            continue

        # Check if this line matches a header pattern
        is_header = False
        matched_header = None
        matched_level = 1

        # Skip blacklisted lines (salutations, closings) before any pattern check
        if _HEADER_BLACKLIST.match(line_stripped):
            current_section["content"].append(line)
            continue

        for pattern, level_override, requires_uppercase in HEADER_PATTERNS:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                # For patterns that require uppercase (e.g. "ALL CAPS" detector),
                # verify the original text is actually uppercase to prevent
                # false positives like "Yours faithfully" or "James Fowler".
                if requires_uppercase:
                    alpha_chars = [c for c in line_stripped if c.isalpha()]
                    if not alpha_chars or not all(c.isupper() for c in alpha_chars):
                        continue

                matched_header = line_stripped

                if matched_header.startswith('#'):
                    matched_header = matched_header.lstrip('#').strip()

                matched_header = matched_header.strip()
                if len(matched_header) < 4:
                    continue
                if len(matched_header) > 100:
                    continue

                is_header = True

                if level_override is not None:
                    matched_level = level_override
                elif line_stripped.startswith('#'):
                    matched_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                break

        if is_header and matched_header:
            # Save current section if it has content
            if current_section["header"]:
                sections.append(current_section)
            elif current_section["content"]:
                # Preserve content that appeared before any header (the
                # letterhead / preamble).  It will be attached to the root
                # node so dates, references, and addresses are not lost.
                preamble_lines = list(current_section["content"])

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
    
    # Build hierarchical tree from detected sections.
    # Attach any preamble (text before the first header) to the root so
    # letterhead info like dates, references, and addresses is preserved.
    preamble_text = '\n'.join(preamble_lines).strip() if preamble_lines else ""
    root = DocumentNode(id="root", level=0, header=title, content=preamble_text)
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
    
    # Post-process: merge tiny adjacent sections into their siblings.
    # Without this, form-style documents fragment titles like
    # "WITNESS SUMMONS" / "TO PRODUCE A RECORD OR THING" / "FORM 48"
    # into separate 15-char sections that lose their semantic connection.
    _merge_tiny_siblings(root)

    # Recount after merges
    total_nodes = _count_nodes(root)

    return DocumentTree(
        title=title,
        root=root,
        total_nodes=total_nodes,
        ingestion_tier=2,
        ingestion_method="pattern_based_headers",
    )


_MERGE_SIBLING_THRESHOLD = 80


def _merge_tiny_siblings(node: DocumentNode) -> None:
    """Merge adjacent tiny leaf sections into their next sibling.

    When a document has header-only sections (e.g. "WITNESS SUMMONS" with
    15 chars, "TO PRODUCE A RECORD OR THING" with 28 chars), merge them
    into the next substantial sibling so the combined content stays together.
    """
    for child in node.children:
        _merge_tiny_siblings(child)

    if len(node.children) < 2:
        return

    merged: list[DocumentNode] = []
    pending_headers: list[str] = []

    for child in node.children:
        total_chars = len(child.header or "") + len(child.content or "")
        is_leaf = not child.children

        if is_leaf and total_chars < _MERGE_SIBLING_THRESHOLD:
            pending_headers.append(child.header)
            if child.content:
                pending_headers.append(child.content)
        else:
            if pending_headers:
                prefix = "\n".join(pending_headers)
                child.content = f"{prefix}\n\n{child.content}" if child.content else prefix
                child.header = pending_headers[0] + " " + child.header
                pending_headers = []
            merged.append(child)

    # If trailing tiny sections remain, attach them to the last kept child
    if pending_headers and merged:
        last = merged[-1]
        suffix = "\n".join(pending_headers)
        last.content = f"{last.content}\n\n{suffix}" if last.content else suffix
    elif pending_headers:
        # All children were tiny -- create one merged node
        combined = DocumentNode(
            id=node.children[0].id,
            level=node.children[0].level,
            header=" | ".join(pending_headers),
            content="\n".join(pending_headers),
        )
        merged.append(combined)

    node.children = merged


def _count_nodes(node: DocumentNode) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def extract_raw_text(pdf_path: Path | str) -> str:
    """
    Extract all text from a PDF as a single string.

    Uses hybrid OCR: pages with embedded text are extracted normally,
    blank (scanned) pages are sent to VLM OCR.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Full text content of the document.
    """
    from rnsr.ingestion.ocr_fallback import hybrid_extract_pages

    pdf_path = Path(pdf_path)
    page_texts = hybrid_extract_pages(pdf_path)
    return "\n\n".join(page_texts)


_VLM_HEADING_PROMPT = """You are a document structure analyst. Given a sample of document text, identify ALL lines that serve as section headings or titles.

Rules:
- Return ONLY the exact heading text, one per line, in the order they appear.
- Include ALL levels of headings (main sections, sub-sections, etc.).
- Do NOT include body text, sentences, or descriptions.
- Do NOT modify the heading text — return it exactly as it appears.
- If there are no clear headings, return the single word: NONE

Document text sample:
---
{sample}
---

Headings (one per line):"""


def _try_vlm_heading_discovery(
    full_text: str,
    title: str,
) -> DocumentTree | None:
    """Use a VLM to discover section headings from the document text.

    Instead of relying on hardcoded regex patterns, this asks the LLM to
    read a sample of the text and identify which lines are headings. The
    discovered headings are then used to split the full text into sections.

    Returns ``None`` if the VLM is unavailable or finds no headings.
    """
    try:
        from rnsr.llm import get_llm, LLMProvider
    except Exception:
        return None

    sample_size = min(len(full_text), 6000)
    sample = full_text[:sample_size]

    try:
        llm = get_llm(provider=LLMProvider.GEMINI, enable_fallback=True)
        prompt = _VLM_HEADING_PROMPT.format(sample=sample)
        response = str(llm.complete(prompt)).strip()
    except Exception as e:
        logger.debug("vlm_heading_discovery_failed", error=str(e))
        return None

    if not response or response.upper() == "NONE":
        return None

    candidate_headings = [
        line.strip() for line in response.split("\n")
        if line.strip() and len(line.strip()) >= 2
    ]

    if len(candidate_headings) < 2:
        return None

    heading_set = set(candidate_headings)

    lines = full_text.split("\n")
    sections: list[dict] = []
    current_content_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped in heading_set:
            if sections or current_content_lines:
                if sections:
                    sections[-1]["content"] = "\n".join(current_content_lines).strip()
                current_content_lines = []
            sections.append({"header": stripped, "content": "", "level": 1})
        else:
            current_content_lines.append(line)

    if sections:
        sections[-1]["content"] = "\n".join(current_content_lines).strip()

    if len(sections) < 2:
        return None

    preamble_text = ""
    if not sections[0].get("content") and current_content_lines:
        pass
    first_section_idx = next(
        (i for i, line in enumerate(lines) if line.strip() in heading_set),
        None,
    )
    if first_section_idx and first_section_idx > 0:
        preamble_text = "\n".join(lines[:first_section_idx]).strip()

    root = DocumentNode(
        id="root", level=0, header=title, content=preamble_text or ""
    )
    for i, sec in enumerate(sections):
        child = DocumentNode(
            id=f"section_{i}",
            level=sec["level"],
            header=sec["header"],
            content=sec["content"] or "",
        )
        root.children.append(child)

    total_nodes = 1 + len(root.children)
    logger.info(
        "vlm_heading_discovery_success",
        path=title,
        sections=len(sections),
        headings_found=len(candidate_headings),
    )
    return DocumentTree(
        title=title,
        root=root,
        total_nodes=total_nodes,
        ingestion_tier=2,
        ingestion_method="vlm_heading_discovery",
    )


def _try_page_level_split(
    pdf_path: Path | str,
) -> DocumentTree | None:
    """Split a multi-page PDF into one node per page.

    This is a better intermediate fallback than pure size-based chunking
    because it preserves natural page boundaries and attaches ``page_num``
    metadata to each node.

    Returns ``None`` for single-page PDFs (nothing to split).
    """
    from rnsr.ingestion.ocr_fallback import hybrid_extract_pages

    page_texts = hybrid_extract_pages(pdf_path)
    non_empty = [t for t in page_texts if t.strip()]

    if len(non_empty) < 2:
        return None

    title = Path(pdf_path).stem

    root = DocumentNode(id="root", level=0, header=title, content="")
    for i, text in enumerate(page_texts):
        if not text.strip():
            continue
        header = _infer_page_header(text, i + 1)
        child = DocumentNode(
            id=f"page_{i + 1:03d}",
            level=1,
            header=header,
            content=text,
            page_num=i,
        )
        root.children.append(child)

    total_nodes = 1 + len(root.children)
    logger.info(
        "page_level_split_success",
        path=str(pdf_path),
        pages=total_nodes - 1,
    )
    return DocumentTree(
        title=title,
        root=root,
        total_nodes=total_nodes,
        ingestion_tier=2,
        ingestion_method="page_split",
    )


def _infer_page_header(text: str, page_num: int) -> str:
    """Derive a short header from the first meaningful line of a page."""
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) >= 3:
            return stripped[:80]
    return f"Page {page_num}"


def try_semantic_splitter_ingestion(
    pdf_path: Path | str,
    embed_provider: str | None = None,
) -> DocumentTree:
    """
    TIER 2 Fallback: Use semantic splitting for flat text documents.
    
    When Font Histogram detects no font variance, this method:
    1. Pattern-based header detection (regex for "1. HEADING", etc.)
    2. VLM heading discovery (LLM identifies headings from a text sample)
    3. Page-level splitting (1 node per page)
    4. Embedding-based semantic splitting
    5. Size-based chunking as last resort
    
    Args:
        pdf_path: Path to the PDF file.
        embed_provider: Embedding provider ("openai", "gemini", or None for auto).
        
    Returns:
        DocumentTree with detected or synthetic sections.
    """
    pdf_path = Path(pdf_path)
    
    logger.info("using_semantic_splitter", path=str(pdf_path))
    
    # Extract raw text (with hybrid OCR for blank pages)
    full_text = extract_raw_text(pdf_path)
    
    if not full_text.strip():
        logger.warning("no_text_extracted", path=str(pdf_path))
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
    
    # =========================================================================
    # STEP 1.5: VLM heading discovery — ask the LLM to identify headings
    # instead of relying on hardcoded regex patterns
    # =========================================================================
    vlm_tree = _try_vlm_heading_discovery(full_text, pdf_path.stem)
    if vlm_tree:
        logger.info(
            "vlm_heading_discovery_used",
            sections=vlm_tree.total_nodes - 1,
            path=str(pdf_path),
        )
        return vlm_tree

    # =========================================================================
    # STEP 2: Page-level splitting for multi-page PDFs
    # Preserves natural page boundaries instead of arbitrary size-based chunks
    # =========================================================================
    if str(pdf_path).lower().endswith(".pdf"):
        page_tree = _try_page_level_split(pdf_path)
        if page_tree:
            return page_tree
    
    logger.debug("all_heading_methods_failed_trying_semantic")
    
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
    Get embedding model with multi-provider support.
    
    Supports: OpenAI, Gemini, auto-detect.
    
    Args:
        provider: "openai", "gemini", or None for auto-detect.
        
    Returns:
        LlamaIndex-compatible embedding model.
    """
    import os
    
    # Auto-detect provider if not specified
    if provider is None:
        if os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError(
                "No embedding API key found. "
                "Set GOOGLE_API_KEY or OPENAI_API_KEY."
            )
    
    provider = provider.lower()
    
    if provider == "gemini":
        try:
            from llama_index.embeddings.gemini import GeminiEmbedding
            
            logger.info("using_gemini_embeddings")
            return GeminiEmbedding(model_name="models/text-embedding-005")
        except ImportError:
            raise ImportError(
                "Gemini embeddings not installed. "
                "Install with: pip install llama-index-embeddings-gemini"
            )
    
    elif provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            logger.info("using_openai_embeddings")
            return OpenAIEmbedding(model="text-embedding-3-small")
        except ImportError:
            raise ImportError(
                "OpenAI embeddings not installed. "
                "Install with: pip install llama-index-embeddings-openai"
            )
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


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

    group_size = 5  # Segments per group
    if len(nodes) < group_size * 1.5:
        flat_items = [(node.text.strip(), i + 1) for i, node in enumerate(nodes)]
        flat_headers = generate_synthetic_headers_batch(flat_items)
        for i, (node, header) in enumerate(zip(nodes, flat_headers), start=1):
            section = DocumentNode(
                id=f"sec_{i:03d}",
                level=1,
                header=header or f"Section {i}",
                content=node.text.strip(),
            )
            root.children.append(section)
    else:
        num_groups = (len(nodes) + group_size - 1) // group_size
        logger.info("processing_groups", total_groups=num_groups)

        group_items: list[tuple[str, int]] = []
        for i in range(num_groups):
            start_index = i * group_size
            group_nodes = nodes[start_index:start_index + group_size]
            group_items.append((group_nodes[0].text.strip(), i + 1))

        group_headers = generate_synthetic_headers_batch(group_items)

        for i, parent_header in enumerate(group_headers):
            start_index = i * group_size
            end_index = start_index + group_size
            group_nodes = nodes[start_index:end_index]

            parent_node = DocumentNode(
                id=f"group_{i}",
                level=1,
                header=parent_header,
            )

            for j, node in enumerate(group_nodes):
                child_node = DocumentNode(
                    id=f"sec_{(start_index + j):03d}",
                    level=2,
                    header=f"Paragraph {j + 1}",
                    content=node.text.strip(),
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
    
    # First pass: split paragraphs into size-bounded chunks (no LLM calls).
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)

    # Second pass: batch-generate headers in parallel.
    header_items = [(c, i + 1) for i, c in enumerate(chunks)]
    headers = generate_synthetic_headers_batch(header_items)

    chunk_num = 0
    for chunk_text, header in zip(chunks, headers):
        chunk_num += 1
        section = DocumentNode(
            id=f"sec_{chunk_num:03d}",
            level=1,
            header=header or f"Section {chunk_num}",
            content=chunk_text,
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


def generate_synthetic_headers_batch(
    items: list[tuple[str, int]],
    *,
    max_workers: int | None = None,
) -> list[str]:
    """Generate synthetic headers for many sections in parallel.

    The single-section LLM call dominates ingestion runtime for large
    PDFs (e.g. a Pfizer 10-K's 350 sections × ~5s/call = ~30 min). The
    LLM call is I/O-bound, so a small thread pool collapses that to
    roughly 30min / max_workers. Each item is `(text, section_num)`;
    results are returned in input order. ``max_workers`` defaults to
    ``RNSR_HEADER_GEN_PARALLELISM`` (env), else 8.
    """
    if not items:
        return []

    if max_workers is None:
        try:
            max_workers = int(os.getenv("RNSR_HEADER_GEN_PARALLELISM", "8"))
        except ValueError:
            max_workers = 8
    max_workers = max(1, max_workers)

    if max_workers == 1 or len(items) == 1:
        return [_generate_synthetic_header(text, idx) for text, idx in items]

    from concurrent.futures import ThreadPoolExecutor

    results: list[str | None] = [None] * len(items)

    def _worker(i: int) -> None:
        text, idx = items[i]
        results[i] = _generate_synthetic_header(text, idx)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_worker, range(len(items))))

    out: list[str] = []
    for i, r in enumerate(results):
        if r is None:
            out.append(_generate_header_heuristic(items[i][0], items[i][1]))
        else:
            out.append(r)
    return out


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
