"""
VLM Structure Extraction - Document structure detection via Vision Language Models.

Replaces LayoutLM (Tier 1b) with a Gemini VLM call for complex-layout PDFs.
Sends page images to the VLM and asks it to identify the hierarchical document
structure (headers, sections, body text), returning a nested DocumentTree.

Advantages over LayoutLM:
- No 1.2 GB local GPU model to load
- No MPS / CUDA thread-safety issues
- Produces nested hierarchy (LayoutLM only gave flat header/body labels)
- Trivially parallelisable across pages (API-bound, not GPU-bound)
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import structlog

from rnsr.models import DocumentNode, DocumentTree, IngestionResult

logger = structlog.get_logger(__name__)

_VLM_STRUCTURE_PROMPT = """\
You are a document structure analyser.  Examine this document page image and \
identify every section header and its associated body text.

Return a JSON array.  Each element must be an object with exactly these keys:
  "header"  – the section heading text (use "" if the text has no heading)
  "level"   – integer nesting depth: 1 = top-level heading, 2 = sub-heading, 3 = sub-sub-heading, 0 = body text with no heading
  "content" – the body text that belongs under this heading (verbatim from the page)

Rules:
- Preserve the reading order of the page.
- Include ALL visible text; do not summarise or omit anything.
- Tables should be transcribed row-by-row inside "content".
- If the page is a single block of body text with no headings, return one \
element with level 0 and an empty header.
- Output ONLY the JSON array – no markdown fences, no commentary.
"""


def _vlm_workers() -> int:
    return max(1, int(os.environ.get("RNSR_VLM_STRUCTURE_WORKERS", "4")))


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array from VLM output, stripping markdown fences if present."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\n?", text)
    if fence:
        text = text[fence.end():]
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        bracket = text.find("[")
        if bracket >= 0:
            parsed = json.loads(text[bracket:])
        else:
            raise

    if not isinstance(parsed, list):
        raise ValueError("VLM response is not a JSON array")
    return parsed


def extract_page_structure(
    llm: Any,
    page_image_bytes: bytes,
    page_num: int,
) -> list[dict[str, Any]]:
    """Send a single page image to the VLM and return structured sections.

    Returns a list of ``{"header": str, "level": int, "content": str}`` dicts
    in reading order.
    """
    try:
        raw = str(llm.complete_with_image(_VLM_STRUCTURE_PROMPT, page_image_bytes)).strip()
        sections = _extract_json_array(raw)
        logger.debug(
            "vlm_structure_page_done",
            page=page_num + 1,
            sections=len(sections),
        )
        return sections
    except Exception as e:
        logger.warning("vlm_structure_page_failed", page=page_num, error=str(e))
        return []


def vlm_structure_ingest(
    pdf_path: Path | str,
    warnings: list[str],
    stats: dict[str, Any],
) -> IngestionResult | None:
    """Full Tier 1b replacement: VLM-based structure detection for complex PDFs.

    1. Renders each page at 150 DPI.
    2. Sends pages to Gemini in parallel via ThreadPoolExecutor.
    3. Builds a nested DocumentTree from the JSON responses.
    """
    import fitz
    from rnsr.llm import get_llm, LLMProvider

    pdf_path = Path(pdf_path)
    logger.debug("trying_vlm_structure", path=str(pdf_path))

    try:
        llm = get_llm(provider=LLMProvider.GEMINI, enable_fallback=True)
        if not hasattr(llm, "complete_with_image"):
            logger.warning("vlm_structure_no_multimodal")
            warnings.append("No multimodal VLM available for structure detection")
            return None
    except Exception as e:
        logger.warning("vlm_structure_llm_init_failed", error=str(e))
        warnings.append(f"VLM init failed: {e}")
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning("vlm_structure_pdf_open_failed", error=str(e))
        warnings.append(f"PDF open failed: {e}")
        return None

    page_count = len(doc)

    page_images: list[bytes] = []
    for page_num in range(page_count):
        pix = doc[page_num].get_pixmap(dpi=150)
        page_images.append(pix.tobytes("png"))
    doc.close()

    logger.info(
        "vlm_structure_start",
        path=str(pdf_path),
        pages=page_count,
    )

    # -- Parallel VLM calls across pages ------------------------------------

    all_page_sections: list[list[dict[str, Any]]] = [[] for _ in range(page_count)]

    def _process_page(pn: int) -> tuple[int, list[dict[str, Any]]]:
        return pn, extract_page_structure(llm, page_images[pn], pn)

    workers = min(_vlm_workers(), page_count)
    if workers <= 1:
        for pn in range(page_count):
            _, sections = _process_page(pn)
            all_page_sections[pn] = sections
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_page, pn): pn for pn in range(page_count)}
            for future in as_completed(futures):
                pn, sections = future.result()
                all_page_sections[pn] = sections

    # -- Build nested DocumentTree ------------------------------------------

    tree = _build_tree_from_vlm_sections(all_page_sections, pdf_path.stem)
    tree.ingestion_tier = 1
    tree.ingestion_method = "vlm_structure"
    tree.page_count = page_count

    logger.info(
        "vlm_structure_success",
        path=str(pdf_path),
        nodes=tree.total_nodes,
        pages=page_count,
    )

    from rnsr.ingestion.pipeline import _add_tables_to_result

    result = IngestionResult(
        tree=tree,
        tier_used=1,
        method="vlm_structure",
        warnings=warnings,
        stats=stats,
    )
    return _add_tables_to_result(result)


def _build_tree_from_vlm_sections(
    all_page_sections: list[list[dict[str, Any]]],
    title: str,
) -> DocumentTree:
    """Convert per-page VLM section lists into a nested DocumentTree.

    Sections with level 1 become direct children of root.
    Sections with level 2 nest under the most recent level-1 node.
    Sections with level 3 nest under the most recent level-2 node.
    Sections with level 0 (body with no heading) attach to the current
    deepest open node.
    """
    root = DocumentNode(id="root", level=0, header=title)
    node_count = 1
    section_idx = 0

    current_l1: DocumentNode | None = None
    current_l2: DocumentNode | None = None

    for page_num, page_sections in enumerate(all_page_sections):
        for sec in page_sections:
            header = (sec.get("header") or "").strip()
            content = (sec.get("content") or "").strip()
            level = sec.get("level", 0)

            if not header and not content:
                continue

            section_idx += 1
            node_id = f"section_{section_idx}"

            if level == 0:
                # Body text with no heading — attach to deepest open node
                target = current_l2 or current_l1 or root
                if target.content:
                    target.content += "\n\n" + content
                else:
                    target.content = content
                continue

            node = DocumentNode(
                id=node_id,
                level=level,
                header=header,
                content=content,
                page_num=page_num,
            )
            node_count += 1

            if level == 1:
                root.children.append(node)
                current_l1 = node
                current_l2 = None
            elif level == 2:
                if current_l1 is not None:
                    current_l1.children.append(node)
                else:
                    root.children.append(node)
                current_l2 = node
            elif level >= 3:
                if current_l2 is not None:
                    current_l2.children.append(node)
                elif current_l1 is not None:
                    current_l1.children.append(node)
                else:
                    root.children.append(node)

    # Edge case: VLM returned nothing useful across all pages
    if not root.children and not root.content:
        root.content = "(No structure detected)"

    return DocumentTree(
        title=title,
        root=root,
        total_nodes=node_count,
    )
