"""
OCR Fallback - TIER 3: For Scanned/Image-Only PDFs

When the document contains no extractable text (scanned PDFs, image-only),
this module applies Vision Language Model (VLM) OCR to generate a text layer.

Primary path: Gemini / Anthropic / OpenAI vision via ``complete_with_image``.
Legacy fallback: pytesseract (only if no VLM provider is configured).

Dependencies (primary):
- PyMuPDF (fitz) — renders PDF pages to images (already a core dependency)
- Any configured VLM provider (Gemini, Anthropic, OpenAI)

Dependencies (legacy fallback):
- pytesseract (OCR engine wrapper)
- pdf2image (PDF to image conversion)
- Tesseract-OCR installed on system
"""

from __future__ import annotations

from pathlib import Path

import structlog

from rnsr.exceptions import OCRError
from rnsr.models import DocumentNode, DocumentTree

logger = structlog.get_logger(__name__)

_VLM_TRANSCRIPTION_PROMPT = (
    "You are a document OCR assistant. Transcribe ALL visible text in this "
    "page image exactly as it appears. Preserve the original layout, line "
    "breaks, paragraph structure, headings, tables, lists, and any other "
    "formatting. Output ONLY the transcribed text — no commentary, no "
    "descriptions of images or logos, no preamble."
)


def check_ocr_available() -> bool:
    """
    Check if any OCR method is available (VLM preferred, tesseract as fallback).

    Returns:
        True if at least one OCR path is usable.
    """
    if _vlm_available():
        return True
    return _tesseract_available()


def _vlm_available() -> bool:
    """True when a VLM provider with ``complete_with_image`` is reachable."""
    try:
        from rnsr.llm import get_llm, LLMProvider
        llm = get_llm(provider=LLMProvider.GEMINI, enable_fallback=True)
        return hasattr(llm, "complete_with_image")
    except Exception:
        return False


def _tesseract_available() -> bool:
    """True when pytesseract + Tesseract binary are installed."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def try_ocr_ingestion(pdf_path: Path | str) -> DocumentTree:
    """
    TIER 3 Fallback: Use VLM (or tesseract) for scanned/image-only PDFs.

    Strategy:
    1. Render each page to a PNG with PyMuPDF.
    2. Send each page image to the configured VLM for transcription.
    3. If VLM is unavailable or fails, fall back to tesseract.
    4. Build a page-based document tree from the resulting text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocumentTree from OCR text.

    Raises:
        OCRError: If all OCR methods fail.
    """
    pdf_path = Path(pdf_path)
    logger.info("using_ocr_fallback", path=str(pdf_path))

    try:
        page_texts = _vlm_ocr(pdf_path)
    except Exception as vlm_err:
        logger.warning(
            "vlm_ocr_failed_trying_tesseract",
            path=str(pdf_path),
            error=str(vlm_err),
        )
        try:
            page_texts = _tesseract_ocr(pdf_path)
        except Exception as tess_err:
            raise OCRError(
                f"All OCR methods failed for {pdf_path.name}. "
                f"VLM error: {vlm_err}  |  Tesseract error: {tess_err}"
            ) from tess_err

    full_text = "\n\n".join(page_texts)
    if not full_text.strip():
        logger.warning("ocr_no_text_found", path=str(pdf_path))
        root = DocumentNode(id="root", level=0, header="Document")
        return DocumentTree(
            title="Empty OCR Result",
            root=root,
            total_nodes=1,
            ingestion_tier=3,
            ingestion_method="vlm_ocr",
        )

    return _build_tree_from_ocr(page_texts, pdf_path.stem)


# ── VLM-based OCR ───────────────────────────────────────────────────────────


def _vlm_ocr(pdf_path: Path) -> list[str]:
    """Render each PDF page to an image and transcribe via VLM."""
    import fitz
    from rnsr.llm import get_llm, LLMProvider

    llm = get_llm(provider=LLMProvider.GEMINI, enable_fallback=True)

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    logger.info("vlm_ocr_start", path=str(pdf_path), pages=page_count)

    page_texts: list[str] = []
    for page_num in range(page_count):
        page = doc[page_num]
        # 300 DPI render (scale factor = 300/72 ≈ 4.17)
        pix = page.get_pixmap(dpi=300)
        image_bytes = pix.tobytes("png")

        try:
            text = str(
                llm.complete_with_image(_VLM_TRANSCRIPTION_PROMPT, image_bytes)
            ).strip()
        except Exception as e:
            logger.warning(
                "vlm_page_failed", page=page_num, error=str(e),
            )
            text = ""

        page_texts.append(text)
        logger.debug(
            "vlm_page_done", page=page_num + 1, total=page_count,
            chars=len(text),
        )

    doc.close()
    logger.info(
        "vlm_ocr_complete",
        path=str(pdf_path),
        pages=page_count,
        total_chars=sum(len(t) for t in page_texts),
    )
    return page_texts


# ── Legacy tesseract OCR ────────────────────────────────────────────────────


def _tesseract_ocr(pdf_path: Path) -> list[str]:
    """Legacy fallback: pdf2image + pytesseract."""
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path, dpi=300)
    logger.info("tesseract_ocr_start", pages=len(images))

    page_texts: list[str] = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        page_texts.append(text)
        logger.debug("tesseract_page_done", page=i + 1)

    return page_texts


# ── Tree builder ────────────────────────────────────────────────────────────


def _build_tree_from_ocr(
    page_texts: list[str],
    title: str,
) -> DocumentTree:
    """
    Build a document tree from OCR output.

    Creates a simple page-based structure since OCR
    doesn't preserve font information.
    """
    root = DocumentNode(
        id="root",
        level=0,
        header=title,
    )

    for page_num, text in enumerate(page_texts, 1):
        text = text.strip()
        if not text:
            continue

        section = DocumentNode(
            id=f"page_{page_num:03d}",
            level=1,
            header=f"Page {page_num}",
            content=text,
            page_num=page_num - 1,  # 0-indexed
        )
        root.children.append(section)

    return DocumentTree(
        title=title,
        root=root,
        total_nodes=len(root.children) + 1,
        ingestion_tier=3,
        ingestion_method="vlm_ocr",
    )


def hybrid_extract_pages(pdf_path: Path | str) -> list[str]:
    """Extract text per page, using VLM OCR for pages with no extractable text.

    This handles mixed PDFs (some pages are text, some are scanned images)
    by running VLM OCR only on the blank pages and using PyMuPDF text
    extraction for pages that have embedded text.

    Returns:
        List of text strings, one per page.
    """
    import fitz

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    page_count = len(doc)

    page_texts: list[str] = []
    blank_indices: list[int] = []

    for i in range(page_count):
        text = doc[i].get_text().strip()
        if text:
            page_texts.append(text)
        else:
            page_texts.append("")
            blank_indices.append(i)

    if not blank_indices:
        doc.close()
        return page_texts

    logger.info(
        "hybrid_ocr_needed",
        path=str(pdf_path),
        total_pages=page_count,
        blank_pages=len(blank_indices),
    )

    try:
        from rnsr.llm import get_llm, LLMProvider

        llm = get_llm(provider=LLMProvider.GEMINI, enable_fallback=True)

        for idx in blank_indices:
            page = doc[idx]
            pix = page.get_pixmap(dpi=300)
            image_bytes = pix.tobytes("png")
            try:
                text = str(
                    llm.complete_with_image(_VLM_TRANSCRIPTION_PROMPT, image_bytes)
                ).strip()
            except Exception as e:
                logger.warning("hybrid_ocr_page_failed", page=idx, error=str(e))
                text = ""
            page_texts[idx] = text
            logger.debug("hybrid_ocr_page_done", page=idx + 1, chars=len(text))
    except Exception as e:
        logger.warning("hybrid_ocr_vlm_unavailable", error=str(e))

    doc.close()
    return page_texts


def has_extractable_text(pdf_path: Path | str) -> bool:
    """
    Check if a PDF has extractable text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        True if text can be extracted, False if OCR is needed.
    """
    import fitz

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    for page in doc:
        text = str(page.get_text()).strip()
        if text:
            doc.close()
            return True

    doc.close()
    return False
