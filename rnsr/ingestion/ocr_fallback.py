"""
OCR Fallback - TIER 3: For Scanned/Image-Only PDFs

When the document contains no extractable text (scanned PDFs, image-only),
this module applies OCR to generate a text layer, then re-runs analysis.

Use this fallback when:
- PDF contains only images (scanned documents)
- No text can be extracted via PyMuPDF
- Document was scanned without OCR processing

Dependencies:
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


def check_ocr_available() -> bool:
    """
    Check if OCR dependencies are available.
    
    Returns:
        True if pytesseract and pdf2image are importable.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        # Test tesseract is installed
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def try_ocr_ingestion(pdf_path: Path | str) -> DocumentTree:
    """
    TIER 3 Fallback: Use OCR for scanned/image-only PDFs.
    
    This method:
    1. Converts PDF pages to images
    2. Applies Tesseract OCR to each page
    3. Builds a document tree from OCR output
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        DocumentTree from OCR text.
        
    Raises:
        OCRError: If OCR fails or dependencies not available.
    """
    pdf_path = Path(pdf_path)
    
    logger.info("using_ocr_fallback", path=str(pdf_path))
    
    # Check dependencies
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        raise OCRError(
            f"OCR dependencies not available: {e}. "
            "Install with: pip install pytesseract pdf2image"
        ) from e
    
    try:
        # Convert PDF pages to images
        logger.debug("converting_pdf_to_images", path=str(pdf_path))
        images = convert_from_path(pdf_path, dpi=300)
        
        logger.info("pdf_converted", pages=len(images))
        
        # OCR each page
        ocr_texts: list[str] = []
        for i, image in enumerate(images):
            logger.debug("processing_page_ocr", page=i)
            text = pytesseract.image_to_string(image)
            ocr_texts.append(text)
        
        # Combine and build tree
        full_text = "\n\n".join(ocr_texts)
        
        if not full_text.strip():
            logger.warning("ocr_no_text_found", path=str(pdf_path))
            root = DocumentNode(id="root", level=0, header="Document")
            return DocumentTree(
                title="Empty OCR Result",
                root=root,
                total_nodes=1,
                ingestion_tier=3,
                ingestion_method="ocr",
            )
        
        # Build tree from OCR text
        return _build_tree_from_ocr(ocr_texts, pdf_path.stem)
        
    except Exception as e:
        raise OCRError(f"OCR processing failed: {e}") from e


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
        
        # Create a section per page
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
        ingestion_method="ocr",
    )


def has_extractable_text(pdf_path: Path | str) -> bool:
    """
    Check if a PDF has extractable text.
    
    Tries PyMuPDF first, then pdfplumber. Some PDFs (e.g. Google Docs
    "Print to PDF") expose text to one extractor but not the other.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        True if text can be extracted, False if OCR is needed.
    """
    pdf_path = Path(pdf_path)
    
    # 1. PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            text = str(page.get_text()).strip()
            if text:
                doc.close()
                return True
        doc.close()
    except Exception:
        pass
    
    # 2. pdfplumber (e.g. Google Docs "Print to PDF" often works here)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as doc:
            for page in doc.pages:
                if page is None:
                    continue
                text = (page.extract_text() or "").strip()
                if text:
                    return True
    except Exception:
        pass
    
    return False
