"""
Vision-Based Retrieval - OCR-Free Document Analysis

This module implements vision-based document retrieval that works directly
on PDF page images without requiring text extraction or OCR.

Inspired by PageIndex's Vision-based Vectorless RAG:
"OCR-free, vision-only RAG with PageIndex's reasoning-native retrieval 
workflow that works directly over PDF page images."

Key Features:
1. Page image extraction from PDFs
2. Vision LLM integration (GPT-4V, Gemini Vision)
3. Page-level navigation using visual reasoning
4. Hybrid text+vision mode for charts/diagrams
5. Image caching for performance

Use Cases:
- Scanned documents where OCR quality is poor
- Documents with complex layouts, charts, or diagrams
- Image-heavy documents (presentations, reports with graphics)
- Documents where visual structure provides context
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VisionConfig:
    """Configuration for vision-based retrieval."""
    
    # Vision model settings
    vision_model: str = "gemini-2.5-flash"  # Model with vision support
    provider: str = "gemini"  # gemini, openai
    
    # Image settings
    image_dpi: int = 150  # DPI for page rendering
    max_image_size: int = 2048  # Max dimension in pixels
    image_format: str = "PNG"  # PNG or JPEG
    
    # Caching
    enable_cache: bool = True
    cache_dir: str = ".rnsr_cache/vision"
    
    # Navigation settings
    max_pages_per_batch: int = 5  # Pages to evaluate per LLM call
    page_selection_threshold: float = 0.3  # Min relevance for selection


# =============================================================================
# Page Image Extractor
# =============================================================================


class PageImageExtractor:
    """
    Extracts page images from PDF documents.
    
    Uses PyMuPDF (fitz) for high-quality page rendering.
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self._cache_dir = Path(config.cache_dir)
        
        if config.enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_page_image(
        self,
        pdf_path: Path | str,
        page_num: int,
    ) -> bytes:
        """
        Extract a single page as an image.
        
        Args:
            pdf_path: Path to PDF file.
            page_num: Page number (0-indexed).
            
        Returns:
            Image bytes (PNG or JPEG).
        """
        pdf_path = Path(pdf_path)
        
        # Check cache
        cache_key = self._get_cache_key(pdf_path, page_num)
        if self.config.enable_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                return cached
        
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")
        
        doc = fitz.open(pdf_path)
        
        if page_num >= len(doc):
            doc.close()
            raise ValueError(f"Page {page_num} does not exist (document has {len(doc)} pages)")
        
        page = doc[page_num]
        
        # Calculate zoom for target DPI
        zoom = self.config.image_dpi / 72  # 72 is default PDF DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Resize if too large
        if max(pix.width, pix.height) > self.config.max_image_size:
            scale = self.config.max_image_size / max(pix.width, pix.height)
            new_width = int(pix.width * scale)
            new_height = int(pix.height * scale)
            
            # Re-render at correct size
            zoom = zoom * scale
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
        
        # Convert to image bytes
        if self.config.image_format.upper() == "PNG":
            image_bytes = pix.tobytes("png")
        else:
            image_bytes = pix.tobytes("jpeg")
        
        doc.close()
        
        # Cache the result
        if self.config.enable_cache:
            self._save_to_cache(cache_key, image_bytes)
        
        logger.debug(
            "page_image_extracted",
            page=page_num,
            width=pix.width,
            height=pix.height,
        )
        
        return image_bytes
    
    def extract_all_pages(
        self,
        pdf_path: Path | str,
        max_pages: int | None = None,
    ) -> list[bytes]:
        """
        Extract all pages as images.
        
        Args:
            pdf_path: Path to PDF file.
            max_pages: Optional limit on number of pages.
            
        Returns:
            List of image bytes.
        """
        pdf_path = Path(pdf_path)
        
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")
        
        doc = fitz.open(pdf_path)
        num_pages = min(len(doc), max_pages) if max_pages else len(doc)
        doc.close()
        
        images = []
        for i in range(num_pages):
            images.append(self.extract_page_image(pdf_path, i))
        
        logger.info("all_pages_extracted", count=len(images))
        return images
    
    def get_page_count(self, pdf_path: Path | str) -> int:
        """Get the number of pages in a PDF."""
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")
        
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    
    def _get_cache_key(self, pdf_path: Path, page_num: int) -> str:
        """Generate a cache key for a page."""
        # Include file modification time for cache invalidation
        mtime = pdf_path.stat().st_mtime
        key_data = f"{pdf_path}:{page_num}:{mtime}:{self.config.image_dpi}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> bytes | None:
        """Load an image from cache."""
        cache_path = self._cache_dir / f"{cache_key}.{self.config.image_format.lower()}"
        if cache_path.exists():
            return cache_path.read_bytes()
        return None
    
    def _save_to_cache(self, cache_key: str, image_bytes: bytes) -> None:
        """Save an image to cache."""
        cache_path = self._cache_dir / f"{cache_key}.{self.config.image_format.lower()}"
        cache_path.write_bytes(image_bytes)


# =============================================================================
# Vision LLM Integration
# =============================================================================


class VisionLLM:
    """
    Vision LLM wrapper supporting multiple providers.
    
    Provides a unified interface for vision-based reasoning.
    """
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self._client: Any = None
    
    def _get_client(self) -> Any:
        """Get or create the vision LLM client."""
        if self._client is not None:
            return self._client
        
        if self.config.provider == "gemini":
            self._client = self._create_gemini_client()
        elif self.config.provider == "openai":
            self._client = self._create_openai_client()
        else:
            raise ValueError(f"Unknown vision provider: {self.config.provider}")
        
        return self._client
    
    def _create_gemini_client(self) -> Any:
        """Create a Gemini vision client."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai not installed. Install with: pip install google-genai")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        return genai.Client(api_key=api_key)
    
    def _create_openai_client(self) -> Any:
        """Create an OpenAI vision client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        return OpenAI(api_key=api_key)
    
    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
    ) -> str:
        """
        Analyze a single image with a prompt.
        
        Args:
            image_bytes: Image as bytes.
            prompt: Question or instruction about the image.
            
        Returns:
            LLM response.
        """
        client = self._get_client()
        
        if self.config.provider == "gemini":
            return self._analyze_gemini(client, image_bytes, prompt)
        elif self.config.provider == "openai":
            return self._analyze_openai(client, image_bytes, prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def analyze_multiple_images(
        self,
        images: list[bytes],
        prompt: str,
    ) -> str:
        """
        Analyze multiple images together.
        
        Args:
            images: List of image bytes.
            prompt: Question about all images.
            
        Returns:
            LLM response.
        """
        client = self._get_client()
        
        if self.config.provider == "gemini":
            return self._analyze_gemini_multi(client, images, prompt)
        elif self.config.provider == "openai":
            return self._analyze_openai_multi(client, images, prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _analyze_gemini(
        self,
        client: Any,
        image_bytes: bytes,
        prompt: str,
    ) -> str:
        """Analyze image using Gemini."""
        from google.genai import types
        
        # Create image part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=f"image/{self.config.image_format.lower()}",
        )
        
        response = client.models.generate_content(
            model=self.config.vision_model,
            contents=[prompt, image_part],
        )
        
        return response.text or ""
    
    def _analyze_gemini_multi(
        self,
        client: Any,
        images: list[bytes],
        prompt: str,
    ) -> str:
        """Analyze multiple images using Gemini."""
        from google.genai import types
        
        # Create content parts
        parts = [prompt]
        for i, img_bytes in enumerate(images):
            parts.append(f"\n\n[Page {i+1}]")
            parts.append(types.Part.from_bytes(
                data=img_bytes,
                mime_type=f"image/{self.config.image_format.lower()}",
            ))
        
        response = client.models.generate_content(
            model=self.config.vision_model,
            contents=parts,
        )
        
        return response.text or ""
    
    def _analyze_openai(
        self,
        client: Any,
        image_bytes: bytes,
        prompt: str,
    ) -> str:
        """Analyze image using OpenAI."""
        # Encode image as base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = f"image/{self.config.image_format.lower()}"
        
        response = client.chat.completions.create(
            model=self.config.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        
        return response.choices[0].message.content or ""
    
    def _analyze_openai_multi(
        self,
        client: Any,
        images: list[bytes],
        prompt: str,
    ) -> str:
        """Analyze multiple images using OpenAI."""
        mime_type = f"image/{self.config.image_format.lower()}"
        
        content = [{"type": "text", "text": prompt}]
        
        for i, img_bytes in enumerate(images):
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "text",
                "text": f"\n[Page {i+1}]",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_image}",
                },
            })
        
        response = client.chat.completions.create(
            model=self.config.vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
        )
        
        return response.choices[0].message.content or ""


# =============================================================================
# Vision-Based Navigator
# =============================================================================


@dataclass
class VisionPage:
    """Represents a page in the vision index."""
    page_num: int
    image_bytes: bytes
    summary: str = ""
    relevance_score: float = 0.0


class VisionNavigator:
    """
    Vision-based document navigator.
    
    Works directly on PDF page images without OCR or text extraction.
    Uses vision LLM to understand and navigate document content.
    """
    
    def __init__(
        self,
        pdf_path: Path | str,
        config: VisionConfig | None = None,
    ):
        self.pdf_path = Path(pdf_path)
        self.config = config or VisionConfig()
        
        # Components
        self.extractor = PageImageExtractor(self.config)
        self.vision_llm = VisionLLM(self.config)
        
        # State
        self.page_count = self.extractor.get_page_count(self.pdf_path)
        self.page_summaries: dict[int, str] = {}
        self.selected_pages: list[int] = []
    
    def navigate(
        self,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Navigate document visually to answer a question.
        
        Args:
            question: The user's question.
            metadata: Optional metadata (e.g., multiple choice options).
            
        Returns:
            Dict with answer, confidence, selected_pages.
        """
        logger.info(
            "vision_navigation_started",
            question=question[:100],
            pages=self.page_count,
        )
        
        trace = []
        
        # Phase 1: Page selection using vision
        relevant_pages = self._select_relevant_pages(question, trace)
        self.selected_pages = relevant_pages
        
        if not relevant_pages:
            return {
                "answer": "Could not identify relevant pages from visual analysis.",
                "confidence": 0.0,
                "selected_pages": [],
                "trace": trace,
            }
        
        # Phase 2: Deep analysis of selected pages
        answer, confidence = self._analyze_selected_pages(
            question,
            relevant_pages,
            metadata,
            trace,
        )
        
        logger.info(
            "vision_navigation_complete",
            selected_pages=relevant_pages,
            confidence=confidence,
        )
        
        return {
            "answer": answer,
            "confidence": confidence,
            "selected_pages": relevant_pages,
            "trace": trace,
        }
    
    def _select_relevant_pages(
        self,
        question: str,
        trace: list[dict],
    ) -> list[int]:
        """Select pages that are likely relevant to the question."""
        relevant = []
        
        # Process pages in batches
        batch_size = self.config.max_pages_per_batch
        
        for batch_start in range(0, self.page_count, batch_size):
            batch_end = min(batch_start + batch_size, self.page_count)
            batch_pages = list(range(batch_start, batch_end))
            
            # Extract images for this batch
            images = [
                self.extractor.extract_page_image(self.pdf_path, p)
                for p in batch_pages
            ]
            
            # Ask vision LLM to evaluate pages
            evaluation_prompt = f"""You are evaluating document pages for relevance to a question.

Question: {question}

For each page shown (numbered starting from {batch_start + 1}), estimate its relevance:
- Does this page contain information that could help answer the question?
- What key content is visible on this page?

OUTPUT FORMAT (JSON):
{{
    "pages": [
        {{"page_num": {batch_start + 1}, "relevance": 0.0-1.0, "summary": "brief description"}}
    ]
}}

Respond with JSON only:"""
            
            try:
                import json
                
                response = self.vision_llm.analyze_multiple_images(
                    images,
                    evaluation_prompt,
                )
                
                # Parse response
                json_match = __import__("re").search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    for page_info in result.get("pages", []):
                        page_num = page_info.get("page_num", 0) - 1  # Convert to 0-indexed
                        relevance = page_info.get("relevance", 0)
                        summary = page_info.get("summary", "")
                        
                        if page_num >= 0 and page_num < self.page_count:
                            self.page_summaries[page_num] = summary
                            
                            if relevance >= self.config.page_selection_threshold:
                                relevant.append(page_num)
                
                trace.append({
                    "action": "page_selection",
                    "batch": f"{batch_start}-{batch_end}",
                    "selected": [p for p in relevant if batch_start <= p < batch_end],
                })
                
            except Exception as e:
                logger.warning("page_evaluation_failed", error=str(e))
                # Fallback: include all pages in batch
                relevant.extend(batch_pages)
        
        # Sort by page number
        relevant.sort()
        
        return relevant  # Return all relevant pages
    
    def _analyze_selected_pages(
        self,
        question: str,
        pages: list[int],
        metadata: dict[str, Any] | None,
        trace: list[dict],
    ) -> tuple[str, float]:
        """Perform deep analysis on selected pages."""
        # Extract images for selected pages
        images = [
            self.extractor.extract_page_image(self.pdf_path, p)
            for p in pages
        ]
        
        # Build analysis prompt
        page_descriptions = "\n".join(
            f"Page {p+1}: {self.page_summaries.get(p, 'No summary')}"
            for p in pages
        )
        
        options = metadata.get("options") if metadata else None
        if options:
            options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            analysis_prompt = f"""Based on the document pages shown, answer this multiple-choice question.

Question: {question}

Options:
{options_text}

Page summaries:
{page_descriptions}

Instructions:
1. Carefully examine ALL page images
2. Find evidence that supports one of the options
3. Respond with the letter and full option text

Your answer (e.g., "A. [option text]"):"""
        else:
            analysis_prompt = f"""Based on the document pages shown, answer the question.

Question: {question}

Page summaries:
{page_descriptions}

Instructions:
1. Carefully examine ALL page images
2. Find specific information that answers the question
3. Cite the page number(s) where you found the answer

Answer:"""
        
        try:
            response = self.vision_llm.analyze_multiple_images(images, analysis_prompt)
            
            trace.append({
                "action": "deep_analysis",
                "pages": pages,
                "response_length": len(response),
            })
            
            # Estimate confidence based on response quality
            confidence = 0.7 if len(response) > 50 else 0.5
            
            # Normalize multiple choice answer
            if options:
                response = self._normalize_mc_answer(response, options)
                confidence = 0.8
            
            return response.strip(), confidence
            
        except Exception as e:
            logger.error("page_analysis_failed", error=str(e))
            return f"Error analyzing pages: {str(e)}", 0.0
    
    def _normalize_mc_answer(self, answer: str, options: list) -> str:
        """Normalize multiple choice answer."""
        answer_lower = answer.lower().strip()
        
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            if answer_lower.startswith(f"{letter.lower()}.") or opt.lower() in answer_lower:
                return opt
        
        return answer


# =============================================================================
# Hybrid Text+Vision Navigator
# =============================================================================


class HybridVisionNavigator:
    """
    Hybrid navigator combining text-based tree navigation with vision analysis.
    
    Best of both worlds:
    - Use tree-based ToT navigation for structured content
    - Use vision analysis for charts, diagrams, complex layouts
    
    The hybrid approach detects when vision analysis would be beneficial:
    - Pages with low text extraction quality
    - Pages with images/charts
    - Pages where text structure is unclear
    """
    
    def __init__(
        self,
        pdf_path: Path | str,
        skeleton: dict | None = None,
        kv_store: Any = None,
        vision_config: VisionConfig | None = None,
    ):
        self.pdf_path = Path(pdf_path)
        self.skeleton = skeleton
        self.kv_store = kv_store
        self.vision_config = vision_config or VisionConfig()
        
        # Vision components
        self.vision_nav = VisionNavigator(pdf_path, self.vision_config)
        
        # Determine when to use vision
        self._vision_pages: set[int] = set()
        self._analyze_pages_for_vision_need()
    
    def _analyze_pages_for_vision_need(self) -> None:
        """Identify pages that would benefit from vision analysis."""
        if self.skeleton is None:
            # No text structure - use vision for all
            self._vision_pages = set(range(self.vision_nav.page_count))
            return
        
        # Check each page for:
        # 1. Low text content
        # 2. Images/figures mentioned in text
        # 3. Tables referenced but not parsed
        
        try:
            import fitz
            doc = fitz.open(self.pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Check for images
                images = page.get_images()
                if len(images) > 2:
                    self._vision_pages.add(page_num)
                    continue
                
                # Check text density
                text = page.get_text()
                if len(text.strip()) < 100:  # Low text content
                    self._vision_pages.add(page_num)
                    continue
                
                # Check for table indicators
                if "table" in text.lower() or "figure" in text.lower():
                    self._vision_pages.add(page_num)
            
            doc.close()
            
        except Exception as e:
            logger.warning("vision_need_analysis_failed", error=str(e))
    
    def navigate(
        self,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Navigate using hybrid text+vision approach.
        
        1. First try text-based navigation
        2. If text mentions charts/figures, or low confidence, use vision
        3. Combine results for final answer
        """
        results = {
            "text_result": None,
            "vision_result": None,
            "combined_answer": None,
            "confidence": 0.0,
            "method_used": "text",
        }
        
        # Try text-based navigation first
        if self.skeleton and self.kv_store:
            try:
                from rnsr.agent.rlm_navigator import run_rlm_navigator
                
                text_result = run_rlm_navigator(
                    question,
                    self.skeleton,
                    self.kv_store,
                    metadata=metadata,
                )
                results["text_result"] = text_result
                
                # Check if vision would help
                needs_vision = self._should_use_vision(text_result, question)
                
                if not needs_vision:
                    results["combined_answer"] = text_result.get("answer")
                    results["confidence"] = text_result.get("confidence", 0.5)
                    results["method_used"] = "text"
                    return results
                    
            except Exception as e:
                logger.warning("text_navigation_failed", error=str(e))
        
        # Use vision navigation
        vision_result = self.vision_nav.navigate(question, metadata)
        results["vision_result"] = vision_result
        
        # Combine results
        if results["text_result"] and vision_result.get("confidence", 0) > 0.3:
            # Both methods produced results - combine
            results["combined_answer"] = self._combine_answers(
                results["text_result"].get("answer"),
                vision_result.get("answer"),
                question,
            )
            results["confidence"] = max(
                results["text_result"].get("confidence", 0),
                vision_result.get("confidence", 0),
            )
            results["method_used"] = "hybrid"
        else:
            # Use vision result
            results["combined_answer"] = vision_result.get("answer")
            results["confidence"] = vision_result.get("confidence", 0)
            results["method_used"] = "vision"
        
        return results
    
    def _should_use_vision(
        self,
        text_result: dict[str, Any],
        question: str,
    ) -> bool:
        """Determine if vision analysis should be used."""
        # Low confidence from text
        if text_result.get("confidence", 0) < 0.5:
            return True
        
        # Question mentions visual elements
        visual_keywords = ["chart", "graph", "figure", "diagram", "image", "table", "picture"]
        question_lower = question.lower()
        if any(kw in question_lower for kw in visual_keywords):
            return True
        
        # Answer mentions visual elements
        answer = str(text_result.get("answer", "")).lower()
        if any(kw in answer for kw in visual_keywords):
            return True
        
        return False
    
    def _combine_answers(
        self,
        text_answer: str | None,
        vision_answer: str | None,
        question: str,
    ) -> str:
        """Combine text and vision answers."""
        if not text_answer:
            return vision_answer or "No answer found"
        if not vision_answer:
            return text_answer
        
        # If answers are similar, use text (usually more precise)
        if text_answer.lower().strip() == vision_answer.lower().strip():
            return text_answer
        
        # Use LLM to combine
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
            
            prompt = f"""Two methods analyzed a document to answer a question.
Both methods found relevant information. Combine their answers.

Question: {question}

Text-based answer: {text_answer}

Vision-based answer: {vision_answer}

Combined answer (choose the most accurate and complete one, or merge if complementary):"""
            
            response = llm.complete(prompt)
            return str(response).strip()
            
        except Exception as e:
            logger.warning("answer_combination_failed", error=str(e))
            # Fallback: return text answer
            return text_answer


# =============================================================================
# Factory Functions
# =============================================================================


def create_vision_navigator(
    pdf_path: Path | str,
    config: VisionConfig | None = None,
) -> VisionNavigator:
    """
    Create a vision-based navigator.
    
    Args:
        pdf_path: Path to PDF file.
        config: Optional vision configuration.
        
    Returns:
        VisionNavigator instance.
        
    Example:
        from rnsr.ingestion.vision_retrieval import create_vision_navigator
        
        nav = create_vision_navigator("scanned_document.pdf")
        result = nav.navigate("What is the total amount?")
        print(result["answer"])
    """
    return VisionNavigator(pdf_path, config)


def create_hybrid_navigator(
    pdf_path: Path | str,
    skeleton: dict | None = None,
    kv_store: Any = None,
    vision_config: VisionConfig | None = None,
) -> HybridVisionNavigator:
    """
    Create a hybrid text+vision navigator.
    
    Args:
        pdf_path: Path to PDF file.
        skeleton: Optional skeleton index for text navigation.
        kv_store: Optional KV store for text content.
        vision_config: Optional vision configuration.
        
    Returns:
        HybridVisionNavigator instance.
    """
    return HybridVisionNavigator(pdf_path, skeleton, kv_store, vision_config)
