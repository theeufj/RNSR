"""Adapters wiring an LLMClient into the ingest pipeline's hooks (§3.1, §3.3).

Ingestion is synchronous; these adapters share one persistent async loop so
provider connection pools survive across documents and across hook types.
Call from synchronous code, outside a running event loop (as the CLI does).
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import json
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

from rnsr.ingest.fallback import VisionExtractor
from rnsr.ingest.model import RawTable
from rnsr.ingest.validate import ProseChecker
from rnsr.llm.base import LLMClient
from rnsr.llm.batch import map_prompts

_YES = re.compile(r"^\s*yes\b", re.IGNORECASE)
_NO = re.compile(r"^\s*no\b", re.IGNORECASE)
_LOG = logging.getLogger(__name__)

# Async provider transports are bound to the loop that first uses them.
# All synchronous ingest hooks share this runner, even when an ingestion
# moves between worker threads. The lock serializes whole hook invocations;
# concurrency within a page/prose batch still runs normally on the loop.
_SYNC_RUNNER = asyncio.Runner()
_SYNC_LOCK = threading.RLock()


def _run_sync(coroutine):
    with _SYNC_LOCK:
        return _SYNC_RUNNER.run(coroutine, context=contextvars.copy_context())


def _close_sync_runner() -> None:
    with _SYNC_LOCK:
        _SYNC_RUNNER.close()


atexit.register(_close_sync_runner)

_VISION_PROMPT = """\
This image is a page from a document containing at least one table.
Extract the LARGEST table as JSON with exactly this shape:
{"header": ["col1", ...], "rows": [["cell", ...], ...]}
Transcribe cell text exactly (keep currency symbols, commas, parentheses).
Use null for empty cells. Return ONLY the JSON object."""


def make_prose_checker(client: LLMClient, model: str, *, concurrency: int = 16) -> ProseChecker:
    """ProseChecker: batch of yes/no prompts -> True/False/None per prompt."""

    def check(prompts: list[str]) -> list[bool | None]:
        responses = _run_sync(
            map_prompts(client, prompts, model=model, max_tokens=16,
                        concurrency=concurrency)
        )
        out: list[bool | None] = []
        for r in responses:
            if r is None:
                out.append(None)
            elif _YES.match(r.text):
                out.append(True)
            elif _NO.match(r.text):
                out.append(False)
            else:
                out.append(None)  # UNCLEAR or malformed — no evidence
        return out

    return check


def rasterize_page(pdf_path: Path, page: int, *, scale: float = 2.0) -> bytes:
    """Render a PDF page or supported standalone image to PNG bytes."""
    import io

    from rnsr.ingest.dispatch import IMAGE_EXTENSIONS

    if page < 1:
        raise ValueError("page numbers are 1-based")
    if Path(pdf_path).suffix.lower() in IMAGE_EXTENSIONS:
        from PIL import Image, ImageOps

        with Image.open(pdf_path) as source:
            source.seek(page - 1)
            image = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            buf = io.BytesIO()
            background.save(buf, format="PNG")
            return buf.getvalue()

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        bitmap = pdf[page - 1].render(scale=scale)
        image = bitmap.to_pil()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()


def _parse_grid(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj.get("header"), list) or not isinstance(obj.get("rows"), list):
        return None
    return obj


_TRANSCRIBE_PROMPT = """\
This image is a scanned page from a document. Transcribe it fully as JSON
with exactly this shape:
{"blocks": [{"kind": "heading"|"text", "text": "..."}, ...],
 "tables": [{"header": ["col", ...], "rows": [["cell", ...], ...]}, ...]}
Rules: transcribe text exactly (keep numbers, currency symbols, commas,
parentheses); reading order top to bottom; each paragraph is one block;
put tabular data in "tables" (use null for empty cells), not in blocks.
Return valid JSON: escape every quotation mark inside a text string as
\\\" and every literal line break inside a string as \\n. Preserve the quoted
words exactly; JSON escaping changes the encoding, not the source text.
Return ONLY the JSON object."""

_TRANSCRIBE_RETRY = """\
The previous response was invalid or empty transcription JSON. Return the complete
transcription again as one valid JSON object. In particular, quotation marks
in the document are part of the text: encode them as \\\" inside JSON strings.
Do not omit quoted clauses, and close every string, array, and object.
Include all legible text in logos, stamps, footers, and margins, even when
most of the page is blank. Never invent text that is not visible.
"""


def _parse_transcription(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("blocks"), list):
        return None
    obj.setdefault("tables", [])
    if (not isinstance(obj["tables"], list)
            or any(not isinstance(block, dict) or not isinstance(block.get("text"), str)
                   for block in obj["blocks"])):
        return None
    for table in obj["tables"]:
        if (not isinstance(table, dict) or not isinstance(table.get("header"), list)
                or not isinstance(table.get("rows"), list)
                or any(not isinstance(row, list) for row in table["rows"])
                or any(isinstance(cell, (dict, list))
                       for cell in table["header"] + [c for row in table["rows"] for c in row])):
            return None
        if table["rows"] and (not table["header"]
                              or any(len(row) != len(table["header"]) for row in table["rows"])):
            return None
    return obj


def _transient_vision_failure(exc: Exception) -> bool:
    import anthropic
    import httpx
    import openai

    from rnsr.llm.governor import is_rate_limit

    # SDKs disable their own retries so every attempt passes the governor.
    # Anthropic/OpenAI expose status_code; Gemini exposes the HTTP code.
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if isinstance(status, int) and 500 <= status < 600:
        return True
    return is_rate_limit(exc) or isinstance(exc, (
        TimeoutError, ConnectionError, httpx.TransportError,
        anthropic.APIConnectionError, openai.APIConnectionError,
    ))


def _trim_blank_margins(png: bytes) -> bytes:
    """Zoom sparse pages for a retry, removing only completely white margins."""
    import io

    from PIL import Image, ImageChops

    with Image.open(io.BytesIO(png)) as source:
        image = source.convert("RGB")
        bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        if bounds is None:
            return png
        left, top, right, bottom = bounds
        bounds = (max(0, left - 16), max(0, top - 16),
                  min(image.width, right + 16), min(image.height, bottom + 16))
        output = io.BytesIO()
        image.crop(bounds).save(output, format="PNG")
        return output.getvalue()


def _has_transcription_content(result: dict) -> bool:
    return (any(block["text"].strip() for block in result["blocks"])
            or any(any(
                str(cell).strip() for cell in table["header"]
                + [c for row in table["rows"] for c in row] if cell is not None)
                   for table in result["tables"]))


# transcribe(pdf_path, pages) -> {page: transcription dict | None}
PageTranscriber = Callable[[Path, list[int]], dict[int, dict | None]]


def make_page_transcriber(client: LLMClient, model: str, *,
                          concurrency: int = 4) -> PageTranscriber:
    """VLM transcription of scanned pages (no OCR engine — spec §3.1 vision
    rung applied to whole pages). Retry malformed/empty JSON once with explicit
    instructions and white margins trimmed, or a transient provider failure once. Usage for
    both attempts remains subject to the same provider governor."""

    async def _run(pdf_path: Path, pages: list[int]) -> dict[int, dict | None]:
        sem = asyncio.Semaphore(concurrency)

        async def one(page: int) -> tuple[int, dict | None]:
            async with sem:
                try:
                    png = rasterize_page(pdf_path, page)
                except Exception as exc:
                    _LOG.warning("Page %s rasterization failed (%s)", page, type(exc).__name__)
                    return page, None
                prompt = _TRANSCRIBE_PROMPT
                for attempt in range(2):
                    try:
                        resp = await client.vision(prompt, png, model=model,
                                                   max_tokens=4096 if attempt == 0 else 8192)
                        result = _parse_transcription(resp.text)
                        if result is not None and _has_transcription_content(result):
                            return page, result
                        _LOG.warning("Page %s transcription returned invalid or empty JSON (attempt %s)",
                                     page, attempt + 1)
                        prompt = _TRANSCRIBE_PROMPT + "\n" + _TRANSCRIBE_RETRY
                        if attempt == 0:
                            png = _trim_blank_margins(png)
                    except Exception as exc:
                        _LOG.warning("Page %s transcription failed (%s, attempt %s)",
                                     page, type(exc).__name__, attempt + 1)
                        if not _transient_vision_failure(exc):
                            break
                return page, None

        return dict(await asyncio.gather(*(one(p) for p in pages)))

    def transcribe(pdf_path: Path, pages: list[int]) -> dict[int, dict | None]:
        return _run_sync(_run(pdf_path, pages))

    transcribe.model = model  # type: ignore[attr-defined]
    return transcribe


def make_vision_extractor(client: LLMClient, model: str) -> VisionExtractor:
    """Rung-2 table extraction: rasterized page crop -> sub-LM -> RawTable."""

    def extract(pdf_path: Path, page: int) -> RawTable | None:
        png = rasterize_page(pdf_path, page)
        resp = _run_sync(client.vision(_VISION_PROMPT, png, model=model))
        grid = _parse_grid(resp.text)
        if grid is None or not grid["rows"]:
            return None
        header = [str(h) if h is not None else "" for h in grid["header"]]
        rows = [[None if c is None else str(c) for c in row] for row in grid["rows"]]
        return RawTable(page=page, header=header, rows=rows, extractor="vision")

    return extract
