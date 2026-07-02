"""Adapters wiring an LLMClient into the ingest pipeline's hooks (§3.1, §3.3).

Ingestion is synchronous; these adapters bridge to the async client with
asyncio.run, so they must not be called from inside a running event loop —
run LLM-assisted ingestion from sync code (the CLI does).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from rnsr.ingest.fallback import VisionExtractor
from rnsr.ingest.model import RawTable
from rnsr.ingest.validate import ProseChecker
from rnsr.llm.base import LLMClient
from rnsr.llm.batch import map_prompts

_YES = re.compile(r"^\s*yes\b", re.IGNORECASE)
_NO = re.compile(r"^\s*no\b", re.IGNORECASE)

_VISION_PROMPT = """\
This image is a page from a document containing at least one table.
Extract the LARGEST table as JSON with exactly this shape:
{"header": ["col1", ...], "rows": [["cell", ...], ...]}
Transcribe cell text exactly (keep currency symbols, commas, parentheses).
Use null for empty cells. Return ONLY the JSON object."""


def make_prose_checker(client: LLMClient, model: str, *, concurrency: int = 16) -> ProseChecker:
    """ProseChecker: batch of yes/no prompts -> True/False/None per prompt."""

    def check(prompts: list[str]) -> list[bool | None]:
        responses = asyncio.run(
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
    """Render one page (1-based) to PNG bytes via pypdfium2."""
    import io

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


def make_vision_extractor(client: LLMClient, model: str) -> VisionExtractor:
    """Rung-2 table extraction: rasterized page crop -> sub-LM -> RawTable."""

    def extract(pdf_path: Path, page: int) -> RawTable | None:
        png = rasterize_page(pdf_path, page)
        resp = asyncio.run(client.vision(_VISION_PROMPT, png, model=model))
        grid = _parse_grid(resp.text)
        if grid is None or not grid["rows"]:
            return None
        header = [str(h) if h is not None else "" for h in grid["header"]]
        rows = [[None if c is None else str(c) for c in row] for row in grid["rows"]]
        return RawTable(page=page, header=header, rows=rows, extractor="vision")

    return extract
