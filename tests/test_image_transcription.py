"""Standalone image pages reach vision transcription and clear health gaps."""
import asyncio
import contextvars
import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from rnsr.db.artifact import CorpusDB
from rnsr.ingest import llm_hooks
from rnsr.ingest.bulk import ingest_bulk
from rnsr.ingest.pipeline import ingest
from rnsr.llm.base import LLMResponse
from rnsr.llm.mock import MockLLM


@pytest.mark.parametrize("extension", ["png", "jpg", "jpeg", "tif", "tiff", "webp"])
def test_rasterize_standalone_image_as_png(tmp_path, extension):
    source = tmp_path / f"scan.{extension}"
    Image.new("RGB", (32, 20), "white").save(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    rendered = llm_hooks.rasterize_page(source, 1)
    assert rendered.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(rendered)) as image:
        assert image.size == (32, 20)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_standalone_image_bulk_ingest_is_transcribed(tmp_path):
    source = tmp_path / "scan.png"
    Image.new("RGB", (32, 20), "white").save(source)
    mock = MockLLM(default=json.dumps({
        "blocks": [{"kind": "text", "text": "Invoice total: $42"}], "tables": []}))
    transcriber = llm_hooks.make_page_transcriber(mock, "mock-vision")
    output = tmp_path / "corpus.db"
    result = ingest_bulk([source], output, workers=1, transcriber=transcriber)
    assert result["new_docs"] == 1
    assert result["scanned_pages_untranscribed"] == 0
    assert len(mock.calls) == 1 and mock.calls[0]["kind"] == "vision"
    with CorpusDB(output) as corpus:
        health = corpus.manifest_get("health")
        assert health["scanned_pages_total"] == 1
        assert health["scanned_pages_untranscribed"] == 0
        assert health["grade"] == "ok"
        assert "Invoice total: $42" in corpus.conn.execute("SELECT text FROM doc_text").fetchone()[0]


@pytest.mark.parametrize("bulk", [False, True])
def test_multipage_tiff_ingests_every_frame_and_reports_complete_health(tmp_path, bulk):
    source = tmp_path / "two-pages.tiff"
    frames = [Image.new("RGB", (32, 20), color) for color in ("red", "blue")]
    frames[0].save(source, save_all=True, append_images=frames[1:])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    text_by_color = {(255, 0, 0): "First page: invoice number 731.",
                     (0, 0, 255): "Second page: payment received."}

    class FrameClient(MockLLM):
        async def vision(self, prompt, image_png, *, model, **kwargs):
            with Image.open(io.BytesIO(image_png)) as page:
                color = page.convert("RGB").getpixel((0, 0))
            self.calls.append({"kind": "vision", "color": color})
            response = {"blocks": [{"kind": "text", "text": text_by_color[color]}],
                        "tables": []}
            return LLMResponse(json.dumps(response), model, self.usage_per_call)

    client = FrameClient()
    transcriber = llm_hooks.make_page_transcriber(client, "mock-vision")
    output = tmp_path / "corpus.db"
    if bulk:
        result = ingest_bulk([source], output, workers=1, transcriber=transcriber)
        assert result["scanned_pages_untranscribed"] == 0
    else:
        report = ingest([source], output, transcriber=transcriber)
        assert report.scanned_pages_transcribed == 2
        assert report.scanned_pages_untranscribed == []
    assert {call["color"] for call in client.calls} == set(text_by_color)
    assert len(client.calls) == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with CorpusDB(output) as corpus:
        assert corpus.conn.execute("SELECT n_pages FROM documents").fetchone()[0] == 2
        pages = corpus.conn.execute("SELECT page,text FROM doc_text ORDER BY page").fetchall()
        assert [tuple(row) for row in pages] == [
            (1, text_by_color[(255, 0, 0)] + "\n"),
            (2, text_by_color[(0, 0, 255)] + "\n"),
        ]
        health = corpus.manifest_get("health")
        assert health["scanned_pages_total"] == 2
        assert health["scanned_pages_untranscribed"] == 0
        assert health["grade"] == "ok"


def test_transparent_scan_uses_white_background(tmp_path):
    source = tmp_path / "transparent.png"
    scan = Image.new("RGBA", (3, 1), (0, 0, 0, 0))
    scan.putpixel((1, 0), (0, 0, 0, 255))
    scan.putpixel((2, 0), (0, 0, 0, 128))
    scan.save(source)
    rendered = llm_hooks.rasterize_page(source, 1)
    with Image.open(io.BytesIO(rendered)) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((1, 0)) == (0, 0, 0)
        assert image.getpixel((2, 0)) == (127, 127, 127)


def test_sync_hooks_share_provider_loop_across_documents_and_threads(tmp_path):
    caller_context = contextvars.ContextVar("ingest_test_job", default="unset")
    observed_contexts = []

    class LoopBoundClient(MockLLM):
        loop = None

        def bind_loop(self):
            observed_contexts.append(caller_context.get())
            loop = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = loop
            assert loop is self.loop and not loop.is_closed()

        async def complete(self, *args, **kwargs):
            self.bind_loop()
            self.default = "YES"
            return await super().complete(*args, **kwargs)

        async def vision(self, prompt, *args, **kwargs):
            self.bind_loop()
            self.default = json.dumps(
                {"header": ["Amount"], "rows": [["42"]]}
                if "LARGEST table" in prompt else
                {"blocks": [{"kind": "text", "text": "Invoice total: $42"}], "tables": []})
            return await super().vision(prompt, *args, **kwargs)

    source = tmp_path / "scan.png"
    Image.new("RGB", (32, 20), "white").save(source)
    client = LoopBoundClient()
    transcribe = llm_hooks.make_page_transcriber(client, "mock-vision")
    check = llm_hooks.make_prose_checker(client, "mock-sub")
    extract = llm_hooks.make_vision_extractor(client, "mock-vision")
    token = caller_context.set("transcription")
    try:
        assert transcribe(source, [1])[1] is not None
        caller_context.set("prose")
        assert check(["Does the text state the total?"]) == [True]
        caller_context.set("vision")
        assert extract(source, 1).rows == [["42"]]
    finally:
        caller_context.reset(token)
    barrier = threading.Barrier(2)

    def another_document(job):
        token = caller_context.set(job)
        try:
            barrier.wait(timeout=5)
            return transcribe(source, [1])[1]
        finally:
            caller_context.reset(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(another_document, f"job-{i}") for i in range(2)]
        assert all(future.result() is not None for future in futures)
    assert len(client.calls) == 5
    assert observed_contexts[:3] == ["transcription", "prose", "vision"]
    assert set(observed_contexts[3:]) == {"job-0", "job-1"}
