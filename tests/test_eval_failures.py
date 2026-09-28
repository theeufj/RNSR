"""A corpus rejection must remain distinguishable from exhausted query budgets."""

import json
from types import SimpleNamespace

import pytest

from rnsr.config import Settings
from rnsr.errors import CorpusHealthError
from rnsr.eval.autopsy import autopsy_run
from rnsr.eval.datasets.base import EvalItem
from rnsr.eval.harness import run_eval
from rnsr.ingest.health import CorpusHealth


@pytest.mark.parametrize("error,cause", [
    (CorpusHealthError(CorpusHealth(grade="blocked")), "ingest"),
    (FileNotFoundError("bad trajectory path"), "execution"),
])
async def test_setup_failure_type_survives_results_and_autopsy(tmp_path, monkeypatch,
                                                             error, cause):
    def fail(*args):
        raise error

    monkeypatch.setattr("rnsr.eval.harness._env_for", fail)
    runner = SimpleNamespace(settings=Settings())
    items = [EvalItem(qid="q", question="What is the amount?", gold="123", context="x")]
    results, summary = await run_eval(items, "docdb", runner, run_dir=tmp_path, judge=False)
    assert results[0].status == "error"
    assert results[0].error_type == type(error).__name__
    assert results[0].cause == cause
    assert summary["cause_counts"] == {cause: 1}
    saved = json.loads((tmp_path / "results.jsonl").read_text())
    assert saved["error_type"] == type(error).__name__
    assert autopsy_run(tmp_path)["cause_counts"][cause] == 1


def test_eval_ingestion_honors_transcription_and_keeps_old_cache(tmp_path, monkeypatch):
    from dataclasses import replace
    from hashlib import sha256

    from rnsr.eval.harness import _corpus_for

    source = tmp_path / "scan.pdf"
    source.write_bytes(b"test source")
    cache = tmp_path / "cache"
    cache.mkdir()
    old_cache = cache / f"corpus_{sha256(source.read_bytes()).hexdigest()[:16]}.db"
    old_cache.write_bytes(b"preserved legacy artifact")
    transcriber = object()
    calls = []

    def resolve(settings):
        return (None, None) if settings.transcribe_scans == "never" else (transcriber, "vision")

    def ingest(sources, path, **kwargs):
        calls.append(kwargs["transcriber"])
        path.write_bytes(b"new artifact")

    monkeypatch.setattr("rnsr.ingest.cost_estimate.resolve_transcriber", resolve)
    monkeypatch.setattr("rnsr.ingest.pipeline.ingest", ingest)
    settings = Settings(transcribe_scans="auto")
    automatic = _corpus_for([source], cache, settings)
    disabled = _corpus_for([source], cache, replace(settings, transcribe_scans="never"))
    assert calls == [transcriber, None]
    assert automatic != disabled != old_cache
    assert old_cache.read_bytes() == b"preserved legacy artifact"


def test_text_ingestion_does_not_reuse_pre_fix_cache(tmp_path):
    from hashlib import sha256

    from rnsr.eval.harness import _text_corpus_for

    context = "One.\nTwo.\nThree.\nThe total liability is limited."
    old_cache = tmp_path / f"corpus_text_{sha256(context.encode()).hexdigest()[:16]}.db"
    old_cache.write_bytes(b"preserved legacy artifact")
    current = _text_corpus_for(context, tmp_path, Settings())
    assert current != old_cache
    assert current.is_file()
    assert old_cache.read_bytes() == b"preserved legacy artifact"


def test_document_cache_does_not_reuse_lost_first_rows(tmp_path, monkeypatch):
    from hashlib import sha256

    from rnsr.eval.harness import _corpus_for

    source = tmp_path / "votes.pdf"
    source.write_bytes(b"headerless voting results")
    identity = {"version": 2, "transcribe_scans": "never", "transcriber_model": None}
    h = sha256(json.dumps(identity, sort_keys=True).encode())
    h.update(json.dumps((str(source.resolve()), sha256(source.read_bytes()).hexdigest())).encode())
    previous = tmp_path / f"corpus_{h.hexdigest()[:16]}.db"
    previous.write_bytes(b"valid older corpus, first vote row missing")
    calls = []

    def ingest(sources, out, **kwargs):
        calls.append(sources)
        out.write_bytes(b"all vote rows retained")

    monkeypatch.setattr("rnsr.ingest.cost_estimate.resolve_transcriber", lambda s: (None, None))
    monkeypatch.setattr("rnsr.eval.harness._corpus_valid", lambda p, n: True)
    monkeypatch.setattr("rnsr.ingest.pipeline.ingest", ingest)
    current = _corpus_for([source], tmp_path, Settings(transcribe_scans="never"))
    assert calls == [[source]]
    assert current != previous
    assert current.read_bytes() == b"all vote rows retained"
    assert previous.read_bytes() == b"valid older corpus, first vote row missing"
