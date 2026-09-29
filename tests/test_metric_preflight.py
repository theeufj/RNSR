"""Unrepairable caller contracts fail before paid answering starts."""
import asyncio
import copy
import hashlib
import json
import threading
import time
from unittest.mock import AsyncMock

import pytest

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.env.calculations import CalculationRegistry
from rnsr.harness.loop import EnvSpec, RootRunner
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest
from rnsr.llm.mock import MockLLM


@pytest.fixture
def contract_corpus(tmp_path):
    rows = [["Revenue", "100"], ["Cost", "40"], ["Zero", "0"]]

    def parse(path):
        return ParsedDocument(doc_id="preflight", source_path=str(path), sha256="e" * 64,
            n_pages=1, parser="test", elements=[Element("heading", "USD 2024", 1, heading_level=1),
                Element("table", "Item | Amount\n" + "\n".join(" | ".join(row) for row in rows), 1)],
            tables=[RawTable(page=1, header=["Item", "Amount"], rows=rows,
                             caption="Contract inputs", extractor="test")])

    path = tmp_path / "corpus.db"
    ingest([tmp_path / "report.pdf"], path, parse=parse)
    with CorpusDB(path) as corpus:
        text = corpus.conn.execute("SELECT text FROM doc_text").fetchone()[0]
        manifest = corpus.manifest_dict()
    selectors = [{"table": "t_preflight_001", "rowid": index, "column": "amount",
                  "label_column": "item", "label": row[0],
                  "unit_span": {"char_start": text.index("USD"), "char_end": text.index("USD") + 3},
                  "period_span": {"char_start": text.index("2024"), "char_end": text.index("2024") + 4}}
                 for index, row in enumerate(rows, 1)]
    contract = {"contract_id": "caller-ratio", "metric": "Revenue to cost",
                "formula": "direct_ratio", "basis": "Caller requests revenue divided by cost",
                "unit": "USD", "period": "2024",
                "sources": {"numerator": selectors[0], "denominator": selectors[1]}}
    return path, manifest, contract, selectors


def runner(root=None, sub=None, **settings):
    return RootRunner(root or MockLLM(), "root", sub or MockLLM(), "sub", Settings(**settings))


@pytest.mark.parametrize("invalid", ["missing_formula", "zero_denominator", "wrong_source_label"])
async def test_invalid_immutable_contract_fails_before_sandbox_or_provider(contract_corpus, tmp_path,
                                                                          monkeypatch, invalid):
    path, manifest, contract, selectors = contract_corpus
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    if invalid == "missing_formula":
        contract.update(formula=None, sources={})
        expected = "no caller-approved formula"
    elif invalid == "zero_denominator":
        contract["sources"]["denominator"] = selectors[2]
        expected = "division by zero"
    else:
        contract["sources"]["numerator"]["label"] = "Unbound source name"
        expected = "source label differs"
    sandbox_factory = AsyncMock(side_effect=AssertionError("sandbox must not start"))
    monkeypatch.setattr("rnsr.harness.loop.SandboxedRepl", sandbox_factory)
    root, sub = MockLLM(), MockLLM()
    with pytest.raises(ValueError, match=f"invalid caller metric contract: .*{expected}"):
        await runner(root, sub).run("Calculate the caller ratio.", EnvSpec(mode="docdb",
            corpus_db=str(path), manifest=manifest, metric_contract=contract), run_dir=tmp_path / "runs")
    assert root.calls == sub.calls == [] and sandbox_factory.call_count == 0
    assert not (tmp_path / "runs").exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


async def test_valid_contract_still_runs_and_is_semantically_reviewed(contract_corpus, tmp_path):
    path, manifest, contract, _ = contract_corpus
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    root = MockLLM(default="```python\nr = calculate_metric()\nFINAL_CALC(r['id'])\n```")
    sub = MockLLM().script("COMPLETE", json.dumps({"reviews": [{"field_id": "f0",
        "verdict": "supported", "evidence_ids": ["f0p0"],
        "reason": "The caller ratio uses revenue and cost from the requested period."}]}))
    result = await runner(root, sub).run("Calculate the caller ratio.", EnvSpec(mode="docdb",
        corpus_db=str(path), manifest=manifest, metric_contract=contract), run_dir=tmp_path / "runs")
    assert result.answer == "2.5" and result.status == "final"
    assert result.claim_review["status"] == "supported"
    assert len(root.calls) == 1 and len(sub.calls) == 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


async def test_generic_question_skips_metric_preflight(monkeypatch, tmp_path):
    preflight = AsyncMock(side_effect=AssertionError("not a caller contract"))
    monkeypatch.setattr("rnsr.harness.loop._validate_metric_contract", preflight)
    root = MockLLM(default="```python\nFINAL('General answer')\n```")
    result = await runner(root, MockLLM(default="COMPLETE")).run(
        "Question?", EnvSpec(mode="classic", context="General answer"), run_dir=tmp_path)
    assert result.answer == "General answer" and preflight.await_count == 0


async def test_contract_preflight_cancellation_stops_local_work(contract_corpus, monkeypatch, tmp_path):
    path, manifest, contract, _ = contract_corpus
    started, stopped = threading.Event(), threading.Event()

    def calculate(self):
        started.set()
        try:
            while True:
                self._check_deadline()
                time.sleep(0.002)
        finally:
            stopped.set()

    monkeypatch.setattr(CalculationRegistry, "calculate_metric", calculate)
    root, sub = MockLLM(), MockLLM()
    task = asyncio.create_task(runner(root, sub).run("Ratio?", EnvSpec(mode="docdb",
        corpus_db=str(path), manifest=manifest, metric_contract=copy.deepcopy(contract)), run_dir=tmp_path / "runs"))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(stopped.wait, 2)
    assert root.calls == sub.calls == []
