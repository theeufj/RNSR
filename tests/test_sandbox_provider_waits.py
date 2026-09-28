"""Uncapped query provider waits must not disable local execution deadlines."""

import asyncio
import sqlite3
import time

import pytest

from rnsr.env.sandbox import SandboxedRepl
from rnsr.errors import PermanentProviderError, SandboxError
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


@pytest.fixture
def corpus(tmp_path):
    def parse(path):
        return ParsedDocument(
            doc_id="waits", source_path=str(path), sha256="e" * 64,
            n_pages=1, parser="test",
            elements=[Element("text", "Revenue was 100 USD in 2022.", 1),
                      Element("table", "Item | Amount\nRevenue | 100\nInterest | 10", 1)],
            tables=[RawTable(page=1, header=["Item", "Amount"],
                             rows=[["Revenue", "100"], ["Interest", "10"]],
                             extractor="test")])

    path = tmp_path / "corpus.db"
    ingest([tmp_path / "source.pdf"], path, parse=parse)
    return path


async def delayed_answer(request):
    await asyncio.sleep(.2)
    return {"results": ["1. yes\n2. yes" for _ in request["prompts"]]}


@pytest.mark.parametrize("paused", [False, True])
async def test_long_provider_wait_requires_explicit_uncapped_mode(paused):
    async with SandboxedRepl(rpc_handlers={"llm_batch": delayed_answer}) as repl:
        await repl.start(mode="classic", context="")
        if paused:
            result = await repl.exec_cell("FINAL(llm_query('answer'))", timeout=.08,
                                          pause_provider_rpc_timeout=True)
            assert result.ok and result.final["value"] == "1. yes\n2. yes"
        else:
            with pytest.raises(SandboxError, match="wall-clock"):
                await repl.exec_cell("llm_query('answer')", timeout=.08)
            assert repl._proc.returncode is not None


async def test_annotation_survives_provider_wait_and_final_uses_resumed_deadline(corpus):
    async with SandboxedRepl(rpc_handlers={"llm_batch": delayed_answer}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        result = await repl.exec_cell(
            "semantic_annotate('t_waits_001', 'label', 'classify')\n"
            "FINAL('100 USD', quotes=['Revenue was 100 USD in 2022.'])",
            timeout=.08, pause_provider_rpc_timeout=True)
        assert result.ok and result.final["verification"]["passed"]
    with sqlite3.connect(corpus) as conn:
        assert conn.execute("SELECT label FROM t_waits_001").fetchall() == [("yes",), ("yes",)]


@pytest.mark.parametrize("annotation", [False, True])
async def test_cancellation_during_suspended_watchdog_stops_provider_and_child(corpus, annotation):
    started, stopped = asyncio.Event(), asyncio.Event()

    async def provider(request):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()

    async with SandboxedRepl(rpc_handlers={"llm_batch": provider}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        code = ("semantic_annotate('t_waits_001', 'label', 'classify')"
                if annotation else "llm_query('answer')")
        task = asyncio.create_task(repl.exec_cell(
            code, timeout=.08, pause_provider_rpc_timeout=True))
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.sleep(.12)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(stopped.wait(), 2)
        assert repl._proc.returncode is not None
    with sqlite3.connect(corpus) as conn:
        assert conn.execute("SELECT count(*) FROM annotation_log").fetchone()[0] == 0
        assert "label" not in {row[1] for row in conn.execute("PRAGMA table_info(t_waits_001)")}


async def test_runaway_local_code_after_provider_wait_still_dies():
    async with SandboxedRepl(rpc_handlers={"llm_batch": delayed_answer}) as repl:
        await repl.start(mode="classic", context="")
        with pytest.raises(SandboxError, match="wall-clock"):
            await repl.exec_cell("llm_query('answer')\nwhile True: pass", timeout=.08,
                                 pause_provider_rpc_timeout=True)
        assert repl._proc.returncode is not None


async def test_empty_provider_rpc_does_not_replenish_local_execution_allowance():
    async def empty_batch(request):
        assert request["prompts"] == []
        return {"results": []}

    async with SandboxedRepl(rpc_handlers={"llm_batch": empty_batch}) as repl:
        await repl.start(mode="classic", context="")
        with pytest.raises(SandboxError, match="wall-clock"):
            await repl.exec_cell(
                "import time\ntime.sleep(.09)\nllm_map([])\ntime.sleep(.09)\nFINAL('late')",
                timeout=.15, pause_provider_rpc_timeout=True)
        assert repl._proc.returncode is not None


async def test_long_provider_wait_excluded_from_cumulative_local_execution_allowance():
    async with SandboxedRepl(rpc_handlers={"llm_batch": delayed_answer}) as repl:
        await repl.start(mode="classic", context="")
        result = await repl.exec_cell(
            "import time\ntime.sleep(.025)\nanswer = llm_query('answer')\n"
            "time.sleep(.025)\nFINAL(answer)",
            timeout=.15, pause_provider_rpc_timeout=True)
        assert result.ok and result.final["value"] == "1. yes\n2. yes"


async def test_calculation_rpc_keeps_its_deadline_in_uncapped_mode(corpus, monkeypatch):
    async with SandboxedRepl(rpc_handlers={"llm_batch": delayed_answer}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        original = repl._calculations.source_number

        def slow(*args, **kwargs):
            result = original(*args, **kwargs)
            time.sleep(.12)
            return result

        monkeypatch.setattr(repl._calculations, "source_number", slow)
        with pytest.raises(SandboxError, match="wall-clock"):
            await repl.exec_cell(
                "llm_query('answer')\nsource_number('t_waits_001', 1, 'amount')",
                timeout=.08, pause_provider_rpc_timeout=True)
        assert repl._proc.returncode is not None
        assert not repl._corpus.conn.in_transaction
        assert repl._calculations._deadline is None


async def test_parent_final_verification_still_has_a_deadline_after_provider_wait(corpus, monkeypatch):
    async with SandboxedRepl(rpc_handlers={"llm_batch": delayed_answer}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        original = repl._calculations.validate_legacy_final

        def slow(*args, **kwargs):
            result = original(*args, **kwargs)
            time.sleep(.12)
            return result

        monkeypatch.setattr(repl._calculations, "validate_legacy_final", slow)
        with pytest.raises(SandboxError, match="source verification exceeded"):
            await repl.exec_cell(
                "llm_query('answer')\n"
                "FINAL('100 USD', quotes=['Revenue was 100 USD in 2022.'])",
                timeout=.08, pause_provider_rpc_timeout=True)
        assert repl._proc.returncode is not None
        assert not repl._corpus.conn.in_transaction
        assert repl._verifier._deadline is None


async def test_permanent_provider_failure_propagates_out_of_rpc():
    async def provider(request):
        raise PermanentProviderError("provider authentication failed")

    async with SandboxedRepl(rpc_handlers={"llm_batch": provider}) as repl:
        await repl.start(mode="classic", context="")
        with pytest.raises(PermanentProviderError, match="authentication"):
            await repl.exec_cell("llm_query('answer')", pause_provider_rpc_timeout=True)
