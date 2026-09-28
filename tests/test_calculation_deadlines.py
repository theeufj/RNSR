"""Parent arithmetic must not escape the cell's existing wall-clock budget."""
import asyncio
import copy
import sqlite3
from unittest.mock import AsyncMock

import pytest

from rnsr.db.artifact import CorpusDB
from rnsr.env import calculations
from rnsr.env import sandbox as sandbox_module
from rnsr.env.calculations import CalculationRegistry
from rnsr.env.evidence import SourceContext
from rnsr.env.lazydoc import LazyDoc
from rnsr.env.sandbox import SandboxedRepl, _document_generation
from rnsr.env.verify import Verifier
from rnsr.errors import SandboxError
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


class Clock:
    value = 100.0

    def monotonic(self):
        return self.value


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    # Do not patch the event loop's real clock or sleep. Only synchronous
    # parent checks see this deterministic clock.
    monkeypatch.setattr(calculations, "time", value)
    monkeypatch.setattr(sandbox_module, "time", value)
    return value


@pytest.fixture
def artifact(tmp_path):
    rows = [["Revenue", "100"], ["Interest", "10"]]

    def parse(path):
        return ParsedDocument(
            doc_id="deadline", source_path=str(path), sha256="d" * 64,
            n_pages=1, parser="test",
            elements=[Element("heading", "USD 2022", 1, heading_level=1),
                      Element("table", "Item | Amount\n" + "\n".join(" | ".join(r) for r in rows), 1)],
            tables=[RawTable(page=1, header=["Item", "Amount"], rows=rows, extractor="test")])

    path = tmp_path / "corpus.db"
    ingest([tmp_path / "source.pdf"], path, parse=parse)
    return path


@pytest.fixture
def corpus(artifact):
    with CorpusDB(artifact) as opened:
        yield opened


def contract(corpus):
    text = corpus.conn.execute("SELECT text FROM doc_text").fetchone()[0]
    return {"contract_id": "deadline-test", "metric": "Declared ratio", "formula": "direct_ratio",
            "basis": "Offline caller test", "unit": "USD", "period": "2022",
            "sources": {role: {
                "table": "t_deadline_001", "rowid": rowid, "column": "amount",
                "label_column": "item", "label": label,
                "unit_span": {"char_start": text.index("USD"), "char_end": text.index("USD") + 3},
                "period_span": {"char_start": text.index("2022"), "char_end": text.index("2022") + 4},
            } for role, rowid, label in (("numerator", 1, "Revenue"), ("denominator", 2, "Interest"))}}


def result(registry, installed_contract=False):
    if installed_contract:
        return registry.calculate_metric()
    source = registry.source_number("t_deadline_001", 1, "amount")
    return registry.calculate("sum", [source["id"]])


def parent(corpus, registry):
    repl = SandboxedRepl()
    repl._corpus, repl._calculations = corpus, registry
    repl._source_generation = _document_generation(corpus.conn)
    stat = corpus.path.stat()
    repl._source_identity = stat.st_dev, stat.st_ino
    repl._verifier = Verifier(LazyDoc(corpus.conn))
    return repl


def assert_cleanup(corpus, registry):
    assert registry._deadline is None
    assert not corpus.conn.in_transaction
    # Many VM steps prove that a stale expired progress handler is gone.
    assert corpus.conn.execute(
        "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<2000) "
        "SELECT sum(x) FROM n").fetchone()[0] == 2_001_000


def test_sql_is_interrupted_and_handler_resets(corpus, clock):
    registry = CalculationRegistry(corpus.conn)

    def ticking():
        clock.value += .001
        return clock.value

    clock.monotonic = ticking
    with pytest.raises(TimeoutError, match="calculation exceeded"), registry.deadline(100.01):
        corpus.conn.execute(
            "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<10000000) "
            "SELECT sum(x) FROM n").fetchone()
    assert_cleanup(corpus, registry)


def test_non_deadline_sql_error_remains_sql_error(corpus, clock):
    registry = CalculationRegistry(corpus.conn)
    with pytest.raises(sqlite3.OperationalError, match="no such table"), registry.deadline(101):
        corpus.conn.execute("SELECT * FROM missing_deadline_fixture")
    assert_cleanup(corpus, registry)


def test_nested_deadline_cannot_extend_outer_and_cancellation_cleans_up(corpus, clock):
    registry = CalculationRegistry(corpus.conn)
    with pytest.raises(asyncio.CancelledError), registry.deadline(101):
        with registry.deadline(200):
            assert registry._deadline == 101
        assert registry._deadline == 101
        raise asyncio.CancelledError
    assert_cleanup(corpus, registry)


@pytest.mark.parametrize("installed_contract", [False, True])
def test_source_validation_cannot_return_after_deadline(corpus, clock, monkeypatch, installed_contract):
    registry = CalculationRegistry(corpus.conn, metric_contract=contract(corpus) if installed_contract else None)
    calculated = result(registry, installed_contract)
    original = registry._read_cell

    def slow(*args):
        value = original(*args)
        clock.value = 102
        return value

    monkeypatch.setattr(registry, "_read_cell", slow)
    with pytest.raises(TimeoutError), registry.deadline(101):
        registry.resolve_final(calculated["id"], calculated["value"])
    assert_cleanup(corpus, registry)
    monkeypatch.setattr(registry, "_read_cell", original)
    with registry.deadline(103):
        assert registry.resolve_final(calculated["id"], calculated["value"])[0] == calculated["value"]


@pytest.mark.parametrize("action", ["source", "compute", "get", "metric"])
async def test_each_rpc_rejects_expired_budget_and_rolls_back(corpus, clock, action):
    registry = CalculationRegistry(corpus.conn, metric_contract=contract(corpus))
    calculated = registry.calculate_metric()
    repl = parent(corpus, registry)
    arguments = {
        "source": {"table": "t_deadline_001", "rowid": 1, "column": "amount"},
        "compute": {"operation": "sum", "operand_ids": [calculated["id"]]},
        "get": {"record_id": calculated["id"]}, "metric": {},
    }
    try:
        with pytest.raises(TimeoutError):
            await repl._calculation({"op": "calculation", "action": action, **arguments[action]}, deadline=99)
        assert_cleanup(corpus, registry)
        assert (await repl._calculation(
            {"op": "calculation", "action": action, **arguments[action]}, deadline=101))["result"]
        assert_cleanup(corpus, registry)
    finally:
        repl._verifier.close()


@pytest.mark.parametrize("installed_contract", [False, True])
async def test_parent_final_rejects_late_resolution_and_next_cell_can_succeed(
        corpus, clock, monkeypatch, installed_contract):
    registry = CalculationRegistry(corpus.conn, metric_contract=contract(corpus) if installed_contract else None)
    calculated = result(registry, installed_contract)
    repl = parent(corpus, registry)
    reply = {"ok": True, "final": {"value": calculated["value"], "verification": {
        "check": "source_bound_calculation", "calculation_id": calculated["id"]}}}

    async def roundtrip(*args, **kwargs):
        assert kwargs["deadline"] == clock.value + .5
        return copy.deepcopy(reply)

    monkeypatch.setattr(repl, "_roundtrip", roundtrip)
    monkeypatch.setattr(repl, "kill", AsyncMock())
    original = registry.resolve_final

    def late(*args):
        resolved = original(*args)
        clock.value += 1
        return resolved

    monkeypatch.setattr(registry, "resolve_final", late)
    try:
        with pytest.raises(SandboxError, match="source verification exceeded"):
            await repl.exec_cell("ignored mock cell", timeout=.5)
        repl.kill.assert_awaited_once()
        assert repl._verifier._deadline is None
        assert_cleanup(corpus, registry)
        monkeypatch.setattr(registry, "resolve_final", original)
        accepted = await repl.exec_cell("ignored mock cell", timeout=.5)
        assert accepted.ok and accepted.final["value"] == calculated["value"]
        assert_cleanup(corpus, registry)
    finally:
        repl._verifier.close()


async def test_final_uses_remaining_budget_not_a_fresh_timeout(corpus, clock, monkeypatch):
    registry = CalculationRegistry(corpus.conn)
    calculated = result(registry)
    repl = parent(corpus, registry)

    async def late_reply(*args, **kwargs):
        clock.value = kwargs["deadline"] + .001
        return {"ok": True, "final": {"value": calculated["value"], "verification": {
            "check": "source_bound_calculation", "calculation_id": calculated["id"]}}}

    monkeypatch.setattr(repl, "_roundtrip", late_reply)
    monkeypatch.setattr(repl, "kill", AsyncMock())
    try:
        with pytest.raises(SandboxError, match="source verification exceeded"):
            await repl.exec_cell("ignored mock cell", timeout=.5)
        assert_cleanup(corpus, registry)
    finally:
        repl._verifier.close()


async def test_real_sandbox_rpc_deadline_kills_child_and_cleans_parent(artifact, clock, monkeypatch):
    async with SandboxedRepl() as repl:
        await repl.start(mode="docdb", corpus_db=str(artifact))
        original = repl._calculations.source_number

        def slow(*args, **kwargs):
            value = original(*args, **kwargs)
            clock.value += 1
            return value

        monkeypatch.setattr(repl._calculations, "source_number", slow)
        with pytest.raises(SandboxError, match="cell exceeded wall-clock"):
            await repl.exec_cell("source_number('t_deadline_001', 1, 'amount')", timeout=.5)
        assert repl._proc.returncode is not None
        assert_cleanup(repl._corpus, repl._calculations)


def test_context_search_cooperates_and_keeps_window_boundary_matches(corpus):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("controlled search deadline")

    context = SourceContext(corpus.conn, check=check)
    text = "a" * (65536 - 3) + "needle" + "a" * 200_000
    assert context._find(text, "needle") == 65536 - 3
    with pytest.raises(TimeoutError, match="controlled search"):
        context._find(text, "missing")


def test_interrupted_context_metadata_does_not_cache_partial_state(corpus):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("controlled metadata deadline")

    context = SourceContext(corpus.conn, check=check)
    with pytest.raises(TimeoutError, match="metadata deadline"):
        context._metadata("deadline")
    assert "deadline" not in context._cache
    context._check = None
    assert context._metadata("deadline")[1]
