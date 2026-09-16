"""Phase C environment: verify, search ladder, semantic_annotate, preload.

Ladder/annotator are tested in-process with a fake RPC; the final test runs
the full docdb preload through the real sandbox.
"""

import sqlite3

import pytest

from rnsr.env.annotate import Annotator
from rnsr.env.search import Ladder
from rnsr.env.verify import Verifier
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


def _corpus_parse(path):
    text = (
        "ACME 2023 results. Net revenue was $3,234 million. "
        "The Widgets segment grew rapidly. Zylon-7 output doubled. "
    )
    return ParsedDocument(
        doc_id="acme", source_path=str(path), sha256="b" * 64, n_pages=1,
        parser="fake",
        elements=[
            Element("heading", "Item 7", 1, heading_level=1),
            Element("text", text, 1),
        ],
        tables=[RawTable(
            page=1,
            header=["Segment", "Revenue ($M)"],
            rows=[["Widgets", "$1,234"], ["Gadgets", "$2,000"], ["Total", "$3,234"]],
            extractor="docling",
        )],
    )


@pytest.fixture
def corpus(tmp_path):
    out = tmp_path / "corpus.db"
    ingest([tmp_path / "acme.pdf"], out, parse=_corpus_parse)
    return out


@pytest.fixture
def env(corpus):
    from rnsr.db.artifact import CorpusDB

    with CorpusDB(corpus) as ro:
        manifest = ro.manifest_dict()
        doc = ro.doc_dict()
    conn = sqlite3.connect(corpus)
    yield conn, doc, manifest
    conn.close()


class FakeRpc:
    def __init__(self, responder=None):
        self.requests = []
        self.responder = responder or (lambda req: {"results": ["NONE"] * len(req.get("prompts", []))})

    def __call__(self, request):
        self.requests.append(request)
        if request["op"] == "log":
            return {}
        return self.responder(request)


class TestVerify:
    def test_exact_quote_passes_with_offsets(self, env):
        _, doc, _ = env
        v = Verifier(doc).verify("3234", ["Net revenue was $3,234 million"])
        assert v["passed"]
        q = v["quotes"][0]
        start, end = q["char_start"], q["char_end"]
        assert doc["acme"][start:end] == "Net revenue was $3,234 million"

    def test_normalization_matrix(self, env):
        _, doc, _ = env
        verifier = Verifier(doc)
        # curly quotes, extra whitespace, case, unicode dash all normalize
        assert verifier.verify("x", ["net  Revenue was $3,234   million"])["passed"]

    def test_fabricated_quote_fails(self, env):
        _, doc, _ = env
        v = Verifier(doc).verify("x", ["Net revenue was $9,999 million"])
        assert not v["passed"]
        assert v["quotes"][0]["matched"] is False

    def test_mixed_quotes_fail_overall(self, env):
        _, doc, _ = env
        v = Verifier(doc).verify("x", ["Zylon-7 output doubled", "made up quote"])
        assert not v["passed"]
        assert [q["matched"] for q in v["quotes"]] == [True, False]


class TestLadder:
    def _ladder(self, env, rpc=None):
        conn, doc, manifest = env
        return Ladder(conn=conn, doc=doc, manifest=manifest, rpc=rpc or FakeRpc())

    def test_rung0_numeric_needle_via_sql(self, env):
        hits = self._ladder(env).search("What was Widgets revenue in $M?", rung=0)
        assert hits and hits[0]["kind"] == "sql"
        assert hits[0]["rows"]["revenue_m"] == 1234
        assert hits[0]["provenance"]["table"] == "t_acme_001"

    def test_rung1_grep_provenance(self, env):
        conn, doc, _ = env
        hits = self._ladder(env).search("Zylon-7", rung=1)
        assert hits
        p = hits[0]["provenance"]
        assert doc[p["doc_id"]][p["char_start"]:p["char_end"]].lower() == "zylon-7"

    def test_rung2_fts(self, env):
        hits = self._ladder(env).search("revenue segment", rung=2)
        assert hits and hits[0]["kind"] == "chunk"
        assert "chunk_id" in hits[0]["provenance"]

    def test_auto_escalation_returns_first_yielding_rung(self, env):
        hits = self._ladder(env).search("Widgets revenue")
        assert hits[0]["rung"] in (0, 1, 2)

    def test_rung3_expansion_finds_via_new_term(self, env):
        def responder(req):
            return {"results": ["Zylon-7\nsuperfiber"] * len(req["prompts"])}

        ladder = self._ladder(env, FakeRpc(responder))
        hits = ladder.search("the doubled experimental material", rung=3)
        assert hits and hits[0]["rung"] == 3

    def test_rung5_gated_behind_estimate(self, env):
        # nothing matches -> auto escalation must NOT run the sweep
        rpc = FakeRpc()
        hits = self._ladder(env, rpc).search("qqqxyzzy unfindable")
        assert hits[0]["kind"] == "estimate"
        assert hits[0]["estimated_sub_calls"] >= 1
        assert all(r["op"] != "llm_batch" or "chunk ids" not in r["prompts"][0]
                   for r in rpc.requests if r["op"] == "llm_batch")

    def test_rung5_explicit_sweep(self, env):
        def responder(req):
            if "chunk ids" in req["prompts"][0]:
                return {"results": ["1"]}
            return {"results": ["NONE"] * len(req["prompts"])}

        hits = self._ladder(env, FakeRpc(responder)).search("anything", rung=5)
        assert hits and hits[0]["rung"] == 5
        assert hits[0]["provenance"]["chunk_id"] == 1

    def test_untrusted_tables_skipped_in_rung0(self, env):
        conn, doc, manifest = env
        for t in manifest["tables"]:
            t["status"] = "untrusted"
        assert Ladder(conn=conn, doc=doc, manifest=manifest,
                      rpc=FakeRpc()).search("Widgets revenue", rung=0) == []


class TestAnnotate:
    def _annotator(self, env, responder):
        conn, _, _ = env
        return Annotator(conn, FakeRpc(responder)), conn

    @staticmethod
    def _label_responder(req):
        # label rows by parsing the numbered JSON lines: Total -> total, else item
        out = []
        for p in req["prompts"]:
            lines = []
            for line in p.splitlines():
                import re

                m = re.match(r"^(\d+)\. (\{.*\})$", line)
                if m:
                    label = "total" if "Total" in m.group(2) else "item"
                    lines.append(f"{m.group(1)}. {label}")
            out.append("\n".join(lines))
        return {"results": out}

    def test_annotate_writes_column_and_log(self, env):
        annotator, conn = self._annotator(env, self._label_responder)
        result = annotator.annotate("t_acme_001", "row_kind", "total row or item?")
        assert result == {"rows": 3, "failed": 0, "coverage": 1.0,
                          "sample": result["sample"]}
        rows = conn.execute(
            "SELECT segment, row_kind FROM t_acme_001 ORDER BY rowid"
        ).fetchall()
        assert rows == [("Widgets", "item"), ("Gadgets", "item"), ("Total", "total")]
        log = conn.execute("SELECT rows_written, rows_failed FROM annotation_log").fetchone()
        assert log == (3, 0)

    def test_idempotent_second_call_noop(self, env):
        annotator, _ = self._annotator(env, self._label_responder)
        annotator.annotate("t_acme_001", "row_kind", "total row or item?")
        n_before = len(annotator.rpc.requests)
        again = annotator.annotate("t_acme_001", "row_kind", "total row or item?")
        assert again["noop"] is True
        assert len(annotator.rpc.requests) == n_before  # zero new sub-calls

    def test_force_reruns(self, env):
        annotator, _ = self._annotator(env, self._label_responder)
        annotator.annotate("t_acme_001", "row_kind", "total row or item?")
        redo = annotator.annotate("t_acme_001", "row_kind", "total row or item?",
                                  force=True)
        assert redo.get("noop") is None and redo["rows"] == 3

    def test_worked_example_annotate_then_exact_sql(self, env):
        # §4.1: the quadratic part is a self-join — exact, instant, free.
        annotator, conn = self._annotator(env, self._label_responder)
        annotator.annotate("t_acme_001", "row_kind", "classify")
        pairs = conn.execute("""
            WITH items AS (SELECT segment FROM t_acme_001 WHERE row_kind = 'item')
            SELECT a.segment, b.segment FROM items a JOIN items b
            ON a.segment < b.segment
        """).fetchall()
        assert pairs == [("Gadgets", "Widgets")]

    def test_malformed_reply_retried_then_counted_failed(self, env):
        calls = {"n": 0}

        def responder(req):
            calls["n"] += 1
            return {"results": ["garbled nonsense"] * len(req["prompts"])}

        annotator, _ = self._annotator(env, responder)
        result = annotator.annotate("t_acme_001", "bad_col", "classify")
        assert calls["n"] == 2          # initial + strict re-ask
        assert result["rows"] == 0 and result["failed"] == 3


class TestDocdbPreloadInSandbox:
    async def test_full_environment_through_sandbox(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        async def llm_batch(req):
            return {"results": ["ok"] * len(req["prompts"])}

        async def log(req):
            return {}

        repl = SandboxedRepl(rpc_handlers={"llm_batch": llm_batch, "log": log})
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell(
                "print(sorted(manifest['tables'][0]['schema'][0].keys()))\n"
                "print(db.execute('SELECT sum(revenue_m) FROM t_acme_001 "
                "WHERE segment != \\'Total\\'').fetchone()[0])\n"
                "hits = search('Widgets revenue', rung=0)\n"
                "print(hits[0]['rows']['revenue_m'])\n"
                "print(verify('x', ['Zylon-7 output doubled'])['passed'])\n"
            )
            assert res.ok, res.error
            lines = res.stdout.strip().splitlines()
            assert lines[1] == "3234"      # exact SQL over source data
            assert lines[2] == "1234"      # ladder rung 0
            assert lines[3] == "True"      # verify against retained text
        finally:
            await repl.close()


class TestVerifiedFinal:
    async def test_final_without_quotes_rejected(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell("FINAL('3234')")
            assert not res.ok
            assert "requires supporting source quotes" in res.error
        finally:
            await repl.close()

    async def test_final_with_fabricated_quote_rejected(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell(
                "FINAL('3234', quotes=['Net revenue was $9,999 million'])")
            assert not res.ok
            assert "FINAL rejected" in res.error
            # loop survives; a corrected FINAL passes
            res = await repl.exec_cell(
                "FINAL('3234', quotes=['Net revenue was $3,234 million'])")
            assert res.final is not None
            assert res.final["verification"]["passed"] is True
        finally:
            await repl.close()

    async def test_final_var_requires_supporting_quotes(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell(
                "total = db.execute('SELECT sum(revenue_m) FROM t_acme_001 "
                "WHERE segment != \\'Total\\'').fetchone()[0]\nFINAL_VAR(total)")
            assert not res.ok and res.final is None
            checked = await repl.exec_cell("FINAL_VAR(total, quotes=['Net revenue was $3,234 million'])")
            assert checked.final["value"] == 3234
            assert checked.final["verification"]["passed"]
        finally:
            await repl.close()

    async def test_repeated_fabricated_quotes_never_accepted(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            r1 = await repl.exec_cell("FINAL('x', quotes=['fabricated one'])")
            r2 = await repl.exec_cell("FINAL('x', quotes=['fabricated two'])")
            assert not r1.ok and not r2.ok
            r3 = await repl.exec_cell("FINAL('x', quotes=['fabricated three'])")
            assert not r3.ok and r3.final is None
        finally:
            await repl.close()


class TestBatchFinalVerified:
    async def test_batch_with_fabricated_quote_rejected(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell(
                "FINAL_BATCH({'q1': '3234'}, quotes={'q1': ['made up text']})")
            assert not res.ok and res.final is None
            assert "FINAL_BATCH rejected" in (res.error or "")
        finally:
            await repl.close()

    async def test_batch_with_matching_quotes_accepted(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell(
                "FINAL_BATCH({'q1': '3234', 'q2': 'NOT_FOUND'}, "
                "quotes={'q1': ['Net revenue was $3,234 million']})")
            assert res.final is not None
            assert res.final["value"] == {"q1": "3234", "q2": "NOT_FOUND"}
            assert res.final["verification"]["q1"]["passed"]
        finally:
            await repl.close()

    async def test_batch_without_quotes_accepted(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell("FINAL_BATCH({'q1': 'Yes'})")
            assert res.final is not None
            assert res.final["value"] == {"q1": "Yes"}
        finally:
            await repl.close()

    async def test_repeated_batch_with_failed_field_never_accepted(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        repl = SandboxedRepl()
        await repl.start(mode="docdb", corpus_db=str(corpus))
        cell = (
            "FINAL_BATCH({'q1': '3234', 'q2': 'nope'}, "
            "quotes={'q1': ['Net revenue was $3,234 million'], "
            "'q2': ['this quote is fabricated']})"
        )
        try:
            assert not (await repl.exec_cell(cell)).ok
            assert not (await repl.exec_cell(cell)).ok
            r3 = await repl.exec_cell(cell)
            assert not r3.ok and r3.final is None
        finally:
            await repl.close()


class TestSchemaMap:
    """schema_map proposes column correspondences; never applies them (§9)."""

    @pytest.fixture
    def two_table_corpus(self, tmp_path):
        def parse(path):
            p = _corpus_parse(path)
            p.tables.append(RawTable(
                page=1,
                header=["Business Unit", "Net Revenue"],   # drifted headers
                rows=[["Widgets", "1,300"], ["Gadgets", "2,100"], ["Total", "3,400"]],
                extractor="docling",
            ))
            return p

        out = tmp_path / "two.db"
        ingest([tmp_path / "acme.pdf"], out, parse=parse)
        return out

    async def test_proposals_surface_in_repl(self, two_table_corpus):
        import json as _json

        from rnsr.env.sandbox import SandboxedRepl

        proposal = _json.dumps([
            {"a": "segment", "b": "business_unit", "confidence": 0.9,
             "reason": "same entity labels"},
            {"a": "revenue_m", "b": "net_revenue", "confidence": 0.8,
             "reason": "revenue figures"},
        ])

        async def llm_batch(req):
            # prompt must describe both tables (columns + sample rows)
            p = req["prompts"][0]
            assert "t_acme_001" in p and "t_acme_002" in p
            assert "segment" in p and "business_unit" in p
            return {"results": [proposal]}

        async def log(req):
            return {}

        repl = SandboxedRepl(rpc_handlers={"llm_batch": llm_batch, "log": log})
        await repl.start(mode="docdb", corpus_db=str(two_table_corpus))
        try:
            res = await repl.exec_cell(
                "props = schema_map('t_acme_001', 't_acme_002')\n"
                "print(len(props), props[0]['a'], props[0]['b'])\n"
                "row = db.execute('SELECT count(*) FROM t_acme_002').fetchone()\n"
                "print(row[0])"
            )
            assert res.ok, res.error
            lines = res.stdout.strip().splitlines()
            assert lines[0] == "2 segment business_unit"
            assert lines[1] == "3"          # tables untouched — proposals only
        finally:
            await repl.close()

    async def test_garbage_reply_yields_empty_list(self, two_table_corpus):
        from rnsr.env.sandbox import SandboxedRepl

        async def llm_batch(req):
            return {"results": ["I think these tables look similar, roughly."]}

        async def log(req):
            return {}

        repl = SandboxedRepl(rpc_handlers={"llm_batch": llm_batch, "log": log})
        await repl.start(mode="docdb", corpus_db=str(two_table_corpus))
        try:
            res = await repl.exec_cell("print(schema_map('t_acme_001', 't_acme_002'))")
            assert res.ok, res.error
            assert res.stdout.strip() == "[]"
        finally:
            await repl.close()


class TestAnnotateVotes:
    def test_majority_overrides_one_bad_pass(self, env):
        conn, _, _ = env
        calls = {"n": 0}

        def responder(req):
            calls["n"] += 1
            out = []
            for p in req["prompts"]:
                import re
                lines = []
                for line in p.splitlines():
                    m = re.match(r"^(\d+)\. (\{.*\})$", line)
                    if m:
                        # pass 2 (calls 2) mislabels everything; 1 and 3 are right
                        if calls["n"] == 2:
                            label = "wrong"
                        else:
                            label = "total" if "Total" in m.group(2) else "item"
                        lines.append(f"{m.group(1)}. {label}")
                out.append("\n".join(lines))
            return {"results": out}

        annotator = Annotator(conn, FakeRpc(responder))
        result = annotator.annotate("t_acme_001", "kind3", "classify", votes=3)
        assert result["coverage"] == 1.0
        rows = conn.execute(
            "SELECT segment, kind3 FROM t_acme_001 ORDER BY rowid").fetchall()
        assert rows == [("Widgets", "item"), ("Gadgets", "item"), ("Total", "total")]
        assert calls["n"] == 3   # three labeling passes

    def test_votes_changes_idempotency_key(self, env):
        conn, _, _ = env
        annotator = Annotator(conn, FakeRpc(TestAnnotate._label_responder))
        annotator.annotate("t_acme_001", "k1", "classify", votes=1)
        # same prompt with different votes is a distinct annotation, not a noop
        second = annotator.annotate("t_acme_001", "k1", "classify", votes=3)
        assert second.get("noop") is None


class TestParentAnnotationBoundary:
    @pytest.mark.parametrize('column', ['rowid', 'oid', '_rowid_', '_page', 'revenue_m', 'SEGMENT'])
    def test_source_and_implicit_columns_cannot_be_annotation_targets(self, env, column):
        conn, _, _ = env
        annotator = Annotator(conn, FakeRpc(TestAnnotate._label_responder))
        with pytest.raises(ValueError, match='cannot overwrite'):
            annotator.annotate('t_acme_001', column, 'classify')

    @pytest.mark.parametrize('predicate', [
        '1; DROP TABLE doc_text',
        'randomblob(1000000000) IS NOT NULL',
        "load_extension('/tmp/untrusted') IS NULL",
    ])
    def test_where_is_readonly_and_disallows_unbounded_functions(self, env, predicate):
        conn, _, _ = env
        annotator = Annotator(conn, FakeRpc(TestAnnotate._label_responder))
        with pytest.raises(sqlite3.Error):
            annotator.annotate('t_acme_001', 'class_label', 'classify', where=predicate)
        assert conn.execute('SELECT count(*) FROM doc_text').fetchone()[0] > 0
        assert conn.execute('SELECT count(*) FROM annotation_log').fetchone()[0] == 0

    async def test_annotation_rpc_persists_only_declared_column(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        async def label(req):
            return TestAnnotate._label_responder(req)

        async with SandboxedRepl(rpc_handlers={'llm_batch': label}) as repl:
            await repl.start(mode='docdb', corpus_db=str(corpus))
            result = await repl.exec_cell(
                "semantic_annotate('t_acme_001', 'category', 'classify')\n"
                "print(db.execute('SELECT count(category) FROM t_acme_001').fetchone()[0])\n"
                "print(any(c['name'] == 'category' for c in manifest['tables'][0]['schema']))")
            assert result.ok, result.error
            assert result.stdout == '3\nTrue\n'

    async def test_timed_out_annotation_cancels_provider_and_no_write(self, corpus):
        import asyncio

        from rnsr.env.sandbox import SandboxedRepl
        from rnsr.errors import SandboxError

        stopped = asyncio.Event()
        async def delayed(req):
            try:
                await asyncio.sleep(30)
            finally:
                stopped.set()

        async with SandboxedRepl(rpc_handlers={'llm_batch': delayed}) as repl:
            await repl.start(mode='docdb', corpus_db=str(corpus))
            with pytest.raises(SandboxError, match='wall-clock'):
                await repl.exec_cell(
                    "semantic_annotate('t_acme_001', 'later', 'classify')", timeout=0.1)
            await asyncio.wait_for(stopped.wait(), timeout=3)
        with sqlite3.connect(corpus) as conn:
            assert conn.execute('SELECT count(*) FROM annotation_log').fetchone()[0] == 0
            assert 'later' not in {r[1] for r in conn.execute('PRAGMA table_info(t_acme_001)')}


class TestFinalQuoteParity:
    async def test_batch_requires_quotes_for_each_value_field(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        async with SandboxedRepl() as repl:
            await repl.start(mode='docdb', corpus_db=str(corpus))
            unquoted = await repl.exec_cell("FINAL_BATCH({'q1': 'Mallory Smith', 'q2': '123 Somewhere St'})")
            assert not unquoted.ok and unquoted.final is None
            partial = await repl.exec_cell(
                "FINAL_BATCH({'q1': '3234', 'q2': 'Invented Address'}, "
                "quotes={'q1': ['Net revenue was $3,234 million']})")
            assert not partial.ok and partial.final is None

    def test_repeated_quote_misses_normalize_corpus_only_once(self):
        class CountingDocs(dict):
            reads = 0
            def __getitem__(self, key):
                self.reads += 1
                return super().__getitem__(key)
        documents = CountingDocs({f'doc-{i}': f'Retained document number {i}.' for i in range(200)})
        verifier = Verifier(documents)
        try:
            report = verifier.verify('x', ['absent one', 'absent two', 'absent three'])
            assert not report['passed']
            assert documents.reads == 200
            verifier.verify('y', ['another missing quote'])
            assert documents.reads == 200
        finally:
            verifier.close()


def test_quote_unicode_casefold_offsets_remain_source_offsets():
    original = 'ΟΔΟΣ İstanbul Straße'
    verifier = Verifier({'unicode': original})
    try:
        report = verifier.verify('street', ['οδός'.replace('ό', 'ο'), 'İstanbul', 'STRASSE'])
        assert report['passed']
        last = report['quotes'][-1]
        assert original[last['char_start']:last['char_end']] == 'Straße'
    finally:
        verifier.close()


class TestConcurrentSourceChanges:
    def test_replacement_during_annotation_cannot_receive_old_labels(self, corpus):
        from rnsr.ingest.lifecycle import replace_document

        def replacement(path):
            parsed = _corpus_parse(path)
            parsed.tables[0].rows = [['Replacement', '$90'], ['Total', '$90']]
            parsed.elements[1].text = 'Replacement source. Revenue is $90.'
            return parsed

        def replace_then_label(request):
            replace_document(corpus, 'acme', corpus.parent / 'replacement.pdf', parse=replacement)
            return TestAnnotate._label_responder(request)

        with sqlite3.connect(corpus) as conn:
            annotator = Annotator(conn, replace_then_label)
            with pytest.raises(ValueError, match='source changed'):
                annotator.annotate('t_acme_001', 'stale_label', 'classify')
            assert conn.execute('SELECT segment FROM t_acme_001 ORDER BY rowid').fetchall() == [
                ('Replacement',), ('Total',)]
            assert 'stale_label' not in {r[1] for r in conn.execute('PRAGMA table_info(t_acme_001)')}
            assert conn.execute('SELECT count(*) FROM annotation_log').fetchone()[0] == 0

    async def test_previous_verified_quote_rejected_after_replacement(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl
        from rnsr.ingest.lifecycle import replace_document

        def replacement(path):
            parsed = _corpus_parse(path)
            parsed.elements[1].text = 'All retained text has been replaced.'
            return parsed

        async with SandboxedRepl() as repl:
            await repl.start(mode='docdb', corpus_db=str(corpus))
            cell = "FINAL('3234', quotes=['Net revenue was $3,234 million'])"
            assert (await repl.exec_cell(cell)).final['verification']['passed']
            replace_document(corpus, 'acme', corpus.parent / 'replacement.pdf', parse=replacement)
            result = await repl.exec_cell(cell)
            assert not result.ok and result.final is None
            assert 'source changed' in result.error


class TestParentWorkLimits:
    @pytest.mark.parametrize('where', [
        "1 UNION ALL SELECT rowid, segment, revenue_m FROM t_acme_001",
        "EXISTS (SELECT 1 FROM t_acme_001)",
        "EXISTS (WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT 1 FROM n)",
    ])
    def test_annotation_predicate_cannot_expand_rows_or_run_subqueries(self, env, where):
        conn, _, _ = env
        with pytest.raises((sqlite3.Error, ValueError)):
            Annotator(conn, FakeRpc()).annotate('t_acme_001', 'label', 'classify', where=where)

    def test_final_field_limit_boundary(self):
        from rnsr.env.finalize import MAX_FINAL_FIELDS, validate_final

        answers = {str(i): 'No' for i in range(MAX_FINAL_FIELDS)}
        assert len(validate_final(answers, {}, None, batch=True)) == MAX_FINAL_FIELDS
        answers['overflow'] = 'No'
        with pytest.raises(ValueError, match='fields'):
            validate_final(answers, {}, None, batch=True)

    def test_final_quote_work_limit_boundaries(self):
        from rnsr.env.finalize import MAX_QUOTE_CHARS, MAX_TOTAL_QUOTE_CHARS, validate_final

        class MatchingVerifier:
            def verify(self, answer, quotes):
                return {'passed': True, 'answer': answer, 'quotes': quotes}
        verifier = MatchingVerifier()
        assert validate_final('v', ['x' * MAX_QUOTE_CHARS], verifier)['passed']
        with pytest.raises(ValueError, match='short source quotes'):
            validate_final('v', ['x' * (MAX_QUOTE_CHARS + 1)], verifier)
        count = MAX_TOTAL_QUOTE_CHARS // MAX_QUOTE_CHARS
        answers = {str(i): 'v' for i in range(count)}
        quotes = {str(i): ['x' * MAX_QUOTE_CHARS] for i in range(count)}
        assert len(validate_final(answers, quotes, verifier, batch=True)) == count
        answers['extra'] = 'v'
        quotes['extra'] = ['x']
        with pytest.raises(ValueError, match='quote characters'):
            validate_final(answers, quotes, verifier, batch=True)

    def test_verifier_deadline_rolls_back_partial_index(self, monkeypatch):
        import time

        class SlowDocs(dict):
            def __getitem__(self, key):
                time.sleep(0.02)
                return super().__getitem__(key)
        verifier = Verifier(SlowDocs({'one': 'One source', 'two': 'Two source'}))
        try:
            verifier.set_deadline(time.monotonic() + 0.01)
            with pytest.raises(TimeoutError, match='wall-clock'):
                verifier.verify('v', ['Two source'])
            verifier.set_deadline(None)
            assert verifier.verify('v', ['Two source'])['passed']
        finally:
            verifier.close()

    async def test_parent_verification_uses_remaining_cell_deadline(self, corpus):
        import time

        from rnsr.env.sandbox import SandboxedRepl
        from rnsr.errors import SandboxError

        async with SandboxedRepl() as repl:
            await repl.start(mode='docdb', corpus_db=str(corpus))
            original = repl._verifier._doc
            class SlowDoc:
                def __iter__(self):
                    return iter(original)
                def __getitem__(self, key):
                    time.sleep(0.08)
                    return original[key]
            repl._verifier._doc = SlowDoc()
            with pytest.raises(SandboxError, match='verification exceeded'):
                await repl.exec_cell(
                    "FINAL('3234', quotes=['Net revenue was $3,234 million'])", timeout=0.05)
