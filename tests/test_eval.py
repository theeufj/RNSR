"""Eval layer: scoring, percentiles, gate logic, end-to-end run on MockLLM."""

import math

import pytest

from rnsr.config import Settings
from rnsr.eval.datasets.oolong import synthetic_oolong
from rnsr.eval.harness import run_eval
from rnsr.eval.metrics import (
    EvalResult,
    gate_report,
    percentile,
    score_answer,
    summarize,
)
from rnsr.harness.loop import RootRunner
from rnsr.llm.mock import MockLLM


class TestScoring:
    def test_exact_text(self):
        assert score_answer("Paris", "paris")
        assert not score_answer("London", "Paris")

    def test_numeric_formats_and_tolerance(self):
        assert score_answer("3,234", "3234")
        assert score_answer("$3,234 million", "3234")
        assert score_answer(3234.0, "3,234")
        assert score_answer("3250", "3234", numeric_rel_tol=0.01)
        assert not score_answer("3300", "3234")

    def test_containment_bounded(self):
        assert score_answer("The answer is Paris.", "Paris")
        assert not score_answer("Paris " + "waffle " * 50, "Paris")

    def test_none_never_correct(self):
        assert not score_answer(None, "42")


class TestAggregates:
    def _result(self, **kw):
        base = dict(qid="q", task_class="numeric", predicted="1", gold="1",
                    correct=True, status="final", latency_s=1.0, cost_usd=0.01,
                    sub_calls=2, iterations=3)
        base.update(kw)
        return EvalResult(**base)

    def test_percentiles(self):
        assert percentile([1, 2, 3, 4], 0.5) == 2.5
        assert math.isnan(percentile([], 0.5))

    def test_summary_by_class(self):
        results = [
            self._result(correct=True, task_class="numeric"),
            self._result(correct=False, task_class="numeric"),
            self._result(correct=True, task_class="lookup"),
        ]
        s = summarize(results)
        assert s["accuracy_by_class"] == {"lookup": 1.0, "numeric": 0.5}
        assert s["status_counts"] == {"final": 3}

    def test_gate_pass_and_fail(self):
        docdb = {"accuracy_by_class": {"numeric": 0.9, "lookup": 0.8},
                 "cost_usd": {"p50": 0.10}}
        classic = {"accuracy_by_class": {"numeric": 0.6, "lookup": 0.8},
                   "cost_usd": {"p50": 0.12}}
        assert gate_report(docdb, classic)["pass"] is True

        worse = {"accuracy_by_class": {"numeric": 0.5, "lookup": 0.8},
                 "cost_usd": {"p50": 0.10}}
        report = gate_report(worse, classic)
        assert report["pass"] is False
        assert report["checks"]["beats_classic[numeric]"] is False


class TestSyntheticOolong:
    def test_deterministic_and_countable(self):
        a, b = synthetic_oolong(n_lines=100, n_items=3), synthetic_oolong(n_lines=100, n_items=3)
        assert [i.gold for i in a] == [i.gold for i in b]
        assert sum(int(i.gold) for i in synthetic_oolong(n_lines=100, n_items=6)) == 100


class TestRunEval:
    async def test_end_to_end_classic_mock(self, tmp_path):
        items = synthetic_oolong(n_lines=30, n_items=2)
        # scripted root: count lines mentioning the target label pattern via
        # a FINAL of the known gold (mock can't reason; verify plumbing only)
        root = MockLLM()
        root.rule(r"How many lines are of type", f"```python\nFINAL({items[0].gold})\n```")
        runner = RootRunner(root_client=root, root_model="m", sub_client=MockLLM(),
                            sub_model="m", settings=Settings(max_root_iters=3))
        results, summary = await run_eval(items[:1], "rlm-classic", runner,
                                          run_dir=tmp_path)
        assert summary["n"] == 1
        assert results[0].correct
        assert (tmp_path / "results.jsonl").exists()
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "trajectories" / f"{items[0].qid}.jsonl").exists()


class TestJudge:
    async def test_judge_yes_no_unparseable(self):
        from rnsr.eval.metrics import judge_answer

        yes = MockLLM(default="YES")
        no = MockLLM(default="NO — the numbers differ")
        weird = MockLLM(default="It depends on interpretation")
        dead = MockLLM(fail_times=99)
        assert await judge_answer(yes, "m", "q", "p", "g") is True
        assert await judge_answer(no, "m", "q", "p", "g") is False
        assert await judge_answer(weird, "m", "q", "p", "g") is None
        assert await judge_answer(dead, "m", "q", "p", "g") is None

    async def test_run_eval_judge_rescues_semantic_match(self, tmp_path):
        items = synthetic_oolong(n_lines=20, n_items=1)
        items[0].gold = "No, the company is managing its CAPEX well."
        root = MockLLM(default="```python\nFINAL('No, it is not capital-intensive.')\n```")
        sub = MockLLM().rule(r"agree with the reference", "YES")
        runner = RootRunner(root_client=root, root_model="m", sub_client=sub,
                            sub_model="m", settings=Settings(max_root_iters=2))
        results, summary = await run_eval(items[:1], "rlm-classic", runner,
                                          run_dir=tmp_path)
        assert results[0].correct and results[0].scored_by == "judge"
        assert summary["scored_by_counts"] == {"judge": 1}

    async def test_string_match_skips_judge(self, tmp_path):
        items = synthetic_oolong(n_lines=20, n_items=1)
        root = MockLLM()
        root.rule(r"How many lines", f"```python\nFINAL({items[0].gold})\n```")
        sub = MockLLM(default="NO")   # would say NO if consulted
        runner = RootRunner(root_client=root, root_model="m", sub_client=sub,
                            sub_model="m", settings=Settings(max_root_iters=2))
        results, _ = await run_eval(items[:1], "rlm-classic", runner,
                                    run_dir=tmp_path)
        assert results[0].correct and results[0].scored_by == "string"
        assert not any(c["kind"] == "complete" and "agree with" in c["prompt"]
                       for c in sub.calls)


class TestNumericGoldClassifier:
    def test_value_golds_numeric(self):
        from rnsr.eval.datasets.financebench import _is_numeric_gold

        for g in ("$1577.00", "8.70", "~1.5x", "25%", "3,234 million",
                  "approximately $ 8.9 billion", "(1,234)"):
            assert _is_numeric_gold(g), g

    def test_sentence_golds_textual(self):
        from rnsr.eval.datasets.financebench import _is_numeric_gold

        for g in ("No, the quick ratio was 0.96 by Jun'23.",
                  "Operating margin decreased by 1.7% in FY2022.",
                  "Yes, they distribute dividends every quarter."):
            assert not _is_numeric_gold(g), g


class TestCorpusCacheValidation:
    def test_empty_cached_corpus_rebuilt(self, tmp_path):
        import sqlite3

        from rnsr.db import schema as dbschema
        from rnsr.eval.harness import _corpus_valid

        empty = tmp_path / "empty.db"
        conn = sqlite3.connect(empty)
        dbschema.create_corpus_db(conn)
        conn.close()
        assert not _corpus_valid(empty, n_sources=1)

    def test_garbage_file_invalid(self, tmp_path):
        from rnsr.eval.harness import _corpus_valid

        junk = tmp_path / "junk.db"
        junk.write_bytes(b"<!DOCTYPE html>not a database")
        assert not _corpus_valid(junk, n_sources=1)


class TestOolongLoader:
    def test_parse_gold_variants(self):
        from rnsr.eval.datasets.oolong import parse_gold

        assert parse_gold("['spam']") == "spam"
        assert parse_gold("['a', 'b']") == "a, b"
        assert parse_gold("[3]") == "3"
        assert parse_gold("plain") == "plain"

    @pytest.mark.live
    def test_real_dataset_loads(self):
        pytest.importorskip("datasets")
        from rnsr.eval.datasets.oolong import load_oolong

        items = load_oolong(limit=14)
        assert len(items) == 14
        lengths = {i.meta["context_len"] for i in items}
        assert len(lengths) >= 6          # round-robin spans buckets
        assert all(1024 <= i.meta["context_len"] <= 65536 for i in items)
        assert all(i.context and i.gold for i in items)
        # unlabeled variant only: contexts must not carry gold label markup
        assert all("label:" not in i.context[:200].lower() for i in items)


class TestIngestText:
    def test_text_corpus_end_to_end(self, tmp_path):
        from rnsr.db import fts
        from rnsr.db.artifact import CorpusDB
        from rnsr.ingest.pipeline import ingest_text

        text = "User 1 asked about spam.\nUser 2 asked about geography.\n" * 30
        report = ingest_text({"conv": text}, tmp_path / "t.db")
        assert report.n_chunks >= 1
        assert len(report.tables) == 1  # the (line_no, text) lines table
        with CorpusDB(tmp_path / "t.db") as corpus:
            assert corpus.doc_ids() == ["conv"]
            assert "geography" in corpus.full_text("conv")
            assert fts.match(corpus.conn, "geography")

    def test_docdb_env_uses_text_corpus(self, tmp_path):
        from rnsr.config import Settings
        from rnsr.eval.datasets.base import EvalItem
        from rnsr.eval.harness import _env_for

        item = EvalItem(qid="q", question="?", gold="g",
                        context="line one\nline two with needle\n" * 20)
        env = _env_for(item, "docdb", tmp_path, Settings())
        assert env.mode == "docdb"
        assert env.corpus_db and env.manifest["documents"]
        # same context -> same cached corpus
        env2 = _env_for(item, "docdb", tmp_path, Settings())
        assert env2.corpus_db == env.corpus_db


class TestIngestTextLinesTable:
    def test_lines_table_supports_annotate_groupby(self, tmp_path):
        import sqlite3

        from rnsr.env.annotate import Annotator
        from rnsr.ingest.pipeline import ingest_text

        text = "spam offer now\nhi mum\nwin prize cash\nsee you at 5\n"
        report = ingest_text({"msgs": text}, tmp_path / "m.db")
        assert report.tables and report.tables[0].status == "trusted"
        table = report.tables[0].name

        def responder(req):
            out = []
            for p in req["prompts"]:
                lines = []
                import re
                for line in p.splitlines():
                    m = re.match(r"^(\d+)\. (\{.*\})$", line)
                    if m:
                        label = "spam" if ("offer" in m.group(2) or "prize" in m.group(2)) else "ham"
                        lines.append(f"{m.group(1)}. {label}")
                out.append("\n".join(lines))
            return {"results": out}

        class Rpc:
            def __call__(self, req):
                return {} if req["op"] == "log" else responder(req)

        conn = sqlite3.connect(tmp_path / "m.db")
        result = Annotator(conn, Rpc()).annotate(table, "label", "spam or ham?")
        assert result["coverage"] == 1.0
        counts = dict(conn.execute(
            f'SELECT label, count(*) FROM "{table}" GROUP BY label').fetchall())
        assert counts == {"spam": 2, "ham": 2}
        # line order preserved via line_no
        first = conn.execute(f'SELECT text FROM "{table}" WHERE line_no = 1').fetchone()[0]
        assert first == "spam offer now"
        conn.close()


class TestLegalLoaders:
    @pytest.mark.live
    def test_cuad_loads_grouped_by_contract(self):
        pytest.importorskip("datasets")
        from rnsr.eval.datasets.legal import load_cuad

        items = load_cuad(limit=12)
        assert len(items) == 12
        assert all(i.context for i in items)
        contracts = {i.meta["contract"] for i in items}
        assert len(contracts) <= 4          # grouped: few corpora to ingest
        assert any(i.task_class == "absent-clause" for i in items) or \
               any(i.task_class == "extraction" for i in items)

    @pytest.mark.live
    def test_contractnli_three_way(self):
        pytest.importorskip("datasets")
        from rnsr.eval.datasets.legal import load_contractnli

        items = load_contractnli(limit=10)
        assert len(items) == 10
        assert all(i.gold in ("entailment", "contradiction", "neutral") for i in items)
        assert all(i.context is None for i in items)   # short reasoning; classic-shaped

    @pytest.mark.live
    def test_legalbench_labels_quoted(self):
        pytest.importorskip("datasets")
        from rnsr.eval.datasets.legal import load_legalbench

        items = load_legalbench(limit=8)
        assert len(items) == 8
        assert all("Answer with exactly one of:" in i.question for i in items)
        assert all(i.task_class.startswith("legalbench:") for i in items)


class TestRagBaselines:
    async def test_bm25_rag_end_to_end(self, tmp_path):
        from rnsr.eval.datasets.base import EvalItem

        item = EvalItem(
            qid="rag-1", question="What was the widget revenue?",
            gold="1234",
            context="Filler line about weather.\nWidget revenue was 1234 dollars.\n" * 5,
        )
        root = MockLLM().rule(r"excerpts", "The widget revenue was 1234 dollars.")
        runner = RootRunner(root_client=root, root_model="m", sub_client=MockLLM(),
                            sub_model="m", settings=Settings())
        results, summary = await run_eval([item], "bm25-rag", runner, run_dir=tmp_path)
        assert results[0].correct
        assert results[0].iterations == 1
        # retrieval fed real excerpts into the single prompt
        assert "Widget revenue was 1234" in root.calls[0]["prompt"]

    async def test_vector_rag_requires_embedder(self, tmp_path):
        from rnsr.eval.datasets.base import EvalItem

        item = EvalItem(qid="rag-2", question="q?", gold="g", context="text\nmore text")
        runner = RootRunner(root_client=MockLLM(), root_model="m",
                            sub_client=MockLLM(), sub_model="m", settings=Settings())
        results, _ = await run_eval([item], "vector-rag", runner, run_dir=tmp_path)
        assert results[0].status == "error"   # recorded, not crashed

    async def test_vector_rag_with_mock_embedder(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        from rnsr.eval.datasets.base import EvalItem

        item = EvalItem(
            qid="rag-3", question="what about the needle topic?",
            gold="needle answer",
            context="\n".join(f"filler sentence number {i} about nothing" for i in range(30))
            + "\nthe needle topic resolves to: needle answer",
        )
        embedder = MockLLM()
        root = MockLLM().rule(r"excerpts", "needle answer")
        runner = RootRunner(root_client=root, root_model="m", sub_client=MockLLM(),
                            sub_model="m", settings=Settings(),
                            embed_client=embedder, embed_model="mock-embed")
        results, _ = await run_eval([item], "vector-rag", runner, run_dir=tmp_path)
        assert results[0].correct


class TestRerankRag:
    async def test_reranker_promotes_relevant_chunk(self, tmp_path):
        from rnsr.eval.datasets.base import EvalItem

        # 30 keyword-matching decoys + 1 true answer chunk; reranker must
        # promote the needle into the top-k
        decoys = "\n".join(
            f"widget revenue report volume {i}: " + "routine notes and filler. " * 60
            for i in range(30))
        context = decoys + "\nwidget revenue was exactly 4321 dollars in the final audit."
        item = EvalItem(qid="rr-1", question="What was the widget revenue?",
                        gold="4321", context=context)
        sub = MockLLM().rule(r"4321", "10").rule(r"routine notes", "1")
        root = MockLLM().rule(r"excerpts", "The widget revenue was 4321 dollars.")
        runner = RootRunner(root_client=root, root_model="m", sub_client=sub,
                            sub_model="m", settings=Settings())
        results, _ = await run_eval([item], "rerank-rag", runner, run_dir=tmp_path)
        assert results[0].correct
        assert results[0].sub_calls > 12          # scored a wide pool
        assert "4321" in root.calls[-1]["prompt"] # needle survived the cut


class TestGraphRag:
    async def test_index_and_answer_end_to_end(self, tmp_path):
        import json as _json

        from rnsr.eval.datasets.base import EvalItem

        context = ("Invoice INV-9001 issued by Vendor Corp for 5000 dollars.\n"
                   "Invoice INV-9002 issued by Vendor Corp for 7000 dollars.\n"
                   "A letter from Client Ltd confirms payment of INV-9001.\n")
        item = EvalItem(qid="g-1", question="What invoices did Vendor Corp issue?",
                        gold="INV-9001, INV-9002", context=context)

        extraction = _json.dumps({
            "entities": [{"name": "INV-9001", "type": "document"},
                         {"name": "Vendor Corp", "type": "org"}],
            "relations": [{"src": "Vendor Corp", "rel": "issued", "dst": "INV-9001"}],
        })
        sub = MockLLM()
        sub.rule(r"Extract entities", extraction)
        sub.rule(r"Summarize this community",
                 "Vendor Corp issued invoices INV-9001 and INV-9002.")
        root = MockLLM().rule(r"COMMUNITY SUMMARIES",
                              "Vendor Corp issued INV-9001 and INV-9002.")
        runner = RootRunner(root_client=root, root_model="m", sub_client=sub,
                            sub_model="m", settings=Settings())
        results, _ = await run_eval([item], "graph-rag", runner, run_dir=tmp_path)
        assert results[0].correct
        assert results[0].sub_calls >= 2         # extraction + summary
        # graph summaries reached the answer prompt
        assert "Vendor Corp issued invoices" in root.calls[0]["prompt"]

    async def test_index_cached_second_question(self, tmp_path):
        import json as _json

        from rnsr.eval.datasets.base import EvalItem

        extraction = _json.dumps({"entities": [{"name": "X", "type": "other"}],
                                  "relations": []})
        sub = MockLLM(default="a summary")
        sub.rule(r"Extract entities", extraction)
        root = MockLLM(default="answer")
        items = [EvalItem(qid=f"g-{i}", question="about X?", gold="answer",
                          context="X did a thing.\nMore text about X here.")
                 for i in range(2)]
        runner = RootRunner(root_client=root, root_model="m", sub_client=sub,
                            sub_model="m", settings=Settings())
        results, _ = await run_eval(items, "graph-rag", runner, run_dir=tmp_path)
        # second question reuses the graph: zero new extraction sub-calls
        assert results[1].sub_calls == 0


class TestAnswerCsvAdapter:
    def test_contract_shape(self, tmp_path, monkeypatch):
        """answers_chunk1.csv: exact header, verbatim order, no empty cells."""
        import csv

        from typer.testing import CliRunner

        from rnsr.cli import app

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # generate one small real PDF so ingest works (docling-free? needs docling)
        import pytest as _pytest

        _pytest.importorskip("docling")
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate

        styles = getSampleStyleSheet()
        SimpleDocTemplate(str(corpus / "doc.pdf"), pagesize=LETTER).build(
            [Paragraph("The secret number is 7714.", styles["BodyText"])])

        questions = tmp_path / "q.csv"
        with open(questions, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ground_truth_question", "extra_col"])
            w.writerow(["What is the secret number?", "x"])
            w.writerow(['A question, with "quotes" and, commas?', "y"])

        # stub the runner so no live LLM is needed
        import rnsr.cli as cli_mod

        class FakeRunner:
            async def run(self, q, env, run_dir=None, query_id=None):
                class R:
                    answer = f"answer to: {q[:20]}"
                    ledger = {"spend_usd": 0, "sub_calls": 0}
                return R()

        monkeypatch.setattr(cli_mod, "_make_runner", lambda s: FakeRunner())

        result = CliRunner().invoke(app, [
            "answer-csv", "--corpus", str(corpus), "--questions", str(questions),
            "--output", str(tmp_path / "out"), "--work-dir", str(tmp_path / "work")])
        assert result.exit_code == 0, result.output

        with open(tmp_path / "out" / "answers_chunk1.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["ground_truth_question", "model_answer"]
        assert rows[1][0] == "What is the secret number?"
        assert rows[2][0] == 'A question, with "quotes" and, commas?'  # verbatim incl. punctuation
        assert all(r[1].strip() for r in rows[1:])
