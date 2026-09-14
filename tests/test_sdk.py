"""Public SDK surface (rnsr.sdk): answer, answer_batch, exports, version.

Runs LLM-free on MockLLM runners over a tiny fake-parsed corpus; the loops
go through the real sandbox, so these are end-to-end minus providers.
"""

import re

import pytest

import rnsr
from rnsr.harness.loop import RootRunner
from rnsr.ingest.model import Element, ParsedDocument
from rnsr.ingest.pipeline import ingest
from rnsr.llm.mock import MockLLM


def _parse(path):
    return ParsedDocument(
        doc_id="acme", source_path=str(path), sha256="b" * 64, n_pages=1,
        parser="fake",
        elements=[
            Element("heading", "Results", 1, heading_level=1),
            Element("text", "ACME 2023 results. Net revenue was $3,234 million.", 1),
        ],
        tables=[],
    )


@pytest.fixture
def corpus(tmp_path):
    out = tmp_path / "corpus.db"
    ingest([tmp_path / "acme.pdf"], out, parse=_parse)
    return out


def make_runner(root: MockLLM, sub: MockLLM | None = None) -> RootRunner:
    return RootRunner(root_client=root, root_model="mock-root",
                      sub_client=sub or MockLLM(default="COMPLETE"),
                      sub_model="mock-sub")


class TestPackageSurface:
    def test_version_is_a_real_version(self):
        # __version__ is the build-time single source (hatch reads it from
        # rnsr/__init__.py) and is stamped into every corpus manifest
        assert re.fullmatch(r"\d+\.\d+\.\d+([a-z]+\d+)?([.+].*)?",
                            rnsr.__version__)

    def test_public_exports_resolve_lazily(self):
        for name in rnsr.__all__:
            assert getattr(rnsr, name) is not None
        assert callable(rnsr.answer)
        assert callable(rnsr.answer_batch_sync)

    def test_ingest_callable_lives_in_sdk(self):
        # the rnsr.ingest SUBPACKAGE shadows any root-level function of
        # that name once imported — the callable is rnsr.sdk.ingest
        from rnsr import sdk
        assert callable(sdk.ingest)

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            rnsr.no_such_symbol  # noqa: B018

    def test_open_corpus(self, corpus):
        with rnsr.open_corpus(corpus) as c:
            assert c.doc_ids() == ["acme"]


class TestAnswer:
    def test_answer_sync_returns_query_result(self, corpus, tmp_path):
        # docdb FINAL requires verbatim quotes verified against source (§6)
        root = MockLLM().script(
            "```python\nFINAL('$3,234 million', "
            "quotes=['Net revenue was $3,234 million.'])\n```"
        )
        result = rnsr.answer_sync("What was net revenue?", corpus,
                                  runner=make_runner(root),
                                  run_dir=tmp_path / "runs")
        assert result.status == "final"
        assert result.answer == "$3,234 million"
        assert result.ledger["spend_usd"] > 0

    async def test_answer_is_awaitable(self, corpus, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL('2023', quotes=['ACME 2023 results.'])\n```"
        )
        result = await rnsr.answer("Which fiscal year?", corpus,
                                   runner=make_runner(root),
                                   run_dir=tmp_path / "runs")
        assert result.status == "final"
        assert result.answer == "2023"


class TestAnswerBatch:
    def test_one_group_answers_in_input_order(self, corpus, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL_BATCH({'q000': 'Paris', 'q001': '42', "
            "'q002': 'NOT_FOUND'})\n```"
        )
        answers = rnsr.answer_batch_sync(
            ["Capital?", "Revenue?", "Weather?"], corpus,
            runner=make_runner(root), run_dir=tmp_path / "runs")
        assert [a.answer for a in answers] == ["Paris", "42", "NOT_FOUND"]
        assert all(a.status == "final" for a in answers)
        assert [a.question for a in answers] == ["Capital?", "Revenue?", "Weather?"]

    def test_unanswered_question_retried_solo(self, corpus, tmp_path):
        # the batch loop never answers q001 (pushback once, then accepted
        # unchanged); the SDK retries it in its own loop
        batch_reply = "```python\nFINAL_BATCH({'q000': 'a', 'q002': 'c'})\n```"
        solo_reply = ("```python\nFINAL('solo-answer', "
                      "quotes=['ACME 2023 results.'])\n```")
        root = MockLLM(default=solo_reply).script(batch_reply, batch_reply)
        answers = rnsr.answer_batch_sync(
            ["First thing?", "What colour is the parrot mascot?", "Third?"],
            corpus, runner=make_runner(root), run_dir=tmp_path / "runs")
        assert [a.answer for a in answers] == ["a", "solo-answer", "c"]

    def test_batch_size_one_runs_solo_loops(self, corpus, tmp_path):
        root = MockLLM(
            default="```python\nFINAL('x', quotes=['ACME 2023 results.'])\n```")
        answers = rnsr.answer_batch_sync(
            ["One?", "Two?"], corpus, batch_size=1,
            runner=make_runner(root), run_dir=tmp_path / "runs")
        assert [a.answer for a in answers] == ["x", "x"]
        assert all(a.status == "final" for a in answers)

    def test_forms_pipeline_build_fanout_score(self, tmp_path):
        # the full form-fill shape, LLM-free: spec -> enriched questions ->
        # (answers) -> fan-out to fields -> scored against a golden set
        import json

        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({
            "context": "Form: Test Form.",
            "fields": [
                {"id": "dom", "title": "Date of marriage",
                 "notes": "FIELD TYPE: date."},
                {"id": "city", "title": "City of marriage", "notes": ""},
            ],
        }))
        items = rnsr.build_questions(spec_path)
        assert [i.field_id for i in items] == ["dom", "city"]

        field_answers, notes = rnsr.fan_out(items, ["12 May 2020", "Sydney"])
        assert field_answers == {"dom": "12 May 2020", "city": "Sydney"}
        assert notes == []

        report = rnsr.score_answers_sync(
            {"dom": ["12 May 2020"], "city": ["Melbourne"]}, field_answers,
            min_accuracy=0.95)
        assert report.correct == 1 and report.total == 2
        assert not report.passed

    def test_group_error_is_reported_not_raised(self, corpus, tmp_path):
        # every root call fails -> run() returns status='error' with no
        # answer; solo retries hit the same wall. Nothing raises; the
        # BatchAnswer rows carry the truth.
        from rnsr.config import Settings
        root = MockLLM(default="no code here, ever.")
        root.fail_times = 10 ** 6
        runner = make_runner(root)
        # a short wall cap keeps the retry/backoff cycle fast in tests
        runner.settings = Settings(max_wall_s=5.0)
        answers = rnsr.answer_batch_sync(
            ["Q?"], corpus, runner=runner,
            run_dir=tmp_path / "runs", batch_size=4)
        assert answers[0].answer is None
        assert answers[0].status in ("unanswered", "error", "budget_exhausted")
