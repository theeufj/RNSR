"""Public SDK surface (rnsr.sdk): answer, answer_batch, exports, version.

Runs LLM-free on MockLLM runners over a tiny fake-parsed corpus; the loops
go through the real sandbox, so these are end-to-end minus providers.
"""

import re

import pytest

import rnsr
from rnsr.config import Settings
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
                      sub_model="mock-sub",
                      # These tests exercise SDK plumbing, not claim review.
                      # Their deliberately synthetic answers are not supported
                      # by their placeholder citations.
                      settings=Settings(claim_review_enabled=False))


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
    @pytest.mark.parametrize("batch", [False, True])
    def test_retention_applies_to_shared_query_entry_points(self, corpus, tmp_path, batch):
        import os
        import time
        from dataclasses import replace

        run_dir = tmp_path / "runs"
        run_dir.mkdir()
        stale = run_dir / "previous.jsonl"
        stale.write_text("{}\n")
        expired = time.time() - 3 * 86400
        os.utime(stale, (expired, expired))
        root = MockLLM(default="```python\nFINAL('2023', quotes=['ACME 2023 results.'])\n```")
        runner = make_runner(root)
        runner.settings = replace(runner.settings, trajectory_retention_days=1)
        if batch:
            results = rnsr.answer_batch_sync(["Which fiscal year?"], corpus,
                batch_size=1, runner=runner, run_dir=run_dir)
            assert results[0].answer == "2023"
        else:
            result = rnsr.answer_sync("Which fiscal year?", corpus,
                runner=runner, run_dir=run_dir)
            assert result.answer == "2023"
        assert not stale.exists()
        assert list(run_dir.glob("*.jsonl"))

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
    def test_batch_exposes_field_citations_review_state_and_audit_path(self, corpus, tmp_path):
        from pathlib import Path

        root = MockLLM(default="```python\nFINAL_BATCH({'q000': '2023', 'q001': '3234'}, "
            "quotes={'q000': ['ACME 2023 results.'], "
            "'q001': ['Net revenue was $3,234 million.']})\n```")
        answers = rnsr.answer_batch_sync(["Which year?", "What revenue?"], corpus,
            runner=make_runner(root), run_dir=tmp_path / "runs")
        assert answers[0].citations[0]["quote"] == "ACME 2023 results."
        assert answers[1].citations[0]["quote"] == "Net revenue was $3,234 million."
        assert all(answer.claim_review["status"] == "disabled" for answer in answers)
        assert all(answer.claim_review["required"] for answer in answers)
        assert all(Path(answer.trajectory_paths[0]).is_file() for answer in answers)
        assert answers[0].health["grade"] == "ok"

    async def test_consensus_metadata_follows_winning_pass_or_tiebreak(self, corpus, tmp_path):
        import hashlib
        import json

        from rnsr.harness.evidence import AnswerEvidence
        from rnsr.harness.loop import ConsensusAnswer, ConsensusBatchResult, QueryResult

        def result(answer, verification, path, status="final"):
            return QueryResult(answer=answer, status=status,
                final={"value": answer, "verification": verification}, ledger={},
                trajectory_path=str(tmp_path / path), iterations=1,
                claim_review={"status": "supported"} if status == "final" else None)

        first = result(None, {}, "first.jsonl", "error")
        winner = result({"q000": "2023", "q001": "other"},
            {"q000": {"quotes": [{"quote": "ACME 2023 results.", "doc_id": "acme"}]}},
            "winner.jsonl")
        winner.claim_review["fields"] = [{"field_id": f"f{i}", "status": "reviewed",
            "verdict": "supported", "qid_sha256": hashlib.sha256(json.dumps(qid).encode()).hexdigest()}
            for i, qid in enumerate(("q000", "q001"))]
        tiebreak = result("3234", {"quotes": [{"quote": "Net revenue was $3,234 million."}]},
            "tiebreak.jsonl")

        class Runner:
            settings = Settings()

            async def run_batch_consensus(self, *args, **kwargs):
                return ConsensusBatchResult(
                    answers={"q000": ConsensusAnswer("2023", "majority", 0.5, [None, "2023"],
                        AnswerEvidence(claim_review_status="supported")),
                        "q001": ConsensusAnswer("3234", "tiebreak", 0.5, [None, "other", "3234"],
                        AnswerEvidence(claim_review_status="supported"))},
                    pass_results=[first, winner], tiebreak_results={"q001": tiebreak})

        answers = await rnsr.answer_batch(["Year?", "Revenue?"], corpus,
            runner=Runner(), consensus=2)
        assert all(answer.status == "final" for answer in answers)
        assert answers[0].citations[0]["quote"] == "ACME 2023 results."
        assert answers[1].citations[0]["quote"] == "Net revenue was $3,234 million."
        assert [record["field_id"] for record in answers[0].claim_review["fields"]] == ["f0"]
        assert answers[0].trajectory_paths == [first.trajectory_path, winner.trajectory_path]
        assert answers[1].trajectory_paths == [first.trajectory_path, winner.trajectory_path,
                                               tiebreak.trajectory_path]

    def test_one_group_answers_in_input_order(self, corpus, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL_BATCH({'q000': 'Paris', 'q001': '$3,234 million', "
            "'q002': 'NOT_FOUND'}, quotes={'q000': ['ACME 2023 results.'], "
            "'q001': ['Net revenue was $3,234 million.']})\n```"
        )
        answers = rnsr.answer_batch_sync(
            ["Capital?", "Revenue?", "Weather?"], corpus,
            runner=make_runner(root), run_dir=tmp_path / "runs")
        assert [a.answer for a in answers] == ["Paris", "$3,234 million", "NOT_FOUND"]
        assert all(a.status == "final" for a in answers)
        assert [a.question for a in answers] == ["Capital?", "Revenue?", "Weather?"]

    def test_unanswered_question_retried_solo(self, corpus, tmp_path):
        # the batch loop never answers q001 (pushback once, then accepted
        # unchanged); the SDK retries it in its own loop
        batch_reply = ("```python\nFINAL_BATCH({'q000': 'a', 'q002': 'c'}, "
                       "quotes={'q000': ['ACME 2023 results.'], "
                       "'q002': ['ACME 2023 results.']})\n```")
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
