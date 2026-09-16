import csv
import json

import pytest
from typer.testing import CliRunner

from rnsr.answer_workflow import AnswerCheckpoint
from rnsr.cli import app
from rnsr.harness.evidence import AnswerEvidence
from rnsr.harness.loop import BatchQueryResult, QueryResult
from rnsr.sdk import BatchAnswer


def test_checkpoint_restores_evidence_retries_failure_and_ignores_changed_corpus(tmp_path):
    ckpt = AnswerCheckpoint(tmp_path / "answers.jsonl", ["one", "two"], "revision")
    good = BatchAnswer("one", "value", "final", tier="medium", evidence={"quotes_total": 1})
    ckpt.record(0, good)
    ckpt.record(1, BatchAnswer("two", None, "error", error="network"))
    assert ckpt.load() == {0: good}
    assert AnswerCheckpoint(ckpt.path, ["one", "two"], "changed").load() == {}
    with ckpt.path.open("a") as out:
        out.write('{"i":')
    assert ckpt.load() == {0: good}
    ckpt.record(1, BatchAnswer("two", "fixed", "final", tier="high"))
    assert len(ckpt.load()) == 2


def test_checkpoint_rejects_negative_indices_and_partial_prior_lines(tmp_path):
    ckpt = AnswerCheckpoint(tmp_path / "answers.jsonl", ["q"], "r")
    ckpt.path.write_text(json.dumps({"i": -1, "q": "q", "revision": "r"}) + "\n")
    assert ckpt.load() == {}
    ckpt.path.write_text("not json\n{}\n")
    with pytest.raises(ValueError, match="record 1"):
        ckpt.load()


def test_cli_checkpoint_resume_keeps_abstention_and_retries_errors(tmp_path, monkeypatch):
    import rnsr.cli as cli

    corpus = tmp_path / "source"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("ACME's yearly revenue was 42 million.")
    qs = tmp_path / "questions.csv"
    qs.write_text("ground_truth_question\none\ntwo\n")
    calls = []

    class Runner:
        async def run(self, question, env, **kwargs):
            calls.append(question)
            if question == "two" and calls.count("two") == 1:
                raise RuntimeError("provider unavailable")
            return QueryResult("42 million", "final", {}, {}, "", 1,
                               evidence=AnswerEvidence(zero_quotes=True))

    monkeypatch.setattr(cli, "_make_runner", lambda _: Runner())
    args = ["answer-csv", "--corpus", str(corpus), "--questions", str(qs),
            "--output", str(tmp_path / "out"), "--work-dir", str(tmp_path / "work"),
            "--batch-size", "1", "--abstain-below", "medium", "--no-transcribe"]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 2, first.output
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0, (second.output, second.exception)
    assert calls == ["one", "two", "two"]
    with (tmp_path / "out" / "answers_chunk1.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["model_answer"] for r in rows] == ["NEEDS REVIEW"] * 2
    with (tmp_path / "out" / "answers_status.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["tier"] for r in rows] == ["medium"] * 2
    assert [r["model_answer"] for r in rows] == ["42 million"] * 2


async def test_sdk_publication_does_not_prevent_solo_retry(tmp_path):
    from rnsr.config import Settings
    from rnsr.harness.loop import EnvSpec
    from rnsr.sdk import answer_batch

    class Runner:
        settings = Settings()
        async def run_batch(self, pairs, env, **kwargs):
            return BatchQueryResult({}, QueryResult(None, "error", None, {}, "", 1))
        async def run(self, question, env, **kwargs):
            return QueryResult("raw", "final", {}, {}, "", 1,
                               evidence=AnswerEvidence(zero_quotes=True))

    rows = await answer_batch(["q"], tmp_path / "unused", runner=Runner(),
                              env=EnvSpec("classic", context=""), abstain_below="medium")
    assert rows[0].answer == "NEEDS REVIEW" and rows[0].raw_answer == "raw"


async def test_eval_decrypts_trajectory_for_autopsy(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet

    import rnsr.eval.harness as harness
    from rnsr.config import Settings
    from rnsr.eval.datasets.base import EvalItem
    from rnsr.harness.trajectory import TrajectoryWriter

    key = Fernet.generate_key().decode()
    observed = []
    real_classify = harness.classify_miss

    def classify(result, records, **kwargs):
        observed.extend(records)
        return real_classify(result, records, **kwargs)

    monkeypatch.setattr(harness, "classify_miss", classify)

    class Runner:
        settings = Settings(trajectory_key=key)
        async def run(self, *args, run_dir, query_id):
            writer = TrajectoryWriter(run_dir, query_id, key=key)
            writer.event("cell", code="print('evidence')", stdout="observed source")
            writer.close()
            return QueryResult("wrong", "final", {}, {"spend_usd": 0, "sub_calls": 0},
                               str(writer.path), 1)

    await harness.run_eval([EvalItem("one", "q", "expected", "text", context="source")],
                           "rlm-classic", Runner(), run_dir=tmp_path, judge=False)
    assert observed[0]["stdout"] == "observed source"


def test_corpus_revision_tracks_retained_text_but_not_annotation_metadata(tmp_path):
    from rnsr.answer_workflow import corpus_revision
    from rnsr.db import schema
    from rnsr.db.artifact import CorpusDB
    from rnsr.ingest.pipeline import ingest

    source = tmp_path / "letter.txt"
    source.write_text("First transcription of the source.")
    path = tmp_path / "corpus.db"
    ingest([source], path)
    before = corpus_revision(path)
    with CorpusDB(path, mode="rw") as corpus:
        corpus.manifest_set("annotation_test", {"column": "new"})
        corpus.conn.commit()
    assert corpus_revision(path) == before
    with CorpusDB(path, mode="rw") as corpus:
        corpus.conn.execute("BEGIN")
        schema.unfreeze_corpus(corpus.conn)
        corpus.conn.execute("UPDATE doc_text SET text = 'Corrected transcription'")
        schema.finalize_corpus(corpus.conn)
        corpus.conn.commit()
    assert corpus_revision(path) != before
