"""CLI argument/exit contracts exercised without live provider requests."""
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rnsr.cli import app
from rnsr.harness.loop import QueryResult

COMMANDS = ["ingest", "query", "trajectory", "gate", "eval-tables", "eval",
            "answer-csv", "build-questions", "regress", "replay", "health",
            "migrate", "autopsy", "audit-export", "review-import", "doctor", "serve", "ablate"]


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_help_contract(command):
    result = CliRunner().invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "letter.txt"
    source.write_text("ACME revenue was 42 million. This is the annual report.")
    output = tmp_path / "corpus.db"
    result = CliRunner().invoke(app, ["ingest", str(source), "--out", str(output), "--no-transcribe"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert output.exists()
    return output


@pytest.mark.parametrize("status", ["error", "budget_exhausted", "final"])
def test_query_exit_codes(corpus, monkeypatch, status):
    import rnsr.cli as cli

    class Runner:
        async def run(self, *args, **kwargs):
            return QueryResult(None, status, None, {"spend_usd": 0, "sub_calls": 0}, "", 0)

    monkeypatch.setattr(cli, "_make_runner", lambda _: Runner())
    result = CliRunner().invoke(app, ["query", str(corpus), "What was revenue?"])
    assert result.exit_code == (0 if status == "final" else 2), (result.output, result.exception)


def test_health_migrate_and_search_replay(corpus, tmp_path):
    runner = CliRunner()
    for command in ("health", "migrate"):
        result = runner.invoke(app, [command, str(corpus)])
        assert result.exit_code == 0, (result.output, result.exception)
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(["revenue"]))
    result = runner.invoke(app, ["replay", "--db", str(corpus), "--queries", str(queries)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert '"n_diffs": 0' in result.output


def test_regression_reports_actual_failed_gate(tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"fields": [{"id": "missing", "golden": []}]}))
    answers = tmp_path / "answers.csv"
    answers.write_text("field_id,answer\nmissing,invented name\n")
    result = CliRunner().invoke(app, ["regress", "--golden", str(gold), "--answers", str(answers),
                                    "--no-judge", "--max-false-positive-rate", "0"])
    assert result.exit_code == 2, result.output
    assert "REGRESSION: false-positive rate" in result.output
    assert "REGRESSION: accuracy" not in result.output


def test_doctor_no_keys_fails(monkeypatch):
    monkeypatch.setattr("rnsr.llm.router.available_providers", lambda: [])
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 1 and "no provider key" in result.output


def test_serve_forwards_host_port(monkeypatch):
    calls = []
    monkeypatch.setattr("rnsr.service.serve", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(app, ["serve", "--host", "127.0.0.1", "--port", "8123"])
    assert result.exit_code == 0, result.output
    assert calls[0]["port"] == 8123 and calls[0]["host"] == "127.0.0.1"


def test_eval_and_gate_dispatch_and_gate_failure(tmp_path, monkeypatch):
    import rnsr.cli as cli
    import rnsr.eval.harness as harness

    calls = []
    monkeypatch.setattr(cli, "_make_runner", lambda _: object())
    monkeypatch.setattr("rnsr.eval.datasets.needle_gen.generate_needle_set", lambda *a, **kw: [])

    async def run(items, system, runner, **kwargs):
        calls.append(system)
        return [], {"accuracy_by_class": {"numeric": .5}, "cost_usd": {"p50": 1}}

    monkeypatch.setattr(harness, "run_eval", run)
    for command, args, expected in [
        ("eval", ["--benchmark", "synthetic-oolong", "--system", "bm25-rag"], 0),
        ("gate", [], 1),
    ]:
        result = CliRunner().invoke(app, [command, *args, "--run-dir", str(tmp_path)])
        assert result.exit_code == expected, (result.output, result.exception)
    assert calls == ["bm25-rag", "docdb", "rlm-classic"]


def test_eval_tables_required_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("rnsr.eval.tables_score.score_labelled_tables", lambda *a, **k:
                        {"results": [], "n_required": 1, "n_required_passed": 0})
    result = CliRunner().invoke(app, ["eval-tables", "--dir", str(tmp_path)])
    assert result.exit_code == 2


def test_ablate_dispatch(corpus, monkeypatch, tmp_path):
    monkeypatch.setattr("rnsr.llm.router.Router.resolve", lambda *a:
                        SimpleNamespace(client=object(), model="mock"))
    monkeypatch.setattr("rnsr.eval.ablation.run_ablation", lambda *a, **k: {"accepts": True})
    report = tmp_path / "ablation.json"
    result = CliRunner().invoke(app, ["ablate", str(corpus), "--report", str(report)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(report.read_text())["accepts"]


def test_question_build_trajectory_audit_review_and_autopsy(tmp_path):
    from rnsr.eval.metrics import EvalResult
    from rnsr.harness.trajectory import TrajectoryWriter

    runner = CliRunner()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"fields": [{"id": "city", "title": "City of marriage", "notes": ""}]}))
    questions = tmp_path / "questions.csv"
    result = runner.invoke(app, ["build-questions", "--spec", str(spec), "--out", str(questions)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(questions.with_suffix(".map.json").read_text())["not_found"] == "Not found in matter corpus"
    work = tmp_path / "work"
    writer = TrajectoryWriter(work / "trajectories", "q1")
    writer.event("start", question="City?")
    writer.event("final", value="Sydney")
    writer.event("end", status="final")
    writer.close()
    result = runner.invoke(app, ["trajectory", str(writer.path), "--kinds", "final"])
    assert result.exit_code == 0 and "Sydney" in result.output
    audit = tmp_path / "audit"
    result = runner.invoke(app, ["audit-export", "--work-dir", str(work), "--out", str(audit)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert (audit / "evidence" / "q1.json").exists()
    review = audit / "review.csv"
    review.write_text("qid,answer,reviewer_mark,corrected,note\nq1,Sydney,wrong,Melbourne,\n")
    result = runner.invoke(app, ["review-import", str(review), "--out", str(audit)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads((audit / "golden" / "from_review.json").read_text())["items"][0]["golden"] == ["Melbourne"]
    evaluated = EvalResult("q1", "text", "Sydney", "Melbourne", False, "final", 0, 0, 0, 1)
    (work / "results.jsonl").write_text(json.dumps(evaluated.to_dict()) + "\n")
    result = runner.invoke(app, ["autopsy", str(work)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert (work / "loss-ledger.md").exists()
