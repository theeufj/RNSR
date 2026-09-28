"""Vote evidence, label contracts and atomic publication use no live providers."""

import hashlib
import json
import re
import sqlite3

import pytest

from rnsr.env.annotate import Annotator
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


@pytest.fixture
def corpus(tmp_path):
    def parse(path):
        return ParsedDocument(
            doc_id="labels", source_path=str(path), sha256="b" * 64,
            n_pages=1, parser="fixture",
            elements=[Element("text", "A generic source used only for annotation tests.", 1)],
            tables=[RawTable(page=1, header=["Item", "Description"],
                             rows=[["first", "private source alpha"],
                                   ["second", "private source beta"],
                                   ["third", "private source gamma"]])])

    path = tmp_path / "corpus.db"
    ingest([tmp_path / "labels.txt"], path, parse=parse)
    return path


@pytest.fixture
def conn(corpus):
    connection = sqlite3.connect(corpus)
    yield connection
    connection.close()


def reply(request, label="yes", *, omit=()):
    return {"results": ["\n".join(f"{rowid}. {label}" for rowid in
            re.findall(r"^(\d+)\. \{", prompt, re.MULTILINE) if int(rowid) not in omit)
            for prompt in request["prompts"]]}


def usage(conn):
    return json.loads(conn.execute("SELECT usage_json FROM annotation_log ORDER BY id DESC")
                      .fetchone()[0])


def test_retains_each_vote_and_exact_order_without_source_text(conn):
    calls = []

    def rpc(request):
        calls.append(request)
        result = reply(request, "no" if len(calls) == 2 else "yes")
        result["response_metadata"] = [{"model": "resolved-fixture", "input_tokens": 42,
                                         "cost_usd": 0.01, "secret": "not retained"}]
        return result

    result = Annotator(conn, rpc).annotate("t_labels_001", "decision", "classify", votes=3)
    assert result["rows"] == 3
    audit = usage(conn)["vote_audit"]
    assert usage(conn)["calls"] == 3
    assert usage(conn)["prompts"] == 3
    assert audit["source"]["row_count"] == 3
    assert len(audit["source"]["selection_sha256"]) == 64
    assert audit["requested_model"] == "sub"
    assert [v["shuffle_seed"] for v in audit["votes"]] == [None, 1, 2]
    assert [r["rowid"] for r in audit["votes"][0]["rows"]] == [1, 2, 3]
    assert [r["rowid"] for r in audit["votes"][1]["rows"]] == [2, 3, 1]
    assert [[r["label"] for r in vote["rows"]] for vote in audit["votes"]] == [
        ["yes"] * 3, ["no"] * 3, ["yes"] * 3]
    assert audit["votes"][0]["attempts"][0]["resolved_model"] == "resolved-fixture"
    assert "private source" not in json.dumps(audit)
    assert "not retained" not in json.dumps(audit)
    assert conn.execute("SELECT decision FROM t_labels_001").fetchall() == [("yes",)] * 3


def test_allowed_labels_retry_and_record_invalid_values(conn):
    calls = []

    def rpc(request):
        calls.append(request)
        return reply(request, "maybe" if len(calls) == 1 else "yes")

    result = Annotator(conn, rpc).annotate(
        "t_labels_001", "decision", "classify", allowed_labels=["yes", "no"])
    assert result["rows"] == 3 and len(calls) == 2
    attempts = usage(conn)["vote_audit"]["votes"][0]["attempts"]
    assert attempts[0]["accepted"] is False
    assert attempts[0]["invalid_labels"] == [{"rowid": i, "label": "maybe"} for i in (1, 2, 3)]
    assert attempts[1]["accepted"] is True and attempts[1]["retry"] is True
    assert '"no", "yes"' in calls[0]["prompts"][0]


def test_missing_and_invalid_votes_are_not_silent_or_fabricated(conn):
    calls = []

    def rpc(request):
        calls.append(request)
        # Pass 1 succeeds; passes 2 and 3 each fail both attempts.
        return reply(request, "yes" if len(calls) == 1 else "invalid", omit=() if len(calls) == 1 else (3,))

    result = Annotator(conn, rpc).annotate(
        "t_labels_001", "decision", "classify", votes=3, allowed_labels=["yes", "no"])
    assert len(calls) == 5 and result["rows"] == 3
    audit = usage(conn)["vote_audit"]
    assert audit["partial_vote_rowids"] == [1, 2, 3]
    assert audit["failed_rowids"] == []
    assert all(row["label"] is None for vote in audit["votes"][1:] for row in vote["rows"])
    assert audit["votes"][1]["attempts"][0]["missing_rowids"] == [3]


def test_all_invalid_votes_fail_and_duplicate_ids_are_rejected(conn):
    def rpc(request):
        return {"results": ["1. yes\n1. no\n2. yes\n3. yes"]}

    result = Annotator(conn, rpc).annotate(
        "t_labels_001", "decision", "classify", allowed_labels=["yes", "no"])
    assert result["rows"] == 0 and result["failed"] == 3
    audit = usage(conn)["vote_audit"]
    assert audit["failed_rowids"] == [1, 2, 3]
    assert audit["votes"][0]["attempts"][0]["duplicate_rowids"] == [1]
    assert conn.execute("SELECT decision FROM t_labels_001").fetchall() == [(None,)] * 3


@pytest.mark.parametrize("labels", [[], "yes", [""], [" yes"], ["yes\nno"],
                                         ["yes", "yes"], [1], [["yes"]]])
def test_invalid_label_contract_refused_before_calls(conn, labels):
    with pytest.raises(ValueError, match="allowed_labels"):
        Annotator(conn, lambda request: pytest.fail("must not call provider")).annotate(
            "t_labels_001", "decision", "classify", allowed_labels=labels)


def test_label_contract_is_idempotent_by_set_and_legacy_hash_stays_compatible(conn):
    annotator = Annotator(conn, reply)
    annotator.annotate("t_labels_001", "decision", "classify")
    assert conn.execute("SELECT prompt_sha256 FROM annotation_log").fetchone()[0] == (
        hashlib.sha256(b"classify|votes=1").hexdigest())
    assert "resolved_model" not in usage(conn)["vote_audit"]["votes"][0]["attempts"][0]
    assert annotator.annotate("t_labels_001", "decision", "classify")["noop"] is True
    assert "noop" not in annotator.annotate(
        "t_labels_001", "decision", "classify", allowed_labels=["yes", "no"])
    assert annotator.annotate("t_labels_001", "decision", "classify",
                              allowed_labels=["no", "yes"])["noop"] is True


def test_force_history_is_bounded_and_prior_votes_remain_auditable(conn):
    annotator = Annotator(conn, reply)
    for i in range(6):
        annotator.annotate("t_labels_001", "decision", "classify", force=i > 0)
    record = usage(conn)
    assert len(record["previous_runs"]) == 3
    assert record["earlier_runs_count"] == 2
    assert len(record["earlier_runs_sha256"]) == 64
    assert all("previous_runs" not in old["usage"] for old in record["previous_runs"])
    assert all(old["usage"]["vote_audit"]["votes"][0]["rows"][0]["label"] == "yes"
               for old in record["previous_runs"])
    assert conn.execute("SELECT count(*) FROM annotation_log").fetchone()[0] == 1


def test_failed_audit_publication_rolls_back_labels_schema_and_old_log(conn):
    annotator = Annotator(conn, reply)
    annotator.annotate("t_labels_001", "decision", "classify")
    before = conn.execute("SELECT usage_json FROM annotation_log").fetchone()[0]
    conn.execute("CREATE TRIGGER reject_audit BEFORE INSERT ON annotation_log "
                 "BEGIN SELECT RAISE(ABORT, 'cannot publish audit'); END")
    conn.commit()
    annotator.rpc = lambda request: reply(request, "no")
    with pytest.raises(sqlite3.IntegrityError, match="cannot publish audit"):
        annotator.annotate("t_labels_001", "decision", "classify", force=True)
    assert not conn.in_transaction
    assert conn.execute("SELECT decision FROM t_labels_001").fetchall() == [("yes",)] * 3
    assert conn.execute("SELECT usage_json FROM annotation_log").fetchone()[0] == before
    with pytest.raises(sqlite3.IntegrityError, match="cannot publish audit"):
        annotator.annotate("t_labels_001", "new_decision", "classify")
    assert "new_decision" not in {r[1] for r in conn.execute("PRAGMA table_info(t_labels_001)")}


def test_long_free_form_label_remains_compatible_with_bounded_audit(conn):
    label = "x" * 5000
    Annotator(conn, lambda request: reply(request, label)).annotate(
        "t_labels_001", "decision", "classify")
    logged = usage(conn)["vote_audit"]["votes"][0]["rows"][0]
    assert len(logged["label"]) == 4096
    assert logged["label_truncated"] is True and logged["label_chars"] == 5000
    assert logged["label_sha256"] == hashlib.sha256(label.encode()).hexdigest()
    assert conn.execute("SELECT decision FROM t_labels_001 LIMIT 1").fetchone()[0] == label


async def test_sandbox_forwards_label_contract_to_parent(corpus):
    from rnsr.env.sandbox import SandboxedRepl

    calls = []

    async def rpc(request):
        calls.append(request)
        return reply(request, "invalid")

    repl = SandboxedRepl(rpc_handlers={"llm_batch": rpc})
    await repl.start(mode="docdb", corpus_db=str(corpus))
    try:
        result = await repl.exec_cell(
            "r = semantic_annotate('t_labels_001', 'decision', 'classify', "
            "allowed_labels=['yes', 'no'])\nprint(r['rows'], r['failed'])")
        assert result.ok, result.error
        assert result.stdout.strip() == "0 3"
        assert len(calls) == 2
    finally:
        await repl.close()
