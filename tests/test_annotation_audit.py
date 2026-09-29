"""Vote evidence, label contracts and atomic publication use no live providers."""

import hashlib
import json
import re
import sqlite3

import numpy as np
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


def classify(annotator, **kwargs):
    return annotator.classify(
        "t_labels_001", "decision", "Classify every instance; use the full vocabulary.",
        allowed_labels=["yes", "no", "unclear"], where="1", expected_count=3, **kwargs)


def test_strict_classification_certifies_exact_scope_and_zero_count_categories(conn):
    annotator = Annotator(conn, reply, model_identities={"sub": "test:model-a"})
    result = classify(annotator)
    assert result["certified"] and result["total"] == 3
    assert result["counts"] == {"no": 0, "unclear": 0, "yes": 3}
    assert result["unresolved_rowids"] == []
    assert result["model_identity"] == "test:model-a"
    assert annotator.classification_counts("t_labels_001", "decision")["counts"] == result["counts"]
    assert classify(annotator)["noop"] is True


@pytest.mark.parametrize("where,expected", [("1", 2), ("item != 'third'", 3)])
def test_strict_classification_rejects_extra_or_missing_instances_before_calls(conn, where, expected):
    annotator = Annotator(conn, lambda _: pytest.fail("must not call provider"))
    with pytest.raises(ValueError, match="instance count mismatch"):
        annotator.classify("t_labels_001", "decision", "classify",
                           allowed_labels=["yes", "no"], where=where, expected_count=expected)
    assert conn.execute("SELECT count(*) FROM annotation_log").fetchone()[0] == 0


@pytest.mark.parametrize("kwargs", [
    {"where": None}, {"where": ""}, {"expected_count": None},
    {"expected_count": True}, {"expected_count": -1}, {"allowed_labels": None},
    {"allowed_labels": []}, {"votes": 0}, {"votes": 2.5},
])
def test_strict_contract_is_required_before_calls(conn, kwargs):
    annotator = Annotator(conn, lambda _: pytest.fail("must not call provider"))
    options = {"where": "1", "expected_count": 3, "allowed_labels": ["yes", "no"]}
    options.update(kwargs)
    with pytest.raises(ValueError):
        annotator.classify("t_labels_001", "decision", "classify", **options)


@pytest.mark.parametrize("expected", [3, np.int64(3), np.int32(3), np.uint64(3)])
def test_integral_count_scalars_use_the_same_json_safe_contract(conn, expected):
    annotator = Annotator(conn, reply)
    options = {"allowed_labels": ["yes", "no"], "where": "1"}
    first = annotator.classify("t_labels_001", "decision", "classify",
                               expected_count=expected, **options)
    assert first["certified"] and first["total"] == 3
    contract = usage(conn)["annotation_state"]["classification"]
    assert type(contract["expected_count"]) is int and contract["expected_count"] == 3
    # Native and dataframe-derived counts identify the exact same cache entry.
    again = annotator.classify("t_labels_001", "decision", "classify",
                               expected_count=3, **options)
    assert again["noop"] and again["annotation_version"] == first["annotation_version"]


@pytest.mark.parametrize("expected", [True, False, np.bool_(True), 3.0, 3.5,
                                       np.float64(3), "3", -1, np.int64(-1)])
def test_nonintegral_or_negative_counts_are_rejected_before_calls(conn, expected):
    annotator = Annotator(conn, lambda _: pytest.fail("must not call provider"))
    with pytest.raises(ValueError, match="expected_count must be a nonnegative integer"):
        annotator.classify("t_labels_001", "decision", "classify",
                           allowed_labels=["yes", "no"], where="1", expected_count=expected)
    assert conn.execute("SELECT count(*) FROM annotation_log").fetchone()[0] == 0


def test_integral_scalar_count_still_requires_exact_instance_coverage(conn):
    annotator = Annotator(conn, lambda _: pytest.fail("must not call provider"))
    with pytest.raises(ValueError, match="instance count mismatch"):
        annotator.classify("t_labels_001", "decision", "classify",
                           allowed_labels=["yes", "no"], where="1", expected_count=np.int64(2))


@pytest.mark.parametrize("expression", ["3", "np.int64(3)", "np.int32(3)", "np.uint64(3)"])
async def test_sandbox_classification_accepts_integral_count(corpus, expression):
    from rnsr.env.sandbox import SandboxedRepl

    async def rpc(request):
        return reply(request)

    async with SandboxedRepl(rpc_handlers={"llm_batch": rpc}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        result = await repl.exec_cell(
            "import numpy as np\n"
            "r = semantic_classify('t_labels_001', 'decision', 'classify', "
            f"allowed_labels=['yes', 'no'], where='1', expected_count={expression})\n"
            "print(r['certified'], r['total'])")
        assert result.ok, result.error
        assert result.stdout.strip() == "True 3"


@pytest.mark.parametrize("expression", ["True", "np.bool_(True)", "3.0", "3.5", "'3'"])
async def test_sandbox_classification_rejects_nonintegral_count(corpus, expression):
    from rnsr.env.sandbox import SandboxedRepl

    async def rpc(request):
        pytest.fail("must not call provider")

    async with SandboxedRepl(rpc_handlers={"llm_batch": rpc}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        result = await repl.exec_cell(
            "import numpy as np\n"
            "semantic_classify('t_labels_001', 'decision', 'classify', "
            f"allowed_labels=['yes', 'no'], where='1', expected_count={expression})")
        assert not result.ok and "expected_count must be a nonnegative integer" in result.error


def test_strict_selection_must_use_source_columns(conn):
    annotator = Annotator(conn, reply)
    annotator.annotate("t_labels_001", "old_label", "classify")
    with pytest.raises(sqlite3.DatabaseError):
        annotator.classify("t_labels_001", "decision", "classify",
                           allowed_labels=["yes", "no"], where="old_label='yes'", expected_count=3)


def test_empty_explicit_instance_set_has_certified_zero_counts_without_calls(conn):
    annotator = Annotator(conn, lambda _: pytest.fail("must not call provider"))
    result = annotator.classify("t_labels_001", "decision", "classify",
                                allowed_labels=["yes", "no"], where="0", expected_count=0)
    assert result["certified"] and result["counts"] == {"no": 0, "yes": 0}
    assert result["total"] == 0


@pytest.mark.parametrize("case", ["invalid", "missing", "tie", "partial"])
def test_unresolved_labels_block_certification(conn, case):
    calls = []

    def rpc(request):
        calls.append(request)
        label = ("invalid" if case == "invalid" or (case == "partial" and len(calls) > 1)
                 else "no" if case == "tie" and len(calls) == 2 else "yes")
        return reply(request, label, omit=(3,) if case == "missing" else ())

    annotator = Annotator(conn, rpc)
    result = classify(annotator, votes=2 if case == "tie" else 3)
    assert result["certified"] is False
    assert result["unresolved_rowids"]
    assert "counts" not in result
    with pytest.raises(ValueError, match="unresolved"):
        annotator.classification_counts("t_labels_001", "decision")


def test_complete_but_semantically_wrong_labels_are_distinct_from_coverage(conn):
    annotator = Annotator(conn, lambda request: reply(request, "no"))
    result = classify(annotator)
    # A scope/schema proof cannot assert correctness of model judgments.
    independent_gold = {1: "yes", 2: "yes", 3: "yes"}
    labels = dict(conn.execute("SELECT rowid, decision FROM t_labels_001"))
    assert result["certified"] and result["coverage"] == 1.0
    assert sum(labels[k] == v for k, v in independent_gold.items()) == 0


def test_force_failure_clears_previous_selected_labels(conn):
    annotator = Annotator(conn, reply)
    classify(annotator)
    annotator.rpc = lambda request: reply(request, "invalid")
    result = classify(annotator, force=True)
    assert result["failed"] == 3 and not result["certified"]
    assert conn.execute("SELECT decision FROM t_labels_001").fetchall() == [(None,)] * 3
    with pytest.raises(ValueError, match="unresolved"):
        annotator.classification_counts("t_labels_001", "decision")


def test_freeform_force_failure_also_clears_previous_selected_labels(conn):
    annotator = Annotator(conn, reply)
    annotator.annotate("t_labels_001", "decision", "classify")
    annotator.rpc = lambda request: {"results": ["malformed"] * len(request["prompts"])}
    result = annotator.annotate("t_labels_001", "decision", "classify", force=True)
    assert result["failed"] == 3
    assert conn.execute("SELECT decision FROM t_labels_001").fetchall() == [(None,)] * 3


def test_overwriting_column_invalidates_historical_cache_entry(conn):
    calls = []

    def rpc(request):
        calls.append(request)
        return reply(request, "no" if len(calls) == 2 else "yes")

    annotator = Annotator(conn, rpc)
    annotator.annotate("t_labels_001", "decision", "instruction A")
    annotator.annotate("t_labels_001", "decision", "instruction B")
    restored = annotator.annotate("t_labels_001", "decision", "instruction A")
    assert not restored.get("noop") and len(calls) == 3
    assert conn.execute("SELECT decision FROM t_labels_001").fetchall() == [("yes",)] * 3


def test_resolved_model_identity_controls_cross_instance_cache(conn):
    first = Annotator(conn, reply, model_identities={"sub": "provider:model-a"})
    first.annotate("t_labels_001", "decision", "classify")
    same = Annotator(conn, lambda _: pytest.fail("must reuse verified result"),
                     model_identities={"sub": "provider:model-a"})
    assert same.annotate("t_labels_001", "decision", "classify")["noop"]
    changed = Annotator(conn, lambda request: reply(request, "no"),
                        model_identities={"sub": "provider:model-b"})
    assert not changed.annotate("t_labels_001", "decision", "classify").get("noop")
    assert conn.execute("SELECT decision FROM t_labels_001").fetchall() == [("no",)] * 3


def test_unresolved_role_identity_cannot_reuse_across_instances(conn):
    Annotator(conn, reply).annotate("t_labels_001", "decision", "classify")
    new = Annotator(conn, lambda request: reply(request, "no"))
    assert not new.annotate("t_labels_001", "decision", "classify").get("noop")


def test_batch_context_and_label_contract_changes_invalidate_cache(conn):
    annotator = Annotator(conn, reply)
    annotator.annotate("t_labels_001", "decision", "classify", batch_size=3,
                       allowed_labels=["yes", "no"])
    assert not annotator.annotate("t_labels_001", "decision", "classify", batch_size=1,
                                  allowed_labels=["yes", "no"]).get("noop")
    assert not annotator.annotate("t_labels_001", "decision", "classify", batch_size=1,
                                  allowed_labels=["yes", "no", "unclear"]).get("noop")


def test_counts_refuse_non_strict_or_replaced_annotations(conn):
    annotator = Annotator(conn, reply)
    annotator.annotate("t_labels_001", "free_form", "classify")
    with pytest.raises(ValueError, match="strict contract"):
        annotator.classification_counts("t_labels_001", "free_form")
    classify(annotator)
    annotator.annotate("t_labels_001", "decision", "different interpretation")
    with pytest.raises(ValueError, match="strict contract"):
        annotator.classification_counts("t_labels_001", "decision")


def test_counts_and_cache_refuse_changed_active_values(conn):
    annotator = Annotator(conn, reply)
    classify(annotator)
    conn.execute("UPDATE t_labels_001 SET decision='no' WHERE rowid=1")
    conn.commit()
    with pytest.raises(ValueError, match="labels changed"):
        annotator.classification_counts("t_labels_001", "decision")
    assert not classify(annotator).get("noop")
    assert annotator.classification_counts("t_labels_001", "decision")["counts"]["yes"] == 3


def test_forced_publication_gets_a_new_version_even_if_sqlite_reuses_log_id(conn):
    annotator = Annotator(conn, reply)
    first = classify(annotator)
    second = classify(annotator, force=True)
    assert first["annotation_version"] != second["annotation_version"]
    assert first["selection_sha256"] == second["selection_sha256"]
    assert first["labels_sha256"] == second["labels_sha256"]


def test_label_digest_distinguishes_reassignment_with_equal_counts(conn):
    assignments = {1: "yes", 2: "no", 3: "unclear"}

    def rpc(request):
        return {"results": ["\n".join(f"{rowid}. {assignments[int(rowid)]}" for rowid in
                re.findall(r"^(\d+)\. \{", prompt, re.MULTILINE))
                for prompt in request["prompts"]]}

    annotator = Annotator(conn, rpc)
    first = classify(annotator)
    assignments.update({1: "no", 2: "yes"})
    second = classify(annotator, force=True)
    assert first["counts"] == second["counts"]
    assert first["selection_sha256"] == second["selection_sha256"]
    assert first["labels_sha256"] != second["labels_sha256"]
    assert second["labels_sha256"] == usage(conn)["annotation_state"]["labels_sha256"]


def test_subset_counts_exclude_old_labels_outside_contract(conn):
    annotator = Annotator(conn, reply)
    classify(annotator)
    annotator.rpc = lambda request: reply(request, "no")
    subset = annotator.classify("t_labels_001", "decision", "classify selected instances",
                                where="item != 'third'", expected_count=2,
                                allowed_labels=["yes", "no"])
    assert subset["counts"] == {"no": 2, "yes": 0}
    assert conn.execute("SELECT decision FROM t_labels_001 WHERE item='third'").fetchone() == ("yes",)


def test_new_unrelated_annotation_does_not_invalidate_source_proof(conn):
    annotator = Annotator(conn, reply)
    classify(annotator)
    annotator.annotate("t_labels_001", "other_property", "different property")
    assert classify(annotator)["noop"]
    assert annotator.classification_counts("t_labels_001", "decision")["total"] == 3


async def test_sandbox_strict_classification_and_parent_count_certificate(corpus):
    from rnsr.env.sandbox import SandboxedRepl

    calls = []

    async def rpc(request):
        calls.append(request)
        return reply(request)

    repl = SandboxedRepl(rpc_handlers={"llm_batch": rpc})
    await repl.start(mode="docdb", corpus_db=str(corpus))
    try:
        rejected = await repl.exec_cell(
            "semantic_classify('t_labels_001', 'decision', 'classify', "
            "allowed_labels=['yes', 'no'], where='1', expected_count=4)")
        assert not rejected.ok and not calls
        result = await repl.exec_cell(
            "r = semantic_classify('t_labels_001', 'decision', 'classify', "
            "allowed_labels=['yes', 'no'], where='1', expected_count=3)\n"
            "c = classification_counts('t_labels_001', 'decision')\n"
            "print(r['certified'], c['total'], c['counts']['yes'], c['counts']['no'])")
        assert result.ok, result.error
        assert result.stdout.strip() == "True 3 3 0"
        assert len(calls) == 3
    finally:
        await repl.close()
