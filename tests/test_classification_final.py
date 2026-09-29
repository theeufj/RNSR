"""Strict scope and parent-derived aggregates, entirely with offline providers."""

import asyncio
import json
import re
import sqlite3

import pytest

from rnsr.env.annotate import Annotator
from rnsr.env.classification import classification_final
from rnsr.env.sandbox import SandboxedRepl
from rnsr.ingest.pipeline import ingest_text

TABLE = "t_instances_001"
SCOPE = "text LIKE 'Instance:%'"
LABELS = ["alpha", "beta", "gamma", "delta"]


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "instances.db"
    ingest_text({"instances": "\n".join([
        "Metadata: exactly three instances follow.",
        "Instance: one alpha", "Instance: two beta", "Instance: three alpha",
        "Footer: this is not an instance.",
    ])}, path)
    return path


def provider(request):
    replies = []
    for prompt in request["prompts"]:
        items = []
        for rowid, raw in re.findall(r"^(\d+)\. (\{.*\})$", prompt, re.MULTILINE):
            label = json.loads(raw)["text"].split()[-1]
            items.append(f"{rowid}. {label}")
        replies.append("\n".join(items))
    return {"results": replies}


def classify(conn, *, where=SCOPE, expected=3, labels=LABELS):
    return Annotator(conn, provider).classify(
        TABLE, "category", "Classify each instance into the full supplied vocabulary.",
        where=where, expected_count=expected, allowed_labels=labels, votes=1)


def classify_code(*, where=SCOPE, expected=3, force=False):
    return (f"semantic_classify({TABLE!r}, 'category', 'Classify each instance', "
            f"allowed_labels={LABELS!r}, where={where!r}, expected_count={expected}, "
            f"votes=1, force={force!r})\n")


@pytest.mark.parametrize("operation,labels,answer", [
    ("count", ["alpha"], "Answer: 2"),
    ("count", ["gamma"], "Answer: 0"),
    ("compare", ["alpha", "beta"], "Answer: alpha is more common than beta"),
    ("compare", ["beta", "alpha"], "Answer: beta is less common than alpha"),
    ("compare", ["gamma", "delta"], "Answer: gamma is same frequency as delta"),
    ("most", None, "Label: alpha"),
    ("least", None, "Label: delta, gamma"),
])
def test_parent_derives_counts_comparisons_and_all_ties(corpus, operation, labels, answer):
    with sqlite3.connect(corpus) as conn:
        classified = classify(conn)
        value, report = classification_final(conn, TABLE, "category", operation, labels)
        assert value == answer and report["answer"] == answer
        assert report["classification"]["annotation_version"] == classified["annotation_version"]
        assert report["classification"]["total"] == 3
        assert report["classification"]["counts"] == {"alpha": 2, "beta": 1, "delta": 0, "gamma": 0}
        assert report["claim_support"] == "not_checked"
        assert not conn.in_transaction


@pytest.mark.parametrize("operation", ["most", "least"])
def test_valid_subset_preserves_all_extreme_ties(corpus, operation):
    with sqlite3.connect(corpus) as conn:
        classify(conn, where=SCOPE + " AND line_no<=3", expected=2, labels=["alpha", "beta"])
        answer, proof = classification_final(conn, TABLE, "category", operation)
        assert answer == "Label: alpha, beta"
        assert proof["classification"]["total"] == 2


@pytest.mark.parametrize("operation,labels", [
    ("count", []), ("count", ["alpha", "beta"]), ("compare", ["alpha"]),
    ("compare", ["alpha", "alpha"]), ("most", ["alpha"]), ("sum", None),
    ("count", ["outside-vocabulary"]), ("count", "alpha"), ("count", [1]),
])
def test_invalid_aggregate_arguments_are_rejected(corpus, operation, labels):
    with sqlite3.connect(corpus) as conn:
        classify(conn)
        with pytest.raises(ValueError):
            classification_final(conn, TABLE, "category", operation, labels)
        assert not conn.in_transaction


@pytest.mark.parametrize("table,column,operation", [
    ([], "category", "count"), (TABLE, {}, "count"), (TABLE, "category", None),
])
def test_non_string_arguments_are_rejected_cleanly(corpus, table, column, operation):
    with sqlite3.connect(corpus) as conn, pytest.raises(ValueError, match="nonempty strings"):
        classification_final(conn, table, column, operation, ["alpha"])


def test_direct_final_keeps_proof_and_metadata_on_one_snapshot(corpus, monkeypatch):
    with sqlite3.connect(corpus) as conn:
        first = classify(conn)
        original = Annotator.classification_counts
        replaced = False

        def concurrent_replace(self, table, column):
            nonlocal replaced
            proof = original(self, table, column)
            if not replaced:
                replaced = True
                with sqlite3.connect(corpus) as writer:
                    Annotator(writer, provider).classify(
                        TABLE, "category", "new instruction", where=SCOPE,
                        expected_count=3, allowed_labels=LABELS, votes=1, force=True)
            return proof

        monkeypatch.setattr(Annotator, "classification_counts", concurrent_replace)
        _, old = classification_final(conn, TABLE, "category", "count", ["alpha"])
        assert old["classification"]["annotation_version"] == first["annotation_version"]
        assert old["classification"]["instruction"] != "new instruction"
        _, new = classification_final(conn, TABLE, "category", "count", ["alpha"])
        assert new["classification"]["annotation_version"] != first["annotation_version"]
        assert new["classification"]["instruction"] == "new instruction"


def test_case_alias_cannot_hide_a_freeform_overwrite_from_strict_proof(corpus):
    with sqlite3.connect(corpus) as conn:
        classify(conn)
        assert classification_final(conn, TABLE, "CATEGORY", "count", ["alpha"])[0] == "Answer: 2"
        # Even identical values written under a different contract invalidate
        # the strict version; column case must not fork its audit history.
        Annotator(conn, provider).annotate(TABLE, "CATEGORY", "unrestricted labels", where=SCOPE)
        with pytest.raises(ValueError, match="strict contract"):
            classification_final(conn, TABLE, "category", "count", ["alpha"])


async def async_provider(request):
    return provider(request)


async def test_sandbox_final_revalidates_and_uses_parent_aggregate(corpus):
    async with SandboxedRepl(rpc_handlers={"llm_batch": async_provider}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        result = await repl.exec_cell(classify_code() +
            f"FINAL_CLASSIFICATION({TABLE!r}, 'category', 'count', ['alpha'])")
        assert result.ok, result.error
        assert result.final["value"] == "Answer: 2"
        assert result.final["verification"]["classification"]["counts"]["alpha"] == 2


async def test_metadata_and_footer_scope_rejected_before_provider_calls(corpus):
    calls = []

    async def counted(request):
        calls.append(request)
        return provider(request)

    async with SandboxedRepl(rpc_handlers={"llm_batch": counted}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        result = await repl.exec_cell(classify_code(where="1", expected=3))
        assert not result.ok and "instance count mismatch" in result.error
        assert not calls
        valid = await repl.exec_cell(classify_code(where=SCOPE + " AND line_no<=3", expected=2) +
            f"FINAL_CLASSIFICATION({TABLE!r}, 'category', 'count', ['alpha'])")
        assert valid.ok and valid.final["value"] == "Answer: 1"


@pytest.mark.parametrize("tamper,error", [
    ("draft.value = 'Answer: 9000'", "differs from parent"),
    ("draft.verification['annotation_version'] = 'forged'", "annotation changed"),
    ("draft.verification['labels'] = ['not-a-category']", "classification vocabulary"),
    ("draft.verification['operation'] = 'sum'", "use count"),
    ("draft.verification['table'] = []", "nonempty strings"),
])
async def test_forged_final_values_and_arguments_fail_parent_check(corpus, tamper, error):
    async with SandboxedRepl(rpc_handlers={"llm_batch": async_provider}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        code = classify_code() + (
            f"try:\n    FINAL_CLASSIFICATION({TABLE!r}, 'category', 'count', ['alpha'])\n"
            f"except Exception as draft:\n    {tamper}\n    raise\n")
        result = await repl.exec_cell(code)
        assert not result.ok and result.final is None
        assert "Parent final verification rejected" in result.error and error in result.error


async def test_changed_annotation_version_rejects_draft_even_when_count_unchanged(corpus):
    async with SandboxedRepl(rpc_handlers={"llm_batch": async_provider}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        replacement = classify_code(force=True).strip()
        code = classify_code() + (
            f"try:\n    FINAL_CLASSIFICATION({TABLE!r}, 'category', 'count', ['alpha'])\n"
            f"except Exception as draft:\n    {replacement}\n    raise\n")
        result = await repl.exec_cell(code)
        assert not result.ok and "annotation changed" in result.error


async def test_child_supplied_counts_are_replaced_with_parent_counts(corpus):
    async with SandboxedRepl(rpc_handlers={"llm_batch": async_provider}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        code = classify_code() + (
            f"try:\n    FINAL_CLASSIFICATION({TABLE!r}, 'category', 'count', ['alpha'])\n"
            "except Exception as draft:\n"
            "    draft.verification['counts'] = {'alpha': 9000}\n"
            "    draft.verification['passed'] = True\n    raise\n")
        result = await repl.exec_cell(code)
        assert result.ok and result.final["value"] == "Answer: 2"
        proof = result.final["verification"]
        assert "counts" not in proof and proof["classification"]["counts"]["alpha"] == 2


async def test_extra_classification_rpc_value_is_rejected(corpus):
    async with SandboxedRepl(rpc_handlers={"llm_batch": async_provider}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        assert (await repl.exec_cell(classify_code())).ok
        with pytest.raises(ValueError, match="unsupported classification arguments"):
            await repl._classification({"op": "classification_final", "table": TABLE,
                "column": "category", "operation": "count", "labels": ["alpha"], "value": 9000})


async def test_classification_provider_wait_uses_uncapped_timeout_handling(corpus):
    async def delayed(request):
        await asyncio.sleep(.2)
        return provider(request)

    async with SandboxedRepl(rpc_handlers={"llm_batch": delayed}) as repl:
        await repl.start(mode="docdb", corpus_db=str(corpus))
        result = await repl.exec_cell(classify_code() +
            f"FINAL_CLASSIFICATION({TABLE!r}, 'category', 'count', ['alpha'])",
            timeout=.08, pause_provider_rpc_timeout=True)
        assert result.ok and result.final["value"] == "Answer: 2"
