"""RLM harness on MockLLM: FINAL path, budgets, damping, recovery, trajectory."""

import json

import pytest

from rnsr.config import Settings
from rnsr.harness.loop import EnvSpec, RootRunner
from rnsr.harness.recovery import rank_candidates
from rnsr.llm.mock import MockLLM


def make_runner(root: MockLLM, sub: MockLLM | None = None, **overrides) -> RootRunner:
    settings = Settings(**overrides)
    return RootRunner(
        root_client=root, root_model="mock-root",
        sub_client=sub or MockLLM(), sub_model="mock-sub",
        settings=settings,
    )


CLASSIC = EnvSpec(mode="classic", context="Fact: the 2023 total was 3234. " * 50)


class TestFinalPath:
    async def test_three_turn_trajectory(self, tmp_path):
        root = MockLLM().script(
            "```python\nprint(len(context))\n```",
            "```python\nimport re\nhits = re.findall(r'total was (\\d+)', context)\n"
            "print(hits[:3])\n```",
            "```python\nFINAL(int(hits[0]))\n```",
        )
        result = await make_runner(root).run("What was the 2023 total?", CLASSIC,
                                             run_dir=tmp_path)
        assert result.status == "final"
        assert result.answer == 3234
        assert result.iterations == 3

    async def test_trajectory_file_written(self, tmp_path):
        root = MockLLM().script("```python\nFINAL('x')\n```")
        result = await make_runner(root).run("q", CLASSIC, run_dir=tmp_path,
                                             query_id="q1")
        from pathlib import Path

        lines = Path(result.trajectory_path).read_text().splitlines()
        events = [json.loads(line) for line in lines]
        kinds = [e["kind"] for e in events]
        assert kinds[0] == "start"
        assert "cell" in kinds and "final" in kinds and kinds[-1] == "end"

    async def test_sub_calls_brokered_and_counted(self, tmp_path):
        root = MockLLM().script(
            "```python\nlabels = llm_map(['classify A', 'classify B'])\n"
            "print(labels)\n```",
            "```python\nFINAL(labels[0])\n```",
        )
        sub = MockLLM(default="LabelX")
        result = await make_runner(root, sub).run("q", CLASSIC, run_dir=tmp_path)
        assert result.answer == "LabelX"
        # 2 llm_map calls + 1 completeness check
        assert result.ledger["sub_calls"] == 3

    async def test_no_code_reply_prompts_again(self, tmp_path):
        root = MockLLM().script(
            "I think the answer is probably in the context somewhere.",
            "```python\nFINAL('done')\n```",
        )
        result = await make_runner(root).run("q", CLASSIC, run_dir=tmp_path)
        assert result.status == "final"
        # second root prompt contains the harness nudge
        assert "one ```python code block" in root.calls[1]["prompt"]


class TestBudgets:
    async def test_iteration_cap_fires_recovery(self, tmp_path):
        root = MockLLM(default="```python\nx = 1\nprint(x)\n```")
        root.rule(r"ended without FINAL", "NONE")
        result = await make_runner(root, max_root_iters=2).run("q", CLASSIC,
                                                               run_dir=tmp_path)
        assert result.status == "budget_exhausted"
        assert result.breached_cap == "max_root_iters"
        assert result.answer is None

    async def test_recovery_confirms_variable(self, tmp_path):
        root = MockLLM(default="```python\nanswer = 3234\nprint('computed')\n```")
        root.rule(r"ended without FINAL", "answer")
        result = await make_runner(root, max_root_iters=1).run("q", CLASSIC,
                                                               run_dir=tmp_path)
        assert result.status == "recovered"
        assert result.answer == 3234
        assert result.breached_cap == "max_root_iters"

    async def test_sub_call_budget_enforced_in_batch(self, tmp_path):
        root = MockLLM().script(
            "```python\nllm_map(['p'] * 500)\n```",   # over the 300 cap
            "```python\nFINAL('gave up on the sweep')\n```",
        )
        result = await make_runner(root).run("q", CLASSIC, run_dir=tmp_path)
        assert result.status == "final"  # loop survives; cell saw the error
        assert result.ledger["sub_calls"] == 1  # only the completeness check


class TestDamping:
    async def test_duplicate_output_injects_confirm_turn(self, tmp_path):
        root = MockLLM().script(
            "```python\nprint(40 + 2)\n```",
            "```python\nprint(sum([40, 2]))\n```",     # same output: 42
            "```python\nFINAL(42)\n```",
        )
        result = await make_runner(root).run("q", CLASSIC, run_dir=tmp_path)
        assert result.status == "final"
        third_prompt = root.calls[2]["prompt"]
        assert "computed this same result before" in third_prompt


class TestRecoveryRanking:
    def test_answerish_names_first(self):
        vars_ = {
            "scratch": {"type": "int", "repr": "1"},
            "final_answer": {"type": "int", "repr": "42"},
            "tmp": {"type": "str", "repr": "'x'"},
        }
        assert rank_candidates(vars_)[0] == "final_answer"

    def test_recency_breaks_ties(self):
        vars_ = {
            "a_result": {"type": "int", "repr": "1"},
            "b_result": {"type": "int", "repr": "2"},
        }
        assert rank_candidates(vars_)[0] == "b_result"


class TestRootResilience:
    async def test_hung_root_call_does_not_eat_wall_budget(self, tmp_path):
        # root client hangs longer than the per-call timeout; with a tiny
        # wall budget both attempts time out and the loop ends gracefully
        # instead of stalling for the provider SDK's 10-minute default.
        root = MockLLM(delay_s=3.0, default="```python\nFINAL('late')\n```")
        result = await make_runner(root, max_wall_s=0.5).run(
            "q", CLASSIC, run_dir=tmp_path)
        assert result.status == "budget_exhausted"
        assert result.breached_cap == "root_timeout"

    async def test_recovery_parses_name_inside_reasoning(self, tmp_path):
        root = MockLLM(default="```python\nanswer = 3234\nprint('done')\n```")
        root.rule(r"ended without FINAL",
                  "Looking at the task, the variable `answer` contains the "
                  "computed total, so that is the one.")
        result = await make_runner(root, max_root_iters=1).run("q", CLASSIC,
                                                               run_dir=tmp_path)
        assert result.status == "recovered"
        assert result.answer == 3234

    async def test_recovery_none_in_reasoning_still_none(self, tmp_path):
        root = MockLLM(default="```python\nx = 1\n```")
        root.rule(r"ended without FINAL",
                  "NONE of these variables answers the task.")
        result = await make_runner(root, max_root_iters=1).run("q", CLASSIC,
                                                               run_dir=tmp_path)
        assert result.status == "budget_exhausted"


class TestRecoveryExcludesToolOutput:
    def test_search_hit_dumps_excluded(self):
        vars_ = {
            "hits": {"type": "list", "repr": "[{'rung': 0, 'kind': 'sql', 'provenance': {...}}]"},
            "revenue_total": {"type": "int", "repr": "3234"},
        }
        assert rank_candidates(vars_) == ["revenue_total"]

    def test_all_junk_yields_empty(self):
        vars_ = {"cur": {"type": "Cursor", "repr": "<sqlite3.Cursor object at 0x10>"}}
        assert rank_candidates(vars_) == []


class TestSandboxRestart:
    async def test_runaway_cell_restarts_sandbox_and_continues(self, tmp_path):
        root = MockLLM().script(
            "```python\nwhile True: pass\n```",
            "```python\nFINAL('recovered after restart')\n```",
        )
        result = await make_runner(root, max_wall_s=15.0, cell_timeout_s=1.5).run(
            "q", CLASSIC, run_dir=tmp_path)
        assert result.status == "final"
        assert result.answer == "recovered after restart"
        # second prompt carries the restart notice
        assert "sandbox was restarted" in root.calls[1]["prompt"].lower()


class TestCompletenessGate:
    async def test_incomplete_final_pushed_back_once(self, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL('the Consumer segment')\n```",
            "```python\nFINAL('the Consumer segment, which shrank 0.9%')\n```",
        )
        sub = MockLLM().script("MISSING: the magnitude of the change", "COMPLETE")
        result = await make_runner(root, sub).run(
            "Which segment shrank, and by how much?", CLASSIC, run_dir=tmp_path)
        assert result.status == "final"
        assert "0.9%" in result.answer
        # pushback text reached the model
        assert "seems incomplete" in root.calls[1]["prompt"]
        # only the first FINAL is checked
        assert sum("Draft answer" in c["prompt"] for c in sub.calls) == 1

    async def test_complete_final_accepted_without_pushback(self, tmp_path):
        root = MockLLM().script("```python\nFINAL('42 (net revenue, $M)')\n```")
        sub = MockLLM(default="COMPLETE")
        result = await make_runner(root, sub).run("q", CLASSIC, run_dir=tmp_path)
        assert result.status == "final" and result.iterations == 1

    async def test_ambiguous_checker_reply_accepts(self, tmp_path):
        root = MockLLM().script("```python\nFINAL('x')\n```")
        sub = MockLLM(default="UNCLEAR")   # not MISSING -> accept
        result = await make_runner(root, sub).run("q", CLASSIC, run_dir=tmp_path)
        assert result.status == "final" and result.iterations == 1

    async def test_resubmission_after_pushback_accepted(self, tmp_path):
        # model insists the answer was complete; second FINAL passes unchecked
        root = MockLLM(default="```python\nFINAL('done')\n```")
        sub = MockLLM(default="MISSING: something imaginary")
        result = await make_runner(root, sub).run("q", CLASSIC, run_dir=tmp_path)
        assert result.status == "final"
        assert result.iterations == 2


class TestBudgetWarning:
    async def test_low_iterations_triggers_converge_nudge(self, tmp_path):
        root = MockLLM().script(
            "```python\nprint('explore 1')\n```",
            "```python\nprint('explore 2')\n```",
            "```python\nFINAL('converged')\n```",
        )
        result = await make_runner(root, max_root_iters=4).run("q", CLASSIC,
                                                               run_dir=tmp_path)
        assert result.status == "final"
        # warning lands when <=2 iterations remain: visible in the 3rd prompt
        assert "BUDGET LOW" in root.calls[2]["prompt"]
        assert "BUDGET LOW" not in root.calls[1]["prompt"]


class TestBatchLoop:
    async def test_batch_final_answers_all(self, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL_BATCH({'q1': '3234', 'q2': 'NOT_FOUND'})\n```",
        )
        br = await make_runner(root).run_batch(
            [("q1", "What was the 2023 total?"), ("q2", "Who is the CFO?")],
            CLASSIC, run_dir=tmp_path)
        assert br.result.status == "final"
        assert br.answers == {"q1": "3234", "q2": "NOT_FOUND"}
        # completeness is structural in batch mode — no sub-LM call
        assert br.result.ledger["sub_calls"] == 0

    async def test_missing_id_pushed_back_then_accepted(self, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL_BATCH({'q1': 'yes'})\n```",
            "```python\nFINAL_BATCH({'q1': 'yes', 'q2': 'no'})\n```",
        )
        br = await make_runner(root).run_batch(
            [("q1", "a?"), ("q2", "b?")], CLASSIC, run_dir=tmp_path)
        assert br.answers == {"q1": "yes", "q2": "no"}
        pushback = root.calls[1]["prompt"]
        assert "q2" in pushback and "FINAL_BATCH" in pushback

    async def test_resubmitted_incomplete_batch_yields_none(self, tmp_path):
        # model insists after one pushback -> accepted; missing qid comes
        # back as None so the caller can retry it solo
        root = MockLLM(default="```python\nFINAL_BATCH({'q1': 'yes'})\n```")
        br = await make_runner(root).run_batch(
            [("q1", "a?"), ("q2", "b?")], CLASSIC, run_dir=tmp_path)
        assert br.answers["q1"] == "yes"
        assert br.answers["q2"] is None

    async def test_non_dict_final_pushed_back(self, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL('just one answer')\n```",
            "```python\nFINAL_BATCH({'q1': 'a', 'q2': 'b'})\n```",
        )
        br = await make_runner(root).run_batch(
            [("q1", "a?"), ("q2", "b?")], CLASSIC, run_dir=tmp_path)
        assert br.answers == {"q1": "a", "q2": "b"}
        assert "not a dict" in root.calls[1]["prompt"]

    async def test_task_prompt_lists_ids_and_batch_hint(self, tmp_path):
        root = MockLLM().script(
            "```python\nFINAL_BATCH({'qA': '1', 'qB': '2'})\n```")
        await make_runner(root).run_batch(
            [("qA", "first?"), ("qB", "second?")], CLASSIC, run_dir=tmp_path)
        prompt = root.calls[0]["prompt"]
        assert "[qA]" in prompt and "[qB]" in prompt
        assert "FINAL_BATCH" in prompt


@pytest.fixture
def docdb_env(tmp_path):
    """Tiny docdb corpus with known facts for negative-audit tests."""
    from rnsr.db.artifact import CorpusDB
    from rnsr.harness.loop import EnvSpec
    from rnsr.ingest.model import Element, ParsedDocument
    from rnsr.ingest.pipeline import ingest

    def parse(path):
        return ParsedDocument(
            doc_id="intake", source_path=str(path), sha256="c" * 64,
            n_pages=1, parser="fake",
            elements=[Element(
                "text",
                "Daniel Robert Mitchell resides at 17 Strathfield Avenue. "
                "Contact email address: daniel@example.com. "
                "Lawyer code SUF123 appears on the letterhead. ", 1)],
            tables=[])

    out = tmp_path / "corpus.db"
    ingest([tmp_path / "intake.pdf"], out, parse=parse)
    with CorpusDB(out) as c:
        manifest = c.manifest_dict()
    return EnvSpec(mode="docdb", corpus_db=str(out), manifest=manifest)


class TestNegativeAudit:
    async def test_lazy_negative_pushed_back_then_corrected(self, tmp_path,
                                                            docdb_env):
        root = MockLLM().script(
            "```python\nFINAL_BATCH({'q1': 'NOT_FOUND', 'q2': 'yes'})\n```",
            "```python\nFINAL_BATCH({'q1': 'daniel@example.com', "
            "'q2': 'yes'})\n```",
        )
        br = await make_runner(root).run_batch(
            [("q1", "What is the email address of Daniel Robert Mitchell?"),
             ("q2", "Does Daniel Robert Mitchell reside at Strathfield?")],
            docdb_env, run_dir=tmp_path)
        assert br.answers["q1"] == "daniel@example.com"
        pushback = root.calls[1]["prompt"]
        assert "answered negatively" in pushback and "q1" in pushback
        # the positive answer is not audited
        assert "q2 (" not in pushback

    async def test_genuine_negative_resubmission_accepted(self, tmp_path,
                                                          docdb_env):
        # the model insists after one pushback -> accepted (audit is one-shot)
        root = MockLLM(
            default="```python\nFINAL_BATCH({'q1': 'NOT_FOUND'})\n```")
        br = await make_runner(root).run_batch(
            [("q1", "What is the email address of Daniel Robert Mitchell?")],
            docdb_env, run_dir=tmp_path)
        assert br.result.status == "final"
        assert br.answers["q1"] == "NOT_FOUND"
        assert br.result.iterations == 2

    async def test_negative_with_no_corpus_hits_not_flagged(self, tmp_path,
                                                            docdb_env):
        # terms absent from the corpus -> probe finds nothing -> no pushback
        root = MockLLM().script(
            "```python\nFINAL_BATCH({'q1': 'NOT_FOUND'})\n```")
        br = await make_runner(root).run_batch(
            [("q1", "Is there a superannuation splitting agreement "
                    "registered in Queensland?")],
            docdb_env, run_dir=tmp_path)
        assert br.result.status == "final"
        assert br.result.iterations == 1

    def test_is_negative_classifier(self):
        from rnsr.harness.loop import _is_negative

        for a in ("No", "no", " unknown ", "N/A", "", "NOT_FOUND",
                  "Not found in matter corpus", "not applicable"):
            assert _is_negative(a), a
        for a in ("Yes", "17 Strathfield Avenue", "14/02/2014", "No. 42"):
            assert not _is_negative(a), a


class TestBatchHelpers:
    def test_scale_budgets_half_per_extra_question(self):
        from rnsr.harness.loop import scale_budgets

        s = Settings()   # 20 iters / 300 sub / 600s / $2
        scaled = scale_budgets(s, 8)   # factor 4.5
        assert scaled.max_root_iters == 90
        assert scaled.max_sub_calls == 1350
        assert scaled.max_wall_s == 2700.0
        assert scaled.max_spend_usd == 9.0
        assert scale_budgets(s, 1) is s

    def test_coerce_batch_parses_dict_and_json_string(self):
        from rnsr.harness.loop import _coerce_batch

        assert _coerce_batch({"a": 1}) == {"a": 1}
        assert _coerce_batch('answers: {"a": "x"} done') == {"a": "x"}
        assert _coerce_batch("no braces here") is None
        assert _coerce_batch("{broken json") is None
        assert _coerce_batch(42) is None


def test_docdb_system_prompt_renders():
    # regression: stray {braces} in prompt text break .format at runtime
    from rnsr.harness.prompts.base import render_system

    s = render_system("docdb", manifest={"tables": [{"table_name": "t_x_001"}]})
    assert "t_x_001" in s and "Reconcile against stated aggregates" in s
