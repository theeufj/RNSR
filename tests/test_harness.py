"""RLM harness on MockLLM: FINAL path, budgets, damping, recovery, trajectory."""

import json

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
        assert result.ledger["sub_calls"] == 2

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
        assert result.ledger["sub_calls"] == 0


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
