"""A generated FINAL must not outrun the runtime evidence it claims to use."""

import json
from pathlib import Path

from rnsr.config import Settings
from rnsr.harness.loop import EnvSpec, RootRunner
from rnsr.llm.mock import MockLLM


def runner(root, **settings):
    return RootRunner(root_client=root, root_model="mock-root",
                      sub_client=MockLLM(default="COMPLETE"), sub_model="mock-sub",
                      settings=Settings(**settings))


ENV = EnvSpec(mode="classic", context="North: 2\nSouth: 3")


async def test_comparison_draft_waits_for_actual_counts(tmp_path):
    root = MockLLM().script(
        "```python\ncounts = dict(line.split(': ') for line in context.splitlines())\n"
        "a, b = int(counts['North']), int(counts['South'])\n"
        "print('North:', a, 'South:', b)\n"
        "# Assumed North 17, South 3\nFINAL('North is more common')\n```",
        "```python\nrelationship = 'more' if a > b else 'less' if a < b else 'equally'\n"
        "FINAL(f'North is {relationship} common')\n```",
    )
    result = await runner(root).run("Compare North and South", ENV, run_dir=tmp_path)
    assert result.status == "final"
    assert result.answer == "North is less common"
    assert "North: 2 South: 3" in root.calls[1]["prompt"]
    assert "pending review" in root.calls[1]["prompt"]
    records = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    assert sum(r["kind"] == "final_observation_review" for r in records) == 1
    assert [r["value"] for r in records if r["kind"] == "final"] == [result.answer]


async def test_batch_draft_waits_for_new_document_evidence(tmp_path):
    root = MockLLM().script(
        "```python\nprint(context)\nFINAL_BATCH({'count': '17', 'larger': 'North'})\n```",
        "```python\nFINAL_BATCH({'count': '2', 'larger': 'South'})\n```",
    )
    result = await runner(root).run_batch(
        [("count", "North count?"), ("larger", "Which is larger?")], ENV, run_dir=tmp_path)
    assert result.answers == {"count": "2", "larger": "South"}
    assert "North: 2\nSouth: 3" in root.calls[1]["prompt"]
    assert "FINAL_BATCH is pending review" in root.calls[1]["prompt"]


async def test_reprinting_observed_evidence_does_not_loop(tmp_path):
    root = MockLLM(default="```python\nprint(context)\nFINAL('South')\n```")
    result = await runner(root).run("Which is larger?", ENV, run_dir=tmp_path)
    assert result.status == "final"
    assert result.iterations == 2


async def test_already_observed_result_needs_no_extra_turn(tmp_path):
    root = MockLLM().script(
        "```python\nprint(context)\n```",
        "```python\nprint(context)\nFINAL('South')\n```",
    )
    result = await runner(root).run("Which is larger?", ENV, run_dir=tmp_path)
    assert result.status == "final"
    assert result.iterations == 2
    assert not any("pending review" in c["prompt"] for c in root.calls)


async def test_last_iteration_does_not_accept_unobserved_wrong_draft(tmp_path):
    root = MockLLM(default="```python\nprint(context)\nFINAL('North is larger')\n```")
    result = await runner(root, max_root_iters=1).run(
        "Which is larger?", ENV, run_dir=tmp_path)
    assert result.status == "budget_exhausted"
    assert result.answer is None


async def test_completeness_pushback_keeps_cell_output(tmp_path):
    root = MockLLM().script(
        "```python\nprint(context)\n```",
        "```python\nprint(context)\nFINAL('South')\n```",
        "```python\nFINAL('South, 3')\n```",
    )
    r = runner(root)
    r.sub_client = MockLLM(default="MISSING: the count")
    result = await r.run("Which is larger and its count?", ENV, run_dir=tmp_path)
    assert result.answer == "South, 3"
    # The current observation remains beside the rejected cell, not merely in
    # an older turn (which may have different source or computational state).
    latest_observation = root.calls[2]["prompt"].split("print(context)")[-1]
    assert "North: 2\nSouth: 3" in latest_observation
    assert "answer seems incomplete" in latest_observation
