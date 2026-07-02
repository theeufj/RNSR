"""Eval layer: scoring, percentiles, gate logic, end-to-end run on MockLLM."""

import math

from rnsr.config import Settings
from rnsr.eval.datasets.oolong import synthetic_oolong
from rnsr.eval.harness import run_eval
from rnsr.eval.metrics import (
    EvalResult,
    gate_report,
    percentile,
    score_answer,
    summarize,
)
from rnsr.harness.loop import RootRunner
from rnsr.llm.mock import MockLLM


class TestScoring:
    def test_exact_text(self):
        assert score_answer("Paris", "paris")
        assert not score_answer("London", "Paris")

    def test_numeric_formats_and_tolerance(self):
        assert score_answer("3,234", "3234")
        assert score_answer("$3,234 million", "3234")
        assert score_answer(3234.0, "3,234")
        assert score_answer("3250", "3234", numeric_rel_tol=0.01)
        assert not score_answer("3300", "3234")

    def test_containment_bounded(self):
        assert score_answer("The answer is Paris.", "Paris")
        assert not score_answer("Paris " + "waffle " * 50, "Paris")

    def test_none_never_correct(self):
        assert not score_answer(None, "42")


class TestAggregates:
    def _result(self, **kw):
        base = dict(qid="q", task_class="numeric", predicted="1", gold="1",
                    correct=True, status="final", latency_s=1.0, cost_usd=0.01,
                    sub_calls=2, iterations=3)
        base.update(kw)
        return EvalResult(**base)

    def test_percentiles(self):
        assert percentile([1, 2, 3, 4], 0.5) == 2.5
        assert math.isnan(percentile([], 0.5))

    def test_summary_by_class(self):
        results = [
            self._result(correct=True, task_class="numeric"),
            self._result(correct=False, task_class="numeric"),
            self._result(correct=True, task_class="lookup"),
        ]
        s = summarize(results)
        assert s["accuracy_by_class"] == {"lookup": 1.0, "numeric": 0.5}
        assert s["status_counts"] == {"final": 3}

    def test_gate_pass_and_fail(self):
        docdb = {"accuracy_by_class": {"numeric": 0.9, "lookup": 0.8},
                 "cost_usd": {"p50": 0.10}}
        classic = {"accuracy_by_class": {"numeric": 0.6, "lookup": 0.8},
                   "cost_usd": {"p50": 0.12}}
        assert gate_report(docdb, classic)["pass"] is True

        worse = {"accuracy_by_class": {"numeric": 0.5, "lookup": 0.8},
                 "cost_usd": {"p50": 0.10}}
        report = gate_report(worse, classic)
        assert report["pass"] is False
        assert report["checks"]["beats_classic[numeric]"] is False


class TestSyntheticOolong:
    def test_deterministic_and_countable(self):
        a, b = synthetic_oolong(n_lines=100, n_items=3), synthetic_oolong(n_lines=100, n_items=3)
        assert [i.gold for i in a] == [i.gold for i in b]
        assert sum(int(i.gold) for i in synthetic_oolong(n_lines=100, n_items=6)) == 100


class TestRunEval:
    async def test_end_to_end_classic_mock(self, tmp_path):
        items = synthetic_oolong(n_lines=30, n_items=2)
        # scripted root: count lines mentioning the target label pattern via
        # a FINAL of the known gold (mock can't reason; verify plumbing only)
        root = MockLLM()
        root.rule(r"How many lines are of type", f"```python\nFINAL({items[0].gold})\n```")
        runner = RootRunner(root_client=root, root_model="m", sub_client=MockLLM(),
                            sub_model="m", settings=Settings(max_root_iters=3))
        results, summary = await run_eval(items[:1], "rlm-classic", runner,
                                          run_dir=tmp_path)
        assert summary["n"] == 1
        assert results[0].correct
        assert (tmp_path / "results.jsonl").exists()
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "trajectories" / f"{items[0].qid}.jsonl").exists()


class TestJudge:
    async def test_judge_yes_no_unparseable(self):
        from rnsr.eval.metrics import judge_answer

        yes = MockLLM(default="YES")
        no = MockLLM(default="NO — the numbers differ")
        weird = MockLLM(default="It depends on interpretation")
        dead = MockLLM(fail_times=99)
        assert await judge_answer(yes, "m", "q", "p", "g") is True
        assert await judge_answer(no, "m", "q", "p", "g") is False
        assert await judge_answer(weird, "m", "q", "p", "g") is None
        assert await judge_answer(dead, "m", "q", "p", "g") is None

    async def test_run_eval_judge_rescues_semantic_match(self, tmp_path):
        items = synthetic_oolong(n_lines=20, n_items=1)
        items[0].gold = "No, the company is managing its CAPEX well."
        root = MockLLM(default="```python\nFINAL('No, it is not capital-intensive.')\n```")
        sub = MockLLM().rule(r"agree with the reference", "YES")
        runner = RootRunner(root_client=root, root_model="m", sub_client=sub,
                            sub_model="m", settings=Settings(max_root_iters=2))
        results, summary = await run_eval(items[:1], "rlm-classic", runner,
                                          run_dir=tmp_path)
        assert results[0].correct and results[0].scored_by == "judge"
        assert summary["scored_by_counts"] == {"judge": 1}

    async def test_string_match_skips_judge(self, tmp_path):
        items = synthetic_oolong(n_lines=20, n_items=1)
        root = MockLLM()
        root.rule(r"How many lines", f"```python\nFINAL({items[0].gold})\n```")
        sub = MockLLM(default="NO")   # would say NO if consulted
        runner = RootRunner(root_client=root, root_model="m", sub_client=sub,
                            sub_model="m", settings=Settings(max_root_iters=2))
        results, _ = await run_eval(items[:1], "rlm-classic", runner,
                                    run_dir=tmp_path)
        assert results[0].correct and results[0].scored_by == "string"
        assert not any(c["kind"] == "complete" and "agree with" in c["prompt"]
                       for c in sub.calls)


class TestNumericGoldClassifier:
    def test_value_golds_numeric(self):
        from rnsr.eval.datasets.financebench import _is_numeric_gold

        for g in ("$1577.00", "8.70", "~1.5x", "25%", "3,234 million",
                  "approximately $ 8.9 billion", "(1,234)"):
            assert _is_numeric_gold(g), g

    def test_sentence_golds_textual(self):
        from rnsr.eval.datasets.financebench import _is_numeric_gold

        for g in ("No, the quick ratio was 0.96 by Jun'23.",
                  "Operating margin decreased by 1.7% in FY2022.",
                  "Yes, they distribute dividends every quarter."):
            assert not _is_numeric_gold(g), g
