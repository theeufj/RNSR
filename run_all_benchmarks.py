#!/usr/bin/env python3
"""
RNSR Generalization Benchmark Suite

Runs RNSR against multiple standard benchmarks to prove generalization
beyond FinanceBench. Produces a combined report with per-benchmark metrics.

Benchmarks:
1. FinanceBench  — Financial document QA (baseline: 100% accuracy)
2. MultiHiertt   — Multi-step hierarchical table reasoning (ACL 2022)
3. TAT-QA        — Tabular + textual financial QA (ACL 2021)
4. QASPER        — Scientific paper QA (NAACL 2021)
5. DocVQA        — Visual document understanding (WACV 2021)

Usage:
    # Run all benchmarks (10 samples each for quick validation)
    python run_all_benchmarks.py --max-samples 10

    # Run specific benchmark
    python run_all_benchmarks.py --benchmarks tatqa qasper --max-samples 50

    # Full run (no sample limit)
    python run_all_benchmarks.py --full
"""

from __future__ import annotations

import argparse
import json
import re
import time
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkScore:
    """Score for a single benchmark."""

    name: str
    total_questions: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_f1: float = 0.0
    avg_latency_s: float = 0.0
    errors: int = 0
    per_type_scores: dict[str, float] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)
    # LLM-as-judge metrics (semantic correctness)
    llm_judge_correct: int = 0
    llm_judge_accuracy: float = 0.0
    llm_judge_avg_score: float = 0.0


@dataclass
class CombinedReport:
    """Combined report across all benchmarks."""

    timestamp: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    max_samples: int | None = None
    benchmarks: list[BenchmarkScore] = field(default_factory=list)
    overall_accuracy: float = 0.0
    overall_f1: float = 0.0
    total_questions: int = 0
    total_correct: int = 0
    total_errors: int = 0
    runtime_s: float = 0.0
    total_llm_judge_correct: int = 0
    overall_llm_judge_accuracy: float = 0.0
    overall_llm_judge_avg_score: float = 0.0

    def compute_overall(self) -> None:
        """Compute aggregate stats across all benchmarks."""
        self.total_questions = sum(b.total_questions for b in self.benchmarks)
        self.total_correct = sum(b.correct for b in self.benchmarks)
        self.total_errors = sum(b.errors for b in self.benchmarks)
        self.total_llm_judge_correct = sum(
            getattr(b, "llm_judge_correct", 0) for b in self.benchmarks
        )
        self.overall_accuracy = (
            self.total_correct / self.total_questions if self.total_questions > 0 else 0
        )
        self.overall_llm_judge_accuracy = (
            self.total_llm_judge_correct / self.total_questions
            if self.total_questions > 0
            else 0
        )
        f1_scores = [b.avg_f1 for b in self.benchmarks if b.total_questions > 0]
        self.overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        judge_scores = [
            getattr(b, "llm_judge_avg_score", 0)
            for b in self.benchmarks
            if b.total_questions > 0
        ]
        self.overall_llm_judge_avg_score = (
            sum(judge_scores) / len(judge_scores) if judge_scores else 0
        )

    def print_report(self) -> None:
        """Print a formatted report to stdout."""
        print("\n" + "=" * 78)
        print("RNSR GENERALIZATION BENCHMARK REPORT")
        print("=" * 78)
        print(f"  Timestamp:    {self.timestamp}")
        print(f"  LLM:          {self.llm_provider} / {self.llm_model}")
        print(f"  Max samples:  {self.max_samples or 'unlimited'}")
        print(f"  Runtime:      {self.runtime_s:.1f}s")
        print("=" * 78)
        print()

        # Per-benchmark results (string metrics + LLM judge)
        print(
            f"{'Benchmark':<20} {'Questions':>10} {'Correct':>10} {'Accuracy':>10} "
            f"{'Judge Acc':>10} {'Judge Avg':>10} {'F1':>10} {'Errors':>8}"
        )
        print("-" * 98)

        for b in self.benchmarks:
            judge_acc = getattr(b, "llm_judge_accuracy", 0)
            judge_avg = getattr(b, "llm_judge_avg_score", 0)
            print(
                f"{b.name:<20} {b.total_questions:>10} {b.correct:>10} "
                f"{b.accuracy:>9.1%} {judge_acc:>9.1%} {judge_avg:>9.3f} "
                f"{b.avg_f1:>9.3f} {b.errors:>8}"
            )

        print("-" * 98)
        print(
            f"{'OVERALL':<20} {self.total_questions:>10} {self.total_correct:>10} "
            f"{self.overall_accuracy:>9.1%} {getattr(self, 'overall_llm_judge_accuracy', 0):>9.1%} "
            f"{getattr(self, 'overall_llm_judge_avg_score', 0):>9.3f} "
            f"{self.overall_f1:>9.3f} {self.total_errors:>8}"
        )
        print("=" * 98)

        # Per-type breakdown if available
        for b in self.benchmarks:
            if b.per_type_scores:
                print(f"\n  {b.name} — Per-type breakdown:")
                for answer_type, score in sorted(b.per_type_scores.items()):
                    print(f"    {answer_type:<30} {score:.1%}")

    def save(self, path: str | Path) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        print(f"\nReport saved to: {path}")

    def save_markdown_report(self, path: str | Path, truncate_prediction: int = 400) -> None:
        """Write a detailed markdown report: summary table + per-question tables (RNSR vs GT, LLM verdict, string match)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# RNSR Generalization Benchmark Report")
        lines.append("")
        lines.append(f"**Timestamp:** {self.timestamp}")
        lines.append(f"**LLM:** {self.llm_provider} / {self.llm_model}")
        lines.append(f"**Max samples:** {self.max_samples or 'unlimited'}")
        lines.append(f"**Runtime:** {self.runtime_s:.1f}s")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Benchmark | Questions | String Accuracy | LLM Judge Accuracy | F1 |")
        lines.append("|----------|-----------|-----------------|--------------------|-----|")
        for b in self.benchmarks:
            judge_acc = getattr(b, "llm_judge_accuracy", 0)
            lines.append(
                f"| {b.name} | {b.total_questions} | {b.accuracy:.1%} | {judge_acc:.1%} | {b.avg_f1:.3f} |"
            )
        lines.append(
            f"| **OVERALL** | {self.total_questions} | {self.overall_accuracy:.1%} | "
            f"{getattr(self, 'overall_llm_judge_accuracy', 0):.1%} | {self.overall_f1:.3f} |"
        )
        lines.append("")
        for b in self.benchmarks:
            lines.append(f"## {b.name}")
            lines.append("")
            lines.append("| # | Question | Ground Truth | RNSR Answer | LLM Judge | String | No content |")
            lines.append("|---|----------|--------------|-------------|-----------|--------|------------|")
            for i, d in enumerate(b.details, 1):
                q_raw = (d.get("question") or "").replace("|", "\\|").replace("\n", " ")
                q = q_raw[:80] + ("..." if len(q_raw) > 80 else "")
                gt_raw = (d.get("expected") or "").replace("|", "\\|").replace("\n", " ")
                gt = gt_raw[:60] + ("..." if len(gt_raw) > 60 else "")
                pred = (d.get("predicted") or "").replace("|", "\\|").replace("\n", " ")
                if len(pred) > truncate_prediction:
                    pred = pred[:truncate_prediction] + "..."
                verdict = d.get("llm_judge_verdict", "—")
                explanation = (d.get("llm_judge_explanation") or "").replace("|", "\\|").replace("\n", " ")[:80]
                if len((d.get("llm_judge_explanation") or "")) > 80:
                    explanation += "..."
                judge_cell = f"{verdict}: {explanation}"
                string_ok = "✓" if d.get("correct") else "✗"
                no_cont = "Yes" if d.get("no_content_found") else ""
                lines.append(f"| {i} | {q} | {gt} | {pred} | {judge_cell} | {string_ok} | {no_cont} |")
            lines.append("")
            # Detailed review: full question, ground truth, RNSR answer, judge verdict/explanation (no truncation)
            lines.append("### Detailed review (full responses for judge audit)")
            lines.append("")
            for i, d in enumerate(b.details, 1):
                lines.append(f"#### {b.name} — Question {i}")
                lines.append("")
                lines.append("**Question:**")
                lines.append("")
                lines.append(_md_block(d.get("question") or ""))
                lines.append("")
                lines.append("**Ground truth:**")
                lines.append("")
                lines.append(_md_block(d.get("expected") or ""))
                lines.append("")
                lines.append("**RNSR answer:**")
                lines.append("")
                lines.append(_md_block(d.get("predicted") or ""))
                lines.append("")
                lines.append(f"**LLM Judge:** {d.get('llm_judge_verdict', '—')} (score: {d.get('llm_judge_score', 0)})")
                lines.append("")
                lines.append(_md_block(d.get("llm_judge_explanation") or ""))
                lines.append("")
                lines.append(f"**String match:** {'✓' if d.get('correct') else '✗'}")
                lines.append("")
                # How we found it: nodes/sources and traversal path
                supp = d.get("supporting_facts") or []
                visited = d.get("nodes_visited") or []
                traversal_path = d.get("traversal_path") or []
                lines.append("**How we found it:**")
                lines.append("")
                nodes_display = (supp or visited)[:20]
                if nodes_display:
                    lines.append(f"- Nodes/sources visited: {nodes_display}" + (" ..." if len(supp or visited) > 20 else ""))
                    if traversal_path:
                        path_preview = traversal_path[:15]
                        path_str = ", ".join(str(x) for x in path_preview) + (" ..." if len(traversal_path) > 15 else "")
                        lines.append(f"- Traversal path: {path_str}")
                else:
                    lines.append("- No nodes read (navigator finished with 0 variables).")
                    if d.get("no_content_found"):
                        lines.append("- Reason: navigator finished with 0 nodes read (ToT dead ends or wrong branches).")
                lines.append("")
                if d.get("no_content_found"):
                    lines.append("*No content:* Answer was \"No relevant content found\" — navigator did not reach any content nodes.")
                lines.append("")
                lines.append("---")
                lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Markdown report saved to: {path}")


def _write_tot_trace_json(output_dir: Path, benchmark_name: str, details: list[dict[str, Any]]) -> None:
    """Write the full tree-of-thought trace for each question to a JSON file for review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{benchmark_name}_tot_trace.json"
    records = []
    for d in details:
        records.append({
            "question_id": d.get("id"),
            "question": d.get("question"),
            "expected": d.get("expected"),
            "predicted": d.get("predicted"),
            "no_content_found": d.get("no_content_found"),
            "nodes_visited": d.get("nodes_visited", []),
            "supporting_facts": d.get("supporting_facts", []),
            "traversal_path": d.get("traversal_path", []),
            "trace": d.get("tot_trace", []),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark": benchmark_name, "questions": records}, f, indent=2, default=str)
    print(f"  ToT trace saved to: {out_path}")


def _md_block(text: str, max_lines: int = 50) -> str:
    """Escape and wrap text for markdown (code block so newlines/pipes don't break tables)."""
    if not text:
        return "_empty_"
    # Truncate if huge to keep report readable
    lines = text.replace("```", "` ` `").split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return "```\n" + "\n".join(lines) + "\n```"


def _normalize_for_f1(text: str) -> list[str]:
    """Normalize text for F1 computation.

    Strips articles, removes commas and dollar signs, collapses trailing
    decimal zeros (e.g. ``1577.00`` -> ``1577``), and removes remaining
    punctuation so that ``$1,577`` and ``$1577.00`` both become ``1577``.
    """
    import string
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove commas (digit-grouping) and dollar signs
    text = text.replace(",", "").replace("$", " ")
    # Normalize trailing decimal zeros: "1577.00" / "1577.0" -> "1577"
    text = re.sub(r'(\d+)\.0+\b', r'\1', text)
    # Strip remaining punctuation
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    return text.split()


def _compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = set(_normalize_for_f1(prediction))
    gt_tokens = set(_normalize_for_f1(ground_truth))

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _is_correct(prediction: str, ground_truth: str, all_answers: list[str] | None = None) -> bool:
    """Check if prediction matches ground truth (flexible matching)."""
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()

    # Exact match
    if pred == gt:
        return True

    # Check all acceptable answers
    if all_answers:
        for ans in all_answers:
            if pred == ans.strip().lower():
                return True

    # Containment check (ground truth is contained in prediction)
    if gt and gt in pred:
        return True

    # Numeric matching (handles "1,234" vs "1234", "$1,234" vs "1234")
    # Try ALL number pairs -- the first number in the prediction may be
    # a year or other incidental digit (e.g. "FY2018 ... $1,577"), not the answer.
    pred_nums = re.findall(r"[\d,.]+", pred)
    gt_nums = re.findall(r"[\d,.]+", gt)
    if pred_nums and gt_nums:
        for pn in pred_nums:
            for gn in gt_nums:
                try:
                    pv = float(pn.replace(",", ""))
                    gv = float(gn.replace(",", ""))
                    if abs(pv - gv) < 0.01:
                        return True
                except ValueError:
                    continue

    return False


def run_benchmark_suite(
    benchmarks: list[str] | None = None,
    max_samples: int | None = 10,
    llm_provider: str = "gemini",
    llm_model: str = "gemini-2.5-flash",
    output_path: str = "benchmark_results/generalization_report.json",
) -> CombinedReport:
    """
    Run the full benchmark suite.

    Args:
        benchmarks: List of benchmark names to run. None = all.
        max_samples: Max samples per benchmark. None = unlimited.
        llm_provider: LLM provider to use.
        llm_model: LLM model to use.
        output_path: Where to save the report.

    Returns:
        CombinedReport with all results.
    """
    from rnsr.benchmarks.evaluation_suite import RNSRBenchmarkAdapter
    from rnsr.benchmarks.standard_benchmarks import LLMJudgeEvaluator

    all_benchmarks = ["financebench", "multihiertt", "tatqa", "qasper", "docvqa"]
    selected = benchmarks or all_benchmarks

    report = CombinedReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_samples=max_samples,
    )

    adapter = RNSRBenchmarkAdapter(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    judge = LLMJudgeEvaluator(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    start_total = time.perf_counter()

    for bench_name in selected:
        print(f"\n{'='*60}")
        print(f"  Running: {bench_name.upper()}")
        print(f"{'='*60}")

        try:
            dataset = _load_dataset(bench_name, max_samples)
        except Exception as e:
            logger.error(f"Failed to load {bench_name}", error=str(e))
            report.benchmarks.append(
                BenchmarkScore(name=bench_name, errors=1)
            )
            continue

        if not dataset.questions:
            print(f"  ⚠ No questions loaded for {bench_name}, skipping.")
            report.benchmarks.append(BenchmarkScore(name=bench_name))
            continue

        print(f"  Loaded {len(dataset.questions)} questions")

        score = BenchmarkScore(
            name=bench_name,
            total_questions=len(dataset.questions),
        )

        f1_scores: list[float] = []
        judge_scores: list[float] = []
        latencies: list[float] = []
        type_correct: dict[str, int] = {}
        type_total: dict[str, int] = {}

        for i, q in enumerate(dataset.questions):
            print(f"  [{i + 1}/{len(dataset.questions)}] {q.question[:80]}...")

            try:
                start = time.perf_counter()

                # Route to PDF or context-based answering
                if q.metadata.get("pdf_path") and Path(q.metadata["pdf_path"]).exists():
                    result = adapter.answer_from_pdf(
                        q.question,
                        Path(q.metadata["pdf_path"]),
                        metadata=q.metadata,
                    )
                else:
                    result = adapter.answer_from_context(
                        q.question,
                        q.context,
                        metadata=q.metadata,
                    )

                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

                # Evaluate: string metrics + LLM judge
                all_answers = q.metadata.get("all_answers")
                correct = _is_correct(result.answer, q.answer, all_answers)
                f1 = _compute_f1(result.answer, q.answer)

                judge_result = judge.evaluate(
                    q.question,
                    result.answer,
                    q.answer,
                    all_acceptable_answers=all_answers,
                )

                if correct:
                    score.correct += 1
                f1_scores.append(f1)
                judge_scores.append(judge_result.score)
                if judge_result.is_correct:
                    score.llm_judge_correct += 1

                # Per-type tracking
                answer_type = q.metadata.get("answer_type", q.reasoning_type)
                type_total[answer_type] = type_total.get(answer_type, 0) + 1
                if correct:
                    type_correct[answer_type] = type_correct.get(answer_type, 0) + 1

                no_content = (result.answer or "").strip().startswith("No relevant content found")
                score.details.append({
                    "id": q.id,
                    "question": q.question,
                    "expected": q.answer,
                    "predicted": result.answer,
                    "correct": correct,
                    "f1": f1,
                    "latency_s": elapsed,
                    "answer_type": answer_type,
                    "llm_judge_verdict": judge_result.verdict,
                    "llm_judge_score": judge_result.score,
                    "llm_judge_explanation": judge_result.explanation,
                    "supporting_facts": getattr(result, "supporting_facts", []) or [],
                    "nodes_visited": getattr(result, "nodes_visited", []) or [],
                    "traversal_path": getattr(result, "traversal_path", []) or [],
                    "no_content_found": no_content,
                    "tot_trace": getattr(result, "trace", []) or [],
                })

                status = "✓" if correct else "✗"
                print(f"    {status} F1={f1:.2f} ({elapsed:.1f}s)")

            except Exception as e:
                score.errors += 1
                logger.warning(
                    "Question failed",
                    question_id=q.id,
                    error=str(e),
                )
                print(f"    ✗ ERROR: {str(e)[:60]}")

        # Compute aggregates
        score.accuracy = score.correct / score.total_questions if score.total_questions else 0
        score.avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        score.llm_judge_accuracy = (
            score.llm_judge_correct / score.total_questions if score.total_questions else 0
        )
        score.llm_judge_avg_score = (
            sum(judge_scores) / len(judge_scores) if judge_scores else 0
        )
        score.avg_latency_s = sum(latencies) / len(latencies) if latencies else 0

        for t, total in type_total.items():
            score.per_type_scores[t] = type_correct.get(t, 0) / total

        # Write full tree-of-thought trace to JSON per benchmark for review
        _write_tot_trace_json(Path(output_path).parent, bench_name, score.details)

        report.benchmarks.append(score)

        print(f"\n  {bench_name}: {score.accuracy:.1%} accuracy, "
              f"F1={score.avg_f1:.3f}, {score.errors} errors")

    report.runtime_s = time.perf_counter() - start_total
    report.compute_overall()
    report.print_report()
    report.save(output_path)
    md_path = Path(output_path).parent / "benchmark_report.md"
    report.save_markdown_report(md_path)

    return report


def _load_dataset(name: str, max_samples: int | None = None) -> Any:
    """Load a benchmark dataset by name."""
    if name == "financebench":
        from rnsr.benchmarks.finance_bench import FinanceBenchLoader
        return FinanceBenchLoader.load(max_samples=max_samples)
    elif name == "multihiertt":
        from rnsr.benchmarks.multihiertt_bench import MultiHierttLoader
        return MultiHierttLoader.load(max_samples=max_samples)
    elif name == "tatqa":
        from rnsr.benchmarks.tatqa_bench import TATQALoader
        return TATQALoader.load(max_samples=max_samples)
    elif name == "qasper":
        from rnsr.benchmarks.qasper_bench import QASPERLoader
        return QASPERLoader.load(max_samples=max_samples)
    elif name == "docvqa":
        from rnsr.benchmarks.docvqa_bench import DocVQALoader
        return DocVQALoader.load(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNSR Generalization Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (10 samples per benchmark)
  python run_all_benchmarks.py

  # Run specific benchmarks
  python run_all_benchmarks.py --benchmarks tatqa qasper

  # Full run with all samples
  python run_all_benchmarks.py --full

  # Use OpenAI instead of Gemini
  python run_all_benchmarks.py --provider openai --model gpt-4o
        """,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["financebench", "multihiertt", "tatqa", "qasper", "docvqa"],
        help="Benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Max samples per benchmark (default: 10)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all samples (overrides --max-samples)",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "anthropic", "ollama"],
        help="LLM provider (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="LLM model name (default: gemini-2.5-flash; for ollama: OLLAMA_MODEL or qwen2.5-coder:32b)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/generalization_report.json",
        help="Output path for the report JSON",
    )

    args = parser.parse_args()

    max_samples = None if args.full else args.max_samples

    # When provider is ollama and user did not pass --model, use default Ollama model
    llm_model = args.model
    if args.provider == "ollama" and args.model == "gemini-2.5-flash":
        import os
        llm_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:32b")

    run_benchmark_suite(
        benchmarks=args.benchmarks,
        max_samples=max_samples,
        llm_provider=args.provider,
        llm_model=llm_model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
