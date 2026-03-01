#!/usr/bin/env python3
"""
Run RNSR on matterAiTests court case directories.

For each subdirectory containing a questions.csv:
  1. Ingest all case documents (excluding files with "question" in name)
  2. Build KG across documents
  3. Ask each question from questions.csv
  4. Compare RNSR answer to expected answer via LLM judge
  5. Write results to questions_results.csv

Usage:
    python run_matter_tests.py                        # first 4 dirs
    python run_matter_tests.py --limit 0              # all dirs
    python run_matter_tests.py --limit 10             # first 10
    python run_matter_tests.py --test-dir ./myTests   # custom root
    python run_matter_tests.py --force-reingest       # re-ingest existing stores
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import gc
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

faulthandler.enable()

import structlog

logger = structlog.get_logger(__name__)

# All file types RNSR can ingest natively
SUPPORTED_EXTENSIONS = {
    ".pdf", ".md", ".txt", ".text", ".markdown", ".docx",
    ".xlsx", ".xls", ".csv", ".msg",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
}

STORE_DIR_NAME = ".rnsr_store"
RESULTS_CSV_NAME = "questions_results.csv"
AGGREGATE_JSON_NAME = "matter_test_results.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question: str
    expected_answer: str
    rnsr_answer: str
    correct: bool | None
    judge_reasoning: str
    time_seconds: float
    nodes_visited: int = 0
    iterations: int = 0
    confidence: float = 0.0
    error: str | None = None


@dataclass
class CaseResult:
    directory: str
    num_documents: int
    ingestion_time_seconds: float
    num_questions: int
    num_correct: int
    num_wrong: int
    num_errors: int
    accuracy: float
    avg_question_time_seconds: float
    total_nodes_visited: int = 0
    total_iterations: int = 0
    avg_nodes_per_question: float = 0.0
    question_results: list[QuestionResult] = field(default_factory=list)
    ingestion_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------


def _find_questions_csv(directory: Path) -> Path | None:
    """Return the questions.csv file in *directory* (case-insensitive)."""
    for f in directory.iterdir():
        if f.is_file() and f.name.lower() == "questions.csv":
            return f
    return None


def discover_case_dirs(root: Path) -> list[Path]:
    """Recursively find all directories that contain a questions.csv."""
    case_dirs: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        lower_names = [f.lower() for f in filenames]
        if "questions.csv" in lower_names:
            case_dirs.append(Path(dirpath))
    case_dirs.sort(key=lambda p: str(p).lower())
    return case_dirs


# ---------------------------------------------------------------------------
# File collection (with exclusion)
# ---------------------------------------------------------------------------


def _has_question_in_name(path: Path) -> bool:
    """True if the filename (case-insensitive) contains 'question'."""
    return "question" in path.name.lower()


def collect_ingestible_files(directory: Path) -> list[Path]:
    """Gather all files eligible for ingestion, excluding question-related files."""
    files: list[Path] = []
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        if _has_question_in_name(f):
            continue
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if f.name.startswith("."):
            continue
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_case(
    case_dir: Path,
    files: list[Path],
    force: bool = False,
) -> tuple[Any, float, list[str]]:
    """
    Create/load a DocumentStore for *case_dir* and ingest *files*.

    Returns (store, elapsed_seconds, errors).
    """
    from rnsr import DocumentStore

    store_path = case_dir / STORE_DIR_NAME
    store = DocumentStore(str(store_path))

    if len(store) > 0 and not force:
        logger.info(
            "store_exists_reusing",
            case=case_dir.name,
            docs=len(store),
        )
        return store, 0.0, []

    errors: list[str] = []
    t0 = time.monotonic()

    # All file types (pdf, docx, md, txt) are now handled natively
    result = store.batch_ingest(
        sources=files,
        build_kg=False,
        skip_existing=True,
    )
    for err in result.errors:
        errors.append(f"{err['file']}: {err['error']}")

    # Build KG after all docs are in
    if len(store) > 0:
        try:
            store.build_workspace_kg()
            store.link_entities_across_documents()
        except Exception as exc:
            errors.append(f"KG build failed: {exc}")

    elapsed = time.monotonic() - t0
    return store, elapsed, errors


# ---------------------------------------------------------------------------
# Q&A evaluation
# ---------------------------------------------------------------------------


def read_questions(csv_path: Path) -> list[tuple[str, str]]:
    """Read (question, answer) pairs from a CSV file."""
    pairs: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            q, a = row[0].strip(), row[1].strip()
            if not q or q.lower() == "question":
                continue
            pairs.append((q, a))
    return pairs


def _judge_answer(
    question: str,
    expected: str,
    actual: str,
    llm: Any,
) -> tuple[bool | None, str]:
    """Use LLM to decide if *actual* is correct given *expected*."""
    if not actual or actual.lower().startswith("unable to"):
        return False, "No substantive answer provided"

    prompt = f"""You are evaluating whether a predicted answer is correct given a question and ground truth.

Question: {question}
Ground Truth Answer: {expected}
Predicted Answer: {actual[:4000]}

Does the predicted answer convey the same information as the ground truth? The predicted answer may be verbose, include source citations, or use different wording - focus on semantic equivalence. Ignore formatting and minor phrasing differences.

**Numeric and derived answers:** Treat numeric answers as correct when the **value** matches the ground truth even if units or format differ (e.g. 8325 thousand = 8.325 million = 8325000; "8325 thousand" vs "$8.325 million"). When the question asks for a derived value (average, total, sum, ratio), treat the prediction as correct if it states or clearly implies the same number, even if the wording differs.

Respond with ONLY valid JSON (no markdown, no extra text):
{{"verdict": "correct"|"partial"|"incorrect", "score": 1.0|0.5|0.0, "explanation": "brief reason"}}

Use: verdict "correct" and score 1.0 when the predicted answer clearly contains the same factual answer (including numerically equivalent values). Use "partial" and 0.5 when it is partly right. Use "incorrect" and 0.0 when it is wrong or does not address the question."""
    try:
        resp = str(llm.complete(prompt))
        m = re.search(r"\{[^}]+\}", resp)
        if m:
            data = json.loads(m.group())
            verdict = data.get("verdict", "incorrect")
            score = float(data.get("score", 0.0))
            explanation = data.get("explanation", "")
            correct = verdict in ("correct", "partial")
            reasoning = f"[{verdict} {score}] {explanation}"
            return correct, reasoning
    except Exception as exc:
        logger.warning("judge_failed", error=str(exc))
    return None, "Judge failed to parse"


def evaluate_questions(
    store: Any,
    qa_pairs: list[tuple[str, str]],
    llm: Any,
) -> list[QuestionResult]:
    """Run each question against the store and judge the answer."""
    results: list[QuestionResult] = []

    for i, (question, expected) in enumerate(qa_pairs):
        print(f"    Q{i + 1}/{len(qa_pairs)}: {question[:80]}...")
        t0 = time.monotonic()
        try:
            resp = store.query_cross_document(question)
            answer = resp.get("answer", "") if isinstance(resp, dict) else str(resp)
            nodes_visited = resp.get("total_nodes_visited", 0) if isinstance(resp, dict) else 0
            iterations = resp.get("total_iterations", 0) if isinstance(resp, dict) else 0
            confidence = resp.get("confidence", 0.0) if isinstance(resp, dict) else 0.0
        except Exception as exc:
            elapsed = time.monotonic() - t0
            results.append(QuestionResult(
                question=question,
                expected_answer=expected,
                rnsr_answer="",
                correct=False,
                judge_reasoning="",
                time_seconds=round(elapsed, 2),
                error=str(exc)[:300],
            ))
            print(f"      ERROR: {exc}")
            continue

        elapsed = time.monotonic() - t0
        correct, reasoning = _judge_answer(question, expected, answer, llm)

        results.append(QuestionResult(
            question=question,
            expected_answer=expected,
            rnsr_answer=answer[:2000],
            correct=correct,
            judge_reasoning=reasoning,
            time_seconds=round(elapsed, 2),
            nodes_visited=nodes_visited,
            iterations=iterations,
            confidence=round(confidence, 3),
        ))

        verdict = "CORRECT" if correct else ("ERROR" if correct is None else "WRONG")
        print(f"      {verdict} ({elapsed:.1f}s, {nodes_visited} nodes, {iterations} iters, conf={confidence:.2f}) — {reasoning[:60]}")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results_csv(csv_path: Path, results: list[QuestionResult]) -> None:
    """Write per-question results alongside the original Q&A."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question", "Expected_Answer", "RNSR_Answer",
            "Correct", "Time_Seconds", "Nodes_Visited", "Iterations",
            "Confidence", "Judge_Reasoning", "Error",
        ])
        for r in results:
            writer.writerow([
                r.question,
                r.expected_answer,
                r.rnsr_answer,
                r.correct,
                r.time_seconds,
                r.nodes_visited,
                r.iterations,
                r.confidence,
                r.judge_reasoning,
                r.error or "",
            ])


def print_summary(case_results: list[CaseResult]) -> None:
    """Print a summary table to the console."""
    print(f"\n{'=' * 110}")
    print("  MATTER AI TEST RESULTS")
    print(f"{'=' * 110}")
    print(
        f"  {'Directory':<45} {'Docs':>4} {'Ingest(s)':>9} {'Qs':>4} "
        f"{'Correct':>7} {'Acc%':>6} {'Avg Q(s)':>8} {'Avg Nodes':>9}"
    )
    print(f"  {'-' * 103}")

    total_q = 0
    total_correct = 0
    total_nodes = 0

    for cr in case_results:
        dir_label = cr.directory[:44]
        print(
            f"  {dir_label:<45} {cr.num_documents:>4} "
            f"{cr.ingestion_time_seconds:>9.1f} {cr.num_questions:>4} "
            f"{cr.num_correct:>4}/{cr.num_questions:<2} "
            f"{cr.accuracy * 100:>5.1f}% {cr.avg_question_time_seconds:>8.1f}"
            f" {cr.avg_nodes_per_question:>9.1f}"
        )
        total_q += cr.num_questions
        total_correct += cr.num_correct
        total_nodes += cr.total_nodes_visited

    print(f"  {'-' * 103}")
    overall_acc = total_correct / total_q * 100 if total_q else 0
    avg_nodes_overall = total_nodes / total_q if total_q else 0
    print(
        f"  {'TOTAL':<45} {'':>4} {'':>9} {total_q:>4} "
        f"{total_correct:>4}/{total_q:<2} {overall_acc:>5.1f}%"
        f" {'':>8} {avg_nodes_overall:>9.1f}"
    )
    print(f"{'=' * 110}\n")


def save_aggregate_json(
    output_path: Path,
    case_results: list[CaseResult],
    wall_time: float,
) -> None:
    """Save aggregate results to JSON."""
    total_q = sum(cr.num_questions for cr in case_results)
    total_correct = sum(cr.num_correct for cr in case_results)

    data = {
        "benchmark": "matter_ai_tests",
        "cases_evaluated": len(case_results),
        "total_questions": total_q,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total_q, 4) if total_q else 0,
        "wall_time_seconds": round(wall_time, 1),
        "cases": [asdict(cr) for cr in case_results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RNSR on matterAiTests")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("matterAiTests"),
        help="Root directory containing case subdirectories (default: matterAiTests)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Max directories to process (0 = all, default: 4)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample directories instead of taking the first N",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (implies --random)",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Re-ingest documents even if a store already exists",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for aggregate JSON (default: <test-dir>/matter_test_results.json)",
    )
    args = parser.parse_args()

    if not args.test_dir.is_dir():
        print(f"Error: {args.test_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (args.test_dir / AGGREGATE_JSON_NAME)

    from rnsr.llm import get_llm

    llm = get_llm()

    # Discover case directories
    case_dirs = discover_case_dirs(args.test_dir)
    if not case_dirs:
        print("No directories with questions.csv found.", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        if args.random or args.seed is not None:
            rng = random.Random(args.seed)
            case_dirs = rng.sample(case_dirs, min(args.limit, len(case_dirs)))
        else:
            case_dirs = case_dirs[: args.limit]

    print(f"\n{'=' * 100}")
    print(f"  Matter AI Tests — {len(case_dirs)} case directories")
    print(f"{'=' * 100}\n")

    case_results: list[CaseResult] = []
    wall_start = time.monotonic()

    for idx, case_dir in enumerate(case_dirs):
        rel = case_dir.relative_to(args.test_dir)
        print(f"\n[{idx + 1}/{len(case_dirs)}] {rel}")
        print(f"  {'-' * 60}")

        # 1. Find questions.csv
        qcsv = _find_questions_csv(case_dir)
        if qcsv is None:
            print("  SKIP — questions.csv not found")
            continue

        # 2. Collect files for ingestion
        files = collect_ingestible_files(case_dir)
        print(f"  Documents to ingest: {len(files)}")
        for f in files:
            print(f"    - {f.name}")

        if not files:
            print("  SKIP — no ingestible documents found")
            continue

        # 3. Ingest
        print(f"  Ingesting...")
        store, ingest_time, ingest_errors = ingest_case(
            case_dir, files, force=args.force_reingest
        )
        print(f"  Ingestion: {ingest_time:.1f}s, {len(store)} docs in store")
        if ingest_errors:
            for e in ingest_errors:
                print(f"    WARNING: {e}")

        # 4. Read Q&A pairs
        qa_pairs = read_questions(qcsv)
        print(f"  Questions: {len(qa_pairs)}")

        if not qa_pairs:
            print("  SKIP — no Q&A pairs found")
            continue

        # 5. Evaluate
        print(f"  Evaluating...")
        q_results = evaluate_questions(store, qa_pairs, llm)

        # 6. Write per-directory results CSV
        results_csv_path = case_dir / RESULTS_CSV_NAME
        write_results_csv(results_csv_path, q_results)
        print(f"  Results written to {results_csv_path}")

        # 7. Summarise
        num_correct = sum(1 for r in q_results if r.correct is True)
        num_wrong = sum(1 for r in q_results if r.correct is False)
        num_errors = sum(1 for r in q_results if r.correct is None)
        scored = [r for r in q_results if r.correct is not None]
        accuracy = num_correct / len(scored) if scored else 0.0
        avg_q_time = (
            sum(r.time_seconds for r in q_results) / len(q_results)
            if q_results
            else 0.0
        )

        total_nodes = sum(r.nodes_visited for r in q_results)
        total_iters = sum(r.iterations for r in q_results)
        avg_nodes = total_nodes / len(q_results) if q_results else 0.0

        case_results.append(CaseResult(
            directory=str(rel),
            num_documents=len(store),
            ingestion_time_seconds=round(ingest_time, 2),
            num_questions=len(qa_pairs),
            num_correct=num_correct,
            num_wrong=num_wrong,
            num_errors=num_errors,
            accuracy=round(accuracy, 4),
            avg_question_time_seconds=round(avg_q_time, 2),
            total_nodes_visited=total_nodes,
            total_iterations=total_iters,
            avg_nodes_per_question=round(avg_nodes, 1),
            question_results=q_results,
            ingestion_errors=ingest_errors,
        ))

        # Save incremental aggregate after each case
        save_aggregate_json(
            output_path, case_results, time.monotonic() - wall_start
        )

        # Release DocumentStore and force GC to reclaim memory between cases
        del store
        gc.collect()

    wall_time = time.monotonic() - wall_start
    print_summary(case_results)
    save_aggregate_json(output_path, case_results, wall_time)


if __name__ == "__main__":
    main()
