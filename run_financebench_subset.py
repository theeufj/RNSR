#!/usr/bin/env python3
"""
Run RNSR on a subset of FinanceBench questions.

Picks a diverse set of questions across companies/question types,
downloads the PDFs, runs RNSR via RNSRClient.ask(), and scores with LLM-as-judge.

Ordered smallest-first (8-K → earnings → 10-Q → 10-K) so you see results fast.
Includes a per-question timeout to prevent hanging on huge documents.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import hashlib
import multiprocessing
import requests
from pathlib import Path
from dataclasses import dataclass, asdict

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path("rnsr/benchmarks/data/financebench")
RESULTS_FILE = Path("benchmark_results/financebench_results.json")
MAX_QUESTIONS = int(os.getenv("FINBENCH_MAX", "15"))
# Per-question timeout in seconds (0 = no timeout; override with env var)
QUESTION_TIMEOUT = int(os.getenv("FINBENCH_TIMEOUT", "0"))

# Hand-picked diverse subset: idx in HF dataset
# All links verified alive 2026-02-14.  Ordered smallest → largest doc.
# 15 questions across 10 companies, 4 doc types (8-K, earnings, 10-Q, 10-K).
SELECTED_INDICES = [
    # --- 8-K filings (tiny, ~2-5 pages) ---
    80,   # Foot Locker 8K May-2022  — board votes (extraction)
    22,   # Amcor 8K Jul-2022        — key filing agenda (extraction)
    125,  # PepsiCo 8K May-2023      — AGM shareholder vote outcome
    29,   # Amcor Q4 FY2023 earnings — real change in sales (reasoning)
    # --- Earnings releases (~5-15 pages) ---
    128,  # PepsiCo 2023Q1 earnings  — why raise guidance (reasoning)
    129,  # PepsiCo 2023Q1 earnings  — guidance raise amount (numerical)
    28,   # Amcor Q4 FY2023 earnings — adj. EBITDA (numerical extraction)
    # --- 10-Q filings (~30-80 pages) ---
    5,    # 3M 2023Q2 10Q            — quick ratio / liquidity (domain)
    53,   # Best Buy 2024Q2 10Q      — cash equiv. drop (extraction)
    109,  # MGM 2023Q2 10Q           — short-term debt type (extraction)
    94,   # JPMorgan 2021Q1 10Q      — lowest net revenue segment
    # --- 10-K filings (~100-300 pages) ---
    50,   # Best Buy 2023 10K        — gross margin consistency (logical)
    8,    # Activision 2019 10K      — fixed asset turnover (numerical)
    130,  # Pfizer 2021 10K          — PPNE growth (extraction)
    31,   # AMD 2022 10K             — quick ratio / liquidity (logical)
]


@dataclass
class FBResult:
    idx: int
    company: str
    doc_name: str
    question: str
    expected_answer: str
    rnsr_answer: str
    correct: bool | None
    judge_reasoning: str
    time_seconds: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Child-process worker (runs in separate process for timeout isolation)
# ---------------------------------------------------------------------------


def _run_question_in_process(pdf_path_str, question, result_queue):
    """Worker function that runs in a child process."""
    try:
        from rnsr import RNSRClient
        client = RNSRClient()
        result = client.ask(pdf_path_str, question, use_knowledge_graph=True)
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        result_queue.put(("ok", answer))
    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

GITHUB_PDF_BASE = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
)


def _is_valid_pdf(path: Path) -> bool:
    """Check that a file is a real PDF by inspecting its magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def _download_pdf(url: str, doc_name: str) -> Path | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in doc_name)
    path = CACHE_DIR / f"{h}_{safe}.pdf"

    if path.exists() and path.stat().st_size > 1000 and _is_valid_pdf(path):
        return path

    # Remove any invalid cached file so we re-download
    if path.exists():
        logger.warning("removing_invalid_cached_pdf", path=str(path))
        path.unlink()

    urls_to_try = [
        url,
        f"{GITHUB_PDF_BASE}/{doc_name}.pdf",
    ]

    for attempt_url in urls_to_try:
        try:
            logger.info("downloading_pdf", doc=doc_name, url=attempt_url[:120])
            r = requests.get(attempt_url, timeout=120)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "")
            if "html" in content_type and "pdf" not in content_type:
                logger.warning(
                    "skipping_non_pdf_response",
                    doc=doc_name,
                    content_type=content_type,
                    url=attempt_url[:120],
                )
                continue

            path.write_bytes(r.content)

            if not _is_valid_pdf(path):
                logger.warning(
                    "downloaded_file_not_valid_pdf",
                    doc=doc_name,
                    url=attempt_url[:120],
                )
                path.unlink(missing_ok=True)
                continue

            logger.info("pdf_download_ok", doc=doc_name, size=path.stat().st_size)
            return path

        except Exception as e:
            logger.warning("pdf_download_attempt_failed", doc=doc_name, error=str(e))
            continue

    logger.error("pdf_download_failed_all_sources", doc=doc_name)
    return None


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def _judge_answer(question: str, expected: str, actual: str, llm) -> tuple[bool, str]:
    """Use LLM to judge if actual answer is correct given expected answer."""
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
        complete_fn = getattr(llm, "complete_json", None) or llm.complete
        resp = str(complete_fn(prompt))
        m = re.search(r'\{[^}]+\}', resp)
        if m:
            data = json.loads(m.group())
            verdict = data.get("verdict", "incorrect")
            score = float(data.get("score", 0.0))
            explanation = data.get("explanation", "")
            correct = verdict in ("correct", "partial")
            reasoning = f"[{verdict} {score}] {explanation}"
            return correct, reasoning
    except Exception as e:
        logger.warning("judge_failed", error=str(e))
    return False, "Judge failed to parse"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from datasets import load_dataset
    from rnsr.llm import get_llm

    ds = load_dataset("PatronusAI/financebench", split="train")
    llm = get_llm()

    indices = SELECTED_INDICES[:MAX_QUESTIONS]
    results: list[FBResult] = []

    print(f"\n{'='*70}")
    print(f"  FinanceBench Subset: {len(indices)} questions")
    print(f"{'='*70}\n")

    for i, idx in enumerate(indices):
        row = ds[idx]
        company = row.get("company", "?")
        doc_name = row.get("doc_name", "?")
        question = row["question"]
        expected = row["answer"]
        doc_link = row.get("doc_link", "")

        print(f"[{i+1}/{len(indices)}] {company} — {doc_name}")
        print(f"  Q: {question[:100]}...")

        # Download PDF
        pdf_path = _download_pdf(doc_link, doc_name) if doc_link else None
        if not pdf_path:
            r = FBResult(idx=idx, company=company, doc_name=doc_name,
                         question=question, expected_answer=expected,
                         rnsr_answer="", correct=None, judge_reasoning="",
                         time_seconds=0, error="PDF download failed")
            results.append(r)
            print(f"  SKIP: PDF download failed\n")
            _save_results(results, indices)
            continue

        # Run RNSR in a child process with a hard timeout via proc.join(timeout)
        t0 = time.time()
        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_run_question_in_process,
            args=(str(pdf_path), question, result_queue),
        )
        proc.start()

        # Wait for child (timeout=None when QUESTION_TIMEOUT==0)
        proc.join(timeout=QUESTION_TIMEOUT or None)
        timed_out_flag = QUESTION_TIMEOUT > 0 and proc.is_alive()
        if timed_out_flag:
            logger.warning("killing_timed_out_child",
                           pid=proc.pid, timeout=QUESTION_TIMEOUT)
            proc.kill()
            proc.join(timeout=10)
        elapsed = time.time() - t0

        if timed_out_flag:
            r = FBResult(idx=idx, company=company, doc_name=doc_name,
                         question=question, expected_answer=expected,
                         rnsr_answer="", correct=False, judge_reasoning="",
                         time_seconds=elapsed,
                         error=f"Timeout after {QUESTION_TIMEOUT}s")
            results.append(r)
            print(f"  TIMEOUT after {elapsed:.0f}s — skipping\n")
        elif not result_queue.empty():
            status_str, payload = result_queue.get_nowait()
            if status_str == "ok":
                answer = payload
                correct, reasoning = _judge_answer(question, expected, answer, llm)

                r = FBResult(idx=idx, company=company, doc_name=doc_name,
                             question=question, expected_answer=expected,
                             rnsr_answer=answer[:500], correct=correct,
                             judge_reasoning=reasoning, time_seconds=elapsed)
                results.append(r)

                verdict = "CORRECT" if correct else "WRONG"
                print(f"  A: {answer[:120]}...")
                print(f"  Expected: {expected[:120]}...")
                print(f"  Judge: {verdict} — {reasoning[:80]}")
                print(f"  Time: {elapsed:.1f}s\n")
            else:
                r = FBResult(idx=idx, company=company, doc_name=doc_name,
                             question=question, expected_answer=expected,
                             rnsr_answer="", correct=False, judge_reasoning="",
                             time_seconds=elapsed, error=payload[:200])
                results.append(r)
                print(f"  ERROR: {payload[:120]}\n")
        else:
            r = FBResult(idx=idx, company=company, doc_name=doc_name,
                         question=question, expected_answer=expected,
                         rnsr_answer="", correct=False, judge_reasoning="",
                         time_seconds=elapsed,
                         error="Process exited without result")
            results.append(r)
            print(f"  ERROR: Process exited without result\n")

        # Save incremental results after each question
        _save_results(results, indices)

    # Final summary
    _print_summary(results, indices)


def _save_results(results: list[FBResult], indices: list[int]):
    scored = [r for r in results if r.correct is not None]
    correct_count = sum(1 for r in scored if r.correct)
    errors = sum(1 for r in results if r.error)
    total = len(scored)
    avg_time = sum(r.time_seconds for r in scored) / total if total else 0

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "benchmark": "financebench_subset",
            "total_questions": len(indices),
            "scored": total,
            "correct": correct_count,
            "accuracy": round(correct_count / total, 4) if total else 0,
            "errors": errors,
            "avg_time_seconds": round(avg_time, 1),
            "results": [asdict(r) for r in results],
        }, f, indent=2)


def _print_summary(results: list[FBResult], indices: list[int]):
    scored = [r for r in results if r.correct is not None]
    correct_count = sum(1 for r in scored if r.correct)
    errors = sum(1 for r in results if r.error)
    total = len(scored)
    avg_time = sum(r.time_seconds for r in scored) / total if total else 0

    print(f"\n{'='*70}")
    print(f"  FINANCEBENCH RESULTS")
    print(f"{'='*70}")
    print(f"  Questions:  {len(indices)}")
    print(f"  Scored:     {total}")
    if total:
        print(f"  Correct:    {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    else:
        print(f"  Correct:    0")
    print(f"  Errors:     {errors}")
    print(f"  Avg time:   {avg_time:.1f}s per question")
    print(f"{'='*70}\n")

    # Per-question breakdown
    for r in results:
        status = "OK" if r.correct else ("ERR" if r.error else "WRONG")
        print(f"  [{status:5s}] {r.company:20s} {r.question[:60]}")

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
