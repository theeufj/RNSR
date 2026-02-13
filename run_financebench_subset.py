#!/usr/bin/env python3
"""
Run RNSR on a subset of FinanceBench questions.

Picks a diverse set of questions across companies/question types,
downloads the PDFs, runs RNSR via RNSRClient.ask(), and scores with LLM-as-judge.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import hashlib
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

# Hand-picked diverse subset: idx in HF dataset
# Covers: 3M, AMD, American Express, Boeing, Best Buy, CVS, J&J, PepsiCo,
#          Pfizer, Verizon, MGM, AES, Amcor, Adobe, Activision
SELECTED_INDICES = [
    0,    # 3M 2018 10K - capex extraction (metrics)
    2,    # 3M 2022 10K - capital intensity (logical reasoning)
    8,    # Activision 2019 10K - fixed asset turnover (numerical)
    15,   # AES 2022 10K - restructuring costs (extraction)
    23,   # Amcor 2023 10K - quick ratio (numerical + logical)
    31,   # AMD 2022 10K - liquidity / quick ratio (logical)
    38,   # AmEx 2022 10K - debt securities (extraction)
    50,   # Best Buy 2023 10K - gross margin consistency (logical)
    76,   # CVS 2022 10K - capital intensity (logical)
    85,   # J&J 2022 10K - high growth assessment (logical)
    90,   # J&J 2023 8K - discontinued operation (extraction)
    106,  # MGM 2022 Q4 earnings - EBITDAR region (extraction)
    120,  # PepsiCo 2022 10K - geographies (extraction)
    130,  # Pfizer 2021 10K - PPNE growth (extraction)
    144,  # Verizon 2022 10K - liquidity / quick ratio (logical)
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
# PDF download
# ---------------------------------------------------------------------------

def _download_pdf(url: str, doc_name: str) -> Path | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in doc_name)
    path = CACHE_DIR / f"{h}_{safe}.pdf"
    if path.exists() and path.stat().st_size > 1000:
        return path
    try:
        logger.info("downloading_pdf", doc=doc_name, url=url[:80])
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        return path
    except Exception as e:
        logger.error("pdf_download_failed", doc=doc_name, error=str(e))
        return None


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def _judge_answer(question: str, expected: str, actual: str, llm) -> tuple[bool, str]:
    """Use LLM to judge if actual answer is correct given expected answer."""
    prompt = f"""You are a financial document QA judge. Determine if the ACTUAL answer is correct
by comparing it to the EXPECTED answer. Focus on factual correctness, not exact wording.

QUESTION: {question}

EXPECTED ANSWER: {expected}

ACTUAL ANSWER: {actual}

Rules:
- If the actual answer contains the key facts from the expected answer, it is CORRECT.
- Minor wording differences are OK.
- If the actual answer says "unable to find" or similar, it is INCORRECT.
- Numerical answers must be approximately correct (within 5% or rounding).

Respond with EXACTLY this JSON:
{{"correct": true/false, "reasoning": "brief explanation"}}
"""
    try:
        complete_fn = getattr(llm, "complete_json", None) or llm.complete
        resp = str(complete_fn(prompt))
        m = re.search(r'\{[^}]+\}', resp)
        if m:
            data = json.loads(m.group())
            return data.get("correct", False), data.get("reasoning", "")
    except Exception as e:
        logger.warning("judge_failed", error=str(e))
    return False, "Judge failed to parse"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from datasets import load_dataset
    from rnsr import RNSRClient
    from rnsr.llm import get_llm

    ds = load_dataset("PatronusAI/financebench", split="train")
    llm = get_llm()
    client = RNSRClient()

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
            continue

        # Run RNSR via client
        try:
            t0 = time.time()
            result = client.ask(
                str(pdf_path),
                question,
                use_knowledge_graph=True,
            )
            elapsed = time.time() - t0

            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            correct, reasoning = _judge_answer(question, expected, answer, llm)

            r = FBResult(idx=idx, company=company, doc_name=doc_name,
                         question=question, expected_answer=expected,
                         rnsr_answer=answer[:500], correct=correct,
                         judge_reasoning=reasoning, time_seconds=elapsed)
            results.append(r)

            status = "CORRECT" if correct else "WRONG"
            print(f"  A: {answer[:120]}...")
            print(f"  Expected: {expected[:120]}...")
            print(f"  Judge: {status} — {reasoning[:80]}")
            print(f"  Time: {elapsed:.1f}s\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = FBResult(idx=idx, company=company, doc_name=doc_name,
                         question=question, expected_answer=expected,
                         rnsr_answer="", correct=False, judge_reasoning="",
                         time_seconds=0, error=str(e)[:200])
            results.append(r)
            print(f"  ERROR: {str(e)[:120]}\n")

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
    main()
