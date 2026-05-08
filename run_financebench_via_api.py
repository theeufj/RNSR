#!/usr/bin/env python3
"""Run the FinanceBench subset against the RNSR FastAPI wrapper.

Differences vs. ``run_financebench_subset.py``:
  * Routes every request through the running FastAPI service (api.main:app),
    so we exercise the wrapper end-to-end (multipart-free; we register PDFs
    by server-side path).
  * Forces a fresh re-ingest on the first question per document by sending
    ``force_reindex=true`` with knowledge-graph build enabled.
  * Captures the FULL navigator trace for any wrong answer into
    ``benchmark_results/finbench_via_api/traces/<idx>.json`` so failures can
    be diagnosed without re-running the question.
  * Saves incremental, resumable results to
    ``benchmark_results/finbench_via_api/results.json``.

Env vars:
  RNSR_API_BASE          API base URL (default: http://127.0.0.1:8123)
  FINBENCH_INDICES       Comma-separated dataset indices to run; defaults to
                         the curated 15-question subset.
  FINBENCH_RESUME        If "1", skip questions already in results.json.
  FINBENCH_OUTDIR        Output directory (default: benchmark_results/finbench_via_api)
  FINBENCH_HTTP_TIMEOUT  Per-request HTTP timeout in seconds (default: 1800)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.getenv("RNSR_API_BASE", "http://127.0.0.1:8123").rstrip("/")
# /ask_advanced timeout: indexing is decoupled to /index (see INDEX_TIMEOUT
# below), so this only needs to cover navigation + verification.
HTTP_TIMEOUT = int(os.getenv("FINBENCH_HTTP_TIMEOUT", "900"))  # 15 min
PDF_CACHE_DIR = Path("rnsr/benchmarks/data/financebench")
OUTDIR = Path(os.getenv("FINBENCH_OUTDIR", "benchmark_results/finbench_via_api"))
RESULTS_FILE = OUTDIR / "results.json"
TRACE_DIR = OUTDIR / "traces"
RESUME = os.getenv("FINBENCH_RESUME", "0") == "1"

# Same curated subset as run_financebench_subset.py — ordered smallest → largest
DEFAULT_INDICES = [
    80, 22, 125,                # 8-Ks
    29, 128, 129, 28,           # earnings releases
    5, 53, 109, 94,             # 10-Qs
    50, 8, 130, 31,             # 10-Ks
]


@dataclass
class FBResult:
    idx: int
    company: str
    doc_name: str
    question: str
    expected_answer: str
    rnsr_answer: str = ""
    correct: bool | None = None
    confidence: float | None = None
    judge_reasoning: str = ""
    time_seconds: float = 0.0
    nodes_visited: int = 0
    iterations: int = 0
    sub_llm_calls: int = 0
    error: str | None = None
    trace_file: str | None = None


# ---------------------------------------------------------------------------
# PDF download (mirrors run_financebench_subset.py so the cache is shared)
# ---------------------------------------------------------------------------
GITHUB_PDF_BASE = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
)


def _is_valid_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def _download_pdf(url: str, doc_name: str) -> Path | None:
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in doc_name)
    path = PDF_CACHE_DIR / f"{h}_{safe}.pdf"

    if path.exists() and path.stat().st_size > 1000 and _is_valid_pdf(path):
        return path
    if path.exists():
        path.unlink()

    for attempt_url in [url, f"{GITHUB_PDF_BASE}/{doc_name}.pdf"]:
        try:
            logger.info("downloading_pdf", doc=doc_name, url=attempt_url[:120])
            r = requests.get(attempt_url, timeout=120)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            if "html" in ct and "pdf" not in ct:
                continue
            path.write_bytes(r.content)
            if not _is_valid_pdf(path):
                path.unlink(missing_ok=True)
                continue
            return path
        except Exception as exc:
            logger.warning("pdf_download_attempt_failed", doc=doc_name, error=str(exc))
            continue
    return None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _wait_for_api(retries: int = 60, sleep_s: float = 1.0) -> None:
    for i in range(retries):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            if r.status_code == 200:
                logger.info("api_ready", base=API_BASE, payload=r.json())
                return
        except requests.RequestException:
            pass
        time.sleep(sleep_s)
    raise RuntimeError(f"API at {API_BASE} did not become ready after {retries}s")


# Map (absolute pdf path) → API doc_id so we register each PDF only once.
_doc_id_cache: dict[str, str] = {}


def _register_pdf(pdf_path: Path) -> str:
    key = str(pdf_path.resolve())
    if key in _doc_id_cache:
        return _doc_id_cache[key]
    r = requests.post(
        f"{API_BASE}/documents",
        data={"path": key},
        timeout=60,
    )
    r.raise_for_status()
    doc_id = r.json()["doc_id"]
    _doc_id_cache[key] = doc_id
    return doc_id


# Track docs we've already pre-warmed (indexed + KG built) in this run, so
# subsequent questions on the same doc skip straight to /ask_advanced.
_prewarmed_docs: set[str] = set()


# Long ceiling: indexing a 200-section 10-K with KG extraction can take many
# minutes. Configurable via FINBENCH_INDEX_TIMEOUT.
INDEX_TIMEOUT = int(os.getenv("FINBENCH_INDEX_TIMEOUT", "5400"))  # 90 min
INDEX_POLL_S = float(os.getenv("FINBENCH_INDEX_POLL_S", "5"))


def _prewarm_index(doc_id: str) -> dict[str, Any]:
    """Trigger /index as a background job and poll /jobs until completion.

    This decouples the slow ingestion+KG build from the HTTP request that
    actually asks the question, so the ask call can use a sane (short)
    HTTP timeout while the indexing still gets all the time it needs.
    """
    if doc_id in _prewarmed_docs:
        return {"cached": True, "doc_id": doc_id}

    # Reuse on-disk cache by default. The runtime navigation/fallback
    # changes we ship don't invalidate previously-built skeletons or KGs,
    # so we'd burn ~30 minutes per large doc reindexing for nothing.
    # Set FINBENCH_FORCE_REINDEX=1 to override (e.g. after an
    # ingestion/header-classifier change).
    force_reindex = os.getenv("FINBENCH_FORCE_REINDEX", "0").lower() in ("1", "true", "yes")
    r = requests.post(
        f"{API_BASE}/documents/{doc_id}/index",
        json={"build_knowledge_graph": True, "force_reindex": force_reindex},
        timeout=60,
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]

    deadline = time.time() + INDEX_TIMEOUT
    while time.time() < deadline:
        time.sleep(INDEX_POLL_S)
        jr = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=15).json()
        status = jr.get("status")
        if status == "completed":
            _prewarmed_docs.add(doc_id)
            return jr.get("result") or {}
        if status == "failed":
            raise RuntimeError(
                f"Index job {job_id} failed: {jr.get('error', '<no error>')}"
            )

    raise TimeoutError(
        f"Index job {job_id} did not finish in {INDEX_TIMEOUT}s"
    )


def _ask_advanced(doc_id: str, question: str) -> dict[str, Any]:
    body = {
        "question": question,
        "use_rlm": True,
        "use_knowledge_graph": True,
        "enable_pre_filtering": True,
        "enable_verification": True,
        "max_recursion_depth": 3,
        # Pre-warm path always built the KG already — never reindex here.
        "force_reindex": False,
    }
    r = requests.post(
        f"{API_BASE}/documents/{doc_id}/ask_advanced",
        json=body,
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# LLM judge (same prompt as the original baseline, called in-process)
# ---------------------------------------------------------------------------
def _judge_answer(question: str, expected: str, actual: str, llm) -> tuple[bool, str]:
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
        m = re.search(r'\{[\s\S]+?\}', resp)
        if m:
            data = json.loads(m.group())
            verdict = data.get("verdict", "incorrect")
            score = float(data.get("score", 0.0))
            explanation = data.get("explanation", "")
            correct = verdict in ("correct", "partial")
            return correct, f"[{verdict} {score}] {explanation}"
    except Exception as exc:
        logger.warning("judge_failed", error=str(exc))
    return False, "Judge failed to parse"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save_results(results: list[FBResult], indices: list[int]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scored = [r for r in results if r.correct is not None]
    correct_count = sum(1 for r in scored if r.correct)
    errors = sum(1 for r in results if r.error)
    total = len(scored)
    avg_time = sum(r.time_seconds for r in scored) / total if total else 0
    payload = {
        "benchmark": "financebench_via_api",
        "api_base": API_BASE,
        "total_questions": len(indices),
        "scored": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 4) if total else 0,
        "errors": errors,
        "avg_time_seconds": round(avg_time, 1),
        "results": [asdict(r) for r in results],
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2))


def _load_existing_results() -> dict[int, FBResult]:
    if not RESULTS_FILE.exists():
        return {}
    try:
        data = json.loads(RESULTS_FILE.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[int, FBResult] = {}
    for raw in data.get("results", []):
        try:
            rec = FBResult(**raw)
            out[rec.idx] = rec
        except TypeError:
            continue
    return out


def _save_trace(idx: int, trace_payload: dict[str, Any]) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{idx}.json"
    path.write_text(json.dumps(trace_payload, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _selected_indices() -> list[int]:
    raw = os.getenv("FINBENCH_INDICES")
    if not raw:
        return DEFAULT_INDICES
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> int:
    from datasets import load_dataset
    from rnsr.llm import get_llm

    _wait_for_api()
    ds = load_dataset("PatronusAI/financebench", split="train")
    llm = get_llm()

    indices = _selected_indices()
    existing = _load_existing_results() if RESUME else {}

    print(f"\n{'=' * 72}")
    print(f"  FinanceBench via FastAPI: {len(indices)} questions")
    print(f"  API base: {API_BASE}")
    print(f"  Output:   {OUTDIR}")
    if RESUME and existing:
        print(f"  Resuming: {len(existing)} prior result(s) on disk")
    print(f"{'=' * 72}\n")

    results: list[FBResult] = []

    for i, idx in enumerate(indices):
        if RESUME and idx in existing and existing[idx].correct is not None:
            print(f"[{i + 1}/{len(indices)}] idx={idx} — skipping (already scored)")
            results.append(existing[idx])
            continue

        row = ds[idx]
        company = row.get("company", "?")
        doc_name = row.get("doc_name", "?")
        question = row["question"]
        expected = row["answer"]
        doc_link = row.get("doc_link", "")

        print(f"[{i + 1}/{len(indices)}] {company} — {doc_name}")
        print(f"  Q: {question[:110]}")

        rec = FBResult(
            idx=idx,
            company=company,
            doc_name=doc_name,
            question=question,
            expected_answer=expected,
        )

        pdf_path = _download_pdf(doc_link, doc_name) if doc_link else None
        if not pdf_path:
            rec.error = "PDF download failed"
            rec.correct = None
            results.append(rec)
            print(f"  SKIP: PDF download failed\n")
            _save_results(results, indices)
            continue

        t0 = time.time()
        try:
            doc_id = _register_pdf(pdf_path)
            prewarm_info = _prewarm_index(doc_id)
            t_index = time.time() - t0
            print(
                f"  indexed in {t_index:.0f}s "
                f"(nodes={prewarm_info.get('nodes', '?')}, "
                f"kg_entities={prewarm_info.get('kg_entities', '?')})"
            )
            api_resp = _ask_advanced(doc_id, question)
        except requests.HTTPError as exc:
            rec.error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            rec.correct = False
            rec.time_seconds = time.time() - t0
            results.append(rec)
            print(f"  ERROR: {rec.error}\n")
            _save_results(results, indices)
            continue
        except Exception as exc:
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.correct = False
            rec.time_seconds = time.time() - t0
            results.append(rec)
            print(f"  ERROR: {rec.error}\n")
            _save_results(results, indices)
            continue
        elapsed = time.time() - t0

        answer = api_resp.get("answer") or ""
        confidence = api_resp.get("confidence")
        raw = api_resp.get("raw") or {}
        nodes_visited = raw.get("visited_nodes") or raw.get("nodes_visited") or []
        iterations = raw.get("iteration") or 0
        sub_calls = raw.get("recursion_call_count") or 0

        correct, reasoning = _judge_answer(question, expected, answer, llm)

        rec.rnsr_answer = answer[:1000]
        rec.confidence = confidence
        rec.correct = correct
        rec.judge_reasoning = reasoning
        rec.time_seconds = elapsed
        rec.nodes_visited = len(nodes_visited) if isinstance(nodes_visited, list) else 0
        rec.iterations = int(iterations) if isinstance(iterations, int) else 0
        rec.sub_llm_calls = int(sub_calls) if isinstance(sub_calls, int) else 0

        if not correct:
            trace_payload = {
                "idx": idx,
                "company": company,
                "doc_name": doc_name,
                "question": question,
                "expected_answer": expected,
                "rnsr_answer": answer,
                "confidence": confidence,
                "judge_reasoning": reasoning,
                "raw_navigator_result": raw,
            }
            trace_path = _save_trace(idx, trace_payload)
            rec.trace_file = str(trace_path)

        results.append(rec)
        verdict = "CORRECT" if correct else "WRONG"
        print(f"  A: {answer[:140]}")
        print(f"  Expected: {expected[:140]}")
        print(f"  Judge: {verdict} — {reasoning[:120]}")
        print(
            f"  Time: {elapsed:.1f}s | nodes={rec.nodes_visited} "
            f"iter={rec.iterations} sub_llm={rec.sub_llm_calls} "
            f"conf={confidence}\n"
        )

        _save_results(results, indices)

    _print_summary(results, indices)
    return 0


def _print_summary(results: list[FBResult], indices: list[int]) -> None:
    scored = [r for r in results if r.correct is not None]
    correct_count = sum(1 for r in scored if r.correct)
    errors = sum(1 for r in results if r.error)
    total = len(scored)
    avg_time = sum(r.time_seconds for r in scored) / total if total else 0

    print(f"\n{'=' * 72}")
    print(f"  FINANCEBENCH (via FastAPI) RESULTS")
    print(f"{'=' * 72}")
    print(f"  Questions:  {len(indices)}")
    print(f"  Scored:     {total}")
    if total:
        print(f"  Correct:    {correct_count}/{total} ({100 * correct_count / total:.1f}%)")
    print(f"  Errors:     {errors}")
    print(f"  Avg time:   {avg_time:.1f}s per question")
    print(f"{'=' * 72}\n")

    for r in results:
        status = "OK" if r.correct else ("ERR" if r.error else "WRONG")
        line = f"  [{status:5s}] idx={r.idx:>3} {r.company:18s} {r.question[:64]}"
        print(line)

    print(f"\nResults: {RESULTS_FILE}")
    print(f"Traces (failures only): {TRACE_DIR}")


if __name__ == "__main__":
    sys.exit(main())
