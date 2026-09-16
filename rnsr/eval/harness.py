"""Evaluation runner (§8): one harness, systems as flags.

rlm-classic is EnvSpec(mode='classic') on the *same* loop/budgets/
trajectory code as docdb — the baseline comparison is apples-to-apples by
construction. Document benchmarks ingest sources once per corpus (cached
by content) before querying.
"""

from __future__ import annotations

import asyncio
import json
import time
from hashlib import sha256
from pathlib import Path

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.eval.autopsy import attach_causes, build_ledger, classify_miss, write_ledger
from rnsr.eval.baselines.rag import rag_answer
from rnsr.eval.datasets.base import EvalItem
from rnsr.eval.metrics import EvalResult, judge_answer, score_answer, summarize
from rnsr.harness.loop import EnvSpec, RootRunner

SYSTEMS = ("docdb", "rlm-classic", "bm25-rag", "vector-rag", "rerank-rag",
           "graph-rag")

def _corpus_valid(path: Path, n_sources: int) -> bool:
    """A cached corpus must hold every source document and some text."""
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            n_docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            n_pages = conn.execute("SELECT count(*) FROM doc_text").fetchone()[0]
        finally:
            conn.close()
        return n_docs >= n_sources and n_pages > 0
    except sqlite3.Error:
        return False


def _corpus_for(sources: list[Path], cache_dir: Path, settings: Settings) -> Path:
    """Ingest sources into a cached corpus.db (keyed by file content).

    Cached artifacts are validated before reuse — an interrupted ingest must
    trigger a rebuild, never an empty environment (seen live)."""
    from rnsr.ingest.pipeline import ingest

    h = sha256()
    for s in sorted(sources):
        h.update(Path(s).read_bytes())
    out = cache_dir / f"corpus_{h.hexdigest()[:16]}.db"
    if out.exists() and not _corpus_valid(out, len(sources)):
        out.unlink()
    if not out.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        ingest(sources, out, config=settings)
    return out


def _text_corpus_for(context: str, cache_dir: Path, settings: Settings) -> Path:
    """Cached corpus for a flat-text context (shared across questions that
    reuse the same context window)."""
    from rnsr.ingest.pipeline import ingest_text

    h = sha256(context.encode()).hexdigest()[:16]
    out = cache_dir / f"corpus_text_{h}.db"
    if out.exists() and not _corpus_valid(out, n_sources=1):
        out.unlink()
    if not out.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        ingest_text({"context": context}, out, config=settings)
    return out


def _env_for(item: EvalItem, system: str, cache_dir: Path, settings: Settings) -> EnvSpec:
    if system == "rlm-classic":
        context = item.context
        if context is None:
            # classic sees the same retained text docdb would, minus structure
            corpus_path = _corpus_for(item.sources, cache_dir, settings)
            with CorpusDB(corpus_path) as corpus:
                context = "\n\n".join(corpus.doc_dict().values())
        return EnvSpec(mode="classic", context=context)

    if system == "docdb":
        if not item.sources:
            if item.context is None:
                return EnvSpec(mode="classic", context=item.context)
            # flat-text benchmarks: ingest the context itself, so docdb gets
            # FTS + semantic_annotate over it instead of a bare string
            corpus_path = _text_corpus_for(item.context, cache_dir, settings)
        else:
            corpus_path = _corpus_for(item.sources, cache_dir, settings)
        from rnsr.ingest.health import enforce_health, load_health

        with CorpusDB(corpus_path) as corpus:
            manifest = corpus.manifest_dict()
            health = load_health(corpus, settings)
        enforce_health(health, settings)
        manifest["health"] = health.to_dict()
        return EnvSpec(mode="docdb", corpus_db=str(corpus_path), manifest=manifest)

    raise ValueError(f"unknown system: {system} (choose from {SYSTEMS})")


async def run_eval(
    items: list[EvalItem],
    system: str,
    runner: RootRunner,
    *,
    run_dir: str | Path,
    limit: int | None = None,
    judge: bool = True,
    concurrency: int = 1,
) -> tuple[list[EvalResult], dict]:
    """Run a benchmark; writes results.jsonl + summary.json under run_dir.

    Scoring: exact/numeric string match first (free); when that fails and
    an answer exists, one sub-LM YES/NO equivalence call decides (string
    matching undercounts essay-style golds — seen live on FinanceBench).
    Judge cost is scoring overhead, not query cost, so it is not added to
    per-query cost_usd.

    Resumable: existing results.jsonl rows are kept and their qids skipped.
    A per-item failure (unparseable filing, ingest crash) records an
    'error' result instead of killing the run.

    concurrency > 1 answers items in parallel (wall-clock win, identical
    LLM spend); corpus ingest stays serialized so concurrent items never
    double-ingest the same sources."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "corpora"
    settings = runner.settings

    results: list[EvalResult] = []
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = EvalResult(**json.loads(line))
            # Void rows — nothing was ever attempted (network outage, parse
            # crash) — are dropped so resume retries them.
            if r.status == "error" or (r.predicted is None and r.iterations == 0):
                continue
            results.append(r)
        results_path.write_text(
            "".join(json.dumps(r.to_dict()) + "\n" for r in results))
    done = {r.qid for r in results}

    ingest_lock = asyncio.Lock()   # heavy + content-cached: build once
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def process(item: EvalItem) -> EvalResult:
        t0 = time.monotonic()
        try:
            if system in ("bm25-rag", "vector-rag", "rerank-rag", "graph-rag"):
                async with ingest_lock:
                    corpus_path = await asyncio.to_thread(
                        lambda: _corpus_for(item.sources, cache_dir, settings)
                        if item.sources
                        else _text_corpus_for(item.context or "", cache_dir,
                                              settings))
                qr = await rag_answer(item, system, corpus_path, runner)
            else:
                async with ingest_lock:
                    env = await asyncio.to_thread(
                        _env_for, item, system, cache_dir, settings)
                qr = await runner.run(item.question, env,
                                      run_dir=run_dir / "trajectories",
                                      query_id=item.qid)
            predicted = None if qr.answer is None else str(qr.answer)
            status = qr.status
            ledger = qr.ledger
            iterations = qr.iterations
            trajectory_path = qr.trajectory_path
            result_tier = qr.evidence.tier if getattr(qr, "evidence", None) else None
        except Exception as e:   # e.g. Docling ConversionError on one filing
            predicted, status = None, "error"
            ledger = {"spend_usd": 0.0, "sub_calls": 0}
            iterations, trajectory_path, result_tier = 0, None, None
            with (run_dir / "errors.log").open("a", encoding="utf-8") as errors:
                errors.write(f"{item.qid}: {type(e).__name__}: {e}\n")
        correct, scored_by = score_answer(predicted, item.gold), "string"
        if judge and not correct and predicted is not None:
            verdict = await judge_answer(runner.sub_client, runner.sub_model,
                                         item.question, predicted, item.gold)
            if verdict is not None:
                correct, scored_by = verdict, "judge"
        return EvalResult(
            qid=item.qid,
            task_class=item.task_class,
            predicted=predicted,
            gold=item.gold,
            correct=correct,
            scored_by=scored_by,
            status=status,
            latency_s=round(time.monotonic() - t0, 3),
            cost_usd=ledger["spend_usd"],
            sub_calls=ledger["sub_calls"],
            iterations=iterations,
            trajectory_path=trajectory_path,
            expect=getattr(item, "expect", "value"),
            tier=result_tier,
        )

    with open(results_path, "a") as out:

        async def run_one(item: EvalItem) -> None:
            async with sem:
                result = await process(item)
            async with write_lock:
                results.append(result)
                out.write(json.dumps(result.to_dict()) + "\n")
                out.flush()

        todo = [item for item in items[: limit or len(items)]
                if item.qid not in done]
        if concurrency > 1:
            await asyncio.gather(*(run_one(item) for item in todo))
        else:
            for item in todo:
                await run_one(item)

    meta_by_qid = {item.qid: item.meta for item in items}
    from rnsr.harness.trajectory import read_trajectory

    classified = []
    for result in results:
        records = []
        if result.trajectory_path and Path(result.trajectory_path).exists():
            try:
                records = read_trajectory(result.trajectory_path, settings.trajectory_key)
            except Exception as exc:
                with (run_dir / "errors.log").open("a", encoding="utf-8") as errors:
                    errors.write(f"{result.qid}: trajectory unavailable ({type(exc).__name__})\n")
                records = []
        classified.append(classify_miss(
            result, records, meta=meta_by_qid.get(result.qid, {})))
    attach_causes(results, classified)
    results_path.write_text(
        "".join(json.dumps(r.to_dict()) + "\n" for r in results))

    summary = summarize(results)
    summary["autopsy"] = {
        "cause_counts": summary["cause_counts"],
        "cause_x_class": summary["cause_x_class"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_ledger(build_ledger(results, classified), run_dir,
                 title=f"Loss ledger — {run_dir.name}")
    return results, summary
