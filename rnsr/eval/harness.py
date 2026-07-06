"""Evaluation runner (§8): one harness, systems as flags.

rlm-classic is EnvSpec(mode='classic') on the *same* loop/budgets/
trajectory code as docdb — the baseline comparison is apples-to-apples by
construction. Document benchmarks ingest sources once per corpus (cached
by content) before querying.
"""

from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.eval.datasets.base import EvalItem
from rnsr.eval.metrics import EvalResult, judge_answer, score_answer, summarize
from rnsr.harness.loop import EnvSpec, RootRunner

SYSTEMS = ("docdb", "rlm-classic", "bm25-rag", "vector-rag", "rerank-rag")

_RERANK_PROMPT = """\
Question: {question}

Candidate excerpt:
{chunk}

Score this excerpt's usefulness for answering the question, 0 (irrelevant)
to 10 (contains the answer). Reply with ONLY the integer."""

_RAG_PROMPT = """\
Answer the question using ONLY the numbered excerpts below, retrieved from
the document corpus. If the excerpts do not contain the answer, say so
plainly rather than guessing.

{excerpts}

Question: {question}

Answer:"""


async def _rag_answer(item: EvalItem, system: str, corpus_path: Path, runner,
                      k: int = 12):
    """Traditional RAG baseline: retrieve top-k chunks, answer in one call.

    bm25-rag uses FTS5 lexical ranking; vector-rag embeds the corpus once
    (write-back cached) and retrieves by cosine. No loop, no tools, no
    verification — that is the point of the baseline.
    """
    import sqlite3

    from rnsr.db import fts as _fts
    from rnsr.harness.loop import QueryResult

    # check_same_thread=False: the embedding build runs in a worker thread
    conn = sqlite3.connect(corpus_path, check_same_thread=False)
    spend = {"usd": 0.0, "sub": 0}
    try:
        if system in ("bm25-rag", "rerank-rag"):
            from rnsr.env.search import _terms

            terms = _terms(item.question)
            match_q = " OR ".join(terms) if terms else item.question
            pool = k * 5 if system == "rerank-rag" else k
            hits = _fts.match(conn, match_q, k=pool)
            chunks = [(h["doc_id"], h["text"]) for h in hits]
            if system == "rerank-rag" and chunks:
                # LLM listwise-style reranking: sub-model scores each
                # candidate; answer from the top-k. Improves precision;
                # cannot widen k (an aggregation beyond k stays beyond k).
                from rnsr.llm.batch import map_prompts

                prompts = [_RERANK_PROMPT.format(question=item.question,
                                                 chunk=text[:1200])
                           for _, text in chunks]
                usage_total = {"n": 0}

                def _count(u):
                    spend["usd"] += u.cost_usd
                    usage_total["n"] += 1

                replies = await map_prompts(runner.sub_client, prompts,
                                            model=runner.sub_model,
                                            max_tokens=6, on_usage=_count)
                spend["sub"] = usage_total["n"]

                def score(reply):
                    try:
                        return int((reply.text if reply else "0").strip().split()[0])
                    except (ValueError, IndexError):
                        return 0

                ranked = sorted(zip(chunks, replies, strict=True),
                                key=lambda p: -score(p[1]))
                chunks = [c for c, _ in ranked[:k]]
        else:  # vector-rag
            if runner.embed_client is None:
                raise RuntimeError(
                    "vector-rag needs an embedding provider (OPENAI_API_KEY "
                    "or GOOGLE_API_KEY)")
            import asyncio as _aio

            from rnsr.env.embeddings import EmbeddingStore

            store = EmbeddingStore(conn)

            def embed_sync(texts: list[str]) -> list[list[float]]:
                return _aio.run(runner.embed_client.embed(
                    texts, model=runner.embed_model))

            # ensure() is sync and batch-heavy; keep the event loop free
            await _aio.to_thread(store.ensure, embed_sync, runner.embed_model)
            qvec = (await runner.embed_client.embed(
                [item.question], model=runner.embed_model))[0]
            scored = store.knn(qvec, k=k)
            rows = {r[0]: r for r in conn.execute(
                "SELECT chunk_id, doc_id, text FROM chunks WHERE chunk_id IN "
                f"({','.join('?' * len(scored))})", [c for c, _ in scored])}
            chunks = [(rows[c][1], rows[c][2]) for c, _ in scored if c in rows]

        excerpts = "\n\n".join(
            f"[{i + 1}] (doc: {doc_id})\n{text}" for i, (doc_id, text) in enumerate(chunks)
        )
        resp = await runner.root_client.complete(
            _RAG_PROMPT.format(excerpts=excerpts, question=item.question),
            model=runner.root_model, max_tokens=1500,
        )
        spend["usd"] += resp.usage.cost_usd
        answer = resp.text.strip()
        status = "final"
    finally:
        conn.close()

    return QueryResult(
        answer=answer, status=status, final=None,
        ledger={"spend_usd": spend["usd"], "sub_calls": spend["sub"]},
        trajectory_path="", iterations=1,
    )


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
        with CorpusDB(corpus_path) as corpus:
            manifest = corpus.manifest_dict()
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
) -> tuple[list[EvalResult], dict]:
    """Run a benchmark; writes results.jsonl + summary.json under run_dir.

    Scoring: exact/numeric string match first (free); when that fails and
    an answer exists, one sub-LM YES/NO equivalence call decides (string
    matching undercounts essay-style golds — seen live on FinanceBench).
    Judge cost is scoring overhead, not query cost, so it is not added to
    per-query cost_usd.

    Resumable: existing results.jsonl rows are kept and their qids skipped.
    A per-item failure (unparseable filing, ingest crash) records an
    'error' result instead of killing the run."""
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

    with open(results_path, "a") as out:
        for item in items[: limit or len(items)]:
            if item.qid in done:
                continue
            t0 = time.monotonic()
            try:
                if system in ("bm25-rag", "vector-rag", "rerank-rag"):
                    corpus_path = (
                        _corpus_for(item.sources, cache_dir, settings)
                        if item.sources
                        else _text_corpus_for(item.context or "", cache_dir, settings)
                    )
                    qr = await _rag_answer(item, system, corpus_path, runner)
                else:
                    env = _env_for(item, system, cache_dir, settings)
                    qr = await runner.run(item.question, env,
                                          run_dir=run_dir / "trajectories",
                                          query_id=item.qid)
                predicted = None if qr.answer is None else str(qr.answer)
                status = qr.status
                ledger = qr.ledger
                iterations = qr.iterations
                trajectory_path = qr.trajectory_path
            except Exception as e:   # e.g. Docling ConversionError on one filing
                predicted, status = None, "error"
                ledger = {"spend_usd": 0.0, "sub_calls": 0}
                iterations, trajectory_path = 0, None
                (run_dir / "errors.log").open("a").write(
                    f"{item.qid}: {type(e).__name__}: {e}\n")
            correct, scored_by = score_answer(predicted, item.gold), "string"
            if judge and not correct and predicted is not None:
                verdict = await judge_answer(runner.sub_client, runner.sub_model,
                                             item.question, predicted, item.gold)
                if verdict is not None:
                    correct, scored_by = verdict, "judge"
            result = EvalResult(
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
            )
            results.append(result)
            out.write(json.dumps(result.to_dict()) + "\n")
            out.flush()

    summary = summarize(results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results, summary
