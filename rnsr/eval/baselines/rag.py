"""Retrieval baselines used by the evaluation runner."""
from __future__ import annotations

from pathlib import Path

from rnsr.eval.datasets.base import EvalItem

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


async def rag_answer(item: EvalItem, system: str, corpus_path: Path, runner,
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
        if system == "graph-rag":
            from rnsr.eval.graphrag import (
                ANSWER_PROMPT,
                build_graph_index,
                graph_retrieve,
            )

            def _count(u):
                spend["usd"] += u.cost_usd
                spend["sub"] += 1

            await build_graph_index(conn, runner.sub_client, runner.sub_model,
                                    on_usage=_count)
            summaries, gchunks = graph_retrieve(conn, item.question, k_chunks=k)
            resp = await runner.root_client.complete(
                ANSWER_PROMPT.format(
                    summaries="\n\n".join(
                        f"[{i + 1}] {s}" for i, s in enumerate(summaries)),
                    excerpts="\n\n".join(
                        f"({d})\n{t[:1200]}" for d, t in gchunks),
                    question=item.question),
                model=runner.root_model, max_tokens=1500)
            spend["usd"] += resp.usage.cost_usd
            return QueryResult(
                answer=resp.text.strip(), status="final", final=None,
                ledger={"spend_usd": spend["usd"], "sub_calls": spend["sub"]},
                trajectory_path="", iterations=1)
        if system in ("bm25-rag", "rerank-rag"):
            from rnsr.env.search import terms

            terms = terms(item.question)
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
