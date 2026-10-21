"""Baseline arms for a matter benchmark directory: the same questions, caps,
models and scorer as the DocDB run, with the corpus presented differently.

    stuffing  every document in the prompt of one call per question
    classic   one flat string in the REPL (the RLM-classic environment)
    dci       one text file per document plus ls/grep/read helpers in the
              REPL (direct corpus interaction, arXiv:2605.05242)

Reads ``<matter-dir>/{questions.csv,golden.json,map.json,work/corpora/corpus.db}``
as laid out by ``docs/benchmarks/matters-2026-10.md`` and writes
``answers_chunk1.csv``, ``answers_status.csv`` (keyed by field id so the
scorer joins tiers) and ``run_report.json`` to ``--out``. Score with
``rnsr regress`` exactly as for the DocDB arm.

    python -m rnsr.eval.matter_arms --matter-dir runs/matters-bench/<M> \
        --arm stuffing --out runs/matters-bench-arms/<M>/stuffing
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import time
from pathlib import Path

from rnsr.answer_semantics import NOT_FOUND

ARMS = ("stuffing", "classic", "dci")

_STUFFING_SYSTEM = """\
You answer questions about one legal matter from its complete document \
corpus, which follows in full. Every document is included; nothing is \
withheld. Answer from the documents only.

Rules:
- Give the answer directly and concisely, then at most one sentence of \
supporting detail naming the document it rests on.
- Where documents disagree, prefer the later or corrected document and say \
that an earlier value was superseded.
- If the corpus does not contain the information asked for, or the question \
rests on a premise the documents contradict, reply exactly: {not_found}
- If the question specifies its own answer format (including its own wording \
for unknown or not-applicable answers), that format governs.

CORPUS
======
{corpus}
"""


def load_documents(corpus_db: Path) -> dict[str, str]:
    """file name -> full retained text, in source order, one entry per document."""
    conn = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True)
    docs: dict[str, str] = {}
    rows = conn.execute(
        "SELECT d.doc_id, d.source_path, group_concat(t.text, char(10) || char(10)) "
        "FROM documents d JOIN (SELECT doc_id, text FROM doc_text ORDER BY doc_id, page) t "
        "ON t.doc_id = d.doc_id GROUP BY d.doc_id ORDER BY d.source_path")
    for doc_id, source_path, text in rows:
        name = Path(source_path or doc_id).stem + ".txt"
        if name in docs:
            name = f"{Path(source_path or doc_id).stem}_{doc_id[-6:]}.txt"
        docs[name] = text or ""
    conn.close()
    return docs


def load_questions(path: Path, id_col: str = "field_id") -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(r[id_col], r["ground_truth_question"]) for r in rows]


def flat_context(docs: dict[str, str]) -> str:
    return "\n\n".join(f"===== {name} =====\n{text}" for name, text in docs.items())


async def _run_stuffing(qs, docs, settings, out: Path, concurrency: int):
    from rnsr.llm.router import Router

    router = Router(settings)
    root = router.resolve("root")
    system = _STUFFING_SYSTEM.format(not_found=NOT_FOUND, corpus=flat_context(docs))
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict] = {}

    async def one(fid: str, question: str) -> None:
        async with sem:
            started = time.monotonic()
            try:
                resp = await root.client.complete(
                    f"QUESTION: {question}", model=root.model, system=system, max_tokens=1024)
                text = (resp.text or "").strip()
                usage = resp.usage
                results[fid] = {"answer": text or None, "status": "final" if text else "unanswered",
                                "error": "", "elapsed_s": round(time.monotonic() - started, 2),
                                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                                "cache_read_tokens": usage.cache_read_tokens,
                                "cost_usd": round(usage.cost_usd, 6)}
            except Exception as exc:  # record, never sink the run
                results[fid] = {"answer": None, "status": "error",
                                "error": f"{type(exc).__name__}: {exc}"[:300],
                                "elapsed_s": round(time.monotonic() - started, 2)}

    # Warm the prefix cache once so the remaining calls read it.
    await one(*qs[0])
    await asyncio.gather(*(one(fid, q) for fid, q in qs[1:]))
    return results, router.governor.snapshot()


async def _run_loop(arm: str, qs, docs, settings, out: Path, batch_size: int, concurrency: int,
                    corpus_db: Path):
    from rnsr.harness.loop import EnvSpec
    from rnsr.llm import governor as _governor
    from rnsr.sdk import answer_batch, make_runner

    if arm == "classic":
        env = EnvSpec(mode="classic", context=flat_context(docs))
    else:
        env = EnvSpec(mode="dci", files=docs)
    runner = make_runner(settings)
    answers = await answer_batch(
        [q for _, q in qs], corpus_db, batch_size=batch_size, concurrency=concurrency,
        settings=settings, runner=runner, env=env,
        question_ids=[fid for fid, _ in qs], run_dir=out / "trajectories")
    results = {}
    for (fid, _), row in zip(qs, answers, strict=True):
        ev = row.evidence or {}
        results[fid] = {"answer": row.answer, "status": row.status, "error": row.error or "",
                        "tier": row.tier or "", "resolved_by": ev.get("resolved_by", ""),
                        "quotes_verified": f"{ev.get('quotes_verified', 0)}/{ev.get('quotes_total', 0)}",
                        "negative_audit": ev.get("negative_audit", "")}
    return results, _governor.current().snapshot()


def write_outputs(arm: str, qs, results: dict, provider: dict, out: Path, wall_s: float,
                  extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    answers = {fid: (r.get("answer") if r.get("answer") not in (None, "", "NOT_FOUND") else NOT_FOUND)
               for fid, r in results.items()}
    with open(out / "answers_chunk1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ground_truth_question", "model_answer"])
        for fid, q in qs:
            w.writerow([q, answers.get(fid, NOT_FOUND)])
    with open(out / "answers_status.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "query_id", "status", "error", "model_answer", "tier",
                    "quotes_verified", "resolved_by", "negative_audit", "cost_usd", "elapsed_s"])
        for i, (fid, _) in enumerate(qs):
            r = results.get(fid, {})
            w.writerow([i, fid, r.get("status", "unanswered"), r.get("error", ""),
                        answers.get(fid, NOT_FOUND), r.get("tier", ""), r.get("quotes_verified", ""),
                        r.get("resolved_by", ""), r.get("negative_audit", ""),
                        r.get("cost_usd", ""), r.get("elapsed_s", "")])
    counts: dict[str, int] = {}
    for fid, _ in qs:
        st = results.get(fid, {}).get("status", "unanswered")
        counts[st] = counts.get(st, 0) + 1
    report = {"arm": arm, "questions": len(qs), "status_counts": counts,
              "not_found": sum(a == NOT_FOUND for a in answers.values()),
              "wall_s": round(wall_s, 1), "provider": provider, **extra}
    (out / "run_report.json").write_text(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    from rnsr.config import Settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matter-dir", type=Path, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--corpus-db", type=Path, default=None,
                    help="corpus artifact (default <matter-dir>/work/corpora/corpus.db)")
    ap.add_argument("--questions", type=Path, default=None,
                    help="questions CSV (default <matter-dir>/questions.csv)")
    ap.add_argument("--id-col", default="field_id", help="question id column in the CSV")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    corpus_db = args.corpus_db or args.matter_dir / "work" / "corpora" / "corpus.db"
    docs = load_documents(corpus_db)
    qs = load_questions(args.questions or args.matter_dir / "questions.csv", args.id_col)
    if args.limit:
        qs = qs[:args.limit]
    corpus_chars = sum(len(t) for t in docs.values())
    print(f"{args.arm}: {len(qs)} questions, {len(docs)} documents, {corpus_chars:,} chars "
          f"(~{corpus_chars // 4:,} tokens)")
    started = time.monotonic()
    if args.arm == "stuffing":
        results, provider = asyncio.run(_run_stuffing(qs, docs, settings, args.out, args.concurrency))
        extra = {"calls_per_question": 1, "root_model": settings.root_model}
    else:
        results, provider = asyncio.run(_run_loop(args.arm, qs, docs, settings, args.out,
                                                  args.batch_size, args.concurrency, corpus_db))
        extra = {"batch_size": args.batch_size, "root_model": settings.root_model,
                 "sub_model": settings.sub_model}
    write_outputs(args.arm, qs, results, provider, args.out, time.monotonic() - started,
                  {**extra, "documents": len(docs), "corpus_chars": corpus_chars,
                   "concurrency": args.concurrency})
    counts = {}
    for r in results.values():
        counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
    print(f"status: {counts}  provider: {provider.get('requests')} request(s), "
          f"${provider.get('spend_usd', 0):.4f}  wall {time.monotonic() - started:.0f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
