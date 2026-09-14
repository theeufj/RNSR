"""Programmatic API for embedding RNSR in workers and services.

The CLI is a thin veneer over this module: anything `rnsr query` or
`rnsr answer-csv` can do, a Celery task, FastAPI handler, or Lambda
function can do by calling these functions directly. The functions are
async because a query is provider-I/O bound; `answer_sync` and
`answer_batch_sync` wrap them for synchronous workers.

Typical worker integration::

    import rnsr

    report = rnsr.ingest(["report.pdf", "ledger.xlsx"], "corpus.db")
    result = rnsr.answer_sync("What was FY2023 segment revenue?", "corpus.db")
    print(result.answer, result.ledger["spend_usd"])

Provider keys and model roles resolve exactly as for the CLI: from
``Settings`` (pass one explicitly, or `Settings.from_env()` is used).
A prebuilt ``RootRunner`` can be reused across calls to skip repeated
provider resolution — pass it as ``runner=``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.harness.loop import EnvSpec, QueryResult, RootRunner

__all__ = [
    "BatchAnswer",
    "answer",
    "answer_batch",
    "answer_batch_sync",
    "answer_sync",
    "build_questions",
    "corpus_env",
    "fan_out",
    "ingest",
    "make_runner",
    "open_corpus",
    "score_answers",
    "score_answers_sync",
]


def ingest(sources, out_db, **kwargs):
    """Ingest documents into a corpus.db artifact (see rnsr.ingest.pipeline).

    Lives here because the ``rnsr.ingest`` *subpackage* shadows any
    same-named function at the package root: importing any
    ``rnsr.ingest.*`` module rebinds the attribute to the module, so a
    root-level ``rnsr.ingest()`` callable cannot exist reliably.
    """
    from rnsr.ingest.pipeline import ingest as _ingest

    return _ingest(sources, out_db, **kwargs)


def open_corpus(path: str | Path, mode: str = "ro") -> CorpusDB:
    """Open an existing corpus.db artifact (``mode``: 'ro' or 'rw')."""
    return CorpusDB(path, mode=mode)


def make_runner(settings: Settings | None = None) -> RootRunner:
    """Resolve provider clients for every role into a ready RootRunner.

    Reusable across queries and threads; building one per call works but
    re-reads provider configuration each time. Raises RuntimeError when no
    provider key is available for the root/sub roles. The embedding role is
    optional — without it, search-ladder rung 4 stays dormant.
    """
    from rnsr.llm.router import Router

    settings = settings or Settings.from_env()
    router = Router(settings)
    root, sub = router.resolve("root"), router.resolve("sub")
    embed_client, embed_model = None, ""
    try:
        embed = router.resolve("embed")
        embed_client, embed_model = embed.client, embed.model
    except RuntimeError:
        pass  # rung 4 stays dormant without an embedding provider
    return RootRunner(root_client=root.client, root_model=root.model,
                      sub_client=sub.client, sub_model=sub.model,
                      embed_client=embed_client, embed_model=embed_model,
                      settings=settings)


def corpus_env(corpus_db: str | Path) -> EnvSpec:
    """Build the docdb EnvSpec for a corpus artifact (manifest included)."""
    with CorpusDB(corpus_db) as c:
        manifest = c.manifest_dict()
    return EnvSpec(mode="docdb", corpus_db=str(corpus_db), manifest=manifest)


async def answer(
    question: str,
    corpus_db: str | Path,
    *,
    settings: Settings | None = None,
    runner: RootRunner | None = None,
    run_dir: str | Path | None = None,
    query_id: str | None = None,
) -> QueryResult:
    """Answer one question against a corpus.db via the RLM loop.

    Returns the full QueryResult: ``.answer`` (None when the loop failed),
    ``.status`` ('final' | 'recovered' | 'budget_exhausted' | 'error'),
    budget ``.ledger``, and ``.trajectory_path`` for the audit record.
    """
    runner = runner or make_runner(settings)
    env = corpus_env(corpus_db)
    return await runner.run(question, env, run_dir=run_dir, query_id=query_id)


@dataclass
class BatchAnswer:
    """One question's outcome from answer_batch, in input order.

    ``answer`` is the model's text — "NOT_FOUND" when it judged the corpus
    lacks the answer — or None when no loop produced an answer (see
    ``status``/``error`` for why). ``agreement`` and ``contested`` are set
    only in consensus mode.
    """

    question: str
    answer: str | None
    status: str                     # QueryResult status, or 'unanswered'
    error: str | None = None
    agreement: float | None = None
    contested: bool = False


async def answer_batch(
    questions: Sequence[str],
    corpus_db: str | Path,
    *,
    batch_size: int = 8,
    concurrency: int = 4,
    consensus: int = 1,
    retry_solo: bool = True,
    settings: Settings | None = None,
    runner: RootRunner | None = None,
    run_dir: str | Path | None = None,
) -> list[BatchAnswer]:
    """Answer many questions over one corpus, sharing exploration.

    Consecutive questions are grouped into shared RLM loops of up to
    ``batch_size`` (1 = one loop per question); up to ``concurrency``
    loops run at once. With ``consensus`` > 1, each group is answered
    that many times independently and voted per question. Questions a
    group failed to answer are retried in their own loop when
    ``retry_solo`` is set.

    This is the library core of ``rnsr answer-csv`` minus the platform
    parts (CSV contract, checkpoint files, work-dir locking) — callers
    embedding this in a worker framework bring their own persistence.
    """
    qs = list(questions)
    qids = [f"q{i:03d}" for i in range(len(qs))]
    runner = runner or make_runner(settings)
    env = corpus_env(corpus_db)
    sem = asyncio.Semaphore(max(1, concurrency))

    out: dict[int, BatchAnswer] = {}

    async def run_group(group: list[int]) -> None:
        pairs = [(qids[i], qs[i]) for i in group]
        group_id = f"b{group[0]:03d}_{group[-1]:03d}"
        async with sem:
            try:
                if consensus > 1:
                    cr = await runner.run_batch_consensus(
                        pairs, env, run_dir=run_dir, query_id=group_id,
                        passes=consensus)
                    status = (cr.pass_results[0].status
                              if cr.pass_results else "error")
                    for i in group:
                        got = cr.answers.get(qids[i])
                        if got is None or got.value is None:
                            continue
                        out[i] = BatchAnswer(
                            question=qs[i], answer=got.value, status=status,
                            agreement=got.agreement, contested=got.contested)
                else:
                    br = await runner.run_batch(pairs, env, run_dir=run_dir,
                                                query_id=group_id)
                    for i in group:
                        text = br.answers.get(qids[i])
                        if text is None:
                            continue
                        out[i] = BatchAnswer(question=qs[i], answer=text,
                                             status=br.result.status)
            except Exception as e:  # a failed group must not sink the run
                error = f"{type(e).__name__}: {e}"[:300]
                for i in group:
                    out.setdefault(i, BatchAnswer(
                        question=qs[i], answer=None, status="error",
                        error=error))
                    out[i].error = out[i].error or error

    async def run_solo(i: int) -> None:
        async with sem:
            try:
                res = await runner.run(qs[i], env, run_dir=run_dir,
                                       query_id=qids[i])
                text = "" if res.answer is None else str(res.answer).strip()
                out[i] = BatchAnswer(question=qs[i], answer=text or None,
                                     status=res.status if text else "unanswered")
            except Exception as e:
                out[i] = BatchAnswer(question=qs[i], answer=None,
                                     status="error",
                                     error=f"{type(e).__name__}: {e}"[:300])

    if batch_size > 1:
        indices = list(range(len(qs)))
        groups = [indices[j:j + batch_size]
                  for j in range(0, len(indices), batch_size)]
        await asyncio.gather(*(run_group(g) for g in groups))
        pending = [i for i in range(len(qs))
                   if i not in out or out[i].answer is None]
        if pending and retry_solo:
            # keep the group error for context if the solo pass also fails
            prior = {i: out[i].error for i in pending if i in out}
            await asyncio.gather(*(run_solo(i) for i in pending))
            for i, err in prior.items():
                out[i].error = out[i].error or err
    else:
        await asyncio.gather(*(run_solo(i) for i in range(len(qs))))

    return [out.get(i) or BatchAnswer(question=qs[i], answer=None,
                                      status="unanswered")
            for i in range(len(qs))]


def answer_sync(question: str, corpus_db: str | Path, **kwargs) -> QueryResult:
    """Synchronous wrapper for answer() — for sync workers (Celery, scripts)."""
    return asyncio.run(answer(question, corpus_db, **kwargs))


def answer_batch_sync(questions: Sequence[str], corpus_db: str | Path,
                      **kwargs) -> list[BatchAnswer]:
    """Synchronous wrapper for answer_batch()."""
    return asyncio.run(answer_batch(questions, corpus_db, **kwargs))


def build_questions(spec):
    """Enriched questions from a form spec (the `rnsr build-questions` core).

    ``spec`` is a FormSpec or a path to a spec JSON. Returns QuestionItems:
    mutually exclusive fields collapsed into one question each, role maps
    and conventions attached. Feed ``[i.question for i in items]`` to
    answer_batch, then fan_out(items, answers) to map group answers back to
    individual fields.
    """
    from rnsr.forms import build_questions as _build
    from rnsr.forms.spec import FormSpec, load_spec

    if not isinstance(spec, FormSpec):
        spec = load_spec(spec)
    return _build(spec)


def fan_out(items, answers):
    """Fan group answers back out to individual field ids.

    ``items`` are QuestionItems (or their dict form from a map file);
    ``answers`` are the model's answer texts in item order. Returns
    (field_id -> value, parse notes). Re-exported from rnsr.forms.
    """
    from dataclasses import asdict, is_dataclass

    from rnsr.forms import fan_out as _fan_out

    items = [asdict(i) if is_dataclass(i) else i for i in items]
    return _fan_out(items, list(answers))


async def score_answers(
    golden,
    field_answers: dict[str, str],
    *,
    min_accuracy: float = 0.0,
    judge: bool = False,
    settings: Settings | None = None,
):
    """Score field answers against a golden set (the `rnsr regress` core).

    ``golden`` is a dict of field_id -> golden values or a path to a
    vendor-shaped golden JSON. String scoring is free and deterministic;
    ``judge=True`` additionally runs the sub-LM equivalence check over
    string failures only (agreement can only go up). Returns a
    RegressionReport — check ``.passed`` against ``min_accuracy`` and
    ``.write(dir)`` for the artifact files.
    """
    from rnsr.eval.regression import judge_disagreements, load_golden, score_run

    if not isinstance(golden, dict):
        golden = load_golden(golden)
    report = score_run(golden, field_answers, min_accuracy=min_accuracy)
    if judge and any(not r.agrees for r in report.results):
        from rnsr.llm.router import Router

        sub = Router(settings or Settings.from_env()).resolve("sub")
        await judge_disagreements(report, sub.client, sub.model)
    return report


def score_answers_sync(golden, field_answers: dict[str, str], **kwargs):
    """Synchronous wrapper for score_answers()."""
    return asyncio.run(score_answers(golden, field_answers, **kwargs))
