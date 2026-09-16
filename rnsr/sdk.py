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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rnsr.eval.regression import RegressionReport
    from rnsr.forms.spec import FormSpec, QuestionItem

from rnsr.answer_semantics import AbstainBelow, QueryStatus, TrustTier, publish_answer
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
    "append",
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


def append(sources, corpus_db, **kwargs):
    """Add documents to an existing corpus.db (see rnsr.ingest.lifecycle)."""
    from rnsr.ingest.lifecycle import append as _append

    return _append(sources, corpus_db, **kwargs)


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


def corpus_env(corpus_db: str | Path, *,
               settings: Settings | None = None) -> EnvSpec:
    """Build the docdb EnvSpec for a corpus artifact (manifest included).

    Enforces the corpus health gate: a blocked corpus raises
    ``CorpusHealthError`` unless ``settings.allow_degraded``.
    """
    from rnsr.ingest.health import enforce_health, load_health

    settings = settings or Settings.from_env()
    with CorpusDB(corpus_db) as c:
        manifest = c.manifest_dict()
        health = load_health(c, settings)
    enforce_health(health, settings)
    manifest["health"] = health.to_dict()
    from rnsr.harness.playbook import discover_playbook

    playbook = discover_playbook(Path(corpus_db).parent, Path(corpus_db).with_suffix(""))
    return EnvSpec(mode="docdb", corpus_db=str(corpus_db), manifest=manifest,
                   playbook=playbook)


async def answer(
    question: str,
    corpus_db: str | Path,
    *,
    settings: Settings | None = None,
    runner: RootRunner | None = None,
    run_dir: str | Path | None = None,
    query_id: str | None = None,
    abstain_below: AbstainBelow = "off",
) -> QueryResult:
    """Answer one question against a corpus.db via the RLM loop.

    Returns the full QueryResult: ``.answer`` (None when the loop failed),
    ``.status`` ('final' | 'recovered' | 'budget_exhausted' | 'error'),
    budget ``.ledger``, and ``.trajectory_path`` for the audit record.
    """
    publish_answer(None, None, abstain_below)
    settings = settings or getattr(runner, "settings", None)
    runner = runner or make_runner(settings)
    env = corpus_env(corpus_db, settings=settings or runner.settings)
    result = await runner.run(question, env, run_dir=run_dir, query_id=query_id)
    health = (env.manifest or {}).get("health")
    result.health = health
    result.raw_answer = result.answer
    result.answer = publish_answer(result.answer, result.evidence.tier if result.evidence else None,
                                   abstain_below)
    return result


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
    status: QueryStatus             # QueryResult status, or 'unanswered'
    error: str | None = None
    agreement: float | None = None
    contested: bool = False
    health: dict | None = None          # corpus health snapshot, if gated
    evidence: dict | None = None
    tier: TrustTier | None = None
    raw_answer: str | None = None


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
    abstain_below: AbstainBelow = "off",
    env: EnvSpec | None = None,
    question_ids: Sequence[str] | None = None,
    on_result: Callable[[int, BatchAnswer], None] | None = None,
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
    if min(batch_size, concurrency, consensus) < 1:
        raise ValueError("batch_size, concurrency and consensus must be positive")
    publish_answer(None, None, abstain_below)  # validate before provider work
    qs = list(questions)
    qids = list(question_ids) if question_ids is not None else [f"q{i:03d}" for i in range(len(qs))]
    if len(qids) != len(qs) or len(set(qids)) != len(qids):
        raise ValueError("question_ids must be unique and match questions")
    if not qs:
        return []
    settings = settings or getattr(runner, "settings", None)
    runner = runner or make_runner(settings)
    env = env or corpus_env(corpus_db, settings=settings or runner.settings)
    health = (env.manifest or {}).get("health")
    sem = asyncio.Semaphore(max(1, concurrency))

    out: dict[int, BatchAnswer] = {}

    def notify(i: int) -> None:
        row = out[i]
        row.raw_answer = row.answer
        row.answer = publish_answer(row.answer, row.tier, abstain_below)
        if on_result:
            on_result(i, row)

    async def run_group(group: list[int]) -> None:
        pairs = [(qids[i], qs[i]) for i in group]
        group_id = f"b{qids[group[0]]}_{qids[group[-1]]}"
        async with sem:
            try:
                if consensus > 1:
                    cr = await runner.run_batch_consensus(
                        pairs, env, run_dir=run_dir, query_id=group_id,
                        passes=consensus)
                    group_status = (cr.pass_results[0].status
                                    if cr.pass_results else "error")
                    for i in group:
                        got = cr.answers.get(qids[i])
                        if got is None or got.value is None:
                            continue
                        ev = got.evidence
                        out[i] = BatchAnswer(
                            question=qs[i], answer=got.value,
                            status=group_status if got.value else "unanswered",
                            agreement=got.agreement, contested=got.contested,
                            health=health,
                            evidence=ev.to_dict() if ev else None,
                            tier=ev.tier if ev else None)
                else:
                    br = await runner.run_batch(pairs, env, run_dir=run_dir,
                                                query_id=group_id)
                    for i in group:
                        text = br.answers.get(qids[i])
                        if text is None:
                            continue
                        ev = br.evidence.get(qids[i])
                        out[i] = BatchAnswer(
                            question=qs[i], answer=text,
                            status=br.result.status,
                            health=health,
                            evidence=ev.to_dict() if ev else None,
                            tier=ev.tier if ev else None)
            except Exception as e:  # a failed group must not sink the run
                error = f"{type(e).__name__}: {e}"[:300]
                for i in group:
                    out.setdefault(i, BatchAnswer(
                        question=qs[i], answer=None, status="error",
                        error=error, health=health))
                    out[i].error = out[i].error or error
        for i in group:
            if i in out:
                notify(i)

    async def run_solo(i: int) -> None:
        async with sem:
            try:
                res = await runner.run(qs[i], env, run_dir=run_dir,
                                       query_id=qids[i])
                text = "" if res.answer is None else str(res.answer).strip()
                ev = res.evidence
                out[i] = BatchAnswer(
                    question=qs[i], answer=text or None,
                    status=res.status,
                    health=health,
                    evidence=ev.to_dict() if ev else None,
                    tier=ev.tier if ev else None)
            except Exception as e:
                out[i] = BatchAnswer(question=qs[i], answer=None,
                                     status="error",
                                     error=f"{type(e).__name__}: {e}"[:300],
                                     health=health)
        notify(i)

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
                                      status="unanswered", health=health)
            for i in range(len(qs))]


def answer_sync(question: str, corpus_db: str | Path, **kwargs) -> QueryResult:
    """Synchronous wrapper for answer() — for sync workers (Celery, scripts)."""
    return asyncio.run(answer(question, corpus_db, **kwargs))


def answer_batch_sync(questions: Sequence[str], corpus_db: str | Path,
                      **kwargs) -> list[BatchAnswer]:
    """Synchronous wrapper for answer_batch()."""
    return asyncio.run(answer_batch(questions, corpus_db, **kwargs))


def build_questions(spec: FormSpec | str | Path) -> list[QuestionItem]:
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


def fan_out(items: Sequence[QuestionItem | dict], answers: Sequence[str]) -> tuple[dict[str, str], list[str]]:
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
    golden: dict[str, list[str]] | str | Path,
    field_answers: dict[str, str],
    *,
    min_accuracy: float = 0.0,
    max_false_positive_rate: float = 1.0,
    min_high_tier_accuracy: float = 0.0,
    judge: bool = False,
    settings: Settings | None = None,
    tiers: dict[str, str] | None = None,
) -> RegressionReport:
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
    report = score_run(
        golden, field_answers, min_accuracy=min_accuracy,
        max_false_positive_rate=max_false_positive_rate,
        min_high_tier_accuracy=min_high_tier_accuracy,
        tiers=tiers,
    )
    if judge and any(not r.agrees for r in report.results):
        from rnsr.llm.router import Router

        sub = Router(settings or Settings.from_env()).resolve("sub")
        await judge_disagreements(report, sub.client, sub.model)
    return report


def score_answers_sync(golden: dict[str, list[str]] | str | Path,
                       field_answers: dict[str, str], **kwargs) -> RegressionReport:
    """Synchronous wrapper for score_answers()."""
    return asyncio.run(score_answers(golden, field_answers, **kwargs))
