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
from rnsr.eval.metrics import EvalResult, score_answer, summarize
from rnsr.harness.loop import EnvSpec, RootRunner

SYSTEMS = ("docdb", "rlm-classic")   # vector-rag / base-lc land in Phase D


def _corpus_for(sources: list[Path], cache_dir: Path, settings: Settings) -> Path:
    """Ingest sources into a cached corpus.db (keyed by file content)."""
    from rnsr.ingest.pipeline import ingest

    h = sha256()
    for s in sorted(sources):
        h.update(Path(s).read_bytes())
    out = cache_dir / f"corpus_{h.hexdigest()[:16]}.db"
    if not out.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        ingest(sources, out, config=settings)
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
            # flat-text benchmarks have no documents to ingest; docdb runs
            # classic-shaped for them (structure is an accelerant, §1.3)
            return EnvSpec(mode="classic", context=item.context)
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
) -> tuple[list[EvalResult], dict]:
    """Run a benchmark; writes results.jsonl + summary.json under run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "corpora"
    settings = runner.settings

    results: list[EvalResult] = []
    with open(run_dir / "results.jsonl", "w") as out:
        for item in items[: limit or len(items)]:
            env = _env_for(item, system, cache_dir, settings)
            t0 = time.monotonic()
            qr = await runner.run(item.question, env, run_dir=run_dir / "trajectories",
                                  query_id=item.qid)
            result = EvalResult(
                qid=item.qid,
                task_class=item.task_class,
                predicted=None if qr.answer is None else str(qr.answer),
                gold=item.gold,
                correct=score_answer(qr.answer, item.gold),
                status=qr.status,
                latency_s=round(time.monotonic() - t0, 3),
                cost_usd=qr.ledger["spend_usd"],
                sub_calls=qr.ledger["sub_calls"],
                iterations=qr.iterations,
                trajectory_path=qr.trajectory_path,
            )
            results.append(result)
            out.write(json.dumps(result.to_dict()) + "\n")
            out.flush()

    summary = summarize(results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results, summary
