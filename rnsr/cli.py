"""rnsr command-line interface.

Commands land with their phases:
    rnsr ingest   — Phase A: documents -> corpus.db + validation report
    rnsr query    — Phase B/C: run the RLM loop against a corpus
    rnsr eval     — §8 evaluation harness (benchmarks, baselines, gate)
    rnsr ablate   — Phase D: rung-4 quantization ablation
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="rnsr", no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def _main(
    log_level: str | None = typer.Option(
        None, "--log-level", help="DEBUG | INFO | WARNING | ERROR "
                                  "(default RNSR_LOG_LEVEL, else INFO)"),
    log_format: str | None = typer.Option(
        None, "--log-format", help="text for terminals, json for log shippers "
                                   "(default RNSR_LOG_FORMAT, else text)"),
) -> None:
    """Structured logs go to stderr, so stdout stays pipeable."""
    from rnsr.config import Settings
    from rnsr.obs import configure_logging

    configure_logging(Settings.from_env(), level=log_level, fmt=log_format,
                      force=True)


@app.command()
def ingest(
    sources: list[Path] = typer.Argument(..., exists=True, readable=True,
                                         help="Documents to ingest (PDF, Word, Excel, "
                                              "PowerPoint, OpenDocument, RTF, EPUB, "
                                              "CSV, Markdown, text, email)"),
    out: Path = typer.Option(Path("corpus.db"), "--out", "-o", help="Output artifact path"),
    report_path: Path | None = typer.Option(None, "--report", help="Write JSON report here"),
    llm: bool = typer.Option(False, "--llm/--no-llm",
                             help="Enable the sub-LM prose cross-check and vision "
                                  "re-extraction rung (§3.3); default is fully LLM-free"),
) -> None:
    """Ingest documents into a corpus.db artifact (Phase A)."""
    from rnsr.config import Settings
    from rnsr.ingest.pipeline import ingest as run_ingest

    settings = Settings.from_env()
    prose_checker = vision = transcriber = None
    if llm:
        from rnsr.ingest.llm_hooks import (
            make_page_transcriber,
            make_prose_checker,
            make_vision_extractor,
        )
        from rnsr.llm.router import Router

        router = Router(settings)
        sub, vis = router.resolve("sub"), router.resolve("vision")
        prose_checker = make_prose_checker(sub.client, sub.model,
                                           concurrency=settings.sub_concurrency)
        vision = make_vision_extractor(vis.client, vis.model)
        transcriber = make_page_transcriber(vis.client, vis.model)

    report = run_ingest(sources, out, config=settings,
                        prose_checker=prose_checker, vision=vision,
                        transcriber=transcriber)

    t = Table(title=f"Ingested -> {out}")
    for col in ("table", "status", "confidence", "extractor", "rows"):
        t.add_column(col)
    for tr in report.tables:
        t.add_row(tr.name, tr.status, f"{tr.confidence:.2f}", tr.extractor, str(tr.n_rows))
    console.print(t)
    console.print(
        f"{len(report.documents)} document(s), {report.n_chunks} chunks, "
        f"validation pass rate {report.validation_pass_rate:.0%}"
    )
    if report.scanned_pages_transcribed:
        console.print(f"{report.scanned_pages_transcribed} scanned page(s) transcribed via VLM")
    if report.scanned_pages_untranscribed:
        console.print(f"[red]untranscribed scanned pages:[/red] {report.scanned_pages_untranscribed}")
    if report.skipped_stages:
        console.print(f"[yellow]skipped:[/yellow] {', '.join(report.skipped_stages)}")
    if report_path:
        report_path.write_text(report.to_json())
        console.print(f"report written to {report_path}")


def _make_runner(settings):
    from rnsr.harness.loop import RootRunner
    from rnsr.llm.router import Router

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


@app.command()
def query(
    corpus: Path = typer.Argument(..., exists=True, help="corpus.db artifact"),
    question: str = typer.Argument(...),
    run_dir: Path = typer.Option(Path("runs/query"), "--run-dir"),
) -> None:
    """Answer a question against a corpus.db via the RLM loop (Phase B/C)."""
    import asyncio

    from rnsr.config import Settings
    from rnsr.db.artifact import CorpusDB
    from rnsr.harness.loop import EnvSpec

    settings = Settings.from_env()
    with CorpusDB(corpus) as c:
        manifest = c.manifest_dict()
    env = EnvSpec(mode="docdb", corpus_db=str(corpus), manifest=manifest)
    result = asyncio.run(_make_runner(settings).run(question, env, run_dir=run_dir))
    console.print(f"[bold]{result.answer}[/bold]")
    console.print(f"status={result.status} iterations={result.iterations} "
                  f"cost=${result.ledger['spend_usd']:.4f} "
                  f"sub_calls={result.ledger['sub_calls']}")
    console.print(f"trajectory: {result.trajectory_path}")


@app.command("trajectory")
def trajectory_cmd(
    path: Path = typer.Argument(..., exists=True, help="trajectory .jsonl(.enc)"),
    kinds: str | None = typer.Option(None, "--kinds",
                                     help="comma-separated event kinds to show"),
) -> None:
    """Print a trajectory, decrypting it when RNSR_TRAJECTORY_KEY is set.

    Encrypted or redacted trajectories are otherwise unreadable, which would
    make the data-protection settings unusable in practice.
    """
    import json

    from rnsr.config import Settings
    from rnsr.harness.trajectory import read_trajectory

    settings = Settings.from_env()
    wanted = {k.strip() for k in kinds.split(",")} if kinds else None
    for record in read_trajectory(path, settings.trajectory_key):
        if wanted is None or record.get("kind") in wanted:
            console.print_json(json.dumps(record))


@app.command()
def gate(
    run_dir: Path = typer.Option(Path("runs/gate"), "--run-dir"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    needle_docs: int = typer.Option(3, "--needle-docs"),
) -> None:
    """§8 go/no-go: docdb vs rlm-classic on the numeric-needle set."""
    import asyncio
    import json

    from rnsr.config import Settings
    from rnsr.eval.datasets.needle_gen import generate_needle_set
    from rnsr.eval.harness import run_eval
    from rnsr.eval.metrics import gate_report

    settings = Settings.from_env()
    items = generate_needle_set(run_dir / "needle_pdfs", n_docs=needle_docs)
    summaries = {}
    for system in ("docdb", "rlm-classic"):
        runner = _make_runner(settings)
        _, summaries[system] = asyncio.run(
            run_eval(items, system, runner, run_dir=run_dir / system, limit=limit)
        )
    report = gate_report(summaries["docdb"], summaries["rlm-classic"])
    (run_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    console.print_json(json.dumps(report["checks"]))
    console.print("[green]GATE PASS[/green]" if report["pass"]
                  else "[red]GATE FAIL[/red]")
    raise typer.Exit(0 if report["pass"] else 1)


@app.command("eval")
def eval_cmd(
    benchmark: str = typer.Option(..., "--benchmark", "-b",
                                  help="synthetic-oolong | oolong | financebench"),
    system: str = typer.Option("docdb", "--system", "-s",
                               help="docdb | rlm-classic"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    run_dir: Path = typer.Option(Path("runs/eval"), "--run-dir"),
    dataset_id: str | None = typer.Option(None, "--dataset-id",
                                          help="HF dataset id override"),
    seed: int = typer.Option(5, "--seed", help="generator seed (matter benchmark)"),
    concurrency: int = typer.Option(1, "--concurrency", "-c",
                                    help="items answered in parallel "
                                         "(wall-clock win, same LLM spend)"),
) -> None:
    """Run the evaluation harness (§8)."""
    import asyncio
    import json

    from rnsr.config import Settings
    from rnsr.eval.harness import run_eval

    if benchmark == "synthetic-oolong":
        from rnsr.eval.datasets.oolong import synthetic_oolong

        items = synthetic_oolong()
    elif benchmark == "oolong":
        from rnsr.eval.datasets.oolong import DEFAULT_DATASET_ID, load_oolong

        items = load_oolong(dataset_id or DEFAULT_DATASET_ID, limit=limit)
    elif benchmark == "financebench":
        from rnsr.eval.datasets.financebench import load_financebench

        items = load_financebench(limit=limit)
    elif benchmark == "cuad":
        from rnsr.eval.datasets.legal import load_cuad

        items = load_cuad(limit=limit)
    elif benchmark == "cuad-long":
        from rnsr.eval.datasets.legal import load_cuad

        # long-contract regime: where context-stuffing strains (~50k-190k tokens)
        items = load_cuad(limit=limit, min_context_chars=200_000,
                          max_context_chars=750_000)
    elif benchmark == "contractnli":
        from rnsr.eval.datasets.legal import load_contractnli

        items = load_contractnli(limit=limit)
    elif benchmark == "legalbench":
        from rnsr.eval.datasets.legal import load_legalbench

        items = load_legalbench(limit=limit)
    elif benchmark == "matter":
        from rnsr.eval.datasets.matter_gen import generate_matter

        items = generate_matter(run_dir / "matter_pdfs", seed=seed)
    else:
        raise typer.BadParameter(f"unknown benchmark: {benchmark}")

    settings = Settings.from_env()
    runner = _make_runner(settings)
    out_dir = run_dir / f"{benchmark}-{system}"
    _, summary = asyncio.run(
        run_eval(items, system, runner, run_dir=out_dir, limit=limit,
                 concurrency=concurrency)
    )
    console.print_json(json.dumps(summary))
    console.print(f"results in {out_dir}")


@app.command("answer-csv")
def answer_csv(
    corpus_dir: Path = typer.Option(..., "--corpus", exists=True, file_okay=False,
                                    help="matter corpus directory (read-only)"),
    questions: Path = typer.Option(..., "--questions", exists=True,
                                   help="questions CSV"),
    output: Path = typer.Option(..., "--output", help="output dir for answers_chunk1.csv"),
    question_col: str = typer.Option("ground_truth_question", "--question-col"),
    work_dir: Path = typer.Option(Path("runs/answer-csv"), "--work-dir",
                                  help="corpus.db cache + trajectories (outside corpus)"),
    concurrency: int = typer.Option(4, "--concurrency",
                                    help="concurrent RLM loops (batches count "
                                         "as one loop each)"),
    batch_size: int = typer.Option(8, "--batch-size",
                                   help="questions answered per RLM loop; "
                                        "consecutive questions share one "
                                        "exploration of the corpus. 1 = one "
                                        "loop per question (slower, original "
                                        "behavior)"),
    consensus: int = typer.Option(
        1, "--consensus",
        help="independent passes per batch, voted per field (1 = off). 2 costs "
             "roughly double for the same wall time and turns disagreements "
             "into flagged fields instead of silent errors"),
    not_found: str = typer.Option("Not found in matter corpus", "--not-found-phrase"),
    max_error_rate: float = typer.Option(
        0.0, "--max-error-rate",
        help="fail the run (exit 2) when more than this fraction of questions "
             "ended in a harness/provider error rather than an answer. 0 means "
             "any error fails the run; 1 disables the gate"),
    llm: bool = typer.Option(False, "--llm/--no-llm",
                             help="LLM-assisted ingest: VLM transcription of scanned "
                                  "pages, vision table re-extraction, prose checks"),
    fast_ingest: bool = typer.Option(False, "--fast-ingest",
                                     help="corpus-scale text-tier ingest (pdfium, no "
                                          "layout ML): use for thousands of files; "
                                          "resumable per document"),
) -> None:
    """Answer a questions CSV over a document corpus (fable-replicate contract).

    Emits output/answers_chunk1.csv with header '<question-col>,model_answer',
    questions verbatim in input order, no empty answers. The corpus is never
    written to; all state lives under --work-dir. Answers checkpoint
    incrementally — rerunning resumes instead of re-paying.

    With --batch-size > 1 (the default), consecutive questions are answered
    in shared RLM loops via FINAL_BATCH; any question a batch fails to
    answer is retried in its own loop before the CSV is written.

    Failures are never disguised as answers: alongside the CSV, the output
    directory gets answers_status.csv (per-question status and error) and
    run_report.json (counts, spend, wall time). A provider outage exits 2
    instead of returning a form full of the not-found phrase.
    """
    import asyncio
    import csv as _csv
    import json as _json
    import time as _time

    from rnsr import obs as _obs
    from rnsr.config import Settings
    from rnsr.db.artifact import CorpusDB
    from rnsr.eval.harness import _corpus_valid
    from rnsr.harness.loop import EnvSpec
    from rnsr.harness.trajectory import prune_trajectories
    from rnsr.llm import governor as _governor
    from rnsr.runlock import WorkDirBusy, WorkDirLock

    settings = Settings.from_env()
    # One writer per work dir: the checkpoint and the corpus artifact are
    # both single-writer, and a second run would interleave with this one.
    try:
        lock = WorkDirLock(work_dir, label=f"answer-csv {questions.name}").acquire()
    except WorkDirBusy as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    pruned = prune_trajectories(work_dir / "trajectories",
                                settings.trajectory_retention_days)
    if pruned:
        console.print(f"retention: pruned {pruned} trajectory file(s)")

    with open(questions, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows or question_col not in rows[0]:
        raise typer.BadParameter(
            f"questions CSV has no column {question_col!r}; "
            f"columns: {list(rows[0].keys()) if rows else 'none'}")
    qs = [r[question_col] for r in rows]
    console.print(f"{len(qs)} questions over corpus {corpus_dir}")

    # ingest once, cached by content (extension-dispatched parsers)
    from rnsr.ingest.dispatch import is_ingestable

    all_files = [p for p in corpus_dir.rglob("*")
                 if p.is_file() and not p.name.startswith(".")]
    files = sorted(p for p in all_files if is_ingestable(p))
    if not files:
        raise typer.BadParameter(f"no ingestable files under {corpus_dir}")
    console.print(f"ingesting {len(files)} documents (cached across runs)")
    n_unsupported = len(all_files) - len(files)
    if n_unsupported:
        exts = sorted({p.suffix.lower() or "(none)" for p in all_files
                       if not is_ingestable(p)})
        console.print(f"[yellow]note:[/yellow] {n_unsupported} file(s) with "
                      f"unsupported extensions skipped: {', '.join(exts)}")

    from hashlib import sha256 as _sha256

    h = _sha256()
    for s in files:  # stat-identity: no byte reads over the corpus
        st = s.stat()
        h.update(f"{s}|{st.st_size}|{st.st_mtime_ns}".encode())
    cache_dir = work_dir / "corpora"
    corpus_path = cache_dir / f"corpus_{h.hexdigest()[:16]}.db"
    if corpus_path.exists() and not _corpus_valid(corpus_path, len(files)):
        corpus_path.unlink()
    if not corpus_path.exists() and fast_ingest:
        cache_dir.mkdir(parents=True, exist_ok=True)
        transcriber = None
        if llm:
            from rnsr.ingest.llm_hooks import make_page_transcriber
            from rnsr.llm.router import Router

            vis = Router(settings).resolve("vision")
            transcriber = make_page_transcriber(vis.client, vis.model)
        from rnsr.ingest.bulk import ingest_bulk

        stats = ingest_bulk(files, corpus_path, config=settings,
                            transcriber=transcriber,
                            progress=lambda s: console.print(f"[dim]{s}[/dim]"))
        console.print(f"bulk ingest: {stats}")
        if stats.get("scanned_pages_untranscribed"):
            console.print(
                f"[red]WARNING:[/red] {stats['scanned_pages_untranscribed']} scanned "
                "pages have no text — rerun with --llm to transcribe, or answers "
                "may miss their content")
    if not corpus_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        prose_checker = vision = transcriber = None
        if llm:
            from rnsr.ingest.llm_hooks import (
                make_page_transcriber,
                make_prose_checker,
                make_vision_extractor,
            )
            from rnsr.llm.router import Router

            router = Router(settings)
            sub, vis = router.resolve("sub"), router.resolve("vision")
            prose_checker = make_prose_checker(sub.client, sub.model,
                                               concurrency=settings.sub_concurrency)
            vision = make_vision_extractor(vis.client, vis.model)
            transcriber = make_page_transcriber(vis.client, vis.model)
        from rnsr.ingest.pipeline import ingest as _ingest

        report = _ingest(files, corpus_path, config=settings,
                         prose_checker=prose_checker, vision=vision,
                         transcriber=transcriber)
        console.print(f"ingested: {report.n_chunks} chunks, validation pass rate "
                      f"{report.validation_pass_rate:.0%}")
        if report.scanned_pages_untranscribed:
            console.print(
                f"[red]WARNING:[/red] scanned pages without text: "
                f"{sum(len(x['pages']) for x in report.scanned_pages_untranscribed)} "
                "pages across "
                f"{len(report.scanned_pages_untranscribed)} docs — rerun with "
                "--llm to transcribe them, or answers may miss their content")
    with CorpusDB(corpus_path) as c:
        manifest = c.manifest_dict()
    env = EnvSpec(mode="docdb", corpus_db=str(corpus_path), manifest=manifest)

    runner = _make_runner(settings)
    sem = asyncio.Semaphore(concurrency)

    # incremental checkpoint: rerun resumes, never re-pays
    ckpt = work_dir / "answers_partial.jsonl"
    done: dict[int, str] = {}
    status: dict[int, str] = {}
    errors: dict[int, str] = {}
    agreements: dict[int, float] = {}     # consensus mode: share of passes agreeing
    contested: set[str] = set()           # query ids the passes disagreed on
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            rec = _json.loads(line)
            if rec.get("q") == qs[rec["i"]] if rec["i"] < len(qs) else False:
                done[rec["i"]] = rec["a"]
                # checkpoints written before status tracking hold answers
                # that were accepted at the time
                status[rec["i"]] = rec.get("status", "final")
                if rec.get("error"):
                    errors[rec["i"]] = rec["error"]
        if done:
            console.print(f"resuming: {len(done)}/{len(qs)} already answered")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt_f = open(ckpt, "a", encoding="utf-8")  # noqa: SIM115 — spans the run

    def record(i: int, answer_text: str, st: str, error: str | None = None) -> None:
        done[i] = answer_text
        status[i] = st
        if error:
            errors[i] = error
        rec = {"i": i, "q": qs[i], "a": answer_text, "status": st}
        if error:
            rec["error"] = error
        ckpt_f.write(_json.dumps(rec) + "\n")
        ckpt_f.flush()

    async def answer(i: int, q: str) -> tuple[int, str]:
        if i in done:
            return i, done[i]
        async with sem:
            try:
                res = await runner.run(q, env, run_dir=work_dir / "trajectories",
                                       query_id=f"q{i:03d}")
                text = "" if res.answer is None else str(res.answer).strip()
                record(i, text or not_found, res.status)
            except Exception as e:
                # the placeholder keeps the CSV contract; the status file and
                # the exit code carry the truth
                record(i, not_found, "error", f"{type(e).__name__}: {e}"[:300])
            return i, done[i]

    async def answer_group(group: list[int]) -> None:
        qid = {i: f"q{i:03d}" for i in group}
        group_status, group_error = "final", None
        agreement: dict[str, float] = {}
        async with sem:
            try:
                if consensus > 1:
                    cr = await runner.run_batch_consensus(
                        [(qid[i], qs[i]) for i in group], env,
                        run_dir=work_dir / "trajectories",
                        query_id=f"b{group[0]:03d}_{group[-1]:03d}",
                        passes=consensus)
                    got = {q: a.value for q, a in cr.answers.items()}
                    agreement = {q: a.agreement for q, a in cr.answers.items()}
                    contested.update(cr.contested_qids)
                    group_status = (cr.pass_results[0].status
                                    if cr.pass_results else "error")
                else:
                    br = await runner.run_batch(
                        [(qid[i], qs[i]) for i in group], env,
                        run_dir=work_dir / "trajectories",
                        query_id=f"b{group[0]:03d}_{group[-1]:03d}")
                    got = br.answers
                    group_status = br.result.status
            except Exception as e:
                got, group_status = {}, "error"
                group_error = f"{type(e).__name__}: {e}"[:300]
        for i in group:
            text = got.get(qid[i])
            if text is None:
                if group_error:      # remember why, in case the solo retry also fails
                    errors.setdefault(i, group_error)
                continue             # unanswered — the solo pass below retries it
            a = not_found if text.upper() == "NOT_FOUND" else text
            if qid[i] in agreement:
                agreements[i] = agreement[qid[i]]
            record(i, a, group_status)

    async def main() -> list[str]:
        if batch_size > 1:
            pending = [i for i in range(len(qs)) if i not in done]
            groups = [pending[j:j + batch_size]
                      for j in range(0, len(pending), batch_size)]
            if groups:
                console.print(f"batched mode: {len(pending)} question(s) in "
                              f"{len(groups)} shared loop(s) of up to "
                              f"{batch_size}")
                await asyncio.gather(*(answer_group(g) for g in groups))
                missing = [i for i in range(len(qs)) if i not in done]
                if missing:
                    console.print(f"retrying {len(missing)} unanswered "
                                  "question(s) in solo loops")
        results = await asyncio.gather(*(answer(i, q) for i, q in enumerate(qs)))
        return [a for _, a in sorted(results)]

    t_start = _time.monotonic()
    try:
        answers = asyncio.run(main())
    finally:
        ckpt_f.close()
        lock.release()

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "answers_chunk1.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow([question_col, "model_answer"])
        for q, a in zip(qs, answers, strict=True):
            w.writerow([q, a])

    status_path = output / "answers_status.csv"
    with open(status_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["row", "query_id", "status", "agreement", "contested",
                    "error", "model_answer"])
        for i in range(len(qs)):
            qid = f"q{i:03d}"
            w.writerow([i, qid, status.get(i, "error"),
                        "" if i not in agreements else f"{agreements[i]:.2f}",
                        "yes" if qid in contested else "",
                        errors.get(i, ""), answers[i]])

    counts: dict[str, int] = {}
    for i in range(len(qs)):
        st = status.get(i, "error")
        counts[st] = counts.get(st, 0) + 1
    n_error = counts.get("error", 0)
    n_nf = sum(a.startswith(not_found) for a in answers)
    error_rate = n_error / len(qs) if qs else 0.0
    gov = _governor.current()
    report = {
        "questions": len(qs),
        "answers_written": len(answers),
        "not_found": n_nf,
        "status_counts": counts,
        "error_rate": round(error_rate, 4),
        "max_error_rate": max_error_rate,
        "wall_s": round(_time.monotonic() - t_start, 1),
        "corpus_db": str(corpus_path),
        "batch_size": batch_size,
        "concurrency": concurrency,
        "consensus_passes": consensus,
        "contested_fields": sorted(contested),
        "provider": gov.snapshot(),
        "metrics": _obs.metrics().snapshot(),
    }
    (output / "run_report.json").write_text(_json.dumps(report, indent=2))

    console.print(f"wrote {out_path} ({len(answers)} rows, {n_nf} not-found)")
    console.print(f"status: {counts} — details in {status_path}")
    if consensus > 1:
        console.print(
            f"consensus: {consensus} passes, {len(contested)} field(s) "
            "contested and settled by a tie-break loop"
            + (f" ({', '.join(sorted(contested)[:8])})" if contested else ""))
    console.print(f"provider: {gov.requests} request(s), "
                  f"${gov.spent_usd:.4f}, {gov.rate_limit_hits} rate-limit hit(s)")
    if n_error:
        for i in sorted(errors)[:5]:
            if status.get(i) == "error":
                console.print(f"[red]row {i} failed:[/red] {errors[i]}")
    if gov.spend_ceiling_usd and gov.spent_usd >= gov.spend_ceiling_usd:
        console.print(
            f"[red]SPEND CEILING REACHED:[/red] ${gov.spent_usd:.2f} of "
            f"${gov.spend_ceiling_usd:.2f}. Calls were refused from that point "
            "on, so later answers are placeholders. Raise "
            "RNSR_RUN_SPEND_CEILING_USD and rerun to resume.")
        raise typer.Exit(2)
    if error_rate > max_error_rate:
        console.print(
            f"[red]FAILED:[/red] {n_error}/{len(qs)} question(s) ended in an "
            f"error ({error_rate:.1%} > --max-error-rate {max_error_rate:.1%}). "
            "The CSV was written so no work is lost, but these answers are "
            "placeholders, not findings — rerun to resume once the cause is "
            "fixed.")
        raise typer.Exit(2)


@app.command("build-questions")
def build_questions_cmd(
    spec_path: Path = typer.Option(..., "--spec", exists=True,
                                   help="form spec JSON (vendor field export "
                                        "plus optional conventions)"),
    out_csv: Path = typer.Option(..., "--out", help="questions CSV for answer-csv"),
    out_map: Path | None = typer.Option(None, "--map",
                                        help="item map JSON for fan-out "
                                             "(default: alongside --out)"),
) -> None:
    """Turn a form spec into enriched questions (roles, groups, conventions).

    Mutually exclusive fields collapse into one question each, so the form's
    alternatives cannot be answered 'yes' several times over; the map file
    records how to fan each answer back out to individual fields.
    """
    import csv as _csv
    import json as _json
    from dataclasses import asdict

    from rnsr.forms import build_questions
    from rnsr.forms.spec import load_spec

    spec = load_spec(spec_path)
    items = build_questions(spec)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["ground_truth_question", "item_id"])
        for item in items:
            w.writerow([item.question, item.item_id])
    map_path = out_map or out_csv.with_suffix(".map.json")
    map_path.write_text(_json.dumps(
        {"form": spec.form, "roles": spec.roles,
         "items": [asdict(i) for i in items]}, indent=1, ensure_ascii=False))

    n_groups = sum(1 for i in items if i.kind == "group")
    n_grouped_fields = sum(len(i.members) for i in items if i.kind == "group")
    console.print(f"wrote {out_csv}: {len(items)} questions "
                  f"({n_groups} groups covering {n_grouped_fields} fields + "
                  f"{len(items) - n_groups} standalone)")
    console.print(f"wrote {map_path}")


@app.command("regress")
def regress_cmd(
    answers: Path = typer.Option(..., "--answers", exists=True,
                                 help="answers CSV from answer-csv"),
    golden: Path = typer.Option(..., "--golden", exists=True,
                                help="golden JSON with per-field values"),
    item_map: Path | None = typer.Option(None, "--map",
                                         help="item map from build-questions; "
                                              "fans group answers out to fields"),
    out_dir: Path | None = typer.Option(None, "--out"),
    min_accuracy: float = typer.Option(0.0, "--min-accuracy",
                                       help="exit 2 below this field accuracy"),
    judge: bool = typer.Option(True, "--judge/--no-judge",
                               help="sub-LM equivalence check for string "
                                    "failures (long answers differ in wording, "
                                    "not meaning)"),
) -> None:
    """Score answers against a golden set and gate on accuracy."""
    import asyncio
    import csv as _csv
    import json as _json

    from rnsr.config import Settings
    from rnsr.eval.regression import (
        judge_disagreements,
        load_field_answers,
        load_golden,
        score_run,
    )
    from rnsr.forms.fanout import fan_out

    gold = load_golden(golden)
    if item_map:
        items = _json.loads(item_map.read_text())["items"]
        with open(answers, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        col = "model_answer" if rows and "model_answer" in rows[0] else None
        if col is None:
            raise typer.BadParameter("answers CSV has no 'model_answer' column")
        field_answers, notes = fan_out(items, [r[col] for r in rows])
        for note in notes:
            console.print(f"[yellow]parse note:[/yellow] {note}")
    else:
        field_answers = load_field_answers(answers)

    report = score_run(gold, field_answers, min_accuracy=min_accuracy)
    if judge and any(not r.agrees for r in report.results):
        from rnsr.llm.router import Router

        sub = Router(Settings.from_env()).resolve("sub")
        asyncio.run(judge_disagreements(report, sub.client, sub.model))

    summary = report.summary()
    sub_correct, sub_total = report.substantive
    console.print(f"agreement: {report.correct}/{report.total} "
                  f"({report.accuracy:.1%})")
    console.print(f"  substantive (golden holds a value): {sub_correct}/{sub_total}")
    console.print(f"  resolved by judge: {summary['scored_by_judge']}")
    for d in summary["disagreements"]:
        console.print(f"[red]DIFF[/red] {d['field_id']}\n"
                      f"    golden: {d['golden']!r}\n"
                      f"    answer: {d['answer']!r}")
    written = report.write(out_dir or answers.parent)
    console.print(f"wrote {written}")
    if not report.passed:
        console.print(f"[red]REGRESSION:[/red] {report.accuracy:.1%} is below "
                      f"--min-accuracy {min_accuracy:.1%}")
        raise typer.Exit(2)


@app.command()
def doctor() -> None:
    """Check provider keys, model names and pricing before a real run.

    Model names rot on the provider's schedule and an unpriced model makes
    every spend cap infinite; both used to surface only mid-run.
    """
    import asyncio

    from rnsr.config import Settings
    from rnsr.llm.router import Router, available_providers
    from rnsr.llm.validate import check_models_live

    settings = Settings.from_env()
    providers = available_providers()
    console.print(f"provider keys found: {', '.join(providers) or 'NONE'}")
    if not providers:
        console.print("[red]no provider key set[/red] — set ANTHROPIC_API_KEY, "
                      "OPENAI_API_KEY or GOOGLE_API_KEY")
        raise typer.Exit(1)

    router = Router(settings)
    console.print(f"active provider: {router.provider}")
    t = Table(title="resolved roles")
    for col in ("role", "model", "provider"):
        t.add_column(col)
    for role in ("root", "sub", "embed", "vision"):
        try:
            resolved = router.resolve(role)
            t.add_row(role, resolved.model,
                      getattr(resolved.client, "provider", ""))
        except Exception as e:
            t.add_row(role, f"[yellow]unavailable[/yellow]: {e}", "")
    console.print(t)

    findings = asyncio.run(check_models_live(router, roles=("root", "sub")))
    if not findings:
        console.print("[green]models verified against the provider[/green]")
        return
    for f in findings:
        console.print(f"[red]{f['role']}[/red] ({f['model']}): {f['problem']}")
    raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Run the HTTP service (needs the 'service' extra).

    Endpoints: /healthz, /readyz, /metrics, POST /jobs, GET /jobs/{id}.
    """
    from rnsr.config import Settings
    from rnsr.service import serve as _serve

    console.print(f"rnsr service on http://{host}:{port}")
    _serve(host=host, port=port, settings=Settings.from_env())


@app.command()
def ablate(
    corpus: Path = typer.Argument(..., exists=True, help="corpus.db artifact"),
    n_queries: int = typer.Option(20, "--queries", "-q"),
    rescore_pool: int = typer.Option(4000, "--pool"),
    report_path: Path | None = typer.Option(None, "--report"),
) -> None:
    """Rung-4 quantization ablation: int8 recall@10/50 vs exact fp32 (§8)."""
    import asyncio
    import json
    import sqlite3

    from rnsr.config import Settings
    from rnsr.eval.ablation import run_ablation
    from rnsr.llm.router import Router

    settings = Settings.from_env()
    embed = Router(settings).resolve("embed")

    def embed_fn(texts):
        return asyncio.run(embed.client.embed(texts, model=embed.model))

    conn = sqlite3.connect(corpus)
    queries = [r[0] for r in conn.execute(
        "SELECT substr(text, 1, 200) FROM chunks ORDER BY random() LIMIT ?",
        (n_queries,))]
    conn.close()

    report = run_ablation(corpus, embed_fn, queries, rescore_pool=rescore_pool)
    console.print_json(json.dumps(report))
    if report_path:
        report_path.write_text(json.dumps(report, indent=2))
    console.print("[green]int8 ACCEPTS — polar path stays dormant[/green]"
                  if report["accepts"] else
                  "[red]int8 below bar — evaluate the polar quantizer[/red]")


if __name__ == "__main__":
    app()
