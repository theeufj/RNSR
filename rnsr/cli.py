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
                                              "CSV, Markdown, text, email, HTML, zip, "
                                              "images, Outlook .msg)"),
    out: Path = typer.Option(Path("corpus.db"), "--out", "-o", help="Output artifact path"),
    report_path: Path | None = typer.Option(None, "--report", help="Write JSON report here"),
    llm: bool = typer.Option(False, "--llm/--no-llm",
                             help="Enable the sub-LM prose cross-check and vision "
                                  "re-extraction rung (§3.3); default is fully LLM-free"),
    no_transcribe: bool = typer.Option(
        False, "--no-transcribe",
        help="Leave scanned pages untranscribed even if a vision key is set"),
    append: bool = typer.Option(
        False, "--append",
        help="Add documents to an existing corpus.db (drop freeze, write, refreeze)"),
    replace: str | None = typer.Option(
        None, "--replace",
        help="Delete this doc_id then append the given source(s)"),
) -> None:
    """Ingest documents into a corpus.db artifact (Phase A)."""
    from rnsr.config import Settings
    from rnsr.ingest.cost_estimate import resolve_transcriber
    from rnsr.ingest.lifecycle import append as append_ingest
    from rnsr.ingest.lifecycle import replace_document
    from rnsr.ingest.pipeline import ingest as run_ingest

    settings = Settings.from_env()
    prose_checker = vision = transcriber = None
    transcriber, vision_model = resolve_transcriber(settings, no_transcribe=no_transcribe)
    if transcriber is not None:
        console.print(f"scanned-page transcription: on ({vision_model})")
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
        if transcriber is None:
            transcriber = make_page_transcriber(vis.client, vis.model)

    if replace:
        stats = replace_document(
            out, replace, sources[0], config=settings, transcriber=transcriber)
        if len(sources) > 1:
            stats = append_ingest(
                sources[1:], out, config=settings, transcriber=transcriber)
        console.print(f"replaced {replace}: {stats}")
        return
    if append:
        stats = append_ingest(
            sources, out, config=settings, transcriber=transcriber)
        console.print(f"appended: {stats}")
        return

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
    from rnsr.sdk import make_runner

    return make_runner(settings)


@app.command()
def query(
    corpus: Path = typer.Argument(..., exists=True, help="corpus.db artifact"),
    question: str = typer.Argument(...),
    run_dir: Path = typer.Option(Path("runs/query"), "--run-dir"),
    allow_degraded: bool = typer.Option(
        False, "--allow-degraded",
        help="Answer even when corpus health is blocked; stamp health on the result"),
) -> None:
    """Answer a question against a corpus.db via the RLM loop (Phase B/C)."""
    import asyncio
    from dataclasses import replace

    from rnsr.config import Settings
    from rnsr.errors import CorpusHealthError
    from rnsr.sdk import corpus_env

    settings = Settings.from_env()
    if allow_degraded:
        settings = replace(settings, allow_degraded=True)
    try:
        env = corpus_env(corpus, settings=settings)
    except CorpusHealthError as e:
        console.print(f"[red]corpus health blocked:[/red] {e}")
        raise typer.Exit(2) from e
    health = (env.manifest or {}).get("health") or {}
    if health.get("grade") and health["grade"] != "ok":
        console.print(f"[yellow]corpus health {health['grade']}[/yellow]")
    result = asyncio.run(_make_runner(settings).run(question, env, run_dir=run_dir))
    result.health = health
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


@app.command("eval-tables")
def eval_tables_cmd(
    directory: Path = typer.Option(..., "--dir", exists=True, file_okay=False,
                                   help="directory with labels.json + documents"),
    out_db: Path | None = typer.Option(None, "--out-db",
                                       help="corpus.db path (default: DIR/corpus.db)"),
) -> None:
    """Score table extraction against a labelled set (see testMatter/messy-tables)."""
    import json

    from rnsr.config import Settings
    from rnsr.eval.tables_score import score_labelled_tables

    report = score_labelled_tables(directory, out_db=out_db, config=Settings.from_env())
    console.print_json(json.dumps({k: report[k] for k in report if k != "results"}))
    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        color = "green" if r["passed"] or not r["must_pass"] else "red"
        console.print(f"[{color}]{mark}[/{color}] {r['doc']} {r['checks']}")
    if report["n_required"] and report["n_required_passed"] < report["n_required"]:
        raise typer.Exit(2)


@app.command("eval")
def eval_cmd(
    benchmark: str = typer.Option(..., "--benchmark", "-b",
                                  help="synthetic-oolong | oolong | financebench | "
                                       "matter | office | cuad | contractnli | legalbench"),
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
    elif benchmark == "office":
        from rnsr.eval.datasets.office_gen import generate_office

        items = generate_office(run_dir / "office_docs", seed=seed)
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
    no_transcribe: bool = typer.Option(
        False, "--no-transcribe",
        help="Leave scanned pages untranscribed even if a vision key is set"),
    allow_degraded: bool = typer.Option(
        False, "--allow-degraded",
        help="Answer even when corpus health is blocked; stamp health on the report"),
    abstain_below: str = typer.Option(
        "off", "--abstain-below",
        help="Replace answers at or below this trust tier with 'NEEDS REVIEW' "
             "in the answers CSV (off | medium | high). Status file keeps the "
             "raw answer."),
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
    from dataclasses import replace as _replace

    from rnsr import obs as _obs
    from rnsr.config import Settings
    from rnsr.db.artifact import CorpusDB
    from rnsr.eval.harness import _corpus_valid
    from rnsr.harness.loop import EnvSpec
    from rnsr.harness.trajectory import prune_trajectories
    from rnsr.llm import governor as _governor
    from rnsr.runlock import WorkDirBusy, WorkDirLock

    settings = Settings.from_env()
    if allow_degraded:
        settings = _replace(settings, allow_degraded=True)
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

    from rnsr.ingest.fast_parse import stat_identity as _stat_id
    from rnsr.ingest.lifecycle import append as append_ingest
    from rnsr.ingest.lifecycle import file_index, replace_document

    cache_dir = work_dir / "corpora"
    corpus_path = cache_dir / "corpus.db"
    if corpus_path.exists() and not _corpus_valid(corpus_path, 1):
        corpus_path.unlink()
    if corpus_path.exists():
        index = file_index(corpus_path)
        from rnsr.ingest.parse import _sha256 as _content_sha

        new_files, changed = [], []
        for s in files:
            rec = index.get(str(s.resolve())) or index.get(s.name)
            if rec is None:
                new_files.append(s)
                continue
            known = {rec.get("sha256"), rec.get("content_sha256")} - {None, ""}
            if _stat_id(s) in known or _content_sha(s) in known:
                continue
            changed.append((rec["doc_id"], s))
        for doc_id, src in changed:
            console.print(f"[dim]replacing changed {src.name}[/dim]")
            replace_document(corpus_path, doc_id, src, config=settings)
        if new_files:
            console.print(f"appending {len(new_files)} new document(s)")
            append_ingest(new_files, corpus_path, config=settings)
    if not corpus_path.exists() and fast_ingest:
        cache_dir.mkdir(parents=True, exist_ok=True)
        from rnsr.ingest.cost_estimate import resolve_transcriber

        transcriber, _vision_model = resolve_transcriber(
            settings, no_transcribe=no_transcribe)
        if llm and transcriber is None and not no_transcribe:
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
        from rnsr.ingest.cost_estimate import resolve_transcriber

        prose_checker = vision = None
        transcriber, _vision_model = resolve_transcriber(
            settings, no_transcribe=no_transcribe)
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
            if transcriber is None and not no_transcribe:
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
    from rnsr.errors import CorpusHealthError
    from rnsr.ingest.health import enforce_health, load_health

    with CorpusDB(corpus_path) as c:
        manifest = c.manifest_dict()
        corpus_health = load_health(c, settings)
    try:
        enforce_health(corpus_health, settings)
    except CorpusHealthError as e:
        console.print(f"[red]corpus health blocked:[/red] {e}")
        lock.release()
        raise typer.Exit(2) from e
    manifest["health"] = corpus_health.to_dict()
    if corpus_health.grade != "ok":
        console.print(f"[yellow]corpus health {corpus_health.grade}[/yellow]")
        for f in corpus_health.findings:
            console.print(f"  [{f.severity}] {f.detail}")
    from rnsr.harness.playbook import discover_playbook

    playbook = discover_playbook(corpus_dir, corpus_path.parent, work_dir)
    env = EnvSpec(mode="docdb", corpus_db=str(corpus_path), manifest=manifest,
                  playbook=playbook)

    runner = _make_runner(settings)
    sem = asyncio.Semaphore(concurrency)

    # incremental checkpoint: rerun resumes, never re-pays
    ckpt = work_dir / "answers_partial.jsonl"
    done: dict[int, str] = {}
    status: dict[int, str] = {}
    errors: dict[int, str] = {}
    agreements: dict[int, float] = {}     # consensus mode: share of passes agreeing
    contested: set[str] = set()           # query ids the passes disagreed on
    tiers: dict[int, str] = {}
    quotes_verified: dict[int, str] = {}
    resolved_by: dict[int, str] = {}
    neg_audits: dict[int, str] = {}
    cite_docs: dict[int, str] = {}

    def stamp_evidence(i: int, ev, *, resolved: str | None = None) -> None:
        if ev is None:
            return
        tiers[i] = ev.tier
        quotes_verified[i] = f"{ev.quotes_verified}/{ev.quotes_total}"
        if resolved or ev.resolved_by:
            resolved_by[i] = resolved or ev.resolved_by or ""
        neg_audits[i] = ev.negative_audit
        if ev.docs_cited:
            cite_docs[i] = ", ".join(ev.docs_cited)
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
                stamp_evidence(i, res.evidence)
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
                    for i in group:
                        ans = cr.answers.get(qid[i])
                        if ans is not None:
                            stamp_evidence(i, ans.evidence,
                                           resolved=ans.resolved_by)
                else:
                    br = await runner.run_batch(
                        [(qid[i], qs[i]) for i in group], env,
                        run_dir=work_dir / "trajectories",
                        query_id=f"b{group[0]:03d}_{group[-1]:03d}")
                    got = br.answers
                    group_status = br.result.status
                    for i in group:
                        stamp_evidence(i, br.evidence.get(qid[i]))
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

    _TIER_RANK = {"low": 0, "medium": 1, "high": 2}
    floor = _TIER_RANK.get((abstain_below or "off").lower())

    def maybe_abstain(i: int, text: str) -> str:
        if floor is None:
            return text
        if _TIER_RANK.get(tiers.get(i, "high"), 2) < floor:
            return "NEEDS REVIEW"
        return text

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "answers_chunk1.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow([question_col, "model_answer"])
        for i, (q, a) in enumerate(zip(qs, answers, strict=True)):
            w.writerow([q, maybe_abstain(i, a)])

    status_path = output / "answers_status.csv"
    with open(status_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["row", "query_id", "status", "agreement", "contested",
                    "error", "model_answer", "corpus_health",
                    "tier", "quotes_verified", "resolved_by", "negative_audit"])
        health_grade = corpus_health.grade
        for i in range(len(qs)):
            qid = f"q{i:03d}"
            w.writerow([i, qid, status.get(i, "error"),
                        "" if i not in agreements else f"{agreements[i]:.2f}",
                        "yes" if qid in contested else "",
                        errors.get(i, ""), answers[i], health_grade,
                        tiers.get(i, ""), quotes_verified.get(i, ""),
                        resolved_by.get(i, ""), neg_audits.get(i, "")])

    counts: dict[str, int] = {}
    for i in range(len(qs)):
        st = status.get(i, "error")
        counts[st] = counts.get(st, 0) + 1
    n_error = counts.get("error", 0)
    n_nf = sum(a.startswith(not_found) for a in answers)
    error_rate = n_error / len(qs) if qs else 0.0
    # snapshot() is the GovernorProtocol reporting surface — a custom
    # (e.g. Redis-backed) governor need only provide these keys
    gov = _governor.current().snapshot()
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
        "provider": gov,
        "metrics": _obs.metrics().snapshot(),
        "health": corpus_health.to_dict(),
        "tier_counts": {t: sum(1 for v in tiers.values() if v == t)
                        for t in ("high", "medium", "low")},
        "abstain_below": abstain_below,
    }
    (output / "run_report.json").write_text(_json.dumps(report, indent=2))

    from rnsr.eval.report_card import write_report_card
    from rnsr.eval.xlsx_out import write_answers_xlsx

    xlsx_rows = []
    for i, (q, a) in enumerate(zip(qs, answers, strict=True)):
        xlsx_rows.append({
            question_col: q,
            "value": maybe_abstain(i, a),
            "tier": tiers.get(i, ""),
            "doc": cite_docs.get(i, ""),
            "page": "",
            "quote": quotes_verified.get(i, ""),
        })
    xlsx_path = write_answers_xlsx(output / "answers.xlsx", xlsx_rows,
                                   question_col=question_col)
    write_report_card(output, report=report, health=corpus_health.to_dict())

    console.print(f"wrote {out_path} ({len(answers)} rows, {n_nf} not-found)")
    console.print(f"wrote {xlsx_path} and {output / 'report.md'}")
    console.print(f"status: {counts} — details in {status_path}")
    if consensus > 1:
        console.print(
            f"consensus: {consensus} passes, {len(contested)} field(s) "
            "contested and settled by a tie-break loop"
            + (f" ({', '.join(sorted(contested)[:8])})" if contested else ""))
    console.print(f"provider: {gov['requests']} request(s), "
                  f"${gov['spend_usd']:.4f}, {gov['rate_limit_hits']} "
                  "rate-limit hit(s)")
    if n_error:
        for i in sorted(errors)[:5]:
            if status.get(i) == "error":
                console.print(f"[red]row {i} failed:[/red] {errors[i]}")
    if gov["spend_ceiling_usd"] and gov["spend_usd"] >= gov["spend_ceiling_usd"]:
        console.print(
            f"[red]SPEND CEILING REACHED:[/red] ${gov['spend_usd']:.2f} of "
            f"${gov['spend_ceiling_usd']:.2f}. Calls were refused from that "
            "point on, so later answers are placeholders. Raise "
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
         "not_found": spec.not_found, "date_format": spec.date_format,
         "items": [asdict(i) for i in items]}, indent=1, ensure_ascii=False))

    n_groups = sum(1 for i in items if i.kind == "group")
    n_grouped_fields = sum(len(i.members) for i in items if i.kind == "group")
    console.print(f"wrote {out_csv}: {len(items)} questions "
                  f"({n_groups} groups covering {n_grouped_fields} fields + "
                  f"{len(items) - n_groups} standalone)")
    console.print(f"wrote {map_path}")


@app.command("regress")
def regress_cmd(
    answers: Path | None = typer.Option(None, "--answers", exists=True,
                                        help="answers CSV from answer-csv"),
    golden: Path | None = typer.Option(None, "--golden", exists=True,
                                       help="golden JSON with per-field values"),
    item_map: Path | None = typer.Option(None, "--map",
                                         help="item map from build-questions; "
                                              "fans group answers out to fields"),
    out_dir: Path | None = typer.Option(None, "--out"),
    min_accuracy: float = typer.Option(0.0, "--min-accuracy",
                                       help="exit 2 below this field accuracy"),
    max_false_positive_rate: float = typer.Option(
        1.0, "--max-false-positive-rate",
        help="exit 2 when confident-wrong / absent-items exceeds this"),
    min_high_tier_accuracy: float = typer.Option(
        0.0, "--min-high-tier-accuracy",
        help="exit 2 when accuracy over high-tier answers is below this"),
    judge: bool = typer.Option(True, "--judge/--no-judge",
                               help="sub-LM equivalence check for string "
                                    "failures (long answers differ in wording, "
                                    "not meaning)"),
    from_review: Path | None = typer.Option(
        None, "--from-review", exists=True,
        help="score a filled review.csv from audit-export into a miss report"),
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
        load_golden_notes,
        load_status_tiers,
        score_run,
    )
    from rnsr.forms.fanout import fan_out

    if from_review:
        from rnsr.eval.audit import score_review

        report = score_review(from_review)
        dest = out_dir or from_review.parent
        dest.mkdir(parents=True, exist_ok=True)
        written = dest / "review_misses.json"
        written.write_text(_json.dumps(report, indent=2))
        console.print(f"review: {report['correct']}/{report['marked']} marked-correct "
                      f"({report['accuracy']:.1%}), {report['misses']} miss(es)")
        for miss in report["miss_list"]:
            console.print(f"[red]MISS[/red] {miss['qid']}: {miss['answer']!r}  "
                          f"{miss['note']}")
        console.print(f"wrote {written}")
        if report["misses"]:
            raise typer.Exit(2)
        return

    if answers is None or golden is None:
        raise typer.BadParameter("provide --answers and --golden, or --from-review")

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

    status_csv = answers.parent / "answers_status.csv"
    report = score_run(
        gold, field_answers, min_accuracy=min_accuracy,
        max_false_positive_rate=max_false_positive_rate,
        min_high_tier_accuracy=min_high_tier_accuracy,
        notes=load_golden_notes(golden),
        tiers=load_status_tiers(status_csv),
    )
    if judge and any(not r.agrees for r in report.results):
        from rnsr.llm.router import Router

        sub = Router(Settings.from_env()).resolve("sub")
        asyncio.run(judge_disagreements(report, sub.client, sub.model))

    summary = report.summary()
    sub_correct, sub_total = report.substantive
    console.print(f"agreement: {report.correct}/{report.total} "
                  f"({report.accuracy:.1%})")
    console.print(f"  substantive (golden holds a value): {sub_correct}/{sub_total}")
    console.print(f"  false-positive rate: {summary['false_positive_rate']:.1%} "
                  f"({summary['confident_wrong']} confident-wrong on absent items)")
    console.print(f"  abstain rate (value items): {summary['abstain_rate']:.1%}")
    console.print(f"  high-tier accuracy: {summary['high_tier_accuracy']:.1%}")
    console.print(f"  review recall: {summary['review_recall']:.1%}  "
                  f"auto-accept: {summary['auto_accept_rate']:.1%}")
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
def replay(
    db: Path = typer.Option(..., "--db", exists=True, help="corpus.db artifact"),
    queries: list[Path] = typer.Option(..., "--queries", exists=True,
                                       help="queries.json and/or trajectory JSONL"),
    baseline: Path | None = typer.Option(None, "--baseline",
                                         help="frozen row-set JSON to compare against"),
    write_baseline: Path | None = typer.Option(None, "--write-baseline",
                                               help="write the cells-path snapshot here"),
    k: int = typer.Option(10, "--k"),
) -> None:
    """Replay rung-0 queries; exit 2 on cells/legacy/baseline diffs."""
    import json

    from rnsr.eval.replay import load_queries
    from rnsr.eval.replay import replay as run_replay
    from rnsr.eval.replay import write_baseline as dump

    qs = load_queries(*queries)
    base = json.loads(baseline.read_text()) if baseline else None
    report = run_replay(db, qs, k=k, baseline=base)
    if write_baseline:
        dump(write_baseline, report["snapshot"])
        console.print(f"wrote baseline {write_baseline}")
    console.print_json(json.dumps({"n": report["n"], "n_diffs": report["n_diffs"],
                                   "diffs": report["diffs"][:20]}))
    if report["n_diffs"]:
        raise typer.Exit(2)


@app.command()
def health(
    corpus: Path = typer.Argument(..., exists=True, help="corpus.db artifact"),
) -> None:
    """Print the corpus health report (grade, findings, table/scan counters)."""
    import json

    from rnsr.config import Settings
    from rnsr.db.artifact import CorpusDB
    from rnsr.ingest.health import load_health

    settings = Settings.from_env()
    with CorpusDB(corpus) as c:
        report = load_health(c, settings)
    console.print_json(json.dumps(report.to_dict()))
    color = {"ok": "green", "degraded": "yellow", "blocked": "red"}.get(report.grade, "white")
    console.print(f"[{color}]grade: {report.grade}[/{color}]  "
                  f"source={report.source}  "
                  f"tables={report.tables_total} "
                  f"(untrusted={report.tables_untrusted} "
                  f"unchecked={report.tables_unchecked})  "
                  f"pass_rate={report.validation_pass_rate:.1%}")
    raise typer.Exit(0 if report.grade != "blocked" else 2)


@app.command()
def migrate(
    corpus: Path = typer.Argument(..., exists=True, help="corpus.db artifact"),
) -> None:
    """Stamp user_version / format_version on a pre-versioned artifact."""
    import json

    from rnsr.db.migrate import migrate_artifact

    result = migrate_artifact(corpus)
    console.print_json(json.dumps(result))


@app.command()
def autopsy(
    run_dir: Path = typer.Argument(..., exists=True,
                                   help="eval or answer-csv run directory"),
    golden: Path | None = typer.Option(None, "--golden", exists=True,
                                       help="golden JSON when results.jsonl is absent"),
    review: Path | None = typer.Option(None, "--review", exists=True,
                                       help="filled review.csv (gold-error marks)"),
    out: Path | None = typer.Option(None, "--out",
                                    help="directory for autopsy.json + loss-ledger.md"),
) -> None:
    """Classify every miss in a run (ingest / retrieval / reasoning / …)."""
    import json

    from rnsr.config import Settings
    from rnsr.eval.autopsy import autopsy_run, write_ledger

    settings = Settings.from_env()
    ledger = autopsy_run(run_dir, golden=golden, review=review,
                         key=settings.trajectory_key)
    dest = out or run_dir
    written = write_ledger(ledger, dest, title=f"Loss ledger — {run_dir.name}")
    console.print_json(json.dumps({
        "n": ledger["n"], "n_miss": ledger["n_miss"],
        "accuracy": ledger["accuracy"],
        "cause_counts": ledger["cause_counts"],
        "cause_x_class": ledger["cause_x_class"],
    }))
    console.print(f"wrote {written['json']} and {written['md']}")


@app.command("audit-export")
def audit_export(
    work_dir: Path = typer.Option(..., "--work-dir", exists=True,
                                  help="answer-csv work directory"),
    out: Path = typer.Option(..., "--out", help="directory for evidence + review.csv"),
) -> None:
    """Export a field-trial packet: evidence.json per question, plus review.csv.

    Trajectories stay in the work directory. The packet has answers, verified
    quotes, SQL, pages, status and corpus health — enough to review without
    shipping the full REPL log.
    """
    from rnsr.config import Settings
    from rnsr.eval.audit import export_audit

    settings = Settings.from_env()
    result = export_audit(work_dir, out, key=settings.trajectory_key)
    console.print(f"wrote {result['n']} evidence file(s) under {result['evidence_dir']}")
    console.print(f"review sheet: {result['review']}")


@app.command("review-import")
def review_import_cmd(
    review: Path = typer.Argument(..., exists=True, help="filled review.csv"),
    corpus_dir: Path | None = typer.Option(
        None, "--corpus",
        help="corpus directory to write golden/ and playbook.diff.json"),
    out: Path | None = typer.Option(None, "--out", help="output directory override"),
) -> None:
    """Turn reviewer corrections into corpus-local golden items and playbook diffs."""
    import json

    from rnsr.eval.review_import import import_review

    result = import_review(review, corpus_dir=corpus_dir, out_dir=out)
    console.print_json(json.dumps({
        k: result[k] for k in ("n_ok", "n_miss", "n_unmarked", "n_golden",
                               "golden", "playbook_diff")
    }))


@app.command()
def doctor(
    check_models: bool = typer.Option(
        False, "--check-models",
        help="ask the provider which models exist and fail on retired names"),
    corpus: Path | None = typer.Option(
        None, "--corpus", exists=True,
        help="corpus.db: warn when rung-4 embeddings should be on"),
) -> None:
    """Check provider keys, model names and pricing before a real run.

    Model names rot on the provider's schedule and an unpriced model makes
    every spend cap infinite; both used to surface only mid-run.
    """
    import asyncio

    from rnsr.config import Settings
    from rnsr.llm.cost import PRICES_PER_MTOK
    from rnsr.llm.router import DEFAULT_MODELS, Router, available_providers
    from rnsr.llm.validate import check_models_live, check_pricing

    settings = Settings.from_env()
    providers = available_providers()
    console.print(f"provider keys found: {', '.join(providers) or 'NONE'}")
    if not providers:
        console.print("[red]no provider key set[/red] — set ANTHROPIC_API_KEY, "
                      "OPENAI_API_KEY or GOOGLE_API_KEY")
        raise typer.Exit(1)

    missing = [
        (provider, role, model)
        for provider, roles in DEFAULT_MODELS.items()
        for role, model in roles.items()
        if model and model not in PRICES_PER_MTOK
    ]
    if missing:
        for provider, role, model in missing:
            console.print(f"[red]unpriced[/red] {provider}/{role}: {model}")
        raise typer.Exit(1)

    router = Router(settings)
    console.print(f"active provider: {router.provider}")
    t = Table(title="resolved roles")
    for col in ("role", "model", "provider"):
        t.add_column(col)
    for role in ("root", "sub", "embed", "vision"):
        try:
            resolved = router.resolve(role)
            priced = "yes" if check_pricing(resolved.model, role) else "NO"
            t.add_row(role, f"{resolved.model} (priced={priced})",
                      getattr(resolved.client, "provider", ""))
        except Exception as e:
            t.add_row(role, f"[yellow]unavailable[/yellow]: {e}", "")
    console.print(t)

    if corpus is not None:
        from rnsr.db.artifact import CorpusDB

        with CorpusDB(corpus) as c:
            n_docs = c.conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        threshold = settings.embed_auto_on_docs
        embed_ok = True
        try:
            router.resolve("embed")
        except Exception:
            embed_ok = False
        if n_docs >= threshold and not embed_ok:
            console.print(
                f"[yellow]rung-4 default-on:[/yellow] corpus has {n_docs} docs "
                f"(threshold {threshold}) but no embed provider is configured. "
                "Set an OpenAI/Gemini key so embeddings activate automatically."
            )
        elif n_docs >= threshold:
            console.print(
                f"rung-4 embeddings: on ({n_docs} docs ≥ {threshold})"
            )

    if not check_models:
        return

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
