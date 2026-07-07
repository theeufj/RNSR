"""rnsr command-line interface.

Commands land with their phases:
    rnsr ingest   — Phase A: PDFs -> corpus.db + validation report
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


@app.command()
def ingest(
    sources: list[Path] = typer.Argument(..., exists=True, readable=True,
                                         help="PDF files to ingest"),
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
        run_eval(items, system, runner, run_dir=out_dir, limit=limit)
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
    concurrency: int = typer.Option(3, "--concurrency"),
    not_found: str = typer.Option("Not found in matter corpus", "--not-found-phrase"),
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
    """
    import asyncio
    import csv as _csv
    import json as _json

    from rnsr.config import Settings
    from rnsr.db.artifact import CorpusDB
    from rnsr.eval.harness import _corpus_valid
    from rnsr.harness.loop import EnvSpec

    settings = Settings.from_env()
    with open(questions, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows or question_col not in rows[0]:
        raise typer.BadParameter(
            f"questions CSV has no column {question_col!r}; "
            f"columns: {list(rows[0].keys()) if rows else 'none'}")
    qs = [r[question_col] for r in rows]
    console.print(f"{len(qs)} questions over corpus {corpus_dir}")

    # ingest once, cached by content (PDFs via docling)
    pdfs = sorted(p for p in corpus_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() == ".pdf")
    others = sorted(p for p in corpus_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in
                    (".txt", ".md", ".eml", ".csv"))
    if not pdfs and not others:
        raise typer.BadParameter(f"no ingestable files under {corpus_dir}")
    console.print(f"ingesting {len(pdfs)} PDFs + {len(others)} text-like files "
                  "(cached across runs)")
    if others:
        console.print("[yellow]note:[/yellow] text-like files present; "
                      "current adapter ingests PDFs only — flag if these matter")

    from hashlib import sha256 as _sha256

    h = _sha256()
    for s in pdfs:  # stat-identity: no byte reads over the corpus
        st = s.stat()
        h.update(f"{s}|{st.st_size}|{st.st_mtime_ns}".encode())
    cache_dir = work_dir / "corpora"
    corpus_path = cache_dir / f"corpus_{h.hexdigest()[:16]}.db"
    if corpus_path.exists() and not _corpus_valid(corpus_path, len(pdfs)):
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

        stats = ingest_bulk(pdfs, corpus_path, config=settings,
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

        report = _ingest(pdfs, corpus_path, config=settings,
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
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            rec = _json.loads(line)
            if rec.get("q") == qs[rec["i"]] if rec["i"] < len(qs) else False:
                done[rec["i"]] = rec["a"]
        if done:
            console.print(f"resuming: {len(done)}/{len(qs)} already answered")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt_f = open(ckpt, "a", encoding="utf-8")  # noqa: SIM115 — spans the run

    async def answer(i: int, q: str) -> tuple[int, str]:
        if i in done:
            return i, done[i]
        async with sem:
            try:
                res = await runner.run(q, env, run_dir=work_dir / "trajectories",
                                       query_id=f"q{i:03d}")
                text = "" if res.answer is None else str(res.answer).strip()
                a = text or not_found
            except Exception:
                a = not_found
            ckpt_f.write(_json.dumps({"i": i, "q": q, "a": a}) + "\n")
            ckpt_f.flush()
            return i, a

    async def main() -> list[str]:
        results = await asyncio.gather(*(answer(i, q) for i, q in enumerate(qs)))
        return [a for _, a in sorted(results)]

    answers = asyncio.run(main())
    ckpt_f.close()

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "answers_chunk1.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow([question_col, "model_answer"])
        for q, a in zip(qs, answers, strict=True):
            w.writerow([q, a])
    n_nf = sum(a.startswith(not_found) for a in answers)
    console.print(f"wrote {out_path} ({len(answers)} rows, {n_nf} not-found)")


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
