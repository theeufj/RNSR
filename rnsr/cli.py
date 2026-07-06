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
) -> None:
    """Run the evaluation harness (§8)."""
    import asyncio
    import json

    from rnsr.config import Settings
    from rnsr.eval.harness import run_eval
    from rnsr.harness.loop import RootRunner
    from rnsr.llm.router import Router

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
    elif benchmark == "contractnli":
        from rnsr.eval.datasets.legal import load_contractnli

        items = load_contractnli(limit=limit)
    elif benchmark == "legalbench":
        from rnsr.eval.datasets.legal import load_legalbench

        items = load_legalbench(limit=limit)
    else:
        raise typer.BadParameter(f"unknown benchmark: {benchmark}")

    settings = Settings.from_env()
    router = Router(settings)
    root, sub = router.resolve("root"), router.resolve("sub")
    runner = RootRunner(root_client=root.client, root_model=root.model,
                        sub_client=sub.client, sub_model=sub.model, settings=settings)
    out_dir = run_dir / f"{benchmark}-{system}"
    _, summary = asyncio.run(
        run_eval(items, system, runner, run_dir=out_dir, limit=limit)
    )
    console.print_json(json.dumps(summary))
    console.print(f"results in {out_dir}")


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
