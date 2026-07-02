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
    prose_checker = vision = None
    if llm:
        from rnsr.ingest.llm_hooks import make_prose_checker, make_vision_extractor
        from rnsr.llm.router import Router

        router = Router(settings)
        sub, vis = router.resolve("sub"), router.resolve("vision")
        prose_checker = make_prose_checker(sub.client, sub.model,
                                           concurrency=settings.sub_concurrency)
        vision = make_vision_extractor(vis.client, vis.model)

    report = run_ingest(sources, out, config=settings,
                        prose_checker=prose_checker, vision=vision)

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
    if report.skipped_stages:
        console.print(f"[yellow]skipped:[/yellow] {', '.join(report.skipped_stages)}")
    if report_path:
        report_path.write_text(report.to_json())
        console.print(f"report written to {report_path}")


@app.command()
def query() -> None:
    """Answer a question against a corpus.db via the RLM loop (Phase B/C)."""
    raise typer.Exit("not implemented yet: lands with Phase B (harness/loop.py)")


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
def ablate() -> None:
    """Run the rung-4 quantization ablation (Phase D)."""
    raise typer.Exit("not implemented yet: lands with Phase D")


if __name__ == "__main__":
    app()
