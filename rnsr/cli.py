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
) -> None:
    """Ingest documents into a corpus.db artifact (Phase A)."""
    from rnsr.config import Settings
    from rnsr.ingest.pipeline import ingest as run_ingest

    report = run_ingest(sources, out, config=Settings.from_env())

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
def eval_cmd() -> None:
    """Run the evaluation harness (§8)."""
    raise typer.Exit("not implemented yet: lands with the eval harness")


@app.command()
def ablate() -> None:
    """Run the rung-4 quantization ablation (Phase D)."""
    raise typer.Exit("not implemented yet: lands with Phase D")


if __name__ == "__main__":
    app()
