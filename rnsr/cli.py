"""rnsr command-line interface.

Commands land with their phases:
    rnsr ingest   — Phase A: PDFs -> corpus.db + validation report
    rnsr query    — Phase B/C: run the RLM loop against a corpus
    rnsr eval     — §8 evaluation harness (benchmarks, baselines, gate)
    rnsr ablate   — Phase D: rung-4 quantization ablation
"""

from __future__ import annotations

import typer

app = typer.Typer(name="rnsr", no_args_is_help=True, add_completion=False)


@app.command()
def ingest() -> None:
    """Ingest documents into a corpus.db artifact (Phase A)."""
    raise typer.Exit("not implemented yet: lands with Phase A (ingest/pipeline.py)")


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
