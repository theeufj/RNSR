"""
RNSR CLI - Command Line Interface

Usage:
    python -m rnsr ingest document.pdf
    python -m rnsr query "What are the payment terms?"
    python -m rnsr batch-ingest ./docs/ --recursive --store ./my_store/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

logger = structlog.get_logger(__name__)


def cmd_ingest(args):
    """Ingest a PDF document."""
    from rnsr.ingestion import ingest_document
    
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Ingesting: {pdf_path}")
    result = ingest_document(pdf_path)
    
    print(f"\n✓ Ingestion complete!")
    print(f"  Tier used: {result.tier_used} ({result.method})")
    print(f"  Total nodes: {result.tree.total_nodes}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result.tree.model_dump(), f, indent=2)
        print(f"\nTree saved to: {output_path}")
    
    return result


def cmd_index(args):
    """Build skeleton index from ingested document."""
    from rnsr.indexing import SQLiteKVStore, build_skeleton_index
    from rnsr.ingestion import ingest_document
    
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Ingest first
    print(f"Ingesting: {pdf_path}")
    result = ingest_document(pdf_path)
    
    # Build index
    db_path = args.db or f"{pdf_path.stem}_index.db"
    kv_store = SQLiteKVStore(db_path)
    skeleton, _ = build_skeleton_index(result.tree, kv_store)
    
    print(f"\n✓ Index built!")
    print(f"  Skeleton nodes: {len(skeleton)}")
    print(f"  KV entries: {kv_store.count()}")
    print(f"  Database: {db_path}")
    
    return skeleton, kv_store


def cmd_query(args):
    """Query a document."""
    from rnsr import RNSRClient
    
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Use RNSRClient to respect provider/model/api_key settings
    client = RNSRClient(
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=args.api_key,
    )
    
    print(f"Ingesting: {pdf_path}")
    
    # Run query via the client (handles ingestion + indexing internally)
    print(f"\nQuery: {args.query}")
    print("-" * 40)
    
    result = client.ask_advanced(
        document=pdf_path,
        question=args.query,
        use_knowledge_graph=False,  # Fast mode for CLI
    )
    
    answer = result.get("answer", "No answer found.")
    confidence = result.get("confidence", 0.0)
    nodes_visited = result.get("nodes_visited", [])
    variables_used = result.get("variables_used", [])
    
    print(f"\nAnswer:")
    print(answer)
    print(f"\nConfidence: {confidence:.2f}")
    print(f"Nodes visited: {len(nodes_visited)}")
    print(f"Variables used: {len(variables_used)}")
    
    if args.trace:
        print(f"\nTrace:")
        for entry in answer["trace"]:
            print(f"  [{entry['node_type']}] {entry['action']}")


def cmd_batch_ingest(args):
    """Batch-ingest documents from folders or file lists into a DocumentStore."""
    from rnsr.document_store import DocumentStore, BatchProgress

    store = DocumentStore(args.store)

    sources = args.sources
    if len(sources) == 1:
        sources = sources[0]

    def _on_progress(p: BatchProgress) -> None:
        tag = {"success": "+", "skipped": "~", "error": "!"}[p.status]
        name = Path(p.current_file).name
        print(f"  [{tag}] ({p.completed}/{p.total}) {name}", end="")
        if p.status == "error":
            print(f"  -- {p.error}", end="")
        print()

    print(f"Store: {args.store}")
    print(f"Sources: {args.sources}")
    if args.recursive:
        print(f"Recursive: yes")
    print(f"Glob: {args.glob}")
    print(f"Workers: {args.workers}")
    print("-" * 50)

    result = store.batch_ingest(
        sources=sources,
        recursive=args.recursive,
        glob_pattern=args.glob,
        skip_existing=args.skip_existing,
        max_workers=args.workers,
        build_kg=args.build_kg,
        on_progress=_on_progress,
    )

    print("-" * 50)
    print(f"Total:     {result.total}")
    print(f"Succeeded: {result.succeeded}")
    print(f"Skipped:   {result.skipped}")
    print(f"Failed:    {result.failed}")
    print(f"Elapsed:   {result.elapsed_seconds:.1f}s")

    if result.errors:
        print(f"\nErrors:")
        for err in result.errors:
            print(f"  {err['file']}: {err['error']}")

    if result.doc_ids:
        print(f"\nDocument IDs:")
        for doc_id in result.doc_ids:
            print(f"  {doc_id}")


def cmd_benchmark(args):
    """Run benchmarks on the RNSR system."""
    from .benchmarks import BenchmarkRunner, BenchmarkConfig
    
    # Check files are provided
    if not args.config and not args.files:
        print("❌ Error: Provide --files or --config for benchmarking")
        return
    
    # Load config if provided
    if args.config:
        config = BenchmarkConfig.from_json(args.config)
    else:
        config = BenchmarkConfig(
            pdf_paths=[Path(f) for f in (args.files or [])],
            iterations=args.iterations,
            compute_quality=args.quality or args.all,
        )
    
    print("=" * 60)
    print("RNSR Benchmark Suite")
    print("=" * 60)
    print(f"Files: {len(config.pdf_paths)}")
    print(f"Iterations: {config.iterations}")
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    report = runner.run()
    
    # Print summary
    report.print_summary()
    
    # Save results
    output_dir = args.output or "benchmark_results"
    output_path = Path(output_dir)
    report_file = output_path / f"benchmark_report_{report.timestamp.replace(':', '-')}.json"
    report.to_json(report_file)
    
    print(f"\n📄 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RNSR - Recursive Neural-Symbolic Retriever"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF document")
    ingest_parser.add_argument("file", help="Path to PDF file")
    ingest_parser.add_argument("-o", "--output", help="Output JSON file for tree")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build skeleton index")
    index_parser.add_argument("file", help="Path to PDF file")
    index_parser.add_argument("--db", help="SQLite database path")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a document")
    query_parser.add_argument("file", help="Path to PDF file")
    query_parser.add_argument("query", help="Question to ask")
    query_parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    query_parser.add_argument("--trace", action="store_true", help="Show trace")
    query_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default=None,
        help="LLM provider (default: auto-detect from API key)",
    )
    query_parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (e.g. gpt-5-mini, claude-sonnet-4-5, gemini-2.5-flash)",
    )
    query_parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM provider (default: from environment variable)",
    )
    
    # Batch-ingest command
    batch_parser = subparsers.add_parser(
        "batch-ingest", help="Batch-ingest documents from folders or file lists"
    )
    batch_parser.add_argument(
        "sources", nargs="+",
        help="Folder path(s) or individual file path(s) to ingest",
    )
    batch_parser.add_argument(
        "-s", "--store", default=".rnsr_store",
        help="Path to DocumentStore directory (default: .rnsr_store/)",
    )
    batch_parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recurse into subdirectories when sources are folders",
    )
    batch_parser.add_argument(
        "-g", "--glob", default="*.pdf",
        help="File glob pattern for directory scanning (default: *.pdf)",
    )
    batch_parser.add_argument(
        "-w", "--workers", type=int, default=1,
        help="Number of parallel ingestion workers (default: 1)",
    )
    batch_parser.add_argument(
        "--build-kg", action=argparse.BooleanOptionalAction, default=True,
        help="Build workspace knowledge graph after ingestion (default: --build-kg)",
    )
    batch_parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="Skip documents already in the store (default: --skip-existing)",
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument(
        "--config", "-c",
        help="Path to benchmark config JSON file"
    )
    bench_parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="PDF files to benchmark"
    )
    bench_parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=3,
        help="Number of iterations per benchmark (default: 3)"
    )
    bench_parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    bench_parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Run performance benchmarks"
    )
    bench_parser.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Run quality benchmarks"
    )
    bench_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all benchmarks"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "batch-ingest":
        cmd_batch_ingest(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
