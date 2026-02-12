#!/usr/bin/env python3
"""
RNSR Benchmark & Demo Script

This script demonstrates RNSR's key capabilities and benchmarks performance.
Run this to prepare for demos or to validate the system works correctly.

Usage:
    python scripts/benchmark_demo.py                    # Run all benchmarks
    python scripts/benchmark_demo.py --quick            # Quick demo only
    python scripts/benchmark_demo.py --pdf path/to.pdf  # Test with your PDF
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


def check_environment():
    """Check that the environment is properly configured."""
    console.print("\n[bold blue]🔍 Checking Environment[/bold blue]\n")
    
    checks = []
    
    # Check API keys
    api_keys = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    has_key = False
    for name, value in api_keys.items():
        if value:
            checks.append((name, "✅ Set", "green"))
            has_key = True
        else:
            checks.append((name, "❌ Not set", "red"))
    
    # Check Ollama
    has_ollama = os.getenv("OLLAMA_BASE_URL") or os.getenv("USE_OLLAMA") or os.getenv("LLM_PROVIDER", "").lower() == "ollama"
    if has_ollama:
        checks.append(("Ollama", "✅ Configured", "green"))
    
    # Check imports
    try:
        from rnsr.ingestion.pipeline import ingest_document
        checks.append(("RNSR Import", "✅ OK", "green"))
    except ImportError as e:
        checks.append(("RNSR Import", f"❌ {e}", "red"))
    
    # Display results
    table = Table(title="Environment Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    for name, status, color in checks:
        table.add_row(name, f"[{color}]{status}[/{color}]")
    
    console.print(table)
    
    if not has_key and not has_ollama:
        console.print("\n[red]⚠️  No LLM provider configured! Set a cloud API key or configure Ollama.[/red]")
        return False
    
    return True


def benchmark_ingestion(pdf_path: Path) -> dict:
    """Benchmark document ingestion."""
    from rnsr.ingestion.pipeline import ingest_document
    
    console.print(f"\n[bold blue]📄 Ingesting Document[/bold blue]: {pdf_path.name}\n")
    
    start = time.time()
    result = ingest_document(pdf_path)
    elapsed = time.time() - start
    
    tree = result.tree
    stats = {
        "file": pdf_path.name,
        "ingestion_time_sec": round(elapsed, 2),
        "total_nodes": tree.total_nodes if tree else 0,
        "max_depth": tree.max_depth if tree else 0,
        "root_sections": len(tree.root.children) if tree and tree.root else 0,
    }
    
    # Display results
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("File", stats["file"])
    table.add_row("Time", f"{stats['ingestion_time_sec']}s")
    table.add_row("Total Nodes", str(stats["total_nodes"]))
    table.add_row("Max Depth", str(stats["max_depth"]))
    table.add_row("Root Sections", str(stats["root_sections"]))
    
    console.print(table)
    
    return {"tree": tree, "stats": stats, "result": result}


def benchmark_skeleton_index(tree) -> dict:
    """Benchmark skeleton index building."""
    from rnsr.indexing.skeleton_builder import build_skeleton_index
    
    console.print("\n[bold blue]🦴 Building Skeleton Index[/bold blue]\n")
    
    start = time.time()
    skeleton, kv_store = build_skeleton_index(tree)
    elapsed = time.time() - start
    
    stats = {
        "build_time_sec": round(elapsed, 2),
        "indexed_nodes": len(kv_store) if kv_store else 0,
    }
    
    console.print(f"  ⏱️  Build time: [green]{stats['build_time_sec']}s[/green]")
    console.print(f"  📊 Indexed nodes: [green]{stats['indexed_nodes']}[/green]")
    
    return {"skeleton": skeleton, "kv_store": kv_store, "stats": stats}


def benchmark_query(skeleton, kv_store, query: str, tree=None) -> dict:
    """Benchmark a single query."""
    from rnsr.agent.rlm_navigator import RLMNavigator
    
    console.print(f"\n[bold blue]❓ Query[/bold blue]: {query}\n")
    
    navigator = RLMNavigator(skeleton=skeleton, kv_store=kv_store, tree=tree)
    
    start = time.time()
    result = navigator.answer(query)
    elapsed = time.time() - start
    
    answer = result.get("answer", "No answer")
    
    console.print(Panel(
        answer[:500] + "..." if len(answer) > 500 else answer,
        title="Answer",
        border_style="green",
    ))
    
    console.print(f"\n  ⏱️  Response time: [green]{elapsed:.2f}s[/green]")
    
    if result.get("sources"):
        console.print(f"  📚 Sources used: [green]{len(result['sources'])}[/green]")
    
    return {
        "query": query,
        "answer": answer,
        "time_sec": round(elapsed, 2),
        "sources": result.get("sources", []),
    }


def run_sample_queries(skeleton, kv_store, tree=None) -> list:
    """Run a set of sample queries to demonstrate capabilities."""
    
    sample_queries = [
        "What is the main topic of this document?",
        "Summarize the key findings or conclusions.",
        "What are the main sections in this document?",
    ]
    
    console.print("\n[bold blue]🔍 Running Sample Queries[/bold blue]\n")
    
    results = []
    for query in sample_queries:
        try:
            result = benchmark_query(skeleton, kv_store, query, tree)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Error on query '{query}': {e}[/red]")
            results.append({"query": query, "error": str(e)})
    
    return results


def compare_with_naive_rag(pdf_path: Path, query: str) -> dict:
    """Compare RNSR with naive RAG approach."""
    console.print("\n[bold blue]⚖️  Comparing RNSR vs Naive RAG[/bold blue]\n")
    
    # This would require implementing a naive RAG baseline
    # For now, we'll show the structure
    
    console.print("[yellow]Note: Full comparison requires baseline implementation.[/yellow]")
    console.print("RNSR advantages over naive RAG:")
    console.print("  • Hierarchical context preservation")
    console.print("  • Section-aware retrieval")
    console.print("  • ToT validation reduces hallucination")
    console.print("  • Grounded extraction with citations")
    
    return {"comparison": "pending_implementation"}


def generate_report(results: dict, output_path: Path = None):
    """Generate a benchmark report."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    
    if output_path:
        output_path.write_text(json.dumps(report, indent=2, default=str))
        console.print(f"\n[green]📄 Report saved to: {output_path}[/green]")
    
    # Summary table
    console.print("\n[bold blue]📊 Benchmark Summary[/bold blue]\n")
    
    table = Table(title="Performance Summary")
    table.add_column("Stage", style="cyan")
    table.add_column("Time", style="green")
    table.add_column("Details", style="yellow")
    
    if "ingestion" in results:
        ing = results["ingestion"]["stats"]
        table.add_row(
            "Ingestion",
            f"{ing['ingestion_time_sec']}s",
            f"{ing['total_nodes']} nodes, depth {ing['max_depth']}"
        )
    
    if "skeleton" in results:
        skel = results["skeleton"]["stats"]
        table.add_row(
            "Index Build",
            f"{skel['build_time_sec']}s",
            f"{skel['indexed_nodes']} entries"
        )
    
    if "queries" in results:
        avg_time = sum(q.get("time_sec", 0) for q in results["queries"]) / len(results["queries"])
        table.add_row(
            "Avg Query",
            f"{avg_time:.2f}s",
            f"{len(results['queries'])} queries"
        )
    
    console.print(table)
    
    return report


def run_quick_demo():
    """Run a quick demo using sample documents."""
    console.print(Panel.fit(
        "[bold]RNSR Quick Demo[/bold]\n\n"
        "This demo uses the sample documents included with RNSR.",
        border_style="blue",
    ))
    
    # Check for sample documents
    samples_dir = Path(__file__).parent.parent / "samples"
    
    if not samples_dir.exists():
        console.print("[red]Sample documents not found. Run with --pdf to test your own document.[/red]")
        return
    
    sample_files = list(samples_dir.glob("*.md"))
    
    if not sample_files:
        console.print("[red]No sample files found in samples/[/red]")
        return
    
    console.print(f"\n[green]Found {len(sample_files)} sample documents[/green]\n")
    
    for f in sample_files:
        console.print(f"  • {f.name}")
    
    # Demo the table parser on sample contract
    console.print("\n[bold blue]📊 Table Parsing Demo[/bold blue]\n")
    
    try:
        from rnsr.ingestion.table_parser import TableParser, TableQueryEngine
        
        sample_contract = samples_dir / "sample_contract.md"
        if sample_contract.exists():
            content = sample_contract.read_text()
            
            parser = TableParser()
            tables = parser.parse_from_text(content)
            
            console.print(f"Found [green]{len(tables)}[/green] tables in sample_contract.md")
            
            if tables:
                table = tables[0]
                console.print(f"  Headers: {table.headers}")
                console.print(f"  Rows: {len(table.rows)}")
                
                # Query the table
                engine = TableQueryEngine(table)
                results = engine.select(limit=3)
                console.print(f"  Sample data: {results[:2]}")
    
    except Exception as e:
        console.print(f"[yellow]Table parsing demo skipped: {e}[/yellow]")
    
    # Demo entity extraction patterns
    console.print("\n[bold blue]🔍 Entity Extraction Demo[/bold blue]\n")
    
    try:
        from rnsr.extraction import CandidateExtractor
        
        sample_text = """
        Dr. Sarah Chen, CEO of Acme Technologies Inc., signed the agreement on March 15, 2024.
        The contract value is $2.5 million over 3 years.
        Contact: sarah.chen@acmetech.com
        """
        
        extractor = CandidateExtractor()
        candidates = extractor.extract_candidates(sample_text)
        
        console.print(f"Extracted [green]{len(candidates)}[/green] entity candidates:\n")
        
        for c in candidates[:10]:
            console.print(f"  • [{c.candidate_type}] {c.text}")
    
    except Exception as e:
        console.print(f"[yellow]Entity extraction demo skipped: {e}[/yellow]")
    
    console.print("\n[green]✅ Quick demo complete![/green]")
    console.print("\nTo test with a real PDF, run:")
    console.print("  [cyan]python scripts/benchmark_demo.py --pdf your_document.pdf[/cyan]")


def run_full_benchmark(pdf_path: Path, queries: list = None):
    """Run full benchmark on a PDF."""
    console.print(Panel.fit(
        f"[bold]RNSR Full Benchmark[/bold]\n\n"
        f"Document: {pdf_path.name}",
        border_style="blue",
    ))
    
    results = {}
    
    # 1. Ingestion
    ing_result = benchmark_ingestion(pdf_path)
    results["ingestion"] = {"stats": ing_result["stats"]}
    tree = ing_result["tree"]
    
    if not tree:
        console.print("[red]Ingestion failed. Cannot continue.[/red]")
        return results
    
    # 2. Skeleton Index
    skel_result = benchmark_skeleton_index(tree)
    results["skeleton"] = {"stats": skel_result["stats"]}
    
    # 3. Queries
    if queries:
        results["queries"] = []
        for query in queries:
            q_result = benchmark_query(
                skel_result["skeleton"],
                skel_result["kv_store"],
                query,
                tree
            )
            results["queries"].append(q_result)
    else:
        results["queries"] = run_sample_queries(
            skel_result["skeleton"],
            skel_result["kv_store"],
            tree
        )
    
    # Generate report
    report_path = Path(f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    generate_report(results, report_path)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RNSR Benchmark & Demo")
    parser.add_argument("--pdf", type=Path, help="Path to PDF to benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick demo only")
    parser.add_argument("--query", type=str, action="append", help="Specific query to test")
    parser.add_argument("--check", action="store_true", help="Check environment only")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]RNSR - Recursive Neural-Symbolic Retriever[/bold cyan]\n"
        "Benchmark & Demo Tool",
        border_style="cyan",
    ))
    
    # Check environment
    if not check_environment():
        if not args.quick:
            return 1
    
    if args.check:
        return 0
    
    # Run appropriate benchmark
    if args.quick:
        run_quick_demo()
    elif args.pdf:
        if not args.pdf.exists():
            console.print(f"[red]File not found: {args.pdf}[/red]")
            return 1
        run_full_benchmark(args.pdf, args.query)
    else:
        # Default: run quick demo
        run_quick_demo()
    
    return 0


if __name__ == "__main__":
    try:
        # Check for rich
        import rich
    except ImportError:
        print("Installing rich for pretty output...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "rich"])
        import rich
    
    sys.exit(main())
