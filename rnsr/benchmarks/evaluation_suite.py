"""
RNSR Benchmark Suite - Comprehensive Evaluation Against Standard Baselines

This module runs RNSR against standard RAG benchmarks to validate
the claims in the research paper:

1. Tree traversal is more efficient than flat chunk retrieval
2. Hierarchical indexing preserves context better
3. Multi-hop reasoning benefits from structural navigation
4. Latent TOC extraction improves document understanding

Usage:
    python -m rnsr.benchmarks.evaluation_suite --dataset hotpotqa --samples 100
"""

from __future__ import annotations

import argparse
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog

from rnsr.benchmarks.standard_benchmarks import (
    BenchmarkLoader,
    BenchmarkDataset,
    BenchmarkQuestion,
    NaiveChunkRAG,
    SemanticChunkRAG,
    RAGASEvaluator,
    RAGASMetrics,
    MultiHopMetrics,
    evaluate_multihop,
    compare_rnsr_vs_baseline,
    ComparisonResult,
)
from rnsr.llm import LLMProvider

logger = structlog.get_logger(__name__)


# =============================================================================
# RNSR Wrapper for Benchmark Evaluation
# =============================================================================

@dataclass
class RNSRResult:
    """Result from RNSR system."""
    
    answer: str
    supporting_facts: list[str]
    nodes_visited: list[str]
    traversal_path: list[str]
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    tree_depth_reached: int
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # RLM-specific metrics (Section 2)
    rlm_metrics: dict[str, Any] = field(default_factory=dict)
    
    # Full execution trace
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RLMMetrics:
    """
    Metrics specific to RLM (Recursive Language Model) execution.
    
    Tracks Section 2 implementation effectiveness:
    - Decomposition quality (Section 2.2)
    - Variable stitching usage (Section 2.2)
    - Batch processing efficiency (Section 2.3)
    - REPL interaction patterns
    """
    
    # Query decomposition
    sub_questions_generated: int = 0
    sub_questions_answered: int = 0
    decomposition_method: str = "none"  # "llm", "pattern", "none"
    
    # Variable stitching
    variables_stored: int = 0
    variables_resolved: int = 0
    total_variable_chars: int = 0
    
    # Recursive execution
    sub_llm_calls: int = 0
    batch_calls: int = 0
    batch_efficiency: float = 0.0  # items_processed / api_calls
    
    # REPL execution
    repl_commands_executed: int = 0
    repl_errors: int = 0
    
    # Timing breakdown
    decomposition_time_s: float = 0.0
    navigation_time_s: float = 0.0
    synthesis_time_s: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sub_questions_generated": self.sub_questions_generated,
            "sub_questions_answered": self.sub_questions_answered,
            "decomposition_method": self.decomposition_method,
            "variables_stored": self.variables_stored,
            "variables_resolved": self.variables_resolved,
            "total_variable_chars": self.total_variable_chars,
            "sub_llm_calls": self.sub_llm_calls,
            "batch_calls": self.batch_calls,
            "batch_efficiency": self.batch_efficiency,
            "repl_commands_executed": self.repl_commands_executed,
            "repl_errors": self.repl_errors,
            "decomposition_time_s": self.decomposition_time_s,
            "navigation_time_s": self.navigation_time_s,
            "synthesis_time_s": self.synthesis_time_s,
        }


class RNSRBenchmarkAdapter:
    """
    Adapter to run RNSR on benchmark datasets.
    
    IMPORTANT: This adapter ALWAYS uses the full RLM (Recursive Language Model)
    pipeline. RLM is not optional - it IS RNSR. The key principles:
    
    1. Document stored as variable (DOC_VAR), not stuffed into prompt
    2. LLM navigates via summaries (skeleton index)
    3. Query decomposition for complex questions
    4. Variable stitching prevents context pollution
    
    For benchmarks that provide raw text (not PDFs), we build an ephemeral
    tree structure to enable full RLM processing.
    """
    
    def __init__(
        self,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.5-flash",
        max_iterations: int = 50,
        tot_selection_threshold: float = 0.4,
        tot_dead_end_threshold: float = 0.1,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        self.tot_selection_threshold = tot_selection_threshold
        self.tot_dead_end_threshold = tot_dead_end_threshold
        # Ingestion cache to avoid re-processing PDFs (thread-safe)
        self._ingestion_cache: dict[Path, tuple] = {}
        self._cache_lock = threading.Lock()
    
    def answer_from_pdf(
        self,
        question: str,
        pdf_path: Path,
        metadata: dict | None = None,
    ) -> RNSRResult:
        """
        Answer a question using RNSR's full RLM pipeline on a PDF.
        
        Uses:
        1. Font histogram ingestion → Document tree
        2. Skeleton index (summaries + KV store)
        3. Navigator agent with decomposition + variable stitching
        """
        from rnsr.ingestion import ingest_document
        from rnsr.indexing import build_skeleton_index
        from rnsr.agent import run_navigator
        
        start_total = time.perf_counter()
        
        # Check cache first (thread-safe)
        start_index = time.perf_counter()
        with self._cache_lock:
            cached = self._ingestion_cache.get(pdf_path)
        
        if cached:
            skeleton, kv_store = cached
            index_time = time.perf_counter() - start_index
            logger.debug("using_cached_ingestion", pdf=str(pdf_path))
        else:
            # Ingest and index
            result = ingest_document(pdf_path)
            skeleton, kv_store = build_skeleton_index(result.tree)
            index_time = time.perf_counter() - start_index
            # Store in cache
            with self._cache_lock:
                self._ingestion_cache[pdf_path] = (skeleton, kv_store)
            logger.debug("cached_ingestion", pdf=str(pdf_path))
        
        # Full RLM: Query with navigator (always uses decomposition + variable stitching)
        start_query = time.perf_counter()
        answer_result = run_navigator(
            question=question,
            skeleton=skeleton,
            kv_store=kv_store,
            max_iterations=self.max_iterations,
            metadata=metadata,
            tot_selection_threshold=self.tot_selection_threshold,
            tot_dead_end_threshold=self.tot_dead_end_threshold,
        )
        query_time = time.perf_counter() - start_query
        
        total_time = time.perf_counter() - start_total
        
        # Extract supporting facts from trace
        supporting_facts = []
        traversal_path = []
        max_depth = 0
        
        for entry in answer_result.get("trace", []):
            if entry.get("action") == "read_node":
                node_id = entry.get("node_id", "")
                supporting_facts.append(node_id)
            traversal_path.append(entry.get("node_type", "unknown"))
            max_depth = max(max_depth, entry.get("depth", 0))
        
        return RNSRResult(
            answer=answer_result.get("answer", ""),
            supporting_facts=supporting_facts,
            nodes_visited=answer_result.get("nodes_visited", []),
            traversal_path=traversal_path,
            retrieval_time_s=index_time,
            generation_time_s=query_time,
            total_time_s=total_time,
            tree_depth_reached=max_depth,
            trace=answer_result.get("trace", []),
            metadata={
                "confidence": answer_result.get("confidence", 0),
                "variables_used": len(answer_result.get("variables_used", [])),
            },
        )
    
    def answer_from_context(
        self,
        question: str,
        contexts: list[str],
        metadata: dict | None = None,
    ) -> RNSRResult:
        """
        Answer using pre-provided contexts (for benchmark datasets).
        
        This uses the FULL RLM pipeline:
        1. Build ephemeral tree from text contexts
        2. Create skeleton index with summaries
        3. Run navigator with decomposition + variable stitching
        
        This is NOT traditional RAG (stuffing context into prompt).
        The document is stored as DOC_VAR and navigated structurally.
        """
        from rnsr.ingestion.text_builder import build_tree_from_contexts
        from rnsr.indexing import build_skeleton_index
        from rnsr.agent import run_navigator
        
        start_total = time.perf_counter()
        metadata = metadata or {}
        
        # Check if this is a PDF-based benchmark (e.g. FinanceBench)
        if metadata and "pdf_path" in metadata and metadata["pdf_path"]:
             pdf_path_str = metadata["pdf_path"]
             if Path(pdf_path_str).exists():
                return self.answer_from_pdf(question, Path(pdf_path_str), metadata)
             # Fallback if PDF missing but context provided (rare for FB)
        
        # Step 1: Build ephemeral tree from benchmark contexts
        start_index = time.perf_counter()
        tree = build_tree_from_contexts(contexts, question)
        skeleton, kv_store = build_skeleton_index(tree)
        
        # Step 1b: Attach image to root node if available (for vision-augmented ToT)
        # This stores the image bytes on the tree node so expand_current_node()
        # can use VisionLLM to produce a text analysis during navigation.
        if metadata and metadata.get("image_bytes") and hasattr(kv_store, "put_image"):
            # Attach image to root node (single-page doc images like DocVQA)
            root_id = tree.root.id
            kv_store.put_image(root_id, metadata["image_bytes"])
            # Also attach to all leaf nodes so any traversal path finds it
            for node_id in skeleton:
                node = skeleton[node_id]
                if not node.child_ids:  # leaf node
                    kv_store.put_image(node_id, metadata["image_bytes"])
        
        index_time = time.perf_counter() - start_index
        
        # Step 2: Run full RLM navigator (decomposition + variable stitching)
        start_query = time.perf_counter()
        try:
            # Use Tree of Thoughts reasoning-based navigation (research paper Section 7.2)
            # Semantic search disabled by default per Section 9.1 (optional shortcut only)
            answer_result = run_navigator(
                question=question,
                skeleton=skeleton,
                kv_store=kv_store,
                max_iterations=self.max_iterations,
                use_semantic_search=False,  # ToT primary, embeddings optional
                metadata=metadata,  # Pass options for multiple-choice questions
                tot_selection_threshold=self.tot_selection_threshold,
                tot_dead_end_threshold=self.tot_dead_end_threshold,
            )
            answer = answer_result.get("answer", "")

            # Fallback 1: header-match — present all headers to LLM, pick relevant nodes
            if (
                (answer or "").strip().startswith("No relevant content found")
                and not answer_result.get("variables_used")
            ):
                try:
                    # Header-match fallback is now built into RLMNavigator
                    # (invoked automatically via run_navigator delegation).
                    # This block is kept for explicit retry with fresh state.
                    from rnsr.agent.rlm_navigator import RLMNavigator, RLMConfig
                    hm_nav = RLMNavigator(
                        skeleton=skeleton, kv_store=kv_store,
                        config=RLMConfig(max_iterations=20, enable_verification=False),
                    )
                    hm_result = hm_nav.navigate(question, metadata=metadata)
                    hm_answer = hm_result.get("answer", "")
                    if hm_answer and not hm_answer.strip().startswith("No relevant content found"):
                        answer_result["answer"] = hm_answer
                        answer_result["variables_used"] = hm_result.get("variables_used", [])
                        answer_result["confidence"] = hm_result.get("confidence", 0.0)
                        answer = hm_answer
                        logger.info("used_header_match_fallback", question_preview=question[:60])
                except Exception as hm_e:
                    logger.warning("header_match_fallback_failed", error=str(hm_e))

            # Fallback 2: semantic search — retry with embeddings (existing fallback)
            if (
                (answer or "").strip().startswith("No relevant content found")
                and not answer_result.get("variables_used")
            ):
                try:
                    fallback_result = run_navigator(
                        question=question,
                        skeleton=skeleton,
                        kv_store=kv_store,
                        max_iterations=self.max_iterations,
                        use_semantic_search=True,
                        metadata=metadata,
                        tot_selection_threshold=self.tot_selection_threshold,
                        tot_dead_end_threshold=self.tot_dead_end_threshold,
                    )
                    fallback_answer = fallback_result.get("answer", "")
                    if fallback_answer and not (fallback_answer or "").strip().startswith("No relevant content found"):
                        answer_result = fallback_result
                        answer = fallback_answer
                        logger.debug("used_semantic_search_fallback", question_preview=question[:60])
                except Exception as fb_e:
                    logger.warning("semantic_search_fallback_failed", error=str(fb_e))
        except Exception as e:
            logger.warning("rnsr_navigation_failed", error=str(e))
            answer = f"Error: {str(e)}"
            answer_result = {"trace": [], "nodes_visited": [], "confidence": 0}
        
        query_time = time.perf_counter() - start_query
        total_time = time.perf_counter() - start_total
        
        # Extract trace information
        supporting_facts = []
        traversal_path = []
        max_depth = 0
        
        for entry in answer_result.get("trace", []):
            if entry.get("action") == "read_node":
                supporting_facts.append(entry.get("node_id", ""))
            traversal_path.append(entry.get("node_type", "unknown"))
            max_depth = max(max_depth, entry.get("depth", 0))
        
        return RNSRResult(
            answer=answer,
            supporting_facts=supporting_facts,
            nodes_visited=answer_result.get("nodes_visited", []),
            traversal_path=traversal_path,
            retrieval_time_s=index_time,
            generation_time_s=query_time,
            total_time_s=total_time,
            tree_depth_reached=max_depth,
            trace=answer_result.get("trace", []),
            metadata={
                "num_contexts": len(contexts),
                "confidence": answer_result.get("confidence", 0),
                "variables_used": len(answer_result.get("variables_used", [])),
                "rlm_mode": True,  # Always true now
            },
        )
    
    def _normalize_multiple_choice(self, answer: str, options: list[str]) -> str:
        """Normalize multiple choice answer to match option text."""
        answer_lower = answer.lower().strip()
        
        for i, opt in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D
            
            # Match patterns like "A.", "A)", "(A)", "A. answer text"
            if (answer_lower.startswith(f"{letter.lower()}.") or 
                answer_lower.startswith(f"{letter.lower()})") or 
                answer_lower.startswith(f"({letter.lower()})")):
                return opt
            
            # Check if option number format
            if answer_lower.startswith(f"{i+1}.") or answer_lower.startswith(f"({i+1})"):
                return opt
            
            # Exact match (case insensitive)
            if answer_lower == opt.lower():
                return opt
            
            # Check if answer contains the option text
            if opt.lower() in answer_lower:
                return opt
        
        return answer  # Return original if no match


# =============================================================================
# Baseline RAG Implementations (for fair comparison)
# =============================================================================

@dataclass
class BaselineResult:
    """Result from a baseline RAG approach."""
    answer: str
    retrieved_chunks: list[str]
    total_time_s: float
    method: str


class NaiveChunkBaseline:
    """
    Naive chunking baseline using the SAME LLM as RNSR for fair comparison.
    
    This chunks the context into fixed-size segments, retrieves top-k by
    simple keyword overlap, and generates an answer.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.5-flash",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_provider = llm_provider
        self.llm_model = llm_model
    
    def name(self) -> str:
        return f"naive_chunk_{self.chunk_size}"
    
    def answer_from_context(
        self,
        question: str,
        contexts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> BaselineResult:
        """Answer using naive chunking on the provided context."""
        from rnsr.llm import get_llm
        
        start_total = time.perf_counter()
        
        # Combine all context
        full_text = "\n\n".join(contexts)
        
        # Chunk the text naively
        chunks = []
        for i in range(0, len(full_text), self.chunk_size - self.chunk_overlap):
            chunk = full_text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        # Simple retrieval by keyword overlap
        question_words = set(question.lower().split())
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(question_words & chunk_words) / max(len(question_words), 1)
            scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        retrieved = [c for _, c in scored_chunks[:self.top_k]]
        
        # Generate answer using the SAME LLM as RNSR
        logger.info("provider_detected", provider=self.llm_provider)
        logger.debug("initializing_llm", provider=self.llm_provider, model=self.llm_model)
        llm = get_llm(provider=LLMProvider(self.llm_provider), model=self.llm_model)
        
        # Check if this is multiple choice
        options = metadata.get("options", []) if metadata else []
        if options:
            options_text = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(options)])
            prompt = f"""Answer this multiple-choice question based on the provided text.

Retrieved chunks:
{chr(10).join(retrieved)}

Question: {question}

Options:
{options_text}

Respond with ONLY the text of the correct option, nothing else."""
        else:
            prompt = f"""Answer this question based on the provided text. Be concise.

Retrieved chunks:
{chr(10).join(retrieved)}

Question: {question}

Answer:"""
        
        try:
            response = llm.complete(prompt)
            answer = str(response).strip()
            
            # Match to options if needed
            if options:
                answer_lower = answer.lower()
                for opt in options:
                    if opt.lower() in answer_lower:
                        answer = opt
                        break
        except Exception as e:
            logger.warning("baseline_llm_failed", error=str(e))
            answer = f"Error: {str(e)}"
        
        total_time = time.perf_counter() - start_total
        
        return BaselineResult(
            answer=answer,
            retrieved_chunks=retrieved,
            total_time_s=total_time,
            method=self.name(),
        )


class SemanticChunkBaseline:
    """Semantic chunking baseline - splits on paragraph boundaries."""
    
    def __init__(
        self,
        top_k: int = 5,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.5-flash",
    ):
        self.top_k = top_k
        self.llm_provider = llm_provider
        self.llm_model = llm_model
    
    def name(self) -> str:
        return "semantic_chunk"
    
    def answer_from_context(
        self,
        question: str,
        contexts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> BaselineResult:
        """Answer using paragraph-based chunking."""
        from rnsr.llm import get_llm
        
        start_total = time.perf_counter()
        
        # Split by paragraphs
        chunks = []
        for ctx in contexts:
            paragraphs = ctx.split("\n\n")
            for para in paragraphs:
                if para.strip() and len(para.strip()) > 50:
                    chunks.append(para.strip())
        
        # Retrieve by keyword overlap
        question_words = set(question.lower().split())
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(question_words & chunk_words) / max(len(question_words), 1)
            scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        retrieved = [c for _, c in scored_chunks[:self.top_k]]
        
        # Generate answer
        logger.info("provider_detected", provider=self.llm_provider)
        logger.debug("initializing_llm", provider=self.llm_provider, model=self.llm_model)
        llm = get_llm(provider=LLMProvider(self.llm_provider), model=self.llm_model)
        
        options = metadata.get("options", []) if metadata else []
        if options:
            options_text = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(options)])
            prompt = f"""Answer this multiple-choice question based on the provided text.

Retrieved paragraphs:
{chr(10).join(retrieved)}

Question: {question}

Options:
{options_text}

Respond with ONLY the text of the correct option, nothing else."""
        else:
            prompt = f"""Answer this question based on the provided text. Be concise.

Retrieved paragraphs:
{chr(10).join(retrieved)}

Question: {question}

Answer:"""
        
        try:
            response = llm.complete(prompt)
            answer = str(response).strip()
            
            if options:
                answer_lower = answer.lower()
                for opt in options:
                    if opt.lower() in answer_lower:
                        answer = opt
                        break
        except Exception as e:
            logger.warning("baseline_llm_failed", error=str(e))
            answer = f"Error: {str(e)}"
        
        total_time = time.perf_counter() - start_total
        
        return BaselineResult(
            answer=answer,
            retrieved_chunks=retrieved,
            total_time_s=total_time,
            method=self.name(),
        )


# =============================================================================
# Evaluation Suite
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""
    
    datasets: list[str] = field(default_factory=lambda: ["hotpotqa"])
    max_samples: int = field(default=100)
    baselines: list[str] = field(default_factory=lambda: ["naive_chunk_512"])
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    run_ragas: bool = field(default=True)
    save_predictions: bool = field(default=True)
    llm_provider: str = field(default="gemini")
    llm_model: str = field(default="gemini-2.5-flash")
    chaos_mode: bool = field(default=False)
    chaos_distractors: int = field(default=3)
    tot_selection_threshold: float = field(default=0.4)
    tot_dead_end_threshold: float = field(default=0.1)
    parallel_workers: int = field(default=1)  # Number of parallel workers for question processing


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    
    timestamp: str
    config: dict[str, Any]
    dataset_results: dict[str, dict[str, Any]]
    comparisons: list[dict[str, Any]]
    summary: dict[str, Any]
    
    @property
    def overall_accuracy(self) -> float:
        """Calculate overall accuracy across all datasets."""
        total_acc = 0.0
        count = 0
        for results in self.dataset_results.values():
            metrics = results.get("rnsr_metrics", {})
            if "accuracy" in metrics:
                total_acc += metrics["accuracy"]
                count += 1
            elif "exact_match" in metrics:
                total_acc += metrics["exact_match"]
                count += 1
        return total_acc / max(count, 1)
    
    @property
    def avg_latency_s(self) -> float:
        """Calculate average latency across all datasets."""
        total_latency = 0.0
        count = 0
        for results in self.dataset_results.values():
            metrics = results.get("rnsr_metrics", {})
            if "avg_time_s" in metrics:
                total_latency += metrics["avg_time_s"]
                count += 1
        return total_latency / max(count, 1)
    
    def save(self, path: Path) -> None:
        """Save report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("RNSR BENCHMARK EVALUATION REPORT")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Datasets evaluated: {list(self.dataset_results.keys())}")
        
        for dataset, results in self.dataset_results.items():
            print(f"\n--- {dataset} ---")
            
            rnsr_metrics = results.get("rnsr_metrics", {})
            for metric, value in rnsr_metrics.items():
                if isinstance(value, float):
                    print(f"  RNSR {metric}: {value:.3f}")
                else:
                    print(f"  RNSR {metric}: {value}")
            
            # Print RLM-specific metrics if available
            rlm_metrics = results.get("rlm_metrics", {})
            if rlm_metrics:
                print("\n  📊 RLM Metrics:")
                print(f"    Sub-questions generated: {rlm_metrics.get('total_sub_questions', 0)}")
                print(f"    Variables stored: {rlm_metrics.get('total_variables', 0)}")
                print(f"    Batch calls: {rlm_metrics.get('total_batch_calls', 0)}")
                print(f"    Avg decomposition time: {rlm_metrics.get('avg_decomposition_time', 0):.3f}s")
                print(f"    Avg sub-task time: {rlm_metrics.get('avg_sub_task_time', 0):.3f}s")
            
            for baseline, metrics in results.get("baseline_metrics", {}).items():
                print(f"\n  {baseline}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.3f}")
                    else:
                        print(f"    {metric}: {value}")
        
        print("\n" + "-" * 70)
        print("IMPROVEMENTS OVER BASELINES")
        print("-" * 70)
        
        for comp in self.comparisons:
            print(f"\nvs {comp['baseline_name']} on {comp['dataset_name']}:")
            for metric, delta in comp.get("improvement", {}).items():
                rel = comp.get("relative_improvement", {}).get(metric, 0) * 100
                print(f"  {metric}: {delta:+.3f} ({rel:+.1f}%)")
        
        print("\n" + "=" * 70)


class EvaluationSuite:
    """
    Run comprehensive RNSR evaluation against standard benchmarks.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.rnsr = RNSRBenchmarkAdapter(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            tot_selection_threshold=config.tot_selection_threshold,
            tot_dead_end_threshold=config.tot_dead_end_threshold,
        )
        self.ragas_evaluator = RAGASEvaluator(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
        ) if config.run_ragas else None
        
        # Initialize baselines
        self.baselines = {}
        for baseline_name in config.baselines:
            if baseline_name.startswith("naive_chunk"):
                chunk_size = int(baseline_name.split("_")[-1])
                self.baselines[baseline_name] = NaiveChunkBaseline(
                    chunk_size=chunk_size,
                    llm_provider=config.llm_provider,
                    llm_model=config.llm_model,
                )
            elif baseline_name == "semantic_chunk":
                self.baselines[baseline_name] = SemanticChunkBaseline(
                    llm_provider=config.llm_provider,
                    llm_model=config.llm_model,
                )
    
    def load_dataset(self, name: str) -> BenchmarkDataset:
        """Load a benchmark dataset by name."""
        if name == "hotpotqa":
            return BenchmarkLoader.load_hotpotqa(max_samples=self.config.max_samples)
        elif name.startswith("musique"):
            variant = "full" if "full" in name else "ans"
            return BenchmarkLoader.load_musique(variant=variant, max_samples=self.config.max_samples)
        elif name.startswith("beir_"):
            dataset_name = name.replace("beir_", "")
            return BenchmarkLoader.load_beir_dataset(dataset_name, max_samples=self.config.max_samples)
        elif name == "qasper":
            return BenchmarkLoader.load_qasper(max_samples=self.config.max_samples)
        elif name == "quality":
            return BenchmarkLoader.load_quality(max_samples=self.config.max_samples)
        elif name == "financebench":
            return BenchmarkLoader.load_financebench(split="train", max_samples=self.config.max_samples)
        elif name == "narrativeqa":
            return BenchmarkLoader.load_narrative_qa(max_samples=self.config.max_samples)
        else:
            raise ValueError(f"Unknown dataset: {name}. Available: hotpotqa, musique_ans, musique_full, qasper, quality, narrativeqa, beir_*")
    
    def evaluate_rnsr_on_dataset(
        self,
        dataset: BenchmarkDataset,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Evaluate RNSR on a benchmark dataset.
        
        Returns:
            predictions: List of prediction dicts
            metrics: Aggregated metrics dict (includes RLM metrics if enabled)
        """
        predictions = []
        
        logger.info(
            "evaluating_rnsr", 
            dataset=dataset.name, 
            questions=len(dataset.questions),
            workers=self.config.parallel_workers,
        )
        
        _metrics_set = set(dataset.metrics or []) if getattr(dataset, "metrics", None) else set()
        use_short_answer = bool(
            _metrics_set & {"answer_f1", "f1", "exact_match", "anls"}
        )

        def process_question(idx_question: tuple[int, "BenchmarkQuestion"]) -> dict[str, Any]:
            """Process a single question (helper for parallel execution)."""
            i, question = idx_question
            logger.debug("processing_question", index=i, question=question.question[:50])
            meta = dict(question.metadata or {})
            if use_short_answer:
                meta["use_short_answer"] = True
            # Pass reasoning_type so arithmetic-aware synthesis can detect question type
            if question.reasoning_type:
                meta.setdefault("reasoning_type", question.reasoning_type)
            try:
                result = self.rnsr.answer_from_context(
                    question=question.question,
                    contexts=question.context,
                    metadata=meta,
                )
                
                pred_entry = {
                    "id": question.id,
                    "question": question.question,
                    "answer": result.answer,
                    "supporting_facts": result.supporting_facts,
                    "nodes_visited": result.nodes_visited,
                    "time_s": result.total_time_s,
                    "tree_depth": result.tree_depth_reached,
                    "trace": result.trace,
                    "metadata": result.metadata,
                }
                
                if result.metadata and result.metadata.get("rlm_mode"):
                    pred_entry["rlm_metrics"] = result.metadata.get("rlm_metrics", {})
                
                return pred_entry
                
            except Exception as e:
                logger.error("question_failed", error=str(e), question_id=question.id)
                return {
                    "id": question.id,
                    "question": question.question,
                    "answer": "",
                    "supporting_facts": [],
                    "error": str(e),
                }
        
        # Parallel or sequential execution based on config
        if self.config.parallel_workers > 1:
            # Parallel execution with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = {
                    executor.submit(process_question, (i, q)): i 
                    for i, q in enumerate(dataset.questions)
                }
                
                completed = 0
                for future in as_completed(futures):
                    pred = future.result()
                    predictions.append(pred)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info("progress", completed=completed, total=len(dataset.questions))
            
            # Sort by original order (id may not be sequential)
            predictions.sort(key=lambda p: str(p.get("id", "")))
        else:
            # Sequential execution (original behavior)
            for i, question in enumerate(dataset.questions):
                pred = process_question((i, question))
                predictions.append(pred)
        
        # Compute metrics
        metrics: dict[str, Any] = {}
        metric_set = set(dataset.metrics or [])
        if metric_set & {"answer_f1", "answer_em", "accuracy", "f1", "exact_match"}:
            multi_hop = evaluate_multihop(predictions, dataset.questions)
            metrics = multi_hop.to_dict()
        elif "anls" in metric_set:
            # ANLS scoring for DocVQA
            from rnsr.benchmarks.docvqa_bench import _compute_anls
            anls_scores = []
            for pred, q in zip(predictions, dataset.questions):
                pred_answer = pred.get("answer", "")
                # Ground truth may have multiple acceptable answers (stored in metadata)
                gt_answers = (q.metadata or {}).get("all_answers") or [q.answer]
                max_anls = max((_compute_anls(pred_answer, gt) for gt in gt_answers), default=0.0)
                anls_scores.append(max_anls)
            avg_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0
            metrics = {"anls": avg_anls, "f1": avg_anls}  # Report ANLS as the primary metric
        else:
            # Retrieval metrics (for BEIR)
            metrics = self._compute_retrieval_metrics(predictions, dataset.questions)
        
        # Add timing metrics
        times = [p.get("time_s", 0) for p in predictions if "time_s" in p]
        if times:
            metrics["mean_time_s"] = sum(times) / len(times)
            metrics["total_time_s"] = sum(times)
        
        # Aggregate RLM metrics if in RLM mode
        rlm_preds = [p for p in predictions if p.get("rlm_metrics")]
        if rlm_preds:
            rlm_agg = {
                "total_sub_questions": sum(p["rlm_metrics"].get("sub_questions_generated", 0) for p in rlm_preds),
                "total_variables": sum(p["rlm_metrics"].get("variables_stored", 0) for p in rlm_preds),
                "total_batch_calls": sum(p["rlm_metrics"].get("batch_calls_made", 0) for p in rlm_preds),
                "total_llm_calls": sum(p["rlm_metrics"].get("total_llm_calls", 0) for p in rlm_preds),
                "avg_decomposition_time": sum(p["rlm_metrics"].get("decomposition_time_s", 0) for p in rlm_preds) / len(rlm_preds),
                "avg_sub_task_time": sum(p["rlm_metrics"].get("sub_task_time_s", 0) for p in rlm_preds) / len(rlm_preds),
                "avg_stitching_time": sum(p["rlm_metrics"].get("stitching_time_s", 0) for p in rlm_preds) / len(rlm_preds),
            }
            metrics["rlm_metrics"] = rlm_agg
        
        return predictions, metrics
    
    def _compute_retrieval_metrics(
        self,
        predictions: list[dict],
        questions: list[BenchmarkQuestion],
    ) -> dict[str, float]:
        """Compute retrieval metrics (precision, recall, etc.)."""
        # Simplified retrieval metrics
        precisions = []
        recalls = []
        
        for pred, q in zip(predictions, questions):
            retrieved = set(pred.get("nodes_visited", []))
            relevant = set(q.supporting_facts) if q.supporting_facts else set()
            
            if retrieved and relevant:
                prec = len(retrieved & relevant) / len(retrieved)
                rec = len(retrieved & relevant) / len(relevant)
                precisions.append(prec)
                recalls.append(rec)
        
        n = len(precisions) or 1
        precision = sum(precisions) / n
        recall = sum(recalls) / n
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    def evaluate_baseline_on_dataset(
        self,
        baseline: NaiveChunkBaseline | SemanticChunkBaseline,
        dataset: BenchmarkDataset,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """
        Evaluate a baseline RAG approach on a benchmark dataset.
        
        Uses the same LLM as RNSR for fair comparison.
        """
        predictions = []
        
        logger.info(
            "evaluating_baseline_on_dataset",
            baseline=baseline.name(),
            dataset=dataset.name,
            questions=len(dataset.questions),
        )
        
        for i, question in enumerate(dataset.questions):
            logger.debug(
                "baseline_processing_question",
                baseline=baseline.name(),
                index=i,
                question=question.question[:50],
            )
            
            try:
                result = baseline.answer_from_context(
                    question=question.question,
                    contexts=question.context,
                    metadata=question.metadata,
                )
                
                predictions.append({
                    "id": question.id,
                    "question": question.question,
                    "answer": result.answer,
                    "time_s": result.total_time_s,
                    "method": result.method,
                })
                
            except Exception as e:
                logger.error(
                    "baseline_question_failed",
                    error=str(e),
                    question_id=question.id,
                    baseline=baseline.name(),
                )
                predictions.append({
                    "id": question.id,
                    "question": question.question,
                    "answer": "",
                    "error": str(e),
                    "method": baseline.name(),
                })
        
        # Compute metrics (same as RNSR)
        metric_set = set(dataset.metrics or [])
        if metric_set & {"answer_f1", "answer_em", "accuracy", "f1", "exact_match"}:
            multi_hop = evaluate_multihop(predictions, dataset.questions)
            metrics = multi_hop.to_dict()
        elif "anls" in metric_set:
            from rnsr.benchmarks.docvqa_bench import _compute_anls
            anls_scores = []
            for pred, q in zip(predictions, dataset.questions):
                pred_answer = pred.get("answer", "")
                gt_answers = (q.metadata or {}).get("all_answers") or [q.answer]
                max_anls = max((_compute_anls(pred_answer, gt) for gt in gt_answers), default=0.0)
                anls_scores.append(max_anls)
            avg_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0
            metrics = {"anls": avg_anls, "f1": avg_anls}
        else:
            metrics = {"f1": 0.0}  # Retrieval metrics not applicable
        
        # Add timing metrics
        times = [p.get("time_s", 0) for p in predictions if "time_s" in p]
        if times:
            metrics["mean_time_s"] = sum(times) / len(times)
            metrics["total_time_s"] = sum(times)
        
        return predictions, metrics

    def run(self) -> EvaluationReport:
        """Run full evaluation suite."""
        logger.info("starting_evaluation_suite", datasets=self.config.datasets)
        
        dataset_results = {}
        all_comparisons = []
        
        for dataset_name in self.config.datasets:
            logger.info("loading_dataset", name=dataset_name)
            dataset = self.load_dataset(dataset_name)
            
            if not dataset.questions:
                logger.warning("empty_dataset", name=dataset_name)
                continue
            
            # Application of Chaos Mode
            if self.config.chaos_mode and "financebench" in dataset_name.lower():
                from rnsr.benchmarks.pdf_merger import PDFMerger
                logger.info("applying_chaos_mode", dataset=dataset_name)
                
                # Collect all available PDFs to use as distractors
                all_pdfs = set()
                for q in dataset.questions:
                    if q.metadata and q.metadata.get("pdf_path"):
                        p = Path(q.metadata["pdf_path"])
                        if p.exists():
                            all_pdfs.add(p)
                            
                pool_of_pdfs = list(all_pdfs)
                if len(pool_of_pdfs) < 2:
                    logger.warning("not_enough_pdfs_for_chaos", count=len(pool_of_pdfs))
                else:
                    chaos_dir = self.config.output_dir / "chaos_data"
                    dataset.questions = PDFMerger.create_chaos_dataset(
                        dataset.questions,
                        pool_of_pdfs,
                        chaos_dir,
                        num_distractors=self.config.chaos_distractors
                    )
                    dataset.name = f"{dataset.name}-CHAOS"
                    logger.info("chaos_mode_applied", new_size=len(dataset.questions))

            # Evaluate RNSR
            predictions, rnsr_metrics = self.evaluate_rnsr_on_dataset(dataset)
            
            # Evaluate baselines using the same LLM for fair comparison
            baseline_metrics = {}
            for baseline_name, baseline in self.baselines.items():
                logger.info("evaluating_baseline", baseline=baseline_name, dataset=dataset_name)
                baseline_preds, base_metrics = self.evaluate_baseline_on_dataset(
                    baseline, dataset
                )
                baseline_metrics[baseline_name] = base_metrics
            
            dataset_results[dataset_name] = {
                "rnsr_metrics": rnsr_metrics,
                "baseline_metrics": baseline_metrics,
                "num_questions": len(dataset.questions),
                "predictions": predictions if self.config.save_predictions else [],
            }
            
            # Generate comparisons
            for baseline_name, base_metrics in baseline_metrics.items():
                comparison = compare_rnsr_vs_baseline(
                    rnsr_metrics,
                    base_metrics,
                    dataset_name,
                    baseline_name,
                )
                all_comparisons.append(asdict(comparison))
        
        # Build summary
        summary = self._build_summary(dataset_results, all_comparisons)
        
        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=asdict(self.config),
            dataset_results=dataset_results,
            comparisons=all_comparisons,
            summary=summary,
        )
        
        # Save report
        report_path = self.config.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report.save(report_path)
        logger.info("report_saved", path=str(report_path))
        
        return report
    
    def _build_summary(
        self,
        dataset_results: dict,
        comparisons: list[dict],
    ) -> dict[str, Any]:
        """Build summary statistics."""
        # Aggregate metrics across datasets
        all_f1s = []
        all_times = []
        
        for results in dataset_results.values():
            rnsr = results.get("rnsr_metrics", {})
            if "answer_f1" in rnsr:
                all_f1s.append(rnsr["answer_f1"])
            if "mean_time_s" in rnsr:
                all_times.append(rnsr["mean_time_s"])
        
        # Average improvements over baselines
        avg_improvements = {}
        for comp in comparisons:
            for metric, delta in comp.get("improvement", {}).items():
                if metric not in avg_improvements:
                    avg_improvements[metric] = []
                avg_improvements[metric].append(delta)
        
        for metric in avg_improvements:
            vals = avg_improvements[metric]
            avg_improvements[metric] = sum(vals) / len(vals) if vals else 0
        
        return {
            "datasets_evaluated": len(dataset_results),
            "total_questions": sum(r.get("num_questions", 0) for r in dataset_results.values()),
            "mean_answer_f1": sum(all_f1s) / len(all_f1s) if all_f1s else 0,
            "mean_time_s": sum(all_times) / len(all_times) if all_times else 0,
            "avg_improvement_over_baselines": avg_improvements,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run RNSR against standard RAG benchmarks"
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["hotpotqa"],
        help="Datasets to evaluate (hotpotqa, musique_ans, beir_nfcorpus, etc.)"
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Max samples per dataset"
    )
    
    parser.add_argument(
        "--baselines", "-b",
        nargs="+",
        default=["naive_chunk_512"],
        help="Baselines to compare against"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory"
    )
    
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS evaluation"
    )
    
    parser.add_argument(
        "--llm-provider", "-p",
        type=str,
        default="gemini",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider to use (default: gemini)"
    )
    
    parser.add_argument(
        "--llm-model", "-m",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model name (default: gemini-2.5-flash)"
    )
    
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="Enable chaos mode (merge PDFs with random distractors)"
    )

    parser.add_argument(
        "--tot-threshold",
        type=float,
        default=0.4,
        help="ToT selection threshold (default: 0.4)"
    )

    parser.add_argument(
        "--tot-dead-end",
        type=float,
        default=0.1,
        help="ToT dead end threshold (default: 0.1)"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers for processing questions (default: 1, sequential)"
    )
    
    args = parser.parse_args()
    
    # RNSR always uses the full RLM flow - no mode switching needed
    config = EvaluationConfig(
        datasets=args.datasets,
        max_samples=args.samples,
        baselines=args.baselines,
        output_dir=args.output,
        run_ragas=not args.no_ragas,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        chaos_mode=args.chaos,
        tot_selection_threshold=args.tot_threshold,
        tot_dead_end_threshold=args.tot_dead_end,
        parallel_workers=args.workers,
    )
    
    suite = EvaluationSuite(config)
    report = suite.run()
    report.print_summary()


if __name__ == "__main__":
    main()
