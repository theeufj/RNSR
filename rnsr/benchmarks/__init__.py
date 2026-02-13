"""
RNSR Benchmarking Suite

Measures:
1. Performance: Ingestion speed, query latency, memory usage
2. Quality: Retrieval accuracy, answer relevance (requires ground truth)
3. Comparison: RNSR vs baseline chunking approaches
4. Standard Benchmarks: HotpotQA, MuSiQue, BEIR, RAGAS
5. Comprehensive: All navigator types (standard, RLM, vision, hybrid)

Standard RAG Benchmarks:
- HotpotQA: Multi-hop question answering (EMNLP 2018)
- MuSiQue: Compositional multi-hop QA (TACL 2022)
- BEIR: Information retrieval benchmark (NeurIPS 2021)
- RAGAS: RAG evaluation metrics (faithfulness, relevance, etc.)

Comprehensive Benchmark (PageIndex/RLM-inspired):
- FinanceBench-style: Financial document QA
- OOLONG-style: Long context aggregation
- Multi-hop: Complex relational queries
"""

from rnsr.benchmarks.performance import (
    PerformanceBenchmark,
    BenchmarkResult,
    run_ingestion_benchmark,
    run_query_benchmark,
)
from rnsr.benchmarks.quality import (
    QualityBenchmark,
    QualityMetrics,
    evaluate_retrieval,
)
from rnsr.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    run_full_benchmark,
)
from rnsr.benchmarks.standard_benchmarks import (
    # Baselines
    NaiveChunkRAG,
    SemanticChunkRAG,
    BaselineResult,
    # Benchmark datasets
    BenchmarkLoader,
    BenchmarkDataset,
    BenchmarkQuestion,
    # RAGAS metrics
    RAGASEvaluator,
    RAGASMetrics,
    # Multi-hop metrics
    MultiHopMetrics,
    evaluate_multihop,
    # Comparison
    compare_rnsr_vs_baseline,
)
from rnsr.benchmarks.evaluation_suite import (
    EvaluationSuite,
    EvaluationConfig,
    EvaluationReport,
    RNSRBenchmarkAdapter,
)
from rnsr.benchmarks.comprehensive_benchmark import (
    # Comprehensive benchmark for all navigator types
    ComprehensiveBenchmarkRunner,
    ComprehensiveBenchmarkReport,
    BenchmarkTestCase,
    MethodResults,
    run_comprehensive_benchmark,
    quick_benchmark,
    # Standard test suites
    get_financebench_cases,
    get_oolong_style_cases,
    get_multi_hop_cases,
)

# New benchmark loaders (Phase 1 - generalization proof)
from rnsr.benchmarks.multihiertt_bench import MultiHierttLoader
from rnsr.benchmarks.tatqa_bench import TATQALoader
from rnsr.benchmarks.qasper_bench import QASPERLoader
from rnsr.benchmarks.docvqa_bench import DocVQALoader, _compute_anls

# Feature benchmarks (timeline + contradiction)
from rnsr.benchmarks.timeline_bench import TimelineBenchLoader
from rnsr.benchmarks.contradiction_bench import ContradictionBenchLoader

__all__ = [
    # Performance
    "PerformanceBenchmark",
    "BenchmarkResult",
    "run_ingestion_benchmark",
    "run_query_benchmark",
    # Quality
    "QualityBenchmark",
    "QualityMetrics",
    "evaluate_retrieval",
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "run_full_benchmark",
    # Standard Benchmarks
    "NaiveChunkRAG",
    "SemanticChunkRAG",
    "BaselineResult",
    "BenchmarkLoader",
    "BenchmarkDataset",
    "BenchmarkQuestion",
    "RAGASEvaluator",
    "RAGASMetrics",
    "MultiHopMetrics",
    "evaluate_multihop",
    "compare_rnsr_vs_baseline",
    # Evaluation Suite
    "EvaluationSuite",
    "EvaluationConfig",
    "EvaluationReport",
    "RNSRBenchmarkAdapter",
    # Comprehensive Benchmark (State-of-the-Art)
    "ComprehensiveBenchmarkRunner",
    "ComprehensiveBenchmarkReport",
    "BenchmarkTestCase",
    "MethodResults",
    "run_comprehensive_benchmark",
    "quick_benchmark",
    "get_financebench_cases",
    "get_oolong_style_cases",
    "get_multi_hop_cases",
    # New benchmark loaders
    "MultiHierttLoader",
    "TATQALoader",
    "QASPERLoader",
    "DocVQALoader",
    "_compute_anls",
    # Feature benchmarks
    "TimelineBenchLoader",
    "ContradictionBenchLoader",
]
