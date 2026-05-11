"""
RNSR - Recursive Neural-Symbolic Retriever

State-of-the-art document retrieval system combining:
- PageIndex: Vectorless, reasoning-based tree search
- RLMs: REPL environment with recursive sub-LLM calls
- Vision: OCR-free image-based document analysis

This is the hybrid recursive visual-symbolic retriever that achieves
superior performance on complex document understanding tasks.

Key Features:
- Font Histogram Algorithm (NOT vision models for structure)
- Recursive XY-Cut (Visual-geometric segmentation)
- Hierarchical Clustering (Multi-resolution topics)
- Skeleton Index pattern (summaries + KV store)
- Pointer-based Variable Stitching (prevents context pollution)
- Pre-LLM Filtering (keyword/regex before expensive ToT)
- Deep Recursive Sub-LLM Calls (configurable depth)
- Answer Verification (sub-LLM validation)
- Vision-based Retrieval (OCR-free page image analysis)
- Hybrid Text+Vision Mode (best of both worlds)
- Multi-provider LLM support (OpenAI, Anthropic, Gemini)

Usage:
    from rnsr import RNSRClient

    # Auto-detect provider from env vars or .env file
    client = RNSRClient()

    # Pass API key directly (recommended for PyPI installs)
    client = RNSRClient(api_key="your-key", llm_provider="gemini")

    # Explicit provider + model, key from env
    client = RNSRClient(llm_provider="anthropic", llm_model="claude-sonnet-4-5")

    # Simple one-line Q&A
    answer = client.ask("contract.pdf", "What are the payment terms?")

    # Advanced RLM navigation with full features
    result = client.ask_advanced(
        "complex_report.pdf",
        "Compare liability clauses in sections 5 and 8",
        enable_verification=True,
        max_recursion_depth=3,
    )

    # Low-level API
    from rnsr import ingest_document, build_skeleton_index, run_rlm_navigator

    result = ingest_document("contract.pdf")
    skeleton, kv_store = build_skeleton_index(result.tree)
    answer = run_rlm_navigator("What are the terms?", skeleton, kv_store)

LLM Provider Configuration:
    1. Pass api_key directly to RNSRClient() or DocumentStore()
    2. Place a .env file in your working directory
    3. Set environment variables:
       - GOOGLE_API_KEY (Gemini)
       - OPENAI_API_KEY (OpenAI)
       - ANTHROPIC_API_KEY (Anthropic)
"""

__version__ = "0.5.0"

# Re-export main entry points
from rnsr.ingestion import ingest_document, IngestionResult
from rnsr.ingestion.pipeline import ingest_document_enhanced
from rnsr.indexing import build_skeleton_index, SQLiteKVStore, InMemoryKVStore
from rnsr.indexing import save_index, load_index, get_index_info, list_indexes
from rnsr.indexing.knowledge_graph import KnowledgeGraph, InMemoryKnowledgeGraph
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode
from rnsr.agent import (
    run_navigator,
    VariableStore,
    # RLM Navigator (State-of-the-Art)
    RLMNavigator,
    RLMConfig,
    run_rlm_navigator,
    create_rlm_navigator,
    PreFilterEngine,
    RecursiveSubLLMEngine,
    AnswerVerificationEngine,
)
from rnsr.document_store import DocumentStore
from rnsr.client import RNSRClient
from rnsr.llm import get_llm, get_embed_model, LLMProvider

__all__ = [
    # Version
    "__version__",
    # High-Level Client (Simplest API)
    "RNSRClient",
    # Data Structures (for BYOD usage)
    "SkeletonNode",
    "KnowledgeGraph",
    "InMemoryKnowledgeGraph",
    "KVStore",
    "SQLiteKVStore",
    "InMemoryKVStore",
    # Ingestion
    "ingest_document",
    "ingest_document_enhanced",
    "IngestionResult",
    # Indexing
    "build_skeleton_index",
    # Persistence
    "save_index",
    "load_index",
    "get_index_info",
    "list_indexes",
    # Document Store
    "DocumentStore",
    # Standard Navigator
    "run_navigator",
    "VariableStore",
    # RLM Navigator (State-of-the-Art)
    "RLMNavigator",
    "RLMConfig",
    "run_rlm_navigator",
    "create_rlm_navigator",
    "PreFilterEngine",
    "RecursiveSubLLMEngine",
    "AnswerVerificationEngine",
    # LLM
    "get_llm",
    "get_embed_model",
    "LLMProvider",
]
