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
- Multi-provider LLM support (OpenAI, Anthropic, Gemini, Ollama)

Usage:
    from rnsr import RNSRClient
    
    # Simple one-line Q&A
    client = RNSRClient()
    answer = client.ask("contract.pdf", "What are the payment terms?")
    
    # Advanced RLM navigation with full features
    result = client.ask_advanced(
        "complex_report.pdf",
        "Compare liability clauses in sections 5 and 8",
        enable_verification=True,
        max_recursion_depth=3,
    )
    
    # Vision-based analysis (for scanned docs, charts)
    result = client.ask_vision(
        "scanned_document.pdf",
        "What does the revenue chart show?",
    )
    
    # Low-level API
    from rnsr import ingest_document, build_skeleton_index, run_rlm_navigator
    
    result = ingest_document("contract.pdf")
    skeleton, kv_store = build_skeleton_index(result.tree)
    answer = run_rlm_navigator("What are the terms?", skeleton, kv_store)
    
LLM Provider Configuration:
    Set one of these environment variables:
    - GOOGLE_API_KEY (Gemini)
    - OPENAI_API_KEY (OpenAI)
    - ANTHROPIC_API_KEY (Anthropic)
    - OLLAMA_BASE_URL or USE_OLLAMA (Ollama, local; default model qwen2.5-coder:32b)
"""

__version__ = "0.2.2"  # Ollama timeout env var; suppress API key warning for Ollama

# Re-export main entry points
from rnsr.ingestion import ingest_document, IngestionResult
from rnsr.ingestion.pipeline import ingest_document_enhanced
from rnsr.indexing import build_skeleton_index, SQLiteKVStore, InMemoryKVStore
from rnsr.indexing import save_index, load_index, get_index_info, list_indexes
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
    # Ingestion
    "ingest_document",
    "ingest_document_enhanced",
    "IngestionResult",
    # Indexing
    "build_skeleton_index",
    "SQLiteKVStore",
    "InMemoryKVStore",
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
