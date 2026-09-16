"""Configuration for rnsr.

All defaults trace to the design spec (docdb-rlm-design-spec.md):
budgets from §7, validation thresholds from §3.3, chunking from §3.4,
coercion from §3.2, batching from §4.1, search-ladder bounds from §5.

Environment variables (see .env.example) override defaults via
``Settings.from_env()``. Model roles resolve per provider in
``rnsr.llm.router``.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, get_type_hints

from dotenv import load_dotenv
from pydantic import TypeAdapter

_ENV_PREFIX = "RNSR_"


@dataclass
class Settings:
    # --- model roles (resolved to (provider, model) by rnsr.llm.router) ---
    provider: Literal["openai", "anthropic", "gemini", "auto"] = "auto"
    root_model: str = ""            # empty -> provider default
    sub_model: str = ""
    embed_model: str = ""
    vision_model: str = ""          # empty -> same as sub_model's provider default

    # --- budgets (§7) ---
    max_root_iters: int = 20
    max_sub_calls: int = 300
    max_wall_s: float = 600.0
    max_spend_usd: float = 2.0
    sub_concurrency: int = 16
    cell_timeout_s: float = 120.0   # per-cell wall clock; sandbox killed past this

    # --- run-level provider governance (rnsr.llm.governor) ---
    # Budgets above cap ONE query; these cap the run. 0 disables a limit.
    max_in_flight_requests: int = 24     # concurrent provider requests, all roles
    max_requests_per_minute: int = 0     # sliding-window RPM ceiling
    run_spend_ceiling_usd: float = 0.0   # aggregate USD before calls are refused

    # --- trajectory data protection (rnsr.harness.trajectory) ---
    # Trajectories quote client documents verbatim. 'full' keeps the complete
    # forensic record; 'redacted' replaces document-bearing values with a
    # length + digest; 'metadata' drops them. A Fernet key encrypts each line
    # at rest (needs the 'secure' extra).
    trajectory_content: Literal["full", "redacted", "metadata"] = "full"
    trajectory_key: str = field(default="", repr=False)
    trajectory_retention_days: float = 0.0    # 0 keeps trajectories forever

    # --- containment ---
    # Supplementary Python audit hook. OS isolation is always required,
    # including when this switch is disabled for guard debugging.
    sandbox_fs_guard: bool = True

    # --- ingestion validation (§3.3) ---
    table_confidence_threshold: float = 0.7
    arithmetic_rel_tol: float = 0.005   # 0.5%
    arithmetic_abs_tol: float = 1.0     # 1 unit
    prose_check_cells: int = 3          # k sampled numeric cells per table

    # --- table coercion (§3.2) ---
    coerce_threshold: float = 0.95      # >=95% of non-null cells must coerce

    # --- corpus health gate ---
    # Answering is refused (exit 2 / CorpusHealthError) when a finding is
    # error-severity unless allow_degraded is set. Untranscribed scans are
    # an error by default (max 0).
    health_min_validation_rate: float = 0.7
    health_max_untranscribed_pages: int = 0
    health_max_parse_failed_rate: float = 0.05
    allow_degraded: bool = False
    # auto: transcribe scanned pages when a vision-capable key is present
    # always: require a vision provider (fail if missing)
    # never: leave scans as visible gaps
    transcribe_scans: Literal["auto", "always", "never"] = "auto"

    # --- derived cell index (engine-poc-plan Stage 1) ---
    # Populates `cells` at ingest so rung-0 sweeps run one indexed scan
    # instead of LIKE over every t_* table. Off -> the legacy per-table
    # sweep path (also used automatically for artifacts without cells).
    cells_index: bool = True

    # --- chunking (§3.4) ---
    chunk_chars: int = 1500
    chunk_overlap: int = 200

    # --- sub-LM batching (§4.1) ---
    sub_call_char_budget: int = 200_000
    annotate_batch_size: int = 40

    # --- search ladder (§5) ---
    expansion_max_rounds: int = 3
    rescore_candidates: int = 4000      # int8 KNN pool rescored at fp32 (rung 4)
    # Rung-4 embeddings default-on once the corpus has this many documents
    # and an embed provider is configured. Tantivy/usearch/TOC stay behind
    # the replay parity gate (docs/search-contract.md; Phase 0 ledger).
    embed_auto_on_docs: int = 200

    # HTTP deployments are one authorization scope per process. Both are
    # required by create_app; the liveness endpoint alone is public.
    service_token: str = field(default="", repr=False)
    service_corpus_root: Path | None = None
    service_max_jobs: int = 200

    # --- misc ---
    llm_seed: int = 42
    run_dir: Path = field(default_factory=lambda: Path("runs"))
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for f in fields(self):
            value = TypeAdapter(hints[f.name]).validate_python(getattr(self, f.name))
            setattr(self, f.name, value)
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and (not math.isfinite(value) or value < 0)):
                raise ValueError(f"{f.name} must be finite and non-negative")
        for name in ("max_wall_s", "cell_timeout_s", "sub_concurrency", "chunk_chars",
                     "sub_call_char_budget", "annotate_batch_size", "rescore_candidates",
                     "service_max_jobs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        for name in ("table_confidence_threshold", "coerce_threshold",
                     "health_min_validation_rate", "health_max_parse_failed_rate"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> Settings:
        """Build Settings from environment, loading .env if present.

        Fields map to RNSR_<UPPER_NAME> using their resolved type annotations.
        LLM_PROVIDER is deprecated and will be removed in version 2.0.
        """
        load_dotenv(dotenv_path or Path(".env"), override=False)
        kwargs: dict = {}
        hints = get_type_hints(cls)
        for f in fields(cls):
            raw = os.environ.get(_ENV_PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            kwargs[f.name] = TypeAdapter(hints[f.name]).validate_python(raw)
        if "provider" not in kwargs:
            legacy = os.environ.get("LLM_PROVIDER")
            if legacy:
                warnings.warn("LLM_PROVIDER is deprecated; use RNSR_PROVIDER. "
                              "Compatibility ends in version 2.0.",
                              DeprecationWarning, stacklevel=2)
                kwargs["provider"] = legacy
        return cls(**kwargs)
