"""Configuration for rnsr.

All defaults trace to the design spec (docdb-rlm-design-spec.md):
budgets from §7, validation thresholds from §3.3, chunking from §3.4,
coercion from §3.2, batching from §4.1, search-ladder bounds from §5.

Environment variables (see .env.example) override defaults via
``Settings.from_env()``. Model roles resolve per provider in
``rnsr.llm.router``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

_ENV_PREFIX = "RNSR_"


@dataclass
class Settings:
    # --- model roles (resolved to (provider, model) by rnsr.llm.router) ---
    provider: str = "auto"          # openai | anthropic | gemini | auto
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

    # --- ingestion validation (§3.3) ---
    table_confidence_threshold: float = 0.7
    arithmetic_rel_tol: float = 0.005   # 0.5%
    arithmetic_abs_tol: float = 1.0     # 1 unit
    prose_check_cells: int = 3          # k sampled numeric cells per table

    # --- table coercion (§3.2) ---
    coerce_threshold: float = 0.95      # >=95% of non-null cells must coerce

    # --- chunking (§3.4) ---
    chunk_chars: int = 1500
    chunk_overlap: int = 200

    # --- sub-LM batching (§4.1) ---
    sub_call_char_budget: int = 200_000
    annotate_batch_size: int = 40

    # --- search ladder (§5) ---
    expansion_max_rounds: int = 3
    rescore_candidates: int = 4000      # int8 KNN pool rescored at fp32 (rung 4)

    # --- misc ---
    llm_seed: int = 42
    run_dir: Path = field(default_factory=lambda: Path("runs"))
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> Settings:
        """Build Settings from environment, loading .env if present.

        Numeric/str fields map to RNSR_<UPPER_NAME>; provider also accepts
        the legacy LLM_PROVIDER name.
        """
        load_dotenv(dotenv_path or Path(".env"), override=False)
        kwargs: dict = {}
        for f in fields(cls):
            raw = os.environ.get(_ENV_PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            if f.type in ("int",):
                kwargs[f.name] = int(raw)
            elif f.type in ("float",):
                kwargs[f.name] = float(raw)
            elif f.name == "run_dir":
                kwargs[f.name] = Path(raw)
            else:
                kwargs[f.name] = raw
        if "provider" not in kwargs:
            legacy = os.environ.get("LLM_PROVIDER")
            if legacy:
                kwargs["provider"] = legacy
        return cls(**kwargs)
