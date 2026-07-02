"""Pricing tables and cost computation.

Prices are USD per **million** tokens (input, output), best-effort as of
2026-07; unknown models cost 0 with a recorded warning flag so budgeting
degrades visibly, not silently. Update alongside provider SDK bumps.
"""

from __future__ import annotations

from rnsr.llm.base import Usage

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.2": (1.25, 10.0),
    "gpt-5.2-pro": (15.0, 120.0),
    "gpt-5-mini": (0.25, 2.0),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # Anthropic
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Gemini
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "text-embedding-004": (0.15, 0.0),
}

unknown_models: set[str] = set()


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        unknown_models.add(model)
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def make_usage(model: str, input_tokens: int, output_tokens: int) -> Usage:
    return Usage(input_tokens, output_tokens,
                 compute_cost(model, input_tokens, output_tokens))
