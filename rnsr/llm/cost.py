"""Pricing tables and cost computation.

Prices are USD per **million** tokens (input, output), best-effort as of
2026-10; unknown models cost 0 with a recorded warning flag so budgeting
degrades visibly, not silently. Update alongside provider SDK bumps.
"""

from __future__ import annotations

from rnsr.llm.base import Usage

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI (GPT-5.6 rates as of 2026-07-30 cut; sol unchanged since GA)
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.2": (1.25, 10.0),          # deprecated; kept for old trajectories
    "gpt-5.2-pro": (15.0, 120.0),
    "gpt-5-mini": (0.25, 2.0),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # Anthropic (platform.claude.com/docs/en/about-claude/pricing, list prices
    # effective 2026-10-07). Dateless IDs are pinned snapshots from 4.6 on.
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),  # deprecated 2026-09-30, retires 2026-11-30
    # Haiku 5.5 is tiered by prompt length: $0.10/$0.50 up to 100K input
    # tokens, $0.50/$2.50 above. The flat table records the <=100K tier, so
    # long-prompt calls are under-counted by up to 5x.
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
    # Gemini
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "text-embedding-004": (0.15, 0.0),
    # Gemini Embedding 2 (ai.google.dev/gemini-api/docs/pricing, text input)
    "gemini-embedding-2": (0.20, 0.0),
}

# Prompt-cache multipliers on the base input price: (cache read, 5-minute
# cache write). Anthropic's standard is 0.1x / 1.25x; Opus 5.5 and Sonnet 5.5
# read at 0.05x and Fable 5.1 at 0.025x (pricing page, 2026-10-07). Models
# without an entry use the standard pair; providers that never report cache
# tokens are unaffected because both counts stay zero.
_CACHE_DEFAULT: tuple[float, float] = (0.10, 1.25)
CACHE_MULTIPLIERS: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (0.025, 1.25),
    "claude-opus-5-5": (0.05, 1.25),
    "claude-sonnet-5-5": (0.05, 1.25),
}

unknown_models: set[str] = set()


def compute_cost(model: str, input_tokens: int, output_tokens: int, *,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """List-price cost. ``input_tokens`` here are the *uncached* prompt tokens;
    cached reads and writes are priced at their own multipliers."""
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        unknown_models.add(model)
        return 0.0
    read_mult, write_mult = CACHE_MULTIPLIERS.get(model, _CACHE_DEFAULT)
    return (input_tokens * price[0]
            + cache_read_tokens * price[0] * read_mult
            + cache_write_tokens * price[0] * write_mult
            + output_tokens * price[1]) / 1_000_000


def make_usage(model: str, input_tokens: int, output_tokens: int, *,
               cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> Usage:
    """Usage for one call. ``input_tokens`` are the uncached prompt tokens as
    the provider reports them; the returned ``Usage.input_tokens`` is the
    whole prompt including cached tokens."""
    cost = compute_cost(model, input_tokens, output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens)
    return Usage(input_tokens + cache_read_tokens + cache_write_tokens,
                 output_tokens, cost, cache_read_tokens, cache_write_tokens)
