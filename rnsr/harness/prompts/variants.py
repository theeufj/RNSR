"""Per-provider prompt variants (spec §9: prompts don't transfer across
models; model swaps are re-evaluation events).

Each entry appends provider-tuned lines to the shared system prompt —
above all the batching guardrail, which the paper found is model-specific.
These start as informed defaults and get tuned from observed trajectories
(Phase D); rerun `rnsr gate` after any change here.
"""

from __future__ import annotations

# provider -> extra guardrail lines appended to the system prompt
GUARDRAILS: dict[str, str] = {
    "openai": (
        "Batch aggressively: prefer a single llm_map over sequential calls, "
        "and pack up to the stated character budget per sub-call."
    ),
    "anthropic": (
        "Batch aggressively: prefer a single llm_map over sequential calls. "
        "Do not restate the plan between cells; put reasoning in comments "
        "inside the code block."
    ),
    "gemini": (
        "Batch aggressively: prefer a single llm_map over sequential calls. "
        "Always print() intermediate values you need to see — expressions "
        "alone produce no output."
    ),
}


def guardrail_for(provider: str) -> str:
    return GUARDRAILS.get(provider, "")
