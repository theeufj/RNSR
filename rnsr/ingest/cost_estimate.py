"""Pre-spend estimate for VLM page transcription.

Vision calls are billed per token. We do not know the image-token count
until the provider replies, so this uses a documented per-page assumption
so operators see a number *before* ingest spends.

Assumption (calibrated against Claude Haiku / Gemini Flash page crops at
scale=2.0 PNG plus the transcription prompt): ~1,500 input tokens and
~800 output tokens per page. Real cost follows the provider Usage.
"""

from __future__ import annotations

from rnsr.llm.cost import compute_cost

TOKENS_IN_PER_PAGE = 1500
TOKENS_OUT_PER_PAGE = 800


def resolve_transcriber(settings, *, no_transcribe: bool = False):
    """Return (transcriber, model) according to Settings.transcribe_scans.

    ``auto`` (default): transcribe when a vision-capable key is present.
    ``always``: require a vision provider.
    ``never`` / ``no_transcribe``: leave scans as visible gaps.
    """
    if no_transcribe or getattr(settings, "transcribe_scans", "auto") == "never":
        return None, None
    try:
        from rnsr.ingest.llm_hooks import make_page_transcriber
        from rnsr.llm.router import Router

        vis = Router(settings).resolve("vision")
        return make_page_transcriber(vis.client, vis.model), vis.model
    except RuntimeError:
        if getattr(settings, "transcribe_scans", "auto") == "always":
            raise
        return None, None


def estimate_transcription_usd(n_pages: int, model: str) -> float:
    """Estimated USD to transcribe ``n_pages`` with ``model``."""
    if n_pages <= 0:
        return 0.0
    return compute_cost(
        model,
        n_pages * TOKENS_IN_PER_PAGE,
        n_pages * TOKENS_OUT_PER_PAGE,
    )
