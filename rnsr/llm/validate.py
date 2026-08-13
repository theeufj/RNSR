"""Startup checks for model names and pricing.

Two failure modes have both happened here and neither announced itself:

  1. A model name rots. Providers retire names on their own schedule, so a
     pinned DEFAULT_MODELS entry becomes a 404 on somebody else's release
     day — first noticed only when every root call failed.
  2. A model has no pricing row. compute_cost() returns 0.0 for unknown
     models, so every §7 spend cap silently becomes infinite. A run that
     believes it spent $0.00 will never breach max_spend_usd.

check_pricing() is free and offline, so it runs on every resolve and warns
once per model. check_models_live() asks the provider what exists and is
therefore opt-in, behind `rnsr doctor`.
"""

from __future__ import annotations

import logging

from rnsr.llm.cost import PRICES_PER_MTOK
from rnsr.obs import get_logger, log

_LOG = get_logger("llm.validate")
_warned: set[str] = set()

# Embeddings are billed per input token only and some providers publish no
# per-model rate; a missing row for these is not the silent-budget trap.
_COST_EXEMPT_ROLES = frozenset({"embed"})


def check_pricing(model: str, role: str = "") -> bool:
    """Warn once if `model` has no pricing row. True when pricing is known."""
    if model in PRICES_PER_MTOK:
        return True
    if role in _COST_EXEMPT_ROLES:
        return False
    if model not in _warned:
        _warned.add(model)
        log(_LOG, logging.WARNING, "model.pricing_missing", model=model,
            role=role,
            detail="calls with this model count as $0.00, so per-query spend "
                   "caps cannot fire — add it to rnsr/llm/cost.py")
    return False


async def check_models_live(router, roles=("root", "sub")) -> list[dict]:
    """Ask the provider which models exist; report resolved names that do not.

    Returns one finding per problem: [{role, model, provider, problem}].
    An empty list means every checked role resolves to a live model.
    """
    findings: list[dict] = []
    listings: dict[str, set[str] | None] = {}
    for role in roles:
        try:
            resolved = router.resolve(role)
        except Exception as e:
            findings.append({"role": role, "model": "", "provider": "",
                             "problem": f"unresolvable: {type(e).__name__}: {e}"})
            continue
        provider = getattr(resolved.client, "provider", "")
        if provider not in listings:
            listings[provider] = await _list_models(resolved.client, provider)
        available = listings[provider]
        if available is None:
            findings.append({
                "role": role, "model": resolved.model, "provider": provider,
                "problem": "could not list models (check the API key and "
                           "network); model name not verified"})
        elif not _matches(resolved.model, available):
            findings.append({
                "role": role, "model": resolved.model, "provider": provider,
                "problem": "not offered by the provider — the name is retired "
                           "or misspelled"})
        if not check_pricing(resolved.model, role):
            findings.append({
                "role": role, "model": resolved.model, "provider": provider,
                "problem": "no pricing row: spend caps cannot fire for this "
                           "model"})
    return findings


def _matches(model: str, available: set[str]) -> bool:
    """Providers accept aliases and dated snapshots of one family, so an
    exact miss is not proof of absence: 'claude-sonnet-4-5' may be served as
    'claude-sonnet-4-5-20260101'."""
    return any(name == model or name.startswith(model) or model.startswith(name)
               for name in available)


async def _list_models(client, provider: str) -> set[str] | None:
    inner = getattr(client, "_inner", client)      # unwrap the governor
    try:
        if provider == "openai":
            page = await inner._client.models.list()
            return {m.id for m in page.data}
        if provider == "anthropic":
            page = await inner._client.models.list()
            return {m.id for m in page.data}
        if provider == "gemini":
            models = await inner._client.aio.models.list()
            out = set()
            async for m in models:
                out.add((m.name or "").removeprefix("models/"))
            return out
    except Exception:
        return None
    return None
