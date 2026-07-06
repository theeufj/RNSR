"""Role -> (client, model) resolution.

Roles: root (REPL orchestrator), sub (cheap batched calls), embed (rung 4),
vision (table fallback). Provider chosen from Settings.provider or
auto-detected from available API keys. Anthropic lacks embeddings, so the
embed role falls through to OpenAI/Gemini when a key is available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rnsr.config import Settings
from rnsr.llm.base import LLMClient

DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "openai": {
        "root": "gpt-5.2",
        "sub": "gpt-5-mini",
        "embed": "text-embedding-3-small",
        "vision": "gpt-5-mini",
    },
    "anthropic": {
        "root": "claude-sonnet-4-5",
        "sub": "claude-haiku-4-5",
        "embed": "",                      # unsupported; falls through
        "vision": "claude-haiku-4-5",
    },
    "gemini": {
        "root": "gemini-2.5-pro",
        "sub": "gemini-2.5-flash",
        "embed": "gemini-embedding-001",  # text-embedding-00x line is retired (404)
        "vision": "gemini-2.5-flash",
    },
}

_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def available_providers() -> list[str]:
    return [p for p, env in _KEY_ENV.items() if os.environ.get(env)]


def detect_provider(settings: Settings) -> str:
    if settings.provider != "auto":
        return settings.provider
    found = available_providers()
    if not found:
        raise RuntimeError(
            "no LLM provider configured: set one of "
            + ", ".join(_KEY_ENV.values())
        )
    return found[0]


@dataclass
class Resolved:
    client: LLMClient
    model: str


class Router:
    """Resolves roles to provider clients, constructing each client once."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.provider = detect_provider(self.settings)
        self._clients: dict[str, LLMClient] = {}

    def _client_for(self, provider: str) -> LLMClient:
        if provider not in self._clients:
            if provider == "openai":
                from rnsr.llm.openai_client import OpenAIClient

                self._clients[provider] = OpenAIClient()
            elif provider == "anthropic":
                from rnsr.llm.anthropic_client import AnthropicClient

                self._clients[provider] = AnthropicClient()
            elif provider == "gemini":
                from rnsr.llm.gemini_client import GeminiClient

                self._clients[provider] = GeminiClient()
            else:
                raise ValueError(f"unknown provider: {provider}")
        return self._clients[provider]

    def resolve(self, role: str) -> Resolved:
        if role not in ("root", "sub", "embed", "vision"):
            raise ValueError(f"unknown role: {role}")
        override = getattr(self.settings, f"{role}_model", "")
        provider = self.provider
        # A provider with no embedding API falls through to one that has it
        # BEFORE the override is applied — otherwise an embed-model override
        # would be routed to a client that cannot embed at all.
        if role == "embed" and not DEFAULT_MODELS[provider]["embed"]:
            for alt in ("openai", "gemini"):
                if alt in available_providers():
                    provider = alt
                    break
            else:
                raise RuntimeError(
                    f"provider '{self.provider}' has no embeddings and no "
                    "OPENAI_API_KEY/GOOGLE_API_KEY fallback is set"
                )
        model = override or DEFAULT_MODELS[provider][role]
        return Resolved(self._client_for(provider), model)
