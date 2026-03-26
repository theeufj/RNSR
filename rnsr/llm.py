"""
LLM Configuration - Multi-Provider LLM and Embedding Support

Supports:
- OpenAI (GPT-4, text-embedding-3-small)
- Anthropic (Claude)
- Google Gemini (gemini-pro, text-embedding-005)

Features:
- Automatic rate limit handling with exponential backoff
- Cross-provider fallback on 429/quota errors
- Provider priority chain for resilience

Usage:
    from rnsr.llm import get_llm, get_embed_model, LLMProvider
    
    # Auto-detect based on environment variables
    llm = get_llm()
    embed = get_embed_model()
    
    # Or specify provider explicitly
    llm = get_llm(provider=LLMProvider.GEMINI)
    embed = get_embed_model(provider=LLMProvider.GEMINI)
"""

from __future__ import annotations

import hashlib
import json as _json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar, Union

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

T = TypeVar("T")

# Load .env file if it exists — check CWD first (PyPI installs), then
# the project root relative to the package (dev checkouts).
try:
    from dotenv import load_dotenv

    for _env_candidate in [Path.cwd() / ".env", Path(__file__).parent.parent / ".env"]:
        if _env_candidate.exists():
            load_dotenv(_env_candidate)
            break
except ImportError:
    pass  # dotenv not installed, rely on system environment

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AUTO = "auto"  # Auto-detect from environment


# Default models per provider (updated February 2026)
DEFAULT_MODELS = {
    LLMProvider.OPENAI: {
        "llm": "gpt-5-mini",  # Fast, affordable - use "gpt-5.2" for latest
        "embed": "text-embedding-3-small",
    },
    LLMProvider.ANTHROPIC: {
        "llm": "claude-sonnet-4-5",  # Smart model for agents/coding (alias for claude-sonnet-4-5-20250929)
        "embed": None,  # Anthropic doesn't have embeddings, fall back to OpenAI/Gemini
    },
    LLMProvider.GEMINI: {
        "llm": "gemini-2.5-flash",  # Stable model. Use "gemini-3-flash-preview" for latest.
        "embed": "text-embedding-005",
    },
}

# Fallback chain when a provider hits rate limits
PROVIDER_FALLBACK_CHAIN = {
    LLMProvider.GEMINI: [LLMProvider.OPENAI, LLMProvider.ANTHROPIC],
    LLMProvider.OPENAI: [LLMProvider.ANTHROPIC, LLMProvider.GEMINI],
    LLMProvider.ANTHROPIC: [LLMProvider.OPENAI, LLMProvider.GEMINI],
}


# Per-request HTTP timeout in seconds.  Prevents hung connections from
# blocking the pipeline forever.  Override with RNSR_LLM_TIMEOUT.
REQUEST_TIMEOUT = int(os.getenv("RNSR_LLM_TIMEOUT", "120"))


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit/quota error that should trigger fallback."""
    error_str = str(error).lower()
    
    # Check for common rate limit indicators
    rate_limit_indicators = [
        "429",
        "rate limit",
        "rate_limit",
        "quota exceeded",
        "quota_exceeded",
        "resource exhausted",
        "resourceexhausted",
        "too many requests",
        "overloaded",
    ]
    
    for indicator in rate_limit_indicators:
        if indicator in error_str:
            return True
    
    # Check for specific exception types
    try:
        from google.api_core import exceptions as google_exceptions
        if isinstance(error, (
            google_exceptions.ResourceExhausted,
            google_exceptions.TooManyRequests,
        )):
            return True
    except ImportError:
        pass
    
    return False


def is_connection_error(error: Exception) -> bool:
    """Check if an error is a connection-level failure that should trigger
    immediate fallback (no point retrying the same broken connection)."""
    if isinstance(error, (ConnectionRefusedError, ConnectionResetError,
                          ConnectionAbortedError, TimeoutError)):
        return True
    error_str = str(error).lower()
    conn_indicators = [
        "connection refused",
        "connection reset",
        "connection aborted",
        "connection error",
        "timed out",
        "timeout",
        "errno 61",
        "errno 54",
        "errno 104",
    ]
    return any(ind in error_str for ind in conn_indicators)


def get_available_fallback_providers(primary: LLMProvider) -> list[LLMProvider]:
    """Get list of available fallback providers for a given primary provider."""
    fallbacks = []
    for provider in PROVIDER_FALLBACK_CHAIN.get(primary, []):
        if validate_provider(provider):
            fallbacks.append(provider)
    return fallbacks


def detect_provider() -> LLMProvider:
    """
    Detect LLM provider.

    Priority:
    1. Explicit ``LLM_PROVIDER`` env var (openai / anthropic / gemini)
    2. Auto-detect from available API keys:
       GOOGLE_API_KEY -> Gemini, ANTHROPIC_API_KEY -> Anthropic,
       OPENAI_API_KEY -> OpenAI

    Returns:
        Detected LLMProvider.

    Raises:
        ValueError: If no API key is found.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit and explicit != "auto":
        mapping = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "gemini": LLMProvider.GEMINI,
        }
        if explicit in mapping:
            prov = mapping[explicit]
            if validate_provider(prov):
                logger.info("provider_from_env", provider=explicit)
                return prov
            logger.warning(
                "provider_env_set_but_no_key",
                provider=explicit,
                hint="Falling back to auto-detect",
            )
        else:
            logger.warning("unknown_llm_provider_env", value=explicit)

    if os.getenv("GOOGLE_API_KEY"):
        logger.info("provider_detected", provider="gemini")
        return LLMProvider.GEMINI
    
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("provider_detected", provider="anthropic")
        return LLMProvider.ANTHROPIC
    
    if os.getenv("OPENAI_API_KEY"):
        logger.info("provider_detected", provider="openai")
        return LLMProvider.OPENAI
    
    raise ValueError(
        "No LLM API key found. Set one of: "
        "GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY"
    )


class CachedLLM:
    """Wraps any LLM with a disk-based response cache for deterministic re-runs.

    Enable via ``RNSR_LLM_CACHE=true`` (default: off).
    Cache directory defaults to ``.rnsr_cache/llm`` but can be set with
    ``RNSR_LLM_CACHE_DIR``.
    """

    def __init__(self, llm: Any, cache_dir: Path | None = None, model_tag: str = ""):
        self._llm = llm
        self._model_tag = model_tag
        self._cache_dir = cache_dir or Path(
            os.getenv("RNSR_LLM_CACHE_DIR", ".rnsr_cache/llm")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ---------------------------------------------------------------

    def _cache_key(self, prompt: str, extra: str = "") -> str:
        """Deterministic hash from prompt + model tag + any extra discriminator."""
        blob = f"{self._model_tag}|{extra}|{prompt}"
        return hashlib.sha256(blob.encode()).hexdigest()

    def _read(self, key: str) -> str | None:
        cache_file = self._cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def _write(self, key: str, value: str) -> None:
        cache_file = self._cache_dir / f"{key}.txt"
        cache_file.write_text(value, encoding="utf-8")

    # -- public interface (mirrors LLM methods) --------------------------------

    def complete(self, prompt: str, **kwargs: Any) -> str:
        key = self._cache_key(prompt, extra="complete")
        cached = self._read(key)
        if cached is not None:
            logger.debug("llm_cache_hit", key=key[:12])
            return cached
        result = str(self._llm.complete(prompt, **kwargs))
        self._write(key, result)
        return result

    def complete_json(self, prompt: str, **kwargs: Any) -> str:
        """Cache-aware JSON completion (see ``complete_json`` on providers)."""
        key = self._cache_key(prompt, extra="complete_json")
        cached = self._read(key)
        if cached is not None:
            logger.debug("llm_cache_hit", key=key[:12])
            return cached
        # Delegate to underlying LLM (may or may not have complete_json)
        fn = getattr(self._llm, "complete_json", self._llm.complete)
        result = str(fn(prompt, **kwargs))
        self._write(key, result)
        return result

    def complete_with_image(self, prompt: str, image_bytes: bytes, **kwargs: Any) -> str:
        # Images are large; include a hash of image bytes in the key
        img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        key = self._cache_key(prompt, extra=f"image|{img_hash}")
        cached = self._read(key)
        if cached is not None:
            logger.debug("llm_cache_hit", key=key[:12])
            return cached
        result = str(self._llm.complete_with_image(prompt, image_bytes, **kwargs))
        self._write(key, result)
        return result

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        msgs_str = _json.dumps(messages, default=str)
        key = self._cache_key(msgs_str, extra="chat")
        cached = self._read(key)
        if cached is not None:
            logger.debug("llm_cache_hit", key=key[:12])
            return cached
        result = self._llm.chat(messages, **kwargs)
        self._write(key, str(result))
        return result

    def clear_cache(self) -> int:
        """Remove all cached responses.  Returns the number of files removed."""
        count = 0
        for f in self._cache_dir.glob("*.txt"):
            f.unlink()
            count += 1
        logger.info("llm_cache_cleared", files_removed=count)
        return count

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped LLM."""
        return getattr(self._llm, name)


def _maybe_wrap_cache(llm: Any, model_tag: str = "") -> Any:
    """Optionally wrap *llm* with :class:`CachedLLM` when caching is enabled."""
    if os.getenv("RNSR_LLM_CACHE", "").lower() in ("1", "true", "yes"):
        logger.info("llm_cache_enabled", model_tag=model_tag)
        return CachedLLM(llm, model_tag=model_tag)
    return llm


def get_llm(
    provider: LLMProvider = LLMProvider.AUTO,
    model: str | None = None,
    enable_fallback: bool = True,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Get an LLM instance for the specified provider.
    
    Args:
        provider: LLM provider (openai, anthropic, gemini, or auto).
        model: Model name override. Uses default if not specified.
        enable_fallback: If True, enables cross-provider fallback on rate limits.
        api_key: API key for the provider. If not specified, falls back to
                 the corresponding environment variable (OPENAI_API_KEY,
                 ANTHROPIC_API_KEY, or GOOGLE_API_KEY).
        **kwargs: Additional arguments passed to the LLM constructor.
        
    Returns:
        LlamaIndex-compatible LLM instance with fallback support.
        
    Example:
        llm = get_llm(provider=LLMProvider.GEMINI)
        response = await llm.acomplete("Hello!")
        
        # With explicit API key
        llm = get_llm(provider=LLMProvider.OPENAI, api_key="sk-...")
    """
    if provider == LLMProvider.AUTO:
        provider = detect_provider()
    
    model = model or DEFAULT_MODELS[provider]["llm"]
    
    # Get primary LLM
    primary_llm = _get_raw_llm(provider, model, api_key=api_key, **kwargs)
    
    if not enable_fallback:
        return _maybe_wrap_cache(primary_llm, model_tag=model or "")
    
    # Build fallback chain
    fallback_providers = get_available_fallback_providers(provider)
    if not fallback_providers:
        logger.debug("no_fallback_providers_available", primary=provider.value)
        return _maybe_wrap_cache(primary_llm, model_tag=model or "")
    
    logger.debug(
        "llm_with_fallback_configured",
        primary=provider.value,
        fallbacks=[p.value for p in fallback_providers],
    )
    
    llm = ResilientLLMWrapper(
        primary_llm=primary_llm,
        primary_provider=provider,
        fallback_providers=fallback_providers,
        **kwargs,
    )
    return _maybe_wrap_cache(llm, model_tag=model or "")


def _get_raw_llm(provider: LLMProvider, model: str, api_key: str | None = None, **kwargs: Any) -> Any:
    """Get a raw LLM instance without fallback wrapper."""
    if provider == LLMProvider.OPENAI:
        return _get_openai_llm(model, api_key=api_key, **kwargs)
    elif provider == LLMProvider.ANTHROPIC:
        return _get_anthropic_llm(model, api_key=api_key, **kwargs)
    elif provider == LLMProvider.GEMINI:
        return _get_gemini_llm(model, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embed_model(
    provider: LLMProvider = LLMProvider.AUTO,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Get an embedding model for the specified provider.
    
    Args:
        provider: LLM provider (openai, gemini, or auto).
        model: Model name override. Uses default if not specified.
        api_key: API key for the provider. If not specified, falls back to
                 the corresponding environment variable.
        **kwargs: Additional arguments passed to the embedding constructor.
        
    Returns:
        LlamaIndex-compatible embedding model.
        
    Note:
        Anthropic doesn't have embeddings. Falls back to OpenAI or Gemini.
        
    Example:
        embed = get_embed_model(provider=LLMProvider.GEMINI)
        vector = embed.get_text_embedding("Hello world")
    """
    if provider == LLMProvider.AUTO:
        provider = detect_provider()
    
    # Anthropic doesn't have embeddings, fall back
    if provider == LLMProvider.ANTHROPIC:
        if os.getenv("GOOGLE_API_KEY"):
            provider = LLMProvider.GEMINI
            logger.info("anthropic_no_embeddings", fallback="gemini")
        elif os.getenv("OPENAI_API_KEY"):
            provider = LLMProvider.OPENAI
            logger.info("anthropic_no_embeddings", fallback="openai")
        else:
            raise ValueError(
                "Anthropic doesn't provide embeddings. "
                "Set GOOGLE_API_KEY or OPENAI_API_KEY for embeddings."
            )
    
    model = model or DEFAULT_MODELS[provider]["embed"]
    
    if api_key:
        kwargs["api_key"] = api_key
    
    if provider == LLMProvider.OPENAI:
        return _get_openai_embed(model, **kwargs)
    elif provider == LLMProvider.GEMINI:
        return _get_gemini_embed(model, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# =============================================================================
# Resilient LLM Wrapper with Cross-Provider Fallback
# =============================================================================


class ResilientLLMWrapper:
    """
    LLM wrapper that provides cross-provider fallback on rate limits.
    
    When the primary provider hits a 429/quota error, automatically switches
    to fallback providers in order until one succeeds.
    """
    
    def __init__(
        self,
        primary_llm: Any,
        primary_provider: LLMProvider,
        fallback_providers: list[LLMProvider],
        max_retries: int = 3,
        retry_delay: float = 2.0,
        **kwargs: Any,
    ):
        self.primary_llm = primary_llm
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.kwargs = kwargs
        
        # Lazily initialized fallback LLMs
        self._fallback_llms: dict[LLMProvider, Any] = {}
        
        # Track which provider we're currently using
        self._current_provider = primary_provider
        self._rate_limited_until: dict[LLMProvider, float] = {}
    
    def _get_fallback_llm(self, provider: LLMProvider) -> Any:
        """Get or create a fallback LLM instance."""
        if provider not in self._fallback_llms:
            model = DEFAULT_MODELS[provider]["llm"]
            self._fallback_llms[provider] = _get_raw_llm(provider, model, **self.kwargs)
            logger.info("fallback_llm_initialized", provider=provider.value, model=model)
        return self._fallback_llms[provider]
    
    def _is_rate_limited(self, provider: LLMProvider) -> bool:
        """Check if a provider is currently rate limited."""
        if provider not in self._rate_limited_until:
            return False
        return time.time() < self._rate_limited_until[provider]
    
    def _mark_rate_limited(self, provider: LLMProvider, duration: float = 60.0):
        """Mark a provider as rate limited for a duration."""
        self._rate_limited_until[provider] = time.time() + duration
        logger.warning(
            "provider_rate_limited",
            provider=provider.value,
            cooldown_seconds=duration,
        )
    
    def _get_available_llms(self) -> list[tuple[LLMProvider, Any]]:
        """Get list of available LLMs in priority order."""
        llms = []
        
        # Primary first (if not rate limited)
        if not self._is_rate_limited(self.primary_provider):
            llms.append((self.primary_provider, self.primary_llm))
        
        # Then fallbacks
        for provider in self.fallback_providers:
            if not self._is_rate_limited(provider):
                llms.append((provider, self._get_fallback_llm(provider)))
        
        # If all are rate limited, try primary anyway (it might work now)
        if not llms:
            llms.append((self.primary_provider, self.primary_llm))
        
        return llms
    
    def _call_with_fallback(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method with automatic fallback on rate limits and connection errors."""
        last_error = None
        
        for provider, llm in self._get_available_llms():
            for attempt in range(self.max_retries):
                try:
                    method = getattr(llm, method_name)
                    result = method(*args, **kwargs)
                    
                    # Success - update current provider
                    if provider != self._current_provider:
                        logger.info(
                            "switched_to_fallback_provider",
                            from_provider=self._current_provider.value,
                            to_provider=provider.value,
                        )
                        self._current_provider = provider
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    if is_rate_limit_error(e):
                        logger.warning(
                            "rate_limit_hit",
                            provider=provider.value,
                            attempt=attempt + 1,
                            error=str(e)[:200],
                        )
                        # Mark provider as rate limited and try next
                        self._mark_rate_limited(provider, duration=60.0)
                        break  # Move to next provider

                    if is_connection_error(e):
                        logger.warning(
                            "connection_error_fallback",
                            provider=provider.value,
                            attempt=attempt + 1,
                            error=str(e)[:200],
                        )
                        # Connection-level failure — skip directly to next
                        # provider rather than retrying the same broken conn.
                        self._mark_rate_limited(provider, duration=30.0)
                        break

                    # Other errors — retry with exponential backoff
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.debug(
                            "retrying_after_error",
                            provider=provider.value,
                            attempt=attempt + 1,
                            delay=delay,
                            error=str(e)[:100],
                        )
                        time.sleep(delay)
                    else:
                        # All retries exhausted for this provider
                        break
        
        # All providers failed
        logger.error(
            "all_providers_failed",
            primary=self.primary_provider.value,
            fallbacks=[p.value for p in self.fallback_providers],
        )
        raise last_error or RuntimeError("All LLM providers failed")
    
    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Complete a prompt with fallback support."""
        return self._call_with_fallback("complete", prompt, **kwargs)

    def complete_json(self, prompt: str, **kwargs: Any) -> Any:
        """JSON-mode completion with fallback support.

        Falls back to regular ``complete()`` if the underlying LLM does not
        expose ``complete_json``.
        """
        # Try JSON-specific path first
        for provider, llm in self._get_available_llms():
            if hasattr(llm, "complete_json"):
                try:
                    return llm.complete_json(prompt, **kwargs)
                except Exception as e:
                    if is_rate_limit_error(e) or is_connection_error(e):
                        self._mark_rate_limited(
                            provider,
                            duration=30.0 if is_connection_error(e) else 60.0,
                        )
                        logger.warning(
                            "complete_json_fallback",
                            provider=provider.value,
                            error_type="connection" if is_connection_error(e) else "rate_limit",
                            error=str(e)[:200],
                        )
                        continue
                    raise
        # No provider had complete_json or all were rate-limited; fall back
        return self.complete(prompt, **kwargs)

    def complete_with_image(self, prompt: str, image_bytes: bytes, **kwargs: Any) -> Any:
        """Complete a prompt with an image (multimodal) with fallback support.

        Falls back to text-only ``complete()`` if no provider supports
        multimodal input.
        """
        last_error = None
        for provider, llm in self._get_available_llms():
            if not hasattr(llm, "complete_with_image"):
                continue
            try:
                result = llm.complete_with_image(prompt, image_bytes, **kwargs)
                if provider != self._current_provider:
                    logger.info(
                        "switched_to_fallback_provider",
                        from_provider=self._current_provider.value,
                        to_provider=provider.value,
                    )
                    self._current_provider = provider
                return result
            except Exception as e:
                last_error = e
                if is_rate_limit_error(e) or is_connection_error(e):
                    self._mark_rate_limited(
                        provider,
                        duration=30.0 if is_connection_error(e) else 60.0,
                    )
                    continue
                raise

        # No multimodal-capable provider succeeded — fall back to text-only
        logger.warning("multimodal_unavailable_falling_back_to_text")
        return self.complete(prompt, **kwargs)
    
    def chat(self, messages: Any, **kwargs: Any) -> Any:
        """Chat with fallback support."""
        return self._call_with_fallback("chat", messages, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """Forward other attributes to the current LLM."""
        return getattr(self.primary_llm, name)


# =============================================================================
# Provider-Specific Implementations
# =============================================================================


def _get_openai_llm(model: str, api_key: str | None = None, **kwargs: Any) -> Any:
    """Get OpenAI LLM instance with ``complete_json`` support and HTTP timeout."""
    try:
        from llama_index.llms.openai import OpenAI as _OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI LLM not installed. "
            "Install with: pip install llama-index-llms-openai"
        )
    
    # Use explicit API key if provided, otherwise rely on env var
    if api_key:
        kwargs["api_key"] = api_key
    
    # Set temperature=0 for deterministic outputs unless overridden
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0
    
    # Add seed for best-effort determinism (configurable via env var)
    _seed = int(os.getenv("RNSR_LLM_SEED", "42"))
    if "seed" not in kwargs:
        kwargs["seed"] = _seed

    # HTTP-level timeout so hung connections fail fast
    if "timeout" not in kwargs:
        kwargs["timeout"] = float(REQUEST_TIMEOUT)

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = int(os.getenv("RNSR_MAX_OUTPUT_TOKENS", "16384"))
    
    logger.debug("initializing_llm", provider="openai", model=model, timeout=REQUEST_TIMEOUT)

    class _OpenAIWithJson(_OpenAI):
        """Thin subclass that adds ``complete_json()`` using JSON mode."""

        def complete_json(self, prompt: str, **kw: Any) -> str:
            """Complete expecting a JSON response (uses OpenAI JSON mode)."""
            try:
                from openai import OpenAI as _RawOpenAI
                client = _RawOpenAI(timeout=float(REQUEST_TIMEOUT))
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    seed=_seed,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or ""
            except Exception:
                # Fallback: regular completion
                return str(self.complete(prompt, **kw))

    return _OpenAIWithJson(model=model, **kwargs)


def _get_anthropic_llm(model: str, api_key: str | None = None, **kwargs: Any) -> Any:
    """Get Anthropic LLM instance with HTTP timeout."""
    try:
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Anthropic LLM not installed. "
            "Install with: pip install llama-index-llms-anthropic"
        )
    
    # Use explicit API key if provided, otherwise rely on env var
    if api_key:
        kwargs["api_key"] = api_key
    
    # Set temperature=0 for deterministic outputs unless overridden
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0

    # HTTP-level timeout so hung connections fail fast
    if "timeout" not in kwargs:
        kwargs["timeout"] = float(REQUEST_TIMEOUT)

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = int(os.getenv("RNSR_MAX_OUTPUT_TOKENS", "16384"))
    
    logger.debug("initializing_llm", provider="anthropic", model=model, timeout=REQUEST_TIMEOUT)

    class _AnthropicWithJson(Anthropic):
        """Thin subclass adding soft ``complete_json`` for Anthropic."""

        def complete_json(self, prompt: str, **kw: Any) -> str:
            """Anthropic has no native JSON mode; we append an instruction."""
            json_prompt = (
                prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                "No markdown fences, no commentary."
            )
            return str(self.complete(json_prompt, **kw))

    return _AnthropicWithJson(model=model, **kwargs)


def _get_gemini_llm(model: str, api_key: str | None = None, **kwargs: Any) -> Any:
    """Get Google Gemini LLM instance using the new google-genai SDK."""
    logger.debug("initializing_llm", provider="gemini", model=model)
    
    # Try the new google-genai SDK first (recommended)
    try:
        from google import genai
        from google.genai import types
        
        # ------------------------------------------------------------------
        # Exception categories for retry logic
        #
        # SERVER_RETRY_EXCEPTIONS: Transient server-side errors that are worth
        #   retrying *within* the Gemini provider (e.g. 429, 503, 500).
        #
        # CONNECTION_EXCEPTIONS: Network-level failures (refused, reset,
        #   timeout).  These should NOT be retried within Gemini — they need
        #   to propagate to the ResilientLLMWrapper for cross-provider
        #   fallback (e.g. switch to OpenAI/Anthropic).
        # ------------------------------------------------------------------
        try:
            from google.api_core import exceptions as google_exceptions
            SERVER_RETRY_EXCEPTIONS = (
                google_exceptions.ServiceUnavailable,
                google_exceptions.TooManyRequests,
                google_exceptions.InternalServerError,
                google_exceptions.ResourceExhausted,
                google_exceptions.Aborted,
            )
        except ImportError:
            SERVER_RETRY_EXCEPTIONS = (Exception,)

        CONNECTION_EXCEPTIONS = (
            ConnectionError,
            ConnectionRefusedError,
            ConnectionResetError,
            ConnectionAbortedError,
            TimeoutError,
            OSError,
        )

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Create a wrapper that matches LlamaIndex LLM interface
        class GeminiWrapper:
            """Wrapper for google-genai to match LlamaIndex LLM interface."""
            
            def __init__(self, model_name: str, api_key: str, temperature: float = 0.0):
                # Configure HTTP-level timeout so hung connections fail fast
                http_opts = types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000)  # ms
                self.client = genai.Client(
                    api_key=api_key,
                    http_options=http_opts,
                )
                self.model_name = model_name
                self.fallback_model = "gemini-3-flash-preview"
                # Temperature 0 + seed for deterministic outputs
                _seed = int(os.getenv("RNSR_LLM_SEED", "42"))
                self.generation_config = types.GenerateContentConfig(
                    temperature=temperature,
                    seed=_seed,
                    max_output_tokens=int(os.getenv("RNSR_MAX_OUTPUT_TOKENS", "16384")),
                )
                logger.debug(
                    "gemini_client_initialized",
                    model=model_name,
                    timeout_s=REQUEST_TIMEOUT,
                )
            
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=15),
                retry=retry_if_exception_type(SERVER_RETRY_EXCEPTIONS),
            )
            def complete(self, prompt: str, **kw: Any) -> str:
                # Connection-level errors propagate immediately to
                # ResilientLLMWrapper for cross-provider fallback.
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=self.generation_config,
                    )
                    return response.text or ""
                except CONNECTION_EXCEPTIONS:
                    raise  # Let ResilientLLMWrapper handle provider switch
                except SERVER_RETRY_EXCEPTIONS as e:
                    logger.warning(
                        "primary_llm_overloaded_using_fallback", 
                        primary=self.model_name, 
                        fallback=self.fallback_model,
                        error=str(e)[:200],
                    )
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=prompt,
                        config=self.generation_config,
                    )
                    return response.text or ""

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=15),
                retry=retry_if_exception_type(SERVER_RETRY_EXCEPTIONS),
            )
            def complete_json(self, prompt: str, **kw: Any) -> str:
                """Complete expecting a JSON response (uses Gemini JSON mode)."""
                json_config = types.GenerateContentConfig(
                    temperature=self.generation_config.temperature,
                    seed=self.generation_config.seed,
                    max_output_tokens=self.generation_config.max_output_tokens,
                    response_mime_type="application/json",
                )
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=json_config,
                    )
                    return response.text or ""
                except CONNECTION_EXCEPTIONS:
                    raise
                except SERVER_RETRY_EXCEPTIONS as e:
                    logger.warning(
                        "primary_llm_overloaded_using_fallback",
                        primary=self.model_name,
                        fallback=self.fallback_model,
                        error=str(e)[:200],
                    )
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=prompt,
                        config=json_config,
                    )
                    return response.text or ""

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=15),
                retry=retry_if_exception_type(SERVER_RETRY_EXCEPTIONS),
            )
            def complete_with_image(self, prompt: str, image_bytes: bytes, **kw: Any) -> str:
                """Complete a prompt with an image (multimodal)."""
                image_part = types.Part.from_bytes(
                    data=image_bytes, mime_type="image/png",
                )
                contents = [image_part, prompt]
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=self.generation_config,
                    )
                    return response.text or ""
                except CONNECTION_EXCEPTIONS:
                    raise
                except SERVER_RETRY_EXCEPTIONS as e:
                    logger.warning(
                        "primary_llm_overloaded_using_fallback",
                        primary=self.model_name,
                        fallback=self.fallback_model,
                        error=str(e)[:200],
                    )
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=contents,
                        config=self.generation_config,
                    )
                    return response.text or ""

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=15),
                retry=retry_if_exception_type(SERVER_RETRY_EXCEPTIONS),
            )
            def chat(self, messages: list, **kw: Any) -> str:
                # Convert to genai format
                contents = []
                for msg in messages:
                    role = "user" if msg.get("role") == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
                
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=self.generation_config,
                    ) 
                    return response.text or ""
                except CONNECTION_EXCEPTIONS:
                    raise
                except SERVER_RETRY_EXCEPTIONS as e:
                    logger.warning(
                        "primary_llm_overloaded_using_fallback", 
                        primary=self.model_name, 
                        fallback=self.fallback_model,
                        error=str(e)[:200],
                    )
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=contents,
                        config=self.generation_config,
                    )
                    return response.text or ""
        
        return GeminiWrapper(model, api_key)
        
    except ImportError:
        # Fall back to llama-index-llms-gemini (deprecated)
        try:
            from llama_index.llms.gemini import Gemini
            
            try:
                from google.api_core import exceptions as google_exceptions
                _LEGACY_SERVER_RETRY = (
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.TooManyRequests,
                    google_exceptions.InternalServerError,
                    google_exceptions.ResourceExhausted,
                    google_exceptions.Aborted,
                    google_exceptions.DeadlineExceeded,
                )
            except ImportError:
                _LEGACY_SERVER_RETRY = (Exception,)

            _LEGACY_CONN_ERRORS = (
                ConnectionError, ConnectionRefusedError,
                ConnectionResetError, TimeoutError, OSError,
            )

            class LlamaIndexGeminiWrapper:
                """Wrapper for llama-index Gemini to provide fallback logic."""
                
                def __init__(self, model_name: str, **kwargs):
                    self.model_name = model_name
                    self.primary = Gemini(model=model_name, **kwargs)
                    self.fallback_model = "models/gemini-3-flash-preview"
                    self.fallback = Gemini(model=self.fallback_model, **kwargs)
                
                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=15),
                    retry=retry_if_exception_type(_LEGACY_SERVER_RETRY),
                )
                def complete(self, prompt: str, **kw: Any) -> Any:
                    try:
                        return self.primary.complete(prompt, **kw)
                    except _LEGACY_CONN_ERRORS:
                        raise  # Cross-provider fallback
                    except _LEGACY_SERVER_RETRY as e:
                        logger.warning(
                            "primary_llm_overloaded_using_fallback", 
                            primary=self.model_name, 
                            fallback=self.fallback_model,
                            error=str(e)[:200],
                        )
                        return self.fallback.complete(prompt, **kw)

                def complete_json(self, prompt: str, **kw: Any) -> str:
                    """Soft JSON mode for legacy LlamaIndex Gemini path."""
                    json_prompt = (
                        prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                        "No markdown fences, no commentary."
                    )
                    return str(self.complete(json_prompt, **kw))

                def complete_with_image(self, prompt: str, image_bytes: bytes, **kw: Any) -> str:
                    """Multimodal via the new google-genai SDK (bypass LlamaIndex)."""
                    try:
                        from google import genai
                        from google.genai import types as genai_types
                    except ImportError:
                        logger.warning("google_genai_not_available_for_multimodal")
                        return self.complete(prompt, **kw)

                    api_key = os.getenv("GOOGLE_API_KEY")
                    http_opts = genai_types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000)
                    client = genai.Client(api_key=api_key, http_options=http_opts)
                    image_part = genai_types.Part.from_bytes(
                        data=image_bytes, mime_type="image/png",
                    )
                    contents = [image_part, prompt]
                    response = client.models.generate_content(
                        model=self.model_name.replace("models/", ""),
                        contents=contents,
                    )
                    return response.text or ""

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=15),
                    retry=retry_if_exception_type(_LEGACY_SERVER_RETRY),
                )
                def chat(self, messages: Any, **kw: Any) -> Any:
                    try:
                        return self.primary.chat(messages, **kw)
                    except _LEGACY_CONN_ERRORS:
                        raise
                    except _LEGACY_SERVER_RETRY as e:
                        logger.warning(
                            "primary_llm_overloaded_using_fallback", 
                            primary=self.model_name, 
                            fallback=self.fallback_model,
                            error=str(e)[:200],
                        )
                        return self.fallback.chat(messages, **kw)

                def __getattr__(self, name: str) -> Any:
                    return getattr(self.primary, name)

            return LlamaIndexGeminiWrapper(model, **kwargs)
        except ImportError:
            raise ImportError(
                "Neither google-genai nor llama-index-llms-gemini installed. "
                "Install with: pip install google-genai"
            )


def _get_openai_embed(model: str, **kwargs: Any) -> Any:
    """Get OpenAI embedding model."""
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError:
        raise ImportError(
            "OpenAI embeddings not installed. "
            "Install with: pip install llama-index-embeddings-openai"
        )
    
    logger.debug("initializing_embed", provider="openai", model=model)
    return OpenAIEmbedding(model=model, **kwargs)


def _get_gemini_embed(model: str, **kwargs: Any) -> Any:
    """Get Google Gemini embedding model."""
    try:
        from llama_index.embeddings.gemini import GeminiEmbedding
    except ImportError:
        raise ImportError(
            "Gemini embeddings not installed. "
            "Install with: pip install llama-index-embeddings-gemini"
        )
    
    logger.debug("initializing_embed", provider="gemini", model=model)
    return GeminiEmbedding(model_name=f"models/{model}", **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_provider_info() -> dict[str, Any]:
    """
    Get information about available providers.
    
    Returns:
        Dictionary with provider availability and configuration.
    """
    info = {
        "available": [],
        "default_provider": None,
        "models": DEFAULT_MODELS,
    }
    
    if os.getenv("OPENAI_API_KEY"):
        info["available"].append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        info["available"].append("anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        info["available"].append("gemini")
    
    if info["available"]:
        try:
            info["default_provider"] = detect_provider().value
        except ValueError:
            pass
    
    return info


def validate_provider(provider: LLMProvider, api_key: str | None = None) -> bool:
    """
    Check if a provider is available (has API key set or provided).
    
    Args:
        provider: Provider to check.
        api_key: Optional explicit API key. If provided, the provider
                 is considered available regardless of environment variables.
        
    Returns:
        True if provider is available.
    """
    if api_key:
        return True
    if provider == LLMProvider.OPENAI:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == LLMProvider.ANTHROPIC:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == LLMProvider.GEMINI:
        return bool(os.getenv("GOOGLE_API_KEY"))
    elif provider == LLMProvider.AUTO:
        return any([
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
        ])
    return False
