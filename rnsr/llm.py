"""
LLM Configuration - Multi-Provider LLM and Embedding Support

Supports:
- OpenAI (GPT-4, text-embedding-3-small)
- Anthropic (Claude)
- Google Gemini (gemini-pro, text-embedding-004)
- Ollama (local; default llm qwen2.5-coder:32b, embed nomic-embed-text)

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

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    
    # Look for .env in the project root (parent of rnsr package)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system environment

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
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
        "embed": "text-embedding-004",
    },
    LLMProvider.OLLAMA: {
        "llm": "qwen2.5-coder:32b",  # Override via OLLAMA_MODEL env
        "embed": "nomic-embed-text",  # User pulls via ollama pull nomic-embed-text
    },
}

# Fallback chain when a provider hits rate limits
PROVIDER_FALLBACK_CHAIN = {
    LLMProvider.GEMINI: [LLMProvider.OPENAI, LLMProvider.ANTHROPIC],
    LLMProvider.OPENAI: [LLMProvider.ANTHROPIC, LLMProvider.GEMINI],
    LLMProvider.ANTHROPIC: [LLMProvider.OPENAI, LLMProvider.GEMINI],
    LLMProvider.OLLAMA: [],  # No cloud fallback from Ollama by default
}


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


def get_available_fallback_providers(primary: LLMProvider) -> list[LLMProvider]:
    """Get list of available fallback providers for a given primary provider."""
    fallbacks = []
    for provider in PROVIDER_FALLBACK_CHAIN.get(primary, []):
        if validate_provider(provider):
            fallbacks.append(provider)
    return fallbacks


def detect_provider() -> LLMProvider:
    """
    Auto-detect LLM provider from environment variables.
    
    Checks in order:
    1. GOOGLE_API_KEY -> Gemini
    2. ANTHROPIC_API_KEY -> Anthropic
    3. OPENAI_API_KEY -> OpenAI
    4. OLLAMA_BASE_URL or USE_OLLAMA -> Ollama (local)
    
    Returns:
        Detected LLMProvider.
        
    Raises:
        ValueError: If no provider is configured.
    """
    if os.getenv("GOOGLE_API_KEY"):
        logger.info("provider_detected", provider="gemini")
        return LLMProvider.GEMINI
    
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("provider_detected", provider="anthropic")
        return LLMProvider.ANTHROPIC
    
    if os.getenv("OPENAI_API_KEY"):
        logger.info("provider_detected", provider="openai")
        return LLMProvider.OPENAI
    
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("USE_OLLAMA"):
        logger.info("provider_detected", provider="ollama")
        return LLMProvider.OLLAMA
    
    raise ValueError(
        "No LLM provider found. Set one of: "
        "GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or OLLAMA_BASE_URL/USE_OLLAMA"
    )


def get_llm(
    provider: LLMProvider = LLMProvider.AUTO,
    model: str | None = None,
    enable_fallback: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Get an LLM instance for the specified provider.
    
    Args:
        provider: LLM provider (openai, anthropic, gemini, ollama, or auto).
        model: Model name override. Uses default if not specified.
        enable_fallback: If True, enables cross-provider fallback on rate limits.
        **kwargs: Additional arguments passed to the LLM constructor.
        
    Returns:
        LlamaIndex-compatible LLM instance with fallback support.
        
    Example:
        llm = get_llm(provider=LLMProvider.GEMINI)
        response = await llm.acomplete("Hello!")
    """
    if provider == LLMProvider.AUTO:
        provider = detect_provider()
    
    # For Ollama, allow .env OLLAMA_MODEL to override default
    if provider == LLMProvider.OLLAMA:
        model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODELS[provider]["llm"]
    else:
        model = model or DEFAULT_MODELS[provider]["llm"]
    
    # Get primary LLM
    primary_llm = _get_raw_llm(provider, model, **kwargs)
    
    if not enable_fallback:
        return primary_llm
    
    # Build fallback chain
    fallback_providers = get_available_fallback_providers(provider)
    if not fallback_providers:
        logger.debug("no_fallback_providers_available", primary=provider.value)
        return primary_llm
    
    logger.debug(
        "llm_with_fallback_configured",
        primary=provider.value,
        fallbacks=[p.value for p in fallback_providers],
    )
    
    return ResilientLLMWrapper(
        primary_llm=primary_llm,
        primary_provider=provider,
        fallback_providers=fallback_providers,
        **kwargs,
    )


def _get_raw_llm(provider: LLMProvider, model: str, **kwargs: Any) -> Any:
    """Get a raw LLM instance without fallback wrapper."""
    if provider == LLMProvider.OPENAI:
        return _get_openai_llm(model, **kwargs)
    elif provider == LLMProvider.ANTHROPIC:
        return _get_anthropic_llm(model, **kwargs)
    elif provider == LLMProvider.GEMINI:
        return _get_gemini_llm(model, **kwargs)
    elif provider == LLMProvider.OLLAMA:
        return _get_ollama_llm(model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embed_model(
    provider: LLMProvider = LLMProvider.AUTO,
    model: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Get an embedding model for the specified provider.
    
    Args:
        provider: LLM provider (openai, gemini, ollama, or auto).
        model: Model name override. Uses default if not specified.
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
    
    if provider == LLMProvider.OPENAI:
        return _get_openai_embed(model, **kwargs)
    elif provider == LLMProvider.GEMINI:
        return _get_gemini_embed(model, **kwargs)
    elif provider == LLMProvider.OLLAMA:
        return _get_ollama_embed(model, **kwargs)
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
        """Call a method with automatic fallback on rate limits."""
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
                    else:
                        # Non-rate-limit error - retry with exponential backoff
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
                if is_rate_limit_error(e):
                    self._mark_rate_limited(provider)
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


def _get_openai_llm(model: str, **kwargs: Any) -> Any:
    """Get OpenAI LLM instance."""
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI LLM not installed. "
            "Install with: pip install llama-index-llms-openai"
        )
    
    # Set temperature=0 for deterministic outputs unless overridden
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0
    
    logger.debug("initializing_llm", provider="openai", model=model)
    return OpenAI(model=model, **kwargs)


def _get_anthropic_llm(model: str, **kwargs: Any) -> Any:
    """Get Anthropic LLM instance."""
    try:
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Anthropic LLM not installed. "
            "Install with: pip install llama-index-llms-anthropic"
        )
    
    # Set temperature=0 for deterministic outputs unless overridden
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0
    
    logger.debug("initializing_llm", provider="anthropic", model=model)
    return Anthropic(model=model, **kwargs)


def _get_gemini_llm(model: str, **kwargs: Any) -> Any:
    """Get Google Gemini LLM instance using the new google-genai SDK."""
    logger.debug("initializing_llm", provider="gemini", model=model)
    
    # Try the new google-genai SDK first (recommended)
    try:
        from google import genai
        from google.genai import types
        
        # Define exceptions to retry on
        # If google.api_core is available (usually is with google SDKs)
        try:
            from google.api_core import exceptions as google_exceptions
            RETRY_EXCEPTIONS = (
                google_exceptions.ServiceUnavailable,
                google_exceptions.TooManyRequests,
                google_exceptions.InternalServerError,
                google_exceptions.ResourceExhausted,
                google_exceptions.Aborted,
                ConnectionError,
                ConnectionRefusedError,
                TimeoutError,
                OSError,  # Covers [Errno 61] and other socket errors
            )
        except ImportError:
            # Fallback: Retry on any Exception that mentions overload/503/429
            # But simpler to just retry on Exception if we can't import specific ones
            RETRY_EXCEPTIONS = (Exception,)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Create a wrapper that matches LlamaIndex LLM interface
        class GeminiWrapper:
            """Wrapper for google-genai to match LlamaIndex LLM interface."""
            
            def __init__(self, model_name: str, api_key: str, temperature: float = 0.0):
                self.client = genai.Client(api_key=api_key)
                self.model_name = model_name
                self.fallback_model = "gemini-3-flash-preview"
                # Temperature 0 for deterministic outputs
                self.generation_config = types.GenerateContentConfig(
                    temperature=temperature,
                )
            
            @retry(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                retry=retry_if_exception_type(RETRY_EXCEPTIONS),
            )
            def complete(self, prompt: str, **kw: Any) -> str:
                try:
                    # Try primary model first with temperature=0 for determinism
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=self.generation_config,
                    )
                    return response.text or ""
                except RETRY_EXCEPTIONS as e:
                    # Fallback to preview model on overload/exhaustion
                    logger.warning(
                        "primary_llm_overloaded_using_fallback", 
                        primary=self.model_name, 
                        fallback=self.fallback_model,
                        error=str(e)
                    )
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=prompt,
                        config=self.generation_config,
                    )
                    return response.text or ""
            
            @retry(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                retry=retry_if_exception_type(RETRY_EXCEPTIONS),
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
                except RETRY_EXCEPTIONS as e:
                    logger.warning(
                        "primary_llm_overloaded_using_fallback",
                        primary=self.model_name,
                        fallback=self.fallback_model,
                        error=str(e),
                    )
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=contents,
                        config=self.generation_config,
                    )
                    return response.text or ""

            @retry(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                retry=retry_if_exception_type(RETRY_EXCEPTIONS),
            )
            def chat(self, messages: list, **kw: Any) -> str:
                # Convert to genai format
                contents = []
                for msg in messages:
                    role = "user" if msg.get("role") == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
                
                try:
                    # Try primary model first with temperature=0 for determinism
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=self.generation_config,
                    ) 
                    return response.text or ""
                except RETRY_EXCEPTIONS as e:
                    # Fallback to preview model
                    logger.warning(
                        "primary_llm_overloaded_using_fallback", 
                        primary=self.model_name, 
                        fallback=self.fallback_model,
                        error=str(e)
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
            
            # Define exceptions for legacy/llama-index path
            try:
                from google.api_core import exceptions as google_exceptions
                RETRY_EXCEPTIONS_LEGACY = (
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.TooManyRequests,
                    google_exceptions.InternalServerError,
                    google_exceptions.ResourceExhausted,
                    google_exceptions.Aborted,
                    google_exceptions.DeadlineExceeded,
                    ConnectionError,
                    ConnectionRefusedError,
                    TimeoutError,
                    OSError,
                )
            except ImportError:
                RETRY_EXCEPTIONS_LEGACY = (Exception,)

            class LlamaIndexGeminiWrapper:
                """Wrapper for llama-index Gemini to provide fallback logic."""
                
                def __init__(self, model_name: str, **kwargs):
                    self.model_name = model_name
                    self.primary = Gemini(model=model_name, **kwargs)
                    # Fallback to older stable model or preview
                    self.fallback_model = "models/gemini-3-flash-preview"
                    self.fallback = Gemini(model=self.fallback_model, **kwargs)
                
                @retry(
                    stop=stop_after_attempt(5),
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                    retry=retry_if_exception_type(RETRY_EXCEPTIONS_LEGACY),
                )
                def complete(self, prompt: str, **kw: Any) -> Any:
                    try:
                        return self.primary.complete(prompt, **kw)
                    except RETRY_EXCEPTIONS_LEGACY as e:
                        logger.warning(
                            "primary_llm_overloaded_using_fallback", 
                            primary=self.model_name, 
                            fallback=self.fallback_model,
                            error=str(e)
                        )
                        return self.fallback.complete(prompt, **kw)

                def complete_with_image(self, prompt: str, image_bytes: bytes, **kw: Any) -> str:
                    """Multimodal via the new google-genai SDK (bypass LlamaIndex)."""
                    try:
                        from google import genai
                        from google.genai import types as genai_types
                    except ImportError:
                        logger.warning("google_genai_not_available_for_multimodal")
                        return self.complete(prompt, **kw)

                    api_key = os.getenv("GOOGLE_API_KEY")
                    client = genai.Client(api_key=api_key)
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
                    stop=stop_after_attempt(5),
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                    retry=retry_if_exception_type(RETRY_EXCEPTIONS_LEGACY),
                )
                def chat(self, messages: Any, **kw: Any) -> Any:
                    try:
                        return self.primary.chat(messages, **kw)
                    except RETRY_EXCEPTIONS_LEGACY as e:
                        logger.warning(
                            "primary_llm_overloaded_using_fallback", 
                            primary=self.model_name, 
                            fallback=self.fallback_model,
                            error=str(e)
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


def _get_ollama_llm(model: str, **kwargs: Any) -> Any:
    """Get Ollama LLM instance (local)."""
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError:
        raise ImportError(
            "Ollama LLM not installed. "
            "Install with: pip install llama-index-llms-ollama"
        )
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resolved_model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0
    if "request_timeout" not in kwargs:
        kwargs["request_timeout"] = timeout
    logger.debug("initializing_llm", provider="ollama", model=resolved_model, timeout=timeout)
    return Ollama(model=resolved_model, base_url=base_url, **kwargs)


def _get_ollama_embed(model: str, **kwargs: Any) -> Any:
    """Get Ollama embedding model (local)."""
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
    except ImportError:
        raise ImportError(
            "Ollama embeddings not installed. "
            "Install with: pip install llama-index-embeddings-ollama"
        )
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.debug("initializing_embed", provider="ollama", model=model)
    return OllamaEmbedding(model_name=model, base_url=base_url, **kwargs)


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
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("USE_OLLAMA"):
        info["available"].append("ollama")
    
    if info["available"]:
        try:
            info["default_provider"] = detect_provider().value
        except ValueError:
            pass
    
    return info


def validate_provider(provider: LLMProvider) -> bool:
    """
    Check if a provider is available (has API key set).
    
    Args:
        provider: Provider to check.
        
    Returns:
        True if provider is available.
    """
    if provider == LLMProvider.OPENAI:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == LLMProvider.ANTHROPIC:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == LLMProvider.GEMINI:
        return bool(os.getenv("GOOGLE_API_KEY"))
    elif provider == LLMProvider.OLLAMA:
        return bool(os.getenv("OLLAMA_BASE_URL") or os.getenv("USE_OLLAMA"))
    elif provider == LLMProvider.AUTO:
        return any([
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("OLLAMA_BASE_URL"),
            os.getenv("USE_OLLAMA"),
        ])
    return False
