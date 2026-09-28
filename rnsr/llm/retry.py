"""Provider refusals that waiting and retrying cannot repair."""

from __future__ import annotations

from rnsr.errors import PermanentProviderError

_TERMINAL_MARKERS = (
    "insufficient_quota",
    "billing_hard_limit_reached",
    "billing_not_active",
    "credit balance is too low",
    "credit balance too low",
    "insufficient credits",
    "insufficient credit balance",
    "exceeded your current quota, please check your plan and billing details",
)


def is_terminal_provider_error(exc: BaseException) -> bool:
    """Recognize terminal 4xx/credit refusals before transient text matching.

    Provider billing refusals sometimes use HTTP 429, just like a temporary
    rate limit. Check their structured identifiers as well as known messages
    so uncapped queries do not retry an account that needs operator action.
    Ordinary 429s and time-based quota windows remain eligible for retry.
    """
    if isinstance(exc, PermanentProviderError):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429):
        return True
    body = getattr(exc, "body", None)
    details = f"{getattr(exc, 'code', '')} {body if isinstance(body, dict) else ''} {exc}".lower()
    return any(marker in details for marker in _TERMINAL_MARKERS)
