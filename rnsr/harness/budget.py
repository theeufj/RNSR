"""Optional per-query budgets: zero disables a cap, while usage is always metered.

An explicitly configured positive cap fires the variable-recovery fallback
and labels the answer budget_exhausted when reached.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from rnsr.config import Settings
from rnsr.errors import BudgetExhausted
from rnsr.llm.base import Usage

_USAGE_FIELDS = {
    "max_root_iters": "root_iters",
    "max_sub_calls": "sub_calls",
    "max_wall_s": "wall_s",
    "max_spend_usd": "spend_usd",
}


@dataclass
class BudgetLedger:
    max_root_iters: int = 0
    max_sub_calls: int = 0
    max_wall_s: float = 0.0
    max_spend_usd: float = 0.0

    root_iters: int = 0
    sub_calls: int = 0
    spend_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    _t0: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls, s: Settings) -> BudgetLedger:
        return cls(s.max_root_iters, s.max_sub_calls, s.max_wall_s, s.max_spend_usd)

    @property
    def wall_s(self) -> float:
        return time.monotonic() - self._t0

    @property
    def uncapped(self) -> bool:
        """Whether every query limit is disabled."""
        return all(getattr(self, cap) == 0 for cap in _USAGE_FIELDS)

    def add_usage(self, usage: Usage, *, sub_call: bool = False) -> None:
        self.usage = self.usage + usage
        self.spend_usd += usage.cost_usd
        if sub_call:
            self.sub_calls += 1

    def reserve_sub_call(self) -> None:
        """Charge an attempted provider request before dispatch, including retries."""
        for cap in ("max_sub_calls", "max_wall_s", "max_spend_usd"):
            if self.limit_reached(cap):
                raise BudgetExhausted(cap, getattr(self, cap),
                                      getattr(self, _USAGE_FIELDS[cap]))
        self.sub_calls += 1

    def limit_reached(self, name: str) -> bool:
        """Whether a positive query cap has been reached; zero is unlimited."""
        if name not in _USAGE_FIELDS:
            raise ValueError(f"unknown query budget cap: {name}")
        limit = getattr(self, name)
        return limit > 0 and getattr(self, _USAGE_FIELDS[name]) >= limit

    def breached(self) -> str | None:
        """Name of the first reached positive cap, or None."""
        return next((cap for cap in _USAGE_FIELDS if self.limit_reached(cap)), None)

    def remaining_sub_calls(self) -> int | None:
        """Remaining attempted requests, or None when the cap is disabled."""
        if self.max_sub_calls == 0:
            return None
        return max(self.max_sub_calls - self.sub_calls, 0)

    def remaining_root_iters(self) -> int | None:
        """Remaining root iterations, or None when the cap is disabled."""
        if self.max_root_iters == 0:
            return None
        return max(self.max_root_iters - self.root_iters, 0)

    def deadline(self) -> float | None:
        """Monotonic query deadline, or None when query time is unlimited."""
        return self._t0 + self.max_wall_s if self.max_wall_s > 0 else None

    def remaining_wall_s(self) -> float:
        """Seconds to the query deadline; infinity when the cap is disabled."""
        if self.max_wall_s == 0:
            return math.inf
        return max(self.max_wall_s - self.wall_s, 0.0)

    def snapshot(self) -> dict:
        return {
            "root_iters": self.root_iters,
            "sub_calls": self.sub_calls,
            "wall_s": round(self.wall_s, 2),
            "spend_usd": round(self.spend_usd, 6),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
        }
