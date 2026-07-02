"""Hard budget caps per query (spec §7). All configurable; breach fires the
variable-recovery fallback and labels the answer budget_exhausted."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rnsr.config import Settings
from rnsr.llm.base import Usage


@dataclass
class BudgetLedger:
    max_root_iters: int = 20
    max_sub_calls: int = 300
    max_wall_s: float = 600.0
    max_spend_usd: float = 2.0

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

    def add_usage(self, usage: Usage, *, sub_call: bool = False) -> None:
        self.usage = self.usage + usage
        self.spend_usd += usage.cost_usd
        if sub_call:
            self.sub_calls += 1

    def breached(self) -> str | None:
        """Name of the first breached cap, or None."""
        if self.root_iters >= self.max_root_iters:
            return "max_root_iters"
        if self.sub_calls >= self.max_sub_calls:
            return "max_sub_calls"
        if self.wall_s >= self.max_wall_s:
            return "max_wall_s"
        if self.spend_usd >= self.max_spend_usd:
            return "max_spend_usd"
        return None

    def remaining_wall_s(self) -> float:
        return max(self.max_wall_s - self.wall_s, 1.0)

    def snapshot(self) -> dict:
        return {
            "root_iters": self.root_iters,
            "sub_calls": self.sub_calls,
            "wall_s": round(self.wall_s, 2),
            "spend_usd": round(self.spend_usd, 6),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
        }
