"""Exception hierarchy for rnsr."""


class RNSRError(Exception):
    """Base class for all rnsr errors."""


class IngestError(RNSRError):
    """Unrecoverable failure during ingestion."""


class TableValidationError(RNSRError):
    """A table failed checksum validation in a way that cannot be retried."""


class ImmutableTableError(RNSRError):
    """Attempted write to source data protected by immutability triggers."""


class SandboxError(RNSRError):
    """The sandboxed REPL child failed, was killed, or violated a limit."""


class BudgetExhausted(RNSRError):
    """A query-time budget cap (iterations, sub-calls, wall-clock, spend) was breached."""

    def __init__(self, cap: str, limit: float, spent: float):
        self.cap = cap
        self.limit = limit
        self.spent = spent
        super().__init__(f"budget cap '{cap}' exhausted: {spent} >= {limit}")
