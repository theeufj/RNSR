"""Exception hierarchy for rnsr."""


class RNSRError(Exception):
    """Base class for all rnsr errors."""


class IngestError(RNSRError):
    """Unrecoverable failure during ingestion."""


class CorpusHealthError(RNSRError):
    """Answering refused because corpus health is blocked.

    ``health`` is the CorpusHealth that failed the gate. Pass
    ``--allow-degraded`` / ``Settings.allow_degraded`` to proceed anyway;
    answers are then stamped with the health record.
    """

    def __init__(self, health):
        self.health = health
        findings = "; ".join(f.detail for f in health.findings if f.severity == "error")
        super().__init__(
            f"corpus health is blocked ({health.grade}): {findings or 'see health.findings'}"
        )


class TableValidationError(RNSRError):
    """A table failed checksum validation in a way that cannot be retried."""


class ImmutableTableError(RNSRError):
    """Attempted write to source data protected by immutability triggers."""


class ArtifactVersionError(RNSRError):
    """corpus.db is missing required tables or is a different format version.

    ``rnsr migrate <corpus.db>`` stamps ``user_version`` / ``format_version``
    on artifacts created before versioning existed.
    """


class SandboxError(RNSRError):
    """The sandboxed REPL child failed, was killed, or violated a limit."""


class BudgetExhausted(RNSRError):
    """A query-time budget cap (iterations, sub-calls, wall-clock, spend) was breached."""

    def __init__(self, cap: str, limit: float, spent: float):
        self.cap = cap
        self.limit = limit
        self.spent = spent
        super().__init__(f"budget cap '{cap}' exhausted: {spent} >= {limit}")
