"""Logging and metrics.

Until now the only machine-readable output was the per-query trajectory
JSONL, which is superb forensics for one query and useless for operating a
fleet of them: nothing said how many queries were in flight, what the run
had spent, or that a provider had started throttling. Settings.log_level
existed and was read by nothing.

This module supplies both halves, with no new dependencies:

  - configure_logging() installs a handler on the 'rnsr' logger. Text
    format for humans at a terminal; JSON lines for log shippers, where
    every structured field lands as a top-level key.
  - a process-wide Metrics registry of counters and observations, dumped
    into run reports. Counters and percentiles are what a dashboard needs
    and what a trajectory cannot give.

Nothing here changes control flow: an observability call must never be
able to fail a query, so the registry is lock-free arithmetic and the
loggers are stdlib.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field

ROOT_LOGGER = "rnsr"

_configured = False


class JsonFormatter(logging.Formatter):
    """One JSON object per line, structured fields hoisted to top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=repr)


def configure_logging(settings=None, *, level: str | None = None,
                      fmt: str | None = None, force: bool = False) -> None:
    """Install the rnsr log handler. Idempotent unless force=True.

    Writes to stderr so structured logs never contaminate stdout, which
    the CLI uses for results a caller may be piping.
    """
    global _configured
    if _configured and not force:
        return
    if settings is not None:
        level = level or settings.log_level
        fmt = fmt or getattr(settings, "log_format", "text")
    level, fmt = (level or "INFO").upper(), (fmt or "text").lower()

    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False        # the root logger is the caller's to own
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Logger under the rnsr tree; module names arrive already namespaced."""
    return logging.getLogger(name if name.startswith(ROOT_LOGGER)
                             else f"{ROOT_LOGGER}.{name}")


def log(logger: logging.Logger, level: int, event: str, **fields) -> None:
    """Emit `event` with structured fields (top-level keys in JSON format)."""
    logger.log(level, event, extra={"fields": fields})


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[idx]


def _key(name: str, labels: dict) -> str:
    if not labels:
        return name
    inner = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


@dataclass
class Metrics:
    """Counters and observations for one process. Cheap, unsynchronized:
    asyncio gives single-threaded arithmetic, and a lost increment must
    never be worth failing a query over."""

    counters: dict[str, float] = field(default_factory=dict)
    observations: dict[str, list[float]] = field(default_factory=dict)
    max_observations: int = 5000     # bounded memory on long-lived services

    def incr(self, name: str, by: float = 1.0, **labels) -> None:
        key = _key(name, labels)
        self.counters[key] = self.counters.get(key, 0.0) + by

    def observe(self, name: str, value: float, **labels) -> None:
        series = self.observations.setdefault(_key(name, labels), [])
        if len(series) < self.max_observations:
            series.append(float(value))

    def snapshot(self) -> dict:
        out: dict = {"counters": {k: round(v, 6)
                                 for k, v in sorted(self.counters.items())}}
        out["observations"] = {
            key: {
                "count": len(series),
                "p50": round(_percentile(series, 0.50), 3),
                "p95": round(_percentile(series, 0.95), 3),
                "max": round(max(series), 3),
                "sum": round(sum(series), 3),
            }
            for key, series in sorted(self.observations.items())
        }
        return out

    def reset(self) -> None:
        self.counters.clear()
        self.observations.clear()


_METRICS = Metrics()


def metrics() -> Metrics:
    return _METRICS


def reset_metrics() -> Metrics:
    _METRICS.reset()
    return _METRICS
