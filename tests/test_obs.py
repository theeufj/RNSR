"""Logging configuration and the metrics registry."""

import json
import logging

import pytest

from rnsr.config import Settings
from rnsr.obs import (
    ROOT_LOGGER,
    Metrics,
    configure_logging,
    get_logger,
    log,
    metrics,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def clean_metrics():
    reset_metrics()
    yield
    reset_metrics()


class TestLogging:
    def test_json_format_hoists_fields(self, capsys):
        configure_logging(Settings(log_level="INFO", log_format="json"), force=True)
        log(get_logger("test"), logging.INFO, "query.end",
            query_id="q001", status="final", spend_usd=0.12)
        record = json.loads(capsys.readouterr().err.strip())
        assert record["event"] == "query.end"
        assert record["query_id"] == "q001"
        assert record["status"] == "final"
        assert record["level"] == "INFO"
        assert record["logger"] == "rnsr.test"

    def test_log_level_is_honoured(self, capsys):
        # the bug this closes: Settings.log_level was read by nothing
        configure_logging(Settings(log_level="WARNING", log_format="json"),
                          force=True)
        logger = get_logger("test")
        log(logger, logging.INFO, "chatter")
        log(logger, logging.WARNING, "trouble")
        lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "trouble"

    def test_logs_go_to_stderr_not_stdout(self, capsys):
        configure_logging(Settings(log_format="json"), force=True)
        log(get_logger("test"), logging.INFO, "event")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "event" in captured.err

    def test_text_format_is_human_readable(self, capsys):
        configure_logging(Settings(log_format="text"), force=True)
        log(get_logger("test"), logging.INFO, "query.start")
        assert "query.start" in capsys.readouterr().err

    def test_does_not_touch_the_root_logger(self):
        configure_logging(Settings(), force=True)
        assert logging.getLogger(ROOT_LOGGER).propagate is False
        assert logging.getLogger().handlers == logging.getLogger().handlers


class TestMetrics:
    def test_counters_and_labels(self):
        m = Metrics()
        m.incr("queries_finished", status="final")
        m.incr("queries_finished", status="final")
        m.incr("queries_finished", status="error")
        snap = m.snapshot()
        assert snap["counters"]["queries_finished{status=final}"] == 2
        assert snap["counters"]["queries_finished{status=error}"] == 1

    def test_observations_summarise_as_percentiles(self):
        m = Metrics()
        for v in (1, 2, 3, 4, 100):
            m.observe("query_latency_s", v)
        obs = m.snapshot()["observations"]["query_latency_s"]
        assert obs["count"] == 5
        assert obs["max"] == 100
        assert obs["p50"] == 3

    def test_observation_memory_is_bounded(self):
        m = Metrics(max_observations=10)
        for i in range(500):
            m.observe("x", i)
        assert m.snapshot()["observations"]["x"]["count"] == 10

    async def test_loop_records_query_metrics(self, tmp_path):
        from rnsr.harness.loop import EnvSpec, RootRunner
        from rnsr.llm.mock import MockLLM

        root = MockLLM().script("```python\nFINAL('42')\n```")
        runner = RootRunner(root_client=root, root_model="m",
                            sub_client=MockLLM(default="COMPLETE"),
                            sub_model="s", settings=Settings())
        await runner.run("q", EnvSpec(mode="classic", context="x"),
                         run_dir=tmp_path, query_id="q001")
        snap = metrics().snapshot()
        assert snap["counters"]["queries_started{mode=classic}"] == 1
        assert snap["counters"]["queries_finished{status=final}"] == 1
        assert snap["observations"]["query_latency_s"]["count"] == 1
