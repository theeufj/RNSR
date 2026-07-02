"""LLM-backed ingest hooks on MockLLM (+ real rasterization on the fixture PDF)."""

import json

import pytest

from rnsr.ingest.llm_hooks import _parse_grid, make_prose_checker
from rnsr.llm.mock import MockLLM


class TestProseChecker:
    def test_yes_no_unclear_parsing(self):
        mock = MockLLM().script("YES, the prose states it.", "No.", "UNCLEAR")
        check = make_prose_checker(mock, "mock-sub")
        assert check(["p1", "p2", "p3"]) == [True, False, None]

    def test_failed_call_is_none(self):
        mock = MockLLM(fail_times=99)
        check = make_prose_checker(mock, "mock-sub")
        assert check(["p"]) == [None]


class TestGridParsing:
    def test_valid_json_with_fence(self):
        text = "```json\n" + json.dumps(
            {"header": ["A", "B"], "rows": [["1", None]]}
        ) + "\n```"
        grid = _parse_grid(text)
        assert grid["header"] == ["A", "B"]

    def test_garbage_returns_none(self):
        assert _parse_grid("I could not find a table.") is None
        assert _parse_grid('{"rows": "not-a-list", "header": []}') is None


class TestVisionExtractor:
    def test_end_to_end_on_fixture(self, fixture_pdf):
        pytest.importorskip("pypdfium2")
        from rnsr.ingest.llm_hooks import make_vision_extractor

        scripted = json.dumps({
            "header": ["Segment", "Revenue ($M)"],
            "rows": [["Widgets", "$1,234"], ["Total", "$1,234"]],
        })
        mock = MockLLM(default=scripted)
        extract = make_vision_extractor(mock, "mock-vision")
        table = extract(fixture_pdf, 1)
        assert table is not None
        assert table.extractor == "vision"
        assert table.rows[0] == ["Widgets", "$1,234"]
        # a real PNG went up to the model
        assert mock.calls[0]["kind"] == "vision"
        assert mock.calls[0]["bytes"] > 10_000

    def test_no_table_returns_none(self, fixture_pdf):
        pytest.importorskip("pypdfium2")
        from rnsr.ingest.llm_hooks import make_vision_extractor

        mock = MockLLM(default="There is no table on this page.")
        assert make_vision_extractor(mock, "m")(fixture_pdf, 1) is None
