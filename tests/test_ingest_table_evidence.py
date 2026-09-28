"""Headerless observations survive, and table renders cannot validate themselves."""

import sqlite3
from types import SimpleNamespace

import pytest

from rnsr.ingest.fallback import reextract_pdfplumber
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.parse import _grid_from_table, render_table_text
from rnsr.ingest.pipeline import ingest
from rnsr.ingest.validate import assign_table_status, validate_table


def docling_item(grid, header_rows=()):
    cells = [SimpleNamespace(
        start_row_offset_idx=r, end_row_offset_idx=r + 1,
        start_col_offset_idx=c, end_col_offset_idx=c + 1,
        text=value, column_header=r in header_rows,
    ) for r, row in enumerate(grid) for c, value in enumerate(row)]
    return SimpleNamespace(data=SimpleNamespace(
        num_cols=len(grid[0]), num_rows=len(grid), table_cells=cells))


@pytest.mark.parametrize("grid", [
    [["For", "1,125"], ["Against", "65"], ["Abstain", "2"]],
    [["North", "Active"], ["South", "Closed"]],
    [["North", "Not disclosed"], ["South", "20"]],
    [["North", "Pending"], ["South", "20"]],
    [["2023", "10"], ["2024", "20"]],
    [["Only observation", "17"]],
])
def test_unflagged_docling_grid_preserves_every_observation(grid):
    header, rows = _grid_from_table(docling_item(grid))
    assert header == ["column_1", "column_2"]
    assert rows == grid


def test_only_leading_header_flags_are_consumed():
    grid = [["Region", "Sales"], ["North", "10"], ["Region", "Sales"], ["South", "20"]]
    header, rows = _grid_from_table(docling_item(grid, header_rows=(0, 2)))
    assert header == ["Region", "Sales"]
    assert rows == grid[1:]  # a repeated header cannot swallow the preceding data
    validation = validate_table(RawTable(page=1, header=header, rows=rows))
    repeat = next(d for d in validation.checks["structural"].details
                  if d["check"] == "no_header_repeats")
    assert not repeat["passed"]


def test_unflagged_text_header_requires_numeric_type_contrast():
    header, rows = _grid_from_table(docling_item(
        [["Region", "Revenue ($M)"], ["North", "10"], ["South", "20"]]))
    assert header == ["Region", "Revenue ($M)"]
    assert rows == [["North", "10"], ["South", "20"]]
    grid = [["North", None], ["South", "20"]]
    assert _grid_from_table(docling_item(grid))[1] == grid


def test_explicit_multilevel_and_header_only_grids():
    header, rows = _grid_from_table(docling_item(
        [["", "Revenue"], ["Region", "2024"], ["North", "10"]], (0, 1)))
    assert header == ["Region", "Revenue 2024"]
    assert rows == [["North", "10"]]
    assert _grid_from_table(docling_item([["Region", "Revenue"]], (0,)))[1] == []


@pytest.mark.parametrize("grid, expected_header, expected_rows", [
    ([["For", "1,125"], ["Against", "65"]], ["column_1", "column_2"], 2),
    ([["Region", "2024"], ["North", "10"]], ["Region", "2024"], 1),
    ([["Only observation", "$17.50"]], ["column_1", "column_2"], 1),
])
def test_pdfplumber_does_not_reintroduce_numeric_header_loss(
        monkeypatch, grid, expected_header, expected_rows):
    import pdfplumber

    found = SimpleNamespace(bbox=(0, 0, 100, 100), extract=lambda: grid)
    page = SimpleNamespace(height=200, find_tables=lambda: [found])

    class PDF:
        pages = [page]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(pdfplumber, "open", lambda _path: PDF())
    table = reextract_pdfplumber("unused.pdf", 1)
    assert table is not None
    assert table.header == expected_header
    assert len(table.rows) == expected_rows
    if expected_header[0] == "column_1":
        assert table.rows[0] == grid[0]


def test_writer_excludes_self_evidence_without_losing_source_or_cells(tmp_path):
    grid = [["For", "1,125"], ["Against", "65"], ["Abstain", "2"]]
    header, rows = _grid_from_table(docling_item(grid))
    raw = RawTable(page=1, header=header, rows=rows)
    source_text = render_table_text([], grid)
    parsed = ParsedDocument(
        doc_id="votes", source_path="synthetic.pdf", sha256="a" * 64, n_pages=1,
        parser="test", tables=[raw], elements=[Element("table", source_text, 1)])

    def must_not_ask(_prompts):
        pytest.fail("A table-only page has no independent prose to check")

    path = tmp_path / "votes.db"
    report = ingest("synthetic.pdf", path, parse=lambda _: parsed,
                    prose_checker=must_not_ask)
    assert report.tables[0].status == "unchecked"
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT column_1,column_2 FROM t_votes_001').fetchall() == [
            ("For", 1125), ("Against", 65), ("Abstain", 2)]
        assert "For | 1,125" in conn.execute('SELECT text FROM doc_text').fetchone()[0]
        assert conn.execute('SELECT COUNT(*) FROM cells').fetchone()[0] == 6


def test_prompt_identifies_claim_and_distinguishes_absence_from_conflict():
    prompts = []

    def unclear(batch):
        prompts.extend(batch)
        return [None] * len(batch)

    raw = RawTable(page=3, header=["Region", "Revenue"], rows=[["North", "$1.25"]],
                   caption="Revenue in millions")
    result = validate_table(raw, prose_checker=unclear,
                            page_texts={3: "The board approved the accounts."})
    assert assign_table_status(result, 0.7) == "unchecked"
    detail = result.checks["prose"].details[0]
    assert detail["applicable"] is False and detail["skipped"] == "unclear_or_no_support"
    prompt = prompts[0]
    for required in ("North", "Revenue", "$1.25", "1250000", "1000000",
                     "NO only for explicit conflicting", "UNCLEAR if", "not evidence"):
        assert required in prompt

    conflict = validate_table(raw, prose_checker=lambda ps: [False] * len(ps),
                              page_texts={3: "North revenue was $9 million."})
    assert conflict.checks["prose"].applicable == 1
    assert conflict.checks["prose"].passed == 0
    assert assign_table_status(conflict, 0.7) == "untrusted"


def test_relevant_page_and_late_claim_survive_bounded_context():
    prompts = []

    def check(batch):
        prompts.extend(batch)
        return [False] * len(batch)

    raw = RawTable(page=2, header=["Region", "Revenue"], rows=[["Remote division", "10"]])
    page = "Unrelated background. " * 1000 + "Remote division revenue was 999 dollars."
    result = validate_table(raw, prose_checker=check,
                            page_texts={1: "Earlier page. " * 2000, 2: page})
    assert "Remote division revenue was 999 dollars." in prompts[0]
    assert prompts[0].index("[Page 2]") < prompts[0].index("[Page 1]")
    assert len(prompts[0]) < 13500
    assert assign_table_status(result, 0.7) == "untrusted"


def test_pipeline_still_counts_independent_contradictions(tmp_path):
    raw = RawTable(page=1, header=["Region", "Revenue"], rows=[["North", "10"]])
    parsed = ParsedDocument(
        doc_id="conflict", source_path="synthetic.pdf", sha256="b" * 64, n_pages=1,
        parser="test", tables=[raw], elements=[
            Element("table", "Region | Revenue\nNorth | 10", 1),
            Element("text", "North revenue was 999 dollars.", 1)])

    def contradict(prompts):
        assert len(prompts) == 1
        prose = prompts[0].split("Table claim to check")[0]
        assert "999 dollars" in prose
        assert "North | 10" not in prose
        return [False]

    report = ingest("synthetic.pdf", tmp_path / "conflict.db", parse=lambda _: parsed,
                    prose_checker=contradict)
    assert report.tables[0].status == "untrusted"
    assert report.validation_pass_rate == 0
