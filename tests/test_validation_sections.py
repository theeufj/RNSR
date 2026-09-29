"""Checksums follow table structure, never a grouping selected to fit values."""

import pytest

from rnsr.ingest.model import RawTable
from rnsr.ingest.validate import assign_table_status, validate_table


def validate(rows):
    return validate_table(RawTable(page=1, header=["Item", "Amount"],
                                   rows=rows, extractor="docling"))


@pytest.mark.parametrize("finance_total,passed", [(85272, 1), (99999, 0)])
def test_section_total_excludes_prior_operating_cost(finance_total, passed):
    result = validate([
        ["Operating lease cost", "1986853"],
        ["Finance lease costs", None],
        ["Interest expense", "9233"], ["Amortization expense", "76039"],
        ["Total finance lease costs", str(finance_total)],
    ]).checks["arithmetic"]
    assert result.applicable == 1
    assert result.passed == passed
    assert result.details[0]["rows"] == [2, 3]


@pytest.mark.parametrize("cash_total,grand_total,passed", [
    (18001, 302707, 3), (18001, 302800, 3),
    (19000, 303706, 2), (18001, 999999, 2),
])
def test_unlabelled_subtotals_replace_their_components(cash_total, grand_total, passed):
    result = validate([
        ["Cash and cash equivalents:", None],
        ["Money market funds", "12009"], ["Certificates", "5992"],
        ["Cash and cash equivalents", str(cash_total)],
        ["Short-term investments:", None],
        ["Government securities", "56835"], ["Agency securities", "9530"],
        ["Certificates", "4466"], ["Corporate bonds", "213875"],
        ["Short-term investments", "284706"],
        ["Total debt investments", str(grand_total)],
    ]).checks["arithmetic"]
    assert result.applicable == 3
    assert result.passed == passed
    assert result.details[-1]["rows"] == [3, 9]


@pytest.mark.parametrize("grand_total,passed", [(50, 4), (999, 3)])
def test_nested_named_sections_do_not_double_count(grand_total, passed):
    result = validate([
        ["Assets:", None], ["Current assets:", None],
        ["Cash", "10"], ["Receivables", "20"], ["Current assets", "30"],
        ["Noncurrent assets:", None], ["Plant", "5"], ["Equipment", "5"],
        ["Total noncurrent assets", "10"], ["Total assets", "40"],
        ["Adjustment", "10"], ["Grand total", str(grand_total)],
    ]).checks["arithmetic"]
    assert result.applicable == 4
    assert result.passed == passed
    assert result.details[2]["rows"] == [4, 8]
    assert result.details[3]["rows"] == [9, 10]


def test_explicit_independent_totals_can_feed_grand_total():
    result = validate([
        ["A", "10"], ["B", "20"], ["Total first", "30"],
        ["C", "4"], ["D", "6"], ["Total second", "10"],
        ["Grand total", "40"],
    ]).checks["arithmetic"]
    assert result.applicable == result.passed == 3
    assert result.details[-1]["rows"] == [2, 5]


def test_grand_total_replaces_its_children_for_later_adjustments():
    result = validate([
        ["A", "10"], ["B", "20"], ["Total first", "30"],
        ["C", "4"], ["D", "6"], ["Total second", "10"],
        ["Grand total", "40"], ["Adjustment", "2"], ["Grand total", "42"],
    ]).checks["arithmetic"]
    assert result.applicable == result.passed == 4
    assert result.details[-1]["rows"] == [6, 7]


@pytest.mark.parametrize("grand_total,passed", [(60, 4), (999, 3)])
def test_sibling_sections_with_generic_subtotals_feed_grand_total(grand_total, passed):
    result = validate([
        ["Resorts:", None], ["A", "10"], ["B", "20"], ["Subtotal", "30"],
        ["Regional:", None], ["C", "4"], ["D", "6"], ["Subtotal", "10"],
        ["Overseas:", None], ["E", "8"], ["F", "12"], ["Subtotal", "20"],
        ["Grand total", str(grand_total)],
    ]).checks["arithmetic"]
    assert result.applicable == 4 and result.passed == passed
    assert result.details[-1]["rows"] == [3, 7, 11]


def test_generic_subtotal_inside_a_section_does_not_close_it_early():
    result = validate([
        ["Expenses:", None], ["A", "10"], ["B", "20"], ["Subtotal", "30"],
        ["Adjustment", "5"], ["Total expenses", "35"],
    ]).checks["arithmetic"]
    assert result.applicable == result.passed == 2
    assert result.details[-1]["rows"] == [3, 4]


@pytest.mark.parametrize("rows", [
    [["First section", None], ["A", "10"], ["B", "20"],
     ["Other section", None], ["C", "3"], ["D", "7"], ["Total", "40"]],
    [["A", "10"], ["B", "20"], ["", "30"], ["Total", "30"]],
])
def test_ambiguous_scope_is_unchecked_not_false_failure(rows):
    result = validate(rows)
    assert result.checks["arithmetic"].applicable == 0
    assert any(d.get("skipped", "").startswith("ambiguous_")
               for d in result.checks["arithmetic"].details)
    assert assign_table_status(result, 0.7) == "unchecked"


def test_section_label_match_does_not_depend_on_amount():
    good = validate([["Items:", None], ["A", "10"], ["B", "20"],
                     ["Items", "30"]])
    bad = validate([["Items:", None], ["A", "10"], ["B", "20"],
                    ["Items", "999"]])
    assert good.checks["arithmetic"].passed == 1
    assert bad.checks["arithmetic"].applicable == 1
    assert bad.checks["arithmetic"].passed == 0
