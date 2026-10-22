"""The certificate nudge: a numeric final that is neither quoted nor certified goes back once."""
from rnsr.harness.loop import derived_without_certificate


def _lex(quote):
    return {"check": "lexical_source_match", "passed": True, "quotes": [{"quote": quote, "matched": True}]}


def test_derived_without_certificate_flags_only_unquoted_uncertified_numbers():
    final = {"value": {"a": "7.9%, the mean of the two rates", "b": "$1,615.9 million (balance sheet)",
                       "c": "Yes, three segments", "d": "42", "e": "NOT_FOUND"},
             "verification": {"a": _lex("rates were 12.6 and 5.7"),
                              "b": _lex("Trade receivables, net | 1,615.9"),
                              "c": _lex("three segments"),
                              "d": {"check": "source_bound_calculation", "passed": True, "quotes": []},
                              "e": _lex("")}}
    assert derived_without_certificate(final, ["a", "b", "c", "d", "e"]) == ["a"]
    assert derived_without_certificate({"value": "101.5%", "verification": _lex("4,713,500")}, None) == ["answer"]
    assert derived_without_certificate({"value": "101.5%", "verification": _lex("101.5")}, None) == []
    assert derived_without_certificate({"value": {"a": "12"}, "verification": {}}, ["a"]) == []   # no report: not ours to judge


def test_zero_is_an_absence_claim_not_a_derivation():
    assert derived_without_certificate({"value": "0 — no restructuring line item", "verification": _lex("Goodwill")}, None) == []
    assert derived_without_certificate({"value": "0.0%", "verification": _lex("x")}, None) == []


def test_headline_number_prefers_quantities_over_dates_and_years():
    from rnsr.harness.loop import _headline_number

    assert _headline_number("Amcor's FY2020 year-end (June 30, 2020) receivables were $1,615.9 million") == "1615.9"
    assert _headline_number("Block FY2016 working capital ratio = 1.73 (current assets / liabilities)") == "1.73"
    assert _headline_number("Amazon's FY2017 DPO is 93.86 days. Formula: 365 x AP") == "93.86"
    assert _headline_number("In 2022 the company reported 3 segments.") == "3"
    assert _headline_number("Gross margin is not a useful metric. It had 2 lines.") is None   # second sentence
