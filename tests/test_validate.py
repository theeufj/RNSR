"""validate.py: arithmetic/structural/prose checks, confidence, §9 rollback."""

from rnsr.ingest.model import RawTable
from rnsr.ingest.validate import validate_table


def _table(rows, header=None, **kw):
    return RawTable(
        page=1,
        header=header or ["Item", "Amount"],
        rows=rows,
        extractor="docling",
        **kw,
    )


class TestArithmetic:
    def test_good_totals_pass(self):
        v = validate_table(_table([
            ["Widgets", "$1,234"],
            ["Gadgets", "$2,000"],
            ["Total", "$3,234"],
        ]))
        arith = v.checks["arithmetic"]
        assert arith.applicable >= 1
        assert arith.passed == arith.applicable
        assert v.confidence >= 0.9

    def test_corrupted_total_fails(self):
        v = validate_table(_table([
            ["Widgets", "$1,234"],
            ["Gadgets", "$2,000"],
            ["Total", "$9,999"],
        ]))
        arith = v.checks["arithmetic"]
        assert arith.passed < arith.applicable
        assert v.confidence < 0.7  # below re-extraction threshold

    def test_tolerance_absorbs_rounding(self):
        v = validate_table(_table([
            ["A", "10.4"], ["B", "10.4"], ["Total", "20.7"],  # off by 0.1 <= 1 unit
        ]))
        assert v.checks["arithmetic"].passed == v.checks["arithmetic"].applicable

    def test_subtotal_segments(self):
        v = validate_table(_table([
            ["A", "10"], ["B", "20"], ["Subtotal", "30"],
            ["C", "5"], ["D", "5"], ["Total", "40"],
        ]))
        arith = v.checks["arithmetic"]
        assert arith.applicable == 2
        assert arith.passed == 2

    def test_separate_totals_restart_groups(self):
        v = validate_table(_table([
            ["A", "10"], ["B", "20"], ["Total", "30"],
            ["C", "5"], ["D", "5"], ["Total", "10"],
        ]))
        assert v.checks["arithmetic"].applicable == v.checks["arithmetic"].passed == 2

    def test_subtotal_then_adjustment_checks_grand_total(self):
        rows = [["A", "1234.5"], ["B", "800"], ["Subtotal", "2034.5"],
                ["Adjustment", "-34.5"], ["Total", "2000"]]
        good = validate_table(_table(rows)).checks["arithmetic"]
        assert good.applicable == good.passed == 2
        rows[-1][1] = "2034.5"
        bad = validate_table(_table(rows)).checks["arithmetic"]
        assert bad.applicable == 2 and bad.passed == 1

    def test_no_total_rows_not_applicable(self):
        v = validate_table(_table([["A", "10"], ["B", "20"]]))
        assert v.checks["arithmetic"].applicable == 0
        assert v.confidence >= 0.7  # no evidence against; structural carries it

    def test_percent_sums_to_100(self):
        v = validate_table(_table(
            [["A", "45%"], ["B", "55%"], ["Total", "100%"]],
            header=["Item", "Share %"],
        ))
        arith = v.checks["arithmetic"]
        pct = [d for d in arith.details if d.get("check") == "pct_sums_to_100"]
        assert pct and pct[0]["passed"]


class TestStyleRollback:
    def test_eu_column_recovered_by_checksum(self):
        # Majority of cells look like US thousands ("1,234"), so style voting
        # picks US — but the sums only work under the EU reading (§9 hazard).
        v = validate_table(_table([
            ["A", "1,234"],
            ["B", "2,346"],
            ["Total", "3,58"],
        ]))
        arith = v.checks["arithmetic"]
        assert arith.passed == arith.applicable, arith.details
        assert v.style_overrides.get("amount") == "eu"

    def test_us_column_gets_no_override(self):
        v = validate_table(_table([
            ["A", "1,000"], ["B", "2,000"], ["Total", "3,000"],
        ]))
        assert v.style_overrides == {}


class TestStructural:
    def test_empty_table_with_declared_columns_is_not_malformed(self):
        from rnsr.ingest.validate import assign_table_status
        v = validate_table(_table([]))
        assert not v.structural_errors
        assert assign_table_status(v, 0.7) == "unchecked"

    def test_repeated_header_in_body_flagged(self):
        v = validate_table(_table([
            ["A", "1"],
            ["Item", "Amount"],  # symptom of a missed multi-page merge
            ["B", "2"],
        ]))
        s = v.checks["structural"]
        repeats = next(d for d in s.details if d["check"] == "no_header_repeats")
        assert not repeats["passed"]

    def test_monotonic_years_pass(self):
        v = validate_table(_table(
            [["2020", "1"], ["2021", "2"], ["2022", "3"]],
            header=["Year", "Value"],
        ))
        mono = [d for d in v.checks["structural"].details if d["check"] == "monotonic_dates"]
        assert mono and mono[0]["passed"]

    def test_shuffled_years_fail(self):
        v = validate_table(_table(
            [["2022", "1"], ["2020", "2"], ["2021", "3"]],
            header=["Year", "Value"],
        ))
        mono = [d for d in v.checks["structural"].details if d["check"] == "monotonic_dates"]
        assert mono and not mono[0]["passed"]


class TestProse:
    def test_prose_checker_wired(self):
        calls = []

        def ask(prompts):
            calls.extend(prompts)
            return [True] * len(prompts)

        v = validate_table(
            _table([["Widgets", "$1,234"], ["Gadgets", "$2,000"], ["Total", "$3,234"]]),
            prose_checker=ask,
            page_texts={1: "Revenue was $1,234 for widgets and $2,000 for gadgets."},
            prose_cells=2,
        )
        assert len(calls) == 2
        assert "Does the prose above state or imply" in calls[0]
        assert v.checks["prose"].applicable == 2
        assert v.checks["prose"].passed == 2

    def test_contradiction_lowers_confidence(self):
        base = validate_table(
            _table([["Widgets", "$1,234"], ["Gadgets", "$2,000"], ["Total", "$3,234"]]),
        )
        contradicted = validate_table(
            _table([["Widgets", "$1,234"], ["Gadgets", "$2,000"], ["Total", "$3,234"]]),
            prose_checker=lambda ps: [False] * len(ps),
            page_texts={1: "irrelevant"},
        )
        assert contradicted.confidence < base.confidence

    def test_unclear_answers_not_applicable(self):
        v = validate_table(
            _table([["A", "1"], ["B", "2"], ["Total", "3"]]),
            prose_checker=lambda ps: [None] * len(ps),
            page_texts={1: "text"},
        )
        assert v.checks["prose"].applicable == 0

    def test_skipped_without_checker(self):
        v = validate_table(_table([["A", "1"], ["B", "2"], ["Total", "3"]]))
        assert v.checks["prose"].applicable == 0


class TestApplicability:
    def test_account_balances_and_price_adjustment_amounts_remain_additive(self):
        for header, rows in [
            (["Account", "Balance"], [["Cash", "10"], ["Savings", "20"], ["Total", "30"]]),
            (["Component", "Amount"], [["Asset cost", "100"],
                                       ["Purchase price adjustment", "-10"], ["Total", "90"]]),
            (["Item", "Total price"], [["A", "100"], ["B", "200"], ["Total", "300"]]),
        ]:
            raw = _table(rows, header=header)
            good = validate_table(raw).checks["arithmetic"]
            assert good.applicable == good.passed == 1
            raw.rows[-1][-1] = "999"
            bad = validate_table(raw).checks["arithmetic"]
            assert bad.applicable == 1 and bad.passed == 0

    def test_rollforward_balance_snapshots_are_not_summed(self):
        raw = _table([["Opening balance", "100"], ["Cash movement", "20"],
                      ["Closing balance", "120"], ["Total", "120"]])
        assert validate_table(raw).checks["arithmetic"].applicable == 0

    def test_prices_balances_and_identifiers_are_not_summed(self):
        raw = _table([
            ["Jan", "1", "10", "3.50", "100"],
            ["Feb", "2", "20", "4.00", "80"],
            ["Total", "3", "30", "3.83", "80"],
        ], header=["Month", "Record ID", "Shares purchased", "Average price", "Remaining balance"])
        arithmetic = validate_table(raw).checks["arithmetic"]
        assert arithmetic.applicable == arithmetic.passed == 1
        assert {x["column"] for x in arithmetic.details if x.get("passed") is not None} == {2}
        raw.rows[-1][2] = "999"
        bad = validate_table(raw).checks["arithmetic"]
        assert bad.applicable == 1 and bad.passed == 0

    def test_growth_rates_are_not_shares(self):
        v = validate_table(_table([["A", "10%"], ["B", "20%"], ["Total", "15%"]],
                                  header=["Segment", "Organic growth %"]))
        assert v.checks["arithmetic"].applicable == 0

    def test_remaining_repurchase_authorization_is_a_snapshot(self):
        v = validate_table(_table([["April", "100"], ["May", "80"], ["Total May", "80"]],
                                  header=["Period", "Maximum Approximate Dollar Value of Shares "
                                          "that May Yet Be Purchased under the Plans or Programs"]))
        assert v.checks["arithmetic"].applicable == 0

    def test_total_may_is_a_date_label_not_a_prose_modal(self):
        v = validate_table(_table([["A", "10"], ["B", "20"], ["Total May 2018", "30"]]))
        assert v.checks["arithmetic"].applicable == v.checks["arithmetic"].passed == 1

    def test_mixed_metric_summary_is_not_an_assets_checksum(self):
        v = validate_table(_table([
            ["Net sales", "32765"], ["Net income", "5349"],
            ["Earnings per share", "8.89"], ["Dividends per 3M common share", "5.44"],
            ["Total assets", "36500"],
        ], header=["Metric", "2018"]))
        assert v.checks["arithmetic"].applicable == 0

    def test_exhibit_identifiers_do_not_generate_numeric_prose_questions(self):
        def ask(_):
            raise AssertionError("Identifiers are not quantities")
        v = validate_table(_table([["4.1", "Agreement A"], ["4.2", "Agreement B"]],
                                  header=["Exhibit", "Description"]), prose_checker=ask)
        assert v.checks["prose"].applicable == 0

    def test_structural_failure_cannot_be_trusted_by_passing_prose(self):
        from rnsr.ingest.validate import assign_table_status
        raw = _table([[None, "Rent", "1234.5"], [None, "Power", "800"],
                      [None, "Subtotal", "2034.5"], [None, "Adjustment", "-34.5"],
                      [None, "Total", "2000"]],
                     header=["NORDIC GmbH", "Ledger Konto", "Extract Betrag"])
        v = validate_table(raw, prose_checker=lambda ps: [True] * len(ps))
        assert v.checks["arithmetic"].applicable == v.checks["arithmetic"].passed == 2
        assert v.structural_errors
        assert assign_table_status(v, 0.7) == "untrusted"
