"""Coercion (§3.2): normalization, the >=95% rule, style detection, hazards."""

from hypothesis import given
from hypothesis import strategies as st

from rnsr.ingest.coerce import (
    CoercedColumn,
    coerce_cell,
    coerce_column,
    detect_style,
    is_null_cell,
)


class TestCoerceCell:
    def test_plain(self):
        assert coerce_cell("1234") == 1234.0

    def test_thousands(self):
        assert coerce_cell("1,234,567") == 1_234_567.0

    def test_currency(self):
        assert coerce_cell("$1,234.56") == 1234.56
        assert coerce_cell("€500") == 500.0

    def test_parens_negative(self):
        assert coerce_cell("(1,234)") == -1234.0
        assert coerce_cell("($42.50)") == -42.50

    def test_percent(self):
        assert coerce_cell("45%") == 45.0
        assert coerce_cell("12.5%") == 12.5

    def test_unicode_minus(self):
        assert coerce_cell("−42") == -42.0
        assert coerce_cell("–7.5") == -7.5

    def test_currency_before_sign(self):
        assert coerce_cell("$-5") == -5.0

    def test_european_style(self):
        assert coerce_cell("1.234,56", style="eu") == 1234.56
        assert coerce_cell("1.234", style="eu") == 1234.0

    def test_non_numeric(self):
        assert coerce_cell("Widgets") is None
        assert coerce_cell("12 apples") is None
        assert coerce_cell("1.2.3") is None

    @given(st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False))
    def test_roundtrip_us_formatting(self, x):
        formatted = f"{x:,.2f}"
        if x < 0:
            formatted = f"({formatted.lstrip('-')})"
        got = coerce_cell(formatted)
        assert got is not None
        assert abs(got - round(x, 2)) < 1e-6

    @given(st.integers(min_value=0, max_value=10**12))
    def test_roundtrip_currency(self, n):
        assert coerce_cell(f"${n:,}") == float(n)


class TestNullCells:
    def test_dashes_and_markers(self):
        for tok in ("-", "–", "—", "", "  ", "n/a", "N/A", "nm", "nil"):
            assert is_null_cell(tok)

    def test_numbers_are_not_null(self):
        assert not is_null_cell("0")
        assert not is_null_cell("-1")


class TestDetectStyle:
    def test_us(self):
        assert detect_style(["1,234.56", "2,000.00", "15.5"]) == "us"

    def test_eu(self):
        assert detect_style(["1.234,56", "2.000,00", "12,5"]) == "eu"

    def test_ambiguous_defaults_us(self):
        # "1,234" alone is ambiguous (§9 hazard) — conservative default is US;
        # the checksum pass owns the rollback if that call is wrong.
        assert detect_style(["1,234", "5,678"]) == "us"


class TestCoerceColumn:
    def test_clean_numeric_integer(self):
        col = coerce_column(["1,234", "5,678", "-", "9"])
        assert col.sql_type == "INTEGER"
        assert col.values == [1234, 5678, None, 9]
        assert col.rule is not None and col.rule.style == "us"

    def test_decimals_give_real(self):
        col = coerce_column(["1.5", "2.25"])
        assert col.sql_type == "REAL"
        assert col.values == [1.5, 2.25]

    def test_below_threshold_stays_text(self):
        raw = ["100", "200", "three hundred", "400"]  # 75% < 95%
        col = coerce_column(raw)
        assert col.sql_type == "TEXT"
        assert col.values == raw
        assert col.rule is None

    def test_threshold_boundary(self):
        raw = ["1"] * 19 + ["x"]  # exactly 95%
        assert coerce_column(raw).sql_type == "INTEGER"
        raw = ["1"] * 18 + ["x", "y"]  # 90%
        assert coerce_column(raw).sql_type == "TEXT"

    def test_all_null_column_is_text(self):
        col = coerce_column(["-", "—", None, ""])
        assert col.sql_type == "TEXT"
        assert col.coverage == 0.0

    def test_nulls_not_counted_against_threshold(self):
        col = coerce_column(["1", "-", "-", "-", "2"])  # 2/2 non-null coerce
        assert col.is_numeric

    def test_features_recorded(self):
        col = coerce_column(["$1,234", "($567)", "$89"])
        assert col.rule is not None
        assert "currency" in col.rule.features
        assert "parens_negative" in col.rule.features
        assert col.values == [1234, -567, 89]

    def test_percent_column(self):
        col = coerce_column(["45%", "30%", "25%"])
        assert col.is_numeric
        assert col.rule is not None and "percent" in col.rule.features
        assert sum(v for v in col.values if v is not None) == 100

    def test_eu_column_end_to_end(self):
        col = coerce_column(["1.234,50", "2.000,00", "500,25"])
        assert col.sql_type == "REAL"
        assert col.rule is not None and col.rule.style == "eu"
        assert col.values == [1234.5, 2000.0, 500.25]

    def test_forced_style_override(self):
        # The §9 rollback path re-coerces with an explicit style.
        col = coerce_column(["1,234", "5,678"], style="eu")
        assert col.values == [1.234, 5.678]

    @given(st.lists(st.integers(min_value=-10**9, max_value=10**9), min_size=1, max_size=30))
    def test_integer_lists_always_integer(self, xs):
        col = coerce_column([f"{x:,}" for x in xs])
        assert col.sql_type == "INTEGER"
        assert col.values == xs

    @given(st.lists(st.text(alphabet="abcdefg hij", min_size=1), min_size=1, max_size=20))
    def test_pure_text_never_numeric(self, xs):
        col: CoercedColumn = coerce_column(xs)
        assert col.sql_type == "TEXT"
