"""Layout spacers must remain source data without becoming false checksum failures."""
from copy import deepcopy

import pytest

from rnsr.ingest.model import RawTable
from rnsr.ingest.validate import assign_table_status, validate_table


@pytest.mark.parametrize("header", [
    ["", "Field"],
    ["This application was prepared by", "Field"],
])
def test_empty_leading_body_column_in_text_form_is_not_misalignment(header):
    raw = RawTable(page=1, header=header, rows=[["", "Solicitor"], ["", "Firm"]])
    original = deepcopy(raw)
    validation = validate_table(raw)
    assert not validation.structural_errors
    assert assign_table_status(validation, 0.7) == "unchecked"
    assert raw == original


def test_blank_spacer_preserves_numeric_checks_and_source_cells():
    raw = RawTable(page=1, header=["", "Item", "Amount"],
                   rows=[["", "A", "10"], ["", "B", "20"], ["", "Total", "30"]])
    original = deepcopy(raw)
    validation = validate_table(raw)
    assert not validation.structural_errors
    assert assign_table_status(validation, 0.7) == "trusted"
    assert validation.checks["arithmetic"].applicable == 1
    assert validation.checks["arithmetic"].passed == 1
    assert raw == original

    raw.rows[-1][-1] = "999"
    validation = validate_table(raw)
    assert validation.checks["arithmetic"].applicable == 1
    assert validation.checks["arithmetic"].passed == 0
    assert assign_table_status(validation, 0.7) == "untrusted"


@pytest.mark.parametrize("rows", [
    [["", "A", "overflow"]],
    [["", "Field"]],
])
def test_layout_spacer_does_not_hide_width_or_repeated_header_error(rows):
    validation = validate_table(RawTable(page=1, header=["", "Field"], rows=rows))
    assert validation.structural_errors
    assert assign_table_status(validation, 0.7) == "untrusted"
