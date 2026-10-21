"""Dense matter generator: golds are exact and internally consistent."""
from __future__ import annotations

from decimal import Decimal

from rnsr.eval.datasets.matter_dense import Facts, fy


def test_fy_boundaries():
    from datetime import date

    assert fy(date(2024, 6, 30)) == 2024
    assert fy(date(2024, 7, 1)) == 2025


def test_facts_consistent():
    f = Facts(seed=3, n_months=36, invoices_per_month=3)
    live = f.live()
    assert live and all(i["status"] in ("issued", "disputed") for i in live)
    # Every cancelled invoice has exactly one credit note and one replacement.
    cancelled = [i for i in f.invoices if i["status"] == "cancelled"]
    assert len(cancelled) == len(f.credit_notes) == sum(1 for i in f.invoices if i["replaces"])
    for cn in f.credit_notes:
        orig = next(i for i in f.invoices if i["no"] == cn["against"])
        assert cn["amount"] == Facts.total(orig)
    # Arithmetic: subtotal + GST == total, GST is 10% to the cent.
    for i in f.invoices:
        assert Facts.subtotal(i) + Facts.gst(i) == Facts.total(i)
        assert Facts.gst(i) == (Facts.subtotal(i) * Decimal("0.10")).quantize(Decimal("0.01"))
    # Remittances only pay live, non-disputed invoices, each at most once.
    paid = [no for r in f.remittances for no in r["invoices"]]
    assert len(paid) == len(set(paid))
    by_no = {i["no"]: i for i in f.invoices}
    assert all(by_no[no]["status"] == "issued" for no in paid)
    # Outstanding at the end equals live invoices not paid.
    expected = sum((Facts.total(i) for i in live if i["no"] not in f.paid), Decimal(0))
    assert f.outstanding_at(f.as_at) == expected
    # Exactly two rate-error invoices, all dated after the variation.
    err = [i for i in f.invoices if i["rate_error"]]
    assert len(err) == 2 and all(i["date"] >= f.sow_amend_date for i in err)
    assert all(ln["rate"] == f.rates_0["Senior Consultant"] for i in err for ln in i["lines"]
               if ln["role"] == "Senior Consultant")
